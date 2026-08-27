"""Exact boundary snapshots for the shot-buffered agent runtime.

The adapter is intentionally explicit about its inventory.  It can be used by the real
pipeline and by small deterministic test backends.  Snapshot data stays in memory; this is
not a checkpoint format and makes no cross-version serialization promise.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from utils.agent_runtime import TrustPartitionedMemory, fingerprint


def clone_state(value: Any) -> Any:
    """Recursively clone tensor-bearing state without changing device or dtype."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {clone_state(key): clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    if isinstance(value, set):
        return {clone_state(item) for item in value}
    if dataclasses.is_dataclass(value):
        return copy.deepcopy(value)
    try:
        return copy.deepcopy(value)
    except Exception:
        # CUDA events/streams and locks are not checkpointable.  A real active
        # shot-buffered backend must surface their pending logical state through
        # ``vae_pending_state`` rather than passing those handles here.
        return value


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    else:
        state["cuda"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"].clone())
    cuda_states = list(state.get("cuda", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("snapshot contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA RNG state count changed ({len(cuda_states)} != {torch.cuda.device_count()})"
            )
        torch.cuda.set_rng_state_all([item.clone() for item in cuda_states])


def _capture_scheduler(scheduler: Any) -> dict[str, Any]:
    if scheduler is None:
        return {"kind": "none", "state": None}
    if hasattr(scheduler, "state_dict") and callable(scheduler.state_dict):
        return {
            "kind": "state_dict",
            "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "state": clone_state(scheduler.state_dict()),
        }
    return {
        "kind": "dict",
        "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "state": clone_state(vars(scheduler)),
    }


def _restore_scheduler(scheduler: Any, snapshot: Mapping[str, Any]) -> None:
    kind = snapshot.get("kind")
    if kind == "none":
        if scheduler is not None:
            raise RuntimeError("scheduler appeared after a no-scheduler snapshot")
        return
    if scheduler is None:
        raise RuntimeError("cannot restore scheduler state into None")
    expected = snapshot.get("class")
    actual = f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
    if expected != actual:
        raise RuntimeError(f"scheduler class changed ({expected} != {actual})")
    if kind == "state_dict":
        scheduler.load_state_dict(clone_state(snapshot["state"]))
    elif kind == "dict":
        vars(scheduler).clear()
        vars(scheduler).update(clone_state(snapshot["state"]))
    else:
        raise RuntimeError(f"unsupported scheduler snapshot kind {kind!r}")


@dataclass
class BoundarySnapshot:
    shot_index: int
    committed_output_prefix: torch.Tensor
    current_shot_noise: torch.Tensor
    kv_cache_pos: Any
    kv_cache_neg: Any
    crossattn_cache_pos: Any
    crossattn_cache_neg: Any
    kv_rag_pos: Any
    kv_rag_neg: Any
    trust_memory_partitions: Any
    current_start_frame: int
    cache_start_frame: int
    rope_temporal_offset: float
    scheduler_state: Any
    rng_state: Any
    streaming_vae_pending_state: Any
    streaming_vae_output_cursor: int
    extra_indices: Mapping[str, Any]
    inventory: Mapping[str, bool]

    def state_fingerprint(self) -> str:
        return fingerprint(self)


REQUIRED_INVENTORY = frozenset(
    {
        "committed_output_latent",
        "current_shot_noise",
        "kv_cache_pos",
        "kv_cache_neg",
        "crossattn_cache_pos",
        "crossattn_cache_neg",
        "kv_rag_transient",
        "kv_rag_candidate",
        "kv_rag_persistent",
        "kv_rag_quarantine",
        "shot_index",
        "rope_offset",
        "scheduler",
        "cpu_rng",
        "cuda_rng",
        "streaming_vae_pending",
        "streaming_vae_output_cursor",
    }
)


class PipelineBoundaryAdapter:
    """Capture/restore all mutable state needed to regenerate one shot exactly.

    ``output`` and ``noise`` are caller-owned tensors.  Only the committed prefix and
    current shot noise slice are copied, preventing a snapshot from seeing any future
    generated output.  The current shot's authored noise is permitted because generation
    consumes it at this boundary.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        output: torch.Tensor,
        noise: torch.Tensor,
        trust_memory: TrustPartitionedMemory | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.output = output
        self.noise = noise
        self.trust_memory = trust_memory

    @staticmethod
    def _kv_rag_state(bank: Any) -> Any:
        if bank is None:
            return None
        if hasattr(bank, "snapshot_state"):
            return bank.snapshot_state()
        return clone_state(vars(bank))

    @staticmethod
    def _restore_kv_rag(bank: Any, state: Any) -> None:
        if state is None:
            if bank is not None:
                raise RuntimeError("KV-RAG bank appeared after snapshot")
            return
        if bank is None:
            raise RuntimeError("KV-RAG bank disappeared before restore")
        if hasattr(bank, "restore_state"):
            bank.restore_state(state)
            return
        vars(bank).clear()
        vars(bank).update(clone_state(state))

    def capture(
        self,
        *,
        shot_index: int,
        committed_end_frame: int,
        shot_noise_start: int,
        shot_noise_end: int,
        current_start_frame: int,
        cache_start_frame: int,
        scheduler: Any = None,
        vae_pending_state: Any = None,
        vae_output_cursor: int = 0,
        extra_indices: Mapping[str, Any] | None = None,
    ) -> BoundarySnapshot:
        if not 0 <= int(committed_end_frame) <= int(self.output.shape[1]):
            raise ValueError("committed output prefix is outside output tensor")
        if not 0 <= int(shot_noise_start) <= int(shot_noise_end) <= int(self.noise.shape[1]):
            raise ValueError("current shot noise slice is outside noise tensor")
        rag_pos = self._kv_rag_state(getattr(self.pipeline, "kv_rag_pos", None))
        rag_neg = self._kv_rag_state(getattr(self.pipeline, "kv_rag_neg", None))
        trust = None if self.trust_memory is None else self.trust_memory.snapshot_state()
        rng = capture_rng_state()
        inventory = {name: True for name in REQUIRED_INVENTORY}
        inventory["cuda_rng"] = bool(torch.cuda.is_available())
        snapshot = BoundarySnapshot(
            shot_index=int(shot_index),
            committed_output_prefix=self.output[:, :committed_end_frame].detach().clone(),
            current_shot_noise=self.noise[:, shot_noise_start:shot_noise_end].detach().clone(),
            kv_cache_pos=clone_state(getattr(self.pipeline, "kv_cache_pos", None)),
            kv_cache_neg=clone_state(getattr(self.pipeline, "kv_cache_neg", None)),
            crossattn_cache_pos=clone_state(getattr(self.pipeline, "crossattn_cache_pos", None)),
            crossattn_cache_neg=clone_state(getattr(self.pipeline, "crossattn_cache_neg", None)),
            kv_rag_pos=rag_pos,
            kv_rag_neg=rag_neg,
            trust_memory_partitions=clone_state(trust),
            current_start_frame=int(current_start_frame),
            cache_start_frame=int(cache_start_frame),
            rope_temporal_offset=float(
                getattr(getattr(self.pipeline, "_dit_model", self.pipeline), "rope_temporal_offset", 0.0)
            ),
            scheduler_state=_capture_scheduler(scheduler),
            rng_state=rng,
            streaming_vae_pending_state=clone_state(vae_pending_state),
            streaming_vae_output_cursor=int(vae_output_cursor),
            extra_indices=clone_state(dict(extra_indices or {})),
            inventory=inventory,
        )
        missing = REQUIRED_INVENTORY.difference(snapshot.inventory)
        if missing:
            raise AssertionError(f"boundary snapshot inventory missing {sorted(missing)}")
        return snapshot

    def restore(
        self,
        snapshot: BoundarySnapshot,
        *,
        shot_noise_start: int,
        shot_noise_end: int,
        scheduler: Any = None,
    ) -> dict[str, Any]:
        prefix_len = int(snapshot.committed_output_prefix.shape[1])
        self.output[:, :prefix_len].copy_(snapshot.committed_output_prefix)
        # Clear provisional/future output.  This is essential: verifier-rejected
        # frames must not remain readable after rollback.
        self.output[:, prefix_len:].zero_()
        self.noise[:, shot_noise_start:shot_noise_end].copy_(snapshot.current_shot_noise)
        self.pipeline.kv_cache_pos = clone_state(snapshot.kv_cache_pos)
        self.pipeline.kv_cache_neg = clone_state(snapshot.kv_cache_neg)
        self.pipeline.crossattn_cache_pos = clone_state(snapshot.crossattn_cache_pos)
        self.pipeline.crossattn_cache_neg = clone_state(snapshot.crossattn_cache_neg)
        self._restore_kv_rag(getattr(self.pipeline, "kv_rag_pos", None), snapshot.kv_rag_pos)
        self._restore_kv_rag(getattr(self.pipeline, "kv_rag_neg", None), snapshot.kv_rag_neg)
        if self.trust_memory is not None and snapshot.trust_memory_partitions is not None:
            self.trust_memory.restore_state(snapshot.trust_memory_partitions)
        dit = getattr(self.pipeline, "_dit_model", self.pipeline)
        dit.rope_temporal_offset = float(snapshot.rope_temporal_offset)
        _restore_scheduler(scheduler, snapshot.scheduler_state)
        restore_rng_state(snapshot.rng_state)
        return {
            "shot_index": snapshot.shot_index,
            "current_start_frame": snapshot.current_start_frame,
            "cache_start_frame": snapshot.cache_start_frame,
            "vae_pending_state": clone_state(snapshot.streaming_vae_pending_state),
            "vae_output_cursor": snapshot.streaming_vae_output_cursor,
            "extra_indices": clone_state(snapshot.extra_indices),
        }
