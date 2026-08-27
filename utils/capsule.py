"""Typed-memory capsule probe and in-place V editor.

The feature is inert unless ``capsule_probe.enabled`` or
``capsule_edit.enabled`` is set.  It deliberately operates outside the compiled
flash/flex attention kernels: cross-attention masks use a bounded-head einsum
side path, and clean-recache V capture/cache edits happen in the eager cache
update/boundary code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True) or {})
    except ImportError:
        pass
    if isinstance(value, dict):
        return dict(value)
    return dict(vars(value))


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _ints(value: Any, default: Iterable[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(x) for x in default)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(x) for x in value)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in value)


@dataclass(frozen=True)
class CapsuleProbeConfig:
    enabled: bool = False
    output_dir: str = "videos/capsule_probe"
    layers: tuple[int, ...] = (0, 7, 14, 21, 29)
    frame_stride: int = 4
    store_dtype: str = "float16"
    mask_method: str = "cross_attention"
    mask_path: str | None = None
    mask_layers: tuple[int, ...] = (14,)
    mask_max_heads: int = 4
    mask_top_fraction: float = 0.20
    mask_min_relative: float = 0.25
    mask_largest_component: bool = False
    mask_dilation: int = 1
    subject_phrases: tuple[str, ...] = ()
    allow_missing_subject: bool = False
    save_soft_masks: bool = True
    fail_open: bool = False

    @classmethod
    def from_config(cls, value: Any) -> "CapsuleProbeConfig":
        cfg = _plain_dict(value)
        return cls(
            enabled=_bool(cfg.get("enabled"), False),
            output_dir=str(cfg.get("output_dir", cls.output_dir)),
            layers=_ints(cfg.get("layers"), cls.layers),
            frame_stride=max(1, int(cfg.get("frame_stride", cls.frame_stride))),
            store_dtype=str(cfg.get("store_dtype", cls.store_dtype)).lower(),
            mask_method=str(cfg.get("mask_method", cls.mask_method)).lower(),
            mask_path=(
                None if cfg.get("mask_path") is None else str(cfg.get("mask_path"))
            ),
            mask_layers=_ints(cfg.get("mask_layers"), cls.mask_layers),
            mask_max_heads=max(1, int(cfg.get("mask_max_heads", cls.mask_max_heads))),
            mask_top_fraction=float(cfg.get("mask_top_fraction", cls.mask_top_fraction)),
            mask_min_relative=float(cfg.get("mask_min_relative", cls.mask_min_relative)),
            mask_largest_component=_bool(
                cfg.get("mask_largest_component"), cls.mask_largest_component
            ),
            mask_dilation=max(0, int(cfg.get("mask_dilation", cls.mask_dilation))),
            subject_phrases=_strings(cfg.get("subject_phrases")),
            allow_missing_subject=_bool(cfg.get("allow_missing_subject"), False),
            save_soft_masks=_bool(cfg.get("save_soft_masks"), cls.save_soft_masks),
            fail_open=_bool(cfg.get("fail_open"), cls.fail_open),
        )


@dataclass(frozen=True)
class CapsuleEditConfig:
    enabled: bool = False
    output_dir: str = "videos/capsule_edits"
    arm: str = "capsule"
    artifact_path: str | None = None
    layer: int = 14
    layers: tuple[int, ...] = ()
    rank: int = 1
    alpha: float = 0.5
    boundaries: tuple[int, ...] = ()
    random_seed: int = 0
    mask_method: str = "cross_attention"
    mask_path: str | None = None
    mask_layers: tuple[int, ...] = (14,)
    mask_max_heads: int = 4
    mask_top_fraction: float = 0.20
    mask_min_relative: float = 0.25
    mask_largest_component: bool = False
    mask_dilation: int = 1
    subject_phrases: tuple[str, ...] = ()
    allow_missing_subject: bool = False
    fail_open: bool = False

    @classmethod
    def from_config(cls, value: Any) -> "CapsuleEditConfig":
        cfg = _plain_dict(value)
        artifact = cfg.get("artifact_path")
        return cls(
            enabled=_bool(cfg.get("enabled"), False),
            output_dir=str(cfg.get("output_dir", cls.output_dir)),
            arm=str(cfg.get("arm", cls.arm)).lower().replace("-", "_"),
            artifact_path=None if artifact is None else str(artifact),
            layer=int(cfg.get("layer", cls.layer)),
            layers=_ints(cfg.get("layers"), ()),
            rank=max(1, min(16, int(cfg.get("rank", cls.rank)))),
            alpha=float(cfg.get("alpha", cls.alpha)),
            boundaries=_ints(cfg.get("boundaries"), ()),
            random_seed=int(cfg.get("random_seed", cls.random_seed)),
            mask_method=str(cfg.get("mask_method", cls.mask_method)).lower(),
            mask_path=(
                None if cfg.get("mask_path") is None else str(cfg.get("mask_path"))
            ),
            mask_layers=_ints(cfg.get("mask_layers"), cls.mask_layers),
            mask_max_heads=max(1, int(cfg.get("mask_max_heads", cls.mask_max_heads))),
            mask_top_fraction=float(cfg.get("mask_top_fraction", cls.mask_top_fraction)),
            mask_min_relative=float(cfg.get("mask_min_relative", cls.mask_min_relative)),
            mask_largest_component=_bool(
                cfg.get("mask_largest_component"), cls.mask_largest_component
            ),
            mask_dilation=max(0, int(cfg.get("mask_dilation", cls.mask_dilation))),
            subject_phrases=_strings(cfg.get("subject_phrases")),
            allow_missing_subject=_bool(cfg.get("allow_missing_subject"), False),
            fail_open=_bool(cfg.get("fail_open"), cls.fail_open),
        )


def _largest_component(mask: torch.Tensor) -> torch.Tensor:
    """Keep the largest 4-connected component of one small CPU bool mask."""
    source = mask.to(device="cpu", dtype=torch.bool)
    h, w = source.shape
    seen = torch.zeros_like(source)
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not bool(source[y, x]) or bool(seen[y, x]):
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and bool(source[ny, nx])
                        and not bool(seen[ny, nx])
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) > len(best):
                best = component
    output = torch.zeros_like(source)
    for y, x in best:
        output[y, x] = True
    return output


def entity_mask_from_attention(
    soft: torch.Tensor,
    *,
    top_fraction: float,
    min_relative: float,
    largest_component: bool,
    dilation: int,
) -> torch.Tensor:
    """Convert ``[F,H,W]`` subject attention to a deterministic token mask."""
    if soft.dim() != 3:
        raise ValueError(f"expected [F,H,W] attention, got {tuple(soft.shape)}")
    top_fraction = min(1.0, max(1.0 / max(1, soft.shape[-1] * soft.shape[-2]), top_fraction))
    masks: list[torch.Tensor] = []
    for frame in soft.detach().float().cpu():
        frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = frame.min(), frame.max()
        norm = (frame - lo) / (hi - lo + 1e-12)
        quantile = torch.quantile(norm.flatten(), 1.0 - top_fraction)
        threshold = max(float(quantile), float(min_relative))
        selected = norm >= threshold
        if not bool(selected.any()):
            selected.view(-1)[int(torch.argmax(norm).item())] = True
        if largest_component:
            selected = _largest_component(selected)
        if dilation > 0:
            kernel = 2 * dilation + 1
            selected = torch.nn.functional.max_pool2d(
                selected[None, None].float(), kernel_size=kernel, stride=1, padding=dilation
            )[0, 0].bool()
        masks.append(selected)
    return torch.stack(masks)


class CapsuleRuntime:
    """Per-pipeline state for cross-attention masks, V dumps, and boundary edits."""

    def __init__(
        self,
        probe: CapsuleProbeConfig,
        edit: CapsuleEditConfig,
        *,
        frame_seqlen: int,
        grid_h: int,
        grid_w: int,
    ) -> None:
        self.probe = probe
        self.edit = edit
        self.frame_seqlen = int(frame_seqlen)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        if self.grid_h * self.grid_w != self.frame_seqlen:
            raise ValueError(
                "capsule token geometry mismatch: "
                f"{self.grid_h}*{self.grid_w}!={self.frame_seqlen}"
            )
        supported_mask_methods = {
            "cross_attention",
            "clipseg_precomputed",
            "clipseg_precomputed_with_manual_visibility_overrides",
        }
        if probe.enabled and probe.mask_method not in supported_mask_methods:
            raise ValueError(f"unsupported capsule_probe.mask_method={probe.mask_method!r}")
        if edit.enabled and edit.mask_method not in supported_mask_methods:
            raise ValueError(f"unsupported capsule_edit.mask_method={edit.mask_method!r}")
        if edit.enabled and edit.arm not in {"capsule", "full_vector", "random_subspace"}:
            raise ValueError(f"unsupported capsule_edit.arm={edit.arm!r}")
        if edit.enabled and not edit.artifact_path:
            raise ValueError("capsule_edit.enabled requires artifact_path")
        self._mask_method = edit.mask_method if edit.enabled else probe.mask_method
        self._mask_path = edit.mask_path if edit.enabled else probe.mask_path
        self._mask_layers = edit.mask_layers if edit.enabled else probe.mask_layers
        self._mask_max_heads = edit.mask_max_heads if edit.enabled else probe.mask_max_heads
        self._mask_top_fraction = (
            edit.mask_top_fraction if edit.enabled else probe.mask_top_fraction
        )
        self._mask_min_relative = (
            edit.mask_min_relative if edit.enabled else probe.mask_min_relative
        )
        self._mask_largest_component = (
            edit.mask_largest_component if edit.enabled else probe.mask_largest_component
        )
        self._mask_dilation = edit.mask_dilation if edit.enabled else probe.mask_dilation
        self._subject_phrases = edit.subject_phrases if edit.enabled else probe.subject_phrases
        self._allow_missing_subject = (
            edit.allow_missing_subject if edit.enabled else probe.allow_missing_subject
        )
        if not self._subject_phrases:
            raise ValueError(
                "capsule cross-attention masking requires non-empty subject_phrases"
            )
        self._precomputed_masks: dict[str, Any] | None = None
        if self._mask_method in {
            "clipseg_precomputed",
            "clipseg_precomputed_with_manual_visibility_overrides",
        }:
            if not self._mask_path:
                raise ValueError(
                    "clipseg_precomputed mask method requires mask_path"
                )
            payload = torch.load(self._mask_path, map_location="cpu", weights_only=False)
            binary = torch.as_tensor(payload.get("binary"), dtype=torch.bool)
            if binary.dim() != 3 or tuple(binary.shape[1:]) != (self.grid_h, self.grid_w):
                raise ValueError(
                    "precomputed binary masks must be [frames,grid_h,grid_w], got "
                    f"{tuple(binary.shape)}"
                )
            soft = torch.as_tensor(payload.get("soft", binary), dtype=torch.float32)
            if soft.shape != binary.shape:
                raise ValueError(
                    f"precomputed soft mask shape {tuple(soft.shape)} != {tuple(binary.shape)}"
                )
            self._precomputed_masks = {
                "binary": binary,
                "soft": soft,
                "metadata": payload.get("metadata", {}),
            }
        self._artifact_cpu: dict[str, Any] | None = None
        self._edit_layers = (
            tuple(dict.fromkeys(int(layer) for layer in edit.layers))
            if edit.layers
            else (int(edit.layer),)
        )
        if edit.enabled:
            self._artifact_cpu = torch.load(
                edit.artifact_path, map_location="cpu", weights_only=False
            )
            self._validate_artifact(self._artifact_cpu)
        self.sample_name = "sample"
        self.seed = 0
        self.prompts: list[str] = []
        self.sample_dir: Path | None = None
        self.runtime: dict[str, Any] = {}
        self._mask_sums: dict[int, torch.Tensor] = {}
        self._mask_count = 0
        self._current_soft_mask: torch.Tensor | None = None
        self._current_binary_mask: torch.Tensor | None = None
        self._saved_mask_chunks: set[int] = set()
        self._cache_masks: dict[int, torch.Tensor] = {}
        self._artifact_device: dict[str, Any] = {}
        self._started_at = 0.0
        self.stats: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self.probe.enabled or self.edit.enabled

    @property
    def fail_open(self) -> bool:
        return self.edit.fail_open if self.edit.enabled else self.probe.fail_open

    def _validate_artifact(self, artifact: dict[str, Any]) -> None:
        if "per_layer" in artifact:
            per_layer = artifact["per_layer"]
            if not isinstance(per_layer, dict):
                raise ValueError("capsule multi-layer artifact per_layer must be a mapping")
            missing_layers = [
                layer for layer in self._edit_layers if str(layer) not in per_layer
            ]
            if missing_layers:
                raise ValueError(
                    f"capsule multi-layer artifact is missing layers {missing_layers}"
                )
            for layer in self._edit_layers:
                self._validate_layer_artifact(per_layer[str(layer)], layer)
            return
        if len(self._edit_layers) != 1:
            raise ValueError(
                "capsule multi-layer edit requires an artifact with per_layer entries"
            )
        self._validate_layer_artifact(artifact, self._edit_layers[0])

    def _validate_layer_artifact(
        self, artifact: dict[str, Any], expected_layer: int
    ) -> None:
        required = {"layer", "rank", "donor"}
        if self.edit.arm == "capsule":
            required |= {"W", "W_inv", "U"}
        elif self.edit.arm == "random_subspace":
            required |= {"W", "W_inv", "random_R"}
        missing = sorted(required - set(artifact))
        if missing:
            raise ValueError(f"capsule artifact missing fields: {missing}")
        if int(artifact["layer"]) != int(expected_layer):
            raise ValueError(
                f"artifact layer {artifact['layer']} != configured layer {expected_layer}"
            )
        if int(artifact["rank"]) != self.edit.rank:
            raise ValueError(
                f"artifact rank {artifact['rank']} != configured rank {self.edit.rank}"
            )
        donor = torch.as_tensor(artifact["donor"])
        if donor.dim() != 2 or donor.shape[-1] != 128:
            raise ValueError(f"artifact donor must be [heads,128], got {tuple(donor.shape)}")
        heads = int(donor.shape[0])
        if not torch.isfinite(donor).all():
            raise ValueError("artifact donor contains non-finite values")
        for field in ("W", "W_inv"):
            if field not in required:
                continue
            matrix = torch.as_tensor(artifact[field])
            if tuple(matrix.shape) != (heads, 128, 128):
                raise ValueError(
                    f"artifact {field} must be [{heads},128,128], got {tuple(matrix.shape)}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError(f"artifact {field} contains non-finite values")
        basis_field = (
            "U" if self.edit.arm == "capsule"
            else "random_R" if self.edit.arm == "random_subspace"
            else None
        )
        if basis_field is not None:
            basis = torch.as_tensor(artifact[basis_field])
            expected = (heads, 128, self.edit.rank)
            if tuple(basis.shape) != expected:
                raise ValueError(
                    f"artifact {basis_field} must be {expected}, got {tuple(basis.shape)}"
                )
            if not torch.isfinite(basis).all():
                raise ValueError(f"artifact {basis_field} contains non-finite values")
        if self.edit.arm == "random_subspace" and int(
            artifact.get("random_seed", self.edit.random_seed)
        ) != int(self.edit.random_seed):
            raise ValueError(
                f"artifact random_seed {artifact.get('random_seed')} != configured "
                f"random_seed {self.edit.random_seed}"
            )

    def start_sample(self, name: str | None, seed: int, prompts: Iterable[str]) -> None:
        self.sample_name = str(name or "sample")
        if self.edit.enabled and self._artifact_cpu is not None:
            artifact_scene = self._artifact_cpu.get("scene")
            if artifact_scene is not None and str(artifact_scene) != self.sample_name:
                raise ValueError(
                    f"artifact scene {artifact_scene!r} != sample {self.sample_name!r}"
                )
        self.seed = int(seed)
        self.prompts = [str(x) for x in prompts]
        self.runtime = {}
        self._mask_sums.clear()
        self._mask_count = 0
        self._current_soft_mask = None
        self._current_binary_mask = None
        self._saved_mask_chunks.clear()
        self._cache_masks.clear()
        self._artifact_device.clear()
        self._started_at = time.perf_counter()
        self.stats = {
            "cross_attention_calls": 0,
            "dump_files": 0,
            "dump_tokens": 0,
            "mask_files": 0,
            "edits": [],
            "errors": [],
        }
        if self.probe.enabled:
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.sample_name)
            self.sample_dir = Path(self.probe.output_dir) / safe_name / f"seed{self.seed}"
            (self.sample_dir / "masks").mkdir(parents=True, exist_ok=True)
            (self.sample_dir / "v").mkdir(parents=True, exist_ok=True)
        elif self.edit.enabled:
            safe_name = "".join(
                c if c.isalnum() or c in "._-" else "_" for c in self.sample_name
            )
            self.sample_dir = Path(self.edit.output_dir) / safe_name / f"seed{self.seed}"
            (self.sample_dir / "masks").mkdir(parents=True, exist_ok=True)
        else:
            self.sample_dir = None

    def finalize_sample(self, error: str | None = None) -> None:
        if error:
            self.stats.setdefault("errors", []).append(str(error))
        self.stats["wall_seconds"] = time.perf_counter() - self._started_at
        if self.sample_dir is None:
            return
        manifest = {
            "schema_version": 1,
            "sample_name": self.sample_name,
            "seed": self.seed,
            "frame_seqlen": self.frame_seqlen,
            "latent_token_grid": [self.grid_h, self.grid_w],
            "pixel_to_token_mapping": (
                "flattened token = frame*880 + y*40 + x; for 704x1280 decoded frames, "
                "token (y,x) covers the corresponding 32x32 pixel cell"
            ),
            "prompts": self.prompts,
            "probe_config": asdict(self.probe),
            "edit_config": asdict(self.edit),
            "mask_provenance": (
                None
                if self._precomputed_masks is None
                else self._precomputed_masks.get("metadata", {})
            ),
            "stats": self.stats,
        }
        (self.sample_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def set_runtime_context(
        self,
        *,
        chunk_index: int | None,
        shot_index: int | None,
        phase: str | None,
    ) -> None:
        previous_chunk = self.runtime.get("chunk_index")
        self.runtime.update(
            {"chunk_index": chunk_index, "shot_index": shot_index, "phase": phase}
        )
        if chunk_index != previous_chunk:
            self._mask_sums.clear()
            self._mask_count = 0
            self._current_soft_mask = None
            self._current_binary_mask = None

    def set_chunk_prompt(self, prompt: str, tokenizer: Any) -> list[int]:
        """Resolve configured phrases to UMT5 token indices using offsets."""
        raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        encoded = raw_tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise RuntimeError("configured tokenizer does not return offset_mapping")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("configured tokenizer does not return attention_mask")
        lowered = prompt.lower()
        spans: list[tuple[int, int]] = []
        for phrase in self._subject_phrases:
            target = phrase.lower().strip()
            start = 0
            while target:
                found = lowered.find(target, start)
                if found < 0:
                    break
                spans.append((found, found + len(target)))
                start = found + len(target)
        indices: list[int] = []
        for idx, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                indices.append(idx)
        indices = sorted(set(indices))
        if not indices:
            if not self._allow_missing_subject:
                raise RuntimeError(
                    f"none of subject_phrases={self._subject_phrases!r} matched prompt"
                )
            self.runtime["subject_missing"] = True
        else:
            self.runtime["subject_missing"] = False
        self.runtime["prompt"] = prompt
        self.runtime["text_token_count"] = int(sum(int(x) for x in attention_mask))
        self.runtime["subject_token_indices"] = indices
        return indices

    def record_cross_attention(self, q: torch.Tensor, k: torch.Tensor, *, layer: int) -> None:
        """Record subject-word attention through a bounded-head side einsum."""
        if self._mask_method != "cross_attention":
            return
        if int(layer) not in self._mask_layers:
            return
        phase = str(self.runtime.get("phase") or "")
        if phase != "clean_recache":
            return
        indices = self.runtime.get("subject_token_indices") or []
        if not indices:
            if self.runtime.get("subject_missing") and self._allow_missing_subject:
                return
            raise RuntimeError("subject token indices were not set before cross-attention probe")
        expected_tokens = int(self.runtime.get("num_frames", 0)) * self.frame_seqlen
        if expected_tokens <= 0:
            expected_tokens = q.shape[1]
        if q.shape[1] != expected_tokens:
            raise RuntimeError(
                f"cross-attention query length {q.shape[1]} != expected {expected_tokens}"
            )
        heads = q.shape[2]
        take = min(heads, self._mask_max_heads)
        head_indices = torch.linspace(
            0, heads - 1, steps=take, device=q.device
        ).round().long().unique()
        q_sel = q.index_select(2, head_indices)
        valid_text_tokens = min(
            int(self.runtime.get("text_token_count", k.shape[1])), int(k.shape[1])
        )
        if valid_text_tokens <= 0:
            raise RuntimeError("cross-attention probe resolved zero valid text tokens")
        if max(indices) >= valid_text_tokens:
            raise RuntimeError(
                f"subject token index {max(indices)} exceeds valid text length "
                f"{valid_text_tokens}"
            )
        k_sel = k[:, :valid_text_tokens].index_select(2, head_indices)
        with torch.no_grad():
            logits = torch.einsum("bqhd,bkhd->bhqk", q_sel, k_sel)
            logits = logits * (1.0 / math.sqrt(max(1, q.shape[-1])))
            weights = torch.softmax(logits.float(), dim=-1)
            subject_indices = torch.tensor(indices, device=q.device, dtype=torch.long)
            subject = weights.index_select(-1, subject_indices).sum(dim=-1).mean(dim=1)
            frames = expected_tokens // self.frame_seqlen
            soft = subject[0].reshape(frames, self.grid_h, self.grid_w).detach().cpu()
        self._mask_sums[int(layer)] = soft
        self._mask_count = len(self._mask_sums)
        self._current_soft_mask = None
        self._current_binary_mask = None
        self.stats["cross_attention_calls"] += 1

    def _ensure_current_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._current_soft_mask is not None and self._current_binary_mask is not None:
            return self._current_soft_mask, self._current_binary_mask
        if self._mask_method in {
            "clipseg_precomputed",
            "clipseg_precomputed_with_manual_visibility_overrides",
        }:
            assert self._precomputed_masks is not None
            frames = int(self.runtime.get("num_frames", 0))
            start_frame = int(self.runtime.get("start_token", 0)) // self.frame_seqlen
            end_frame = start_frame + frames
            binary = self._precomputed_masks["binary"][start_frame:end_frame]
            soft = self._precomputed_masks["soft"][start_frame:end_frame]
            if binary.shape[0] != frames:
                raise RuntimeError(
                    f"precomputed mask range [{start_frame},{end_frame}) exceeds "
                    f"{self._precomputed_masks['binary'].shape[0]} frames"
                )
            self._current_soft_mask = soft
            self._current_binary_mask = binary
            self._save_current_mask(soft, binary)
            return soft, binary
        if self.runtime.get("subject_missing") and self._allow_missing_subject:
            frames = int(self.runtime.get("num_frames", 0))
            if frames <= 0:
                raise RuntimeError("num_frames is required for an explicit empty subject mask")
            soft = torch.zeros(frames, self.grid_h, self.grid_w)
            binary = torch.zeros_like(soft, dtype=torch.bool)
            self._current_soft_mask = soft
            self._current_binary_mask = binary
            self._save_current_mask(soft, binary)
            return soft, binary
        if not self._mask_sums:
            raise RuntimeError(
                "no cross-attention mask was captured for current clean-recache chunk"
            )
        soft = torch.stack(list(self._mask_sums.values())).mean(dim=0)
        binary = entity_mask_from_attention(
            soft,
            top_fraction=self._mask_top_fraction,
            min_relative=self._mask_min_relative,
            largest_component=self._mask_largest_component,
            dilation=self._mask_dilation,
        )
        self._current_soft_mask = soft
        self._current_binary_mask = binary
        self._save_current_mask(soft, binary)
        return soft, binary

    def _save_current_mask(self, soft: torch.Tensor, binary: torch.Tensor) -> None:
        if self.sample_dir is None:
            return
        chunk = int(self.runtime.get("chunk_index", -1))
        if chunk in self._saved_mask_chunks:
            return
        shot = int(self.runtime.get("shot_index", -1))
        start_frame = int(self.runtime.get("start_token", chunk * soft.shape[0] * self.frame_seqlen)) // self.frame_seqlen
        payload: dict[str, Any] = {
            "binary": binary.to(torch.bool),
            "frame_indices": torch.arange(start_frame, start_frame + soft.shape[0]),
            "chunk_index": chunk,
            "shot_index": shot,
            "subject_token_indices": list(self.runtime.get("subject_token_indices") or []),
            "subject_phrases": list(self._subject_phrases),
            "prompt": self.runtime.get("prompt"),
            "latent_token_grid": [self.grid_h, self.grid_w],
        }
        if self.probe.save_soft_masks:
            payload["soft"] = soft.to(torch.float16)
        path = self.sample_dir / "masks" / f"chunk_{chunk:03d}_shot_{shot:02d}.pt"
        torch.save(payload, path)
        self._saved_mask_chunks.add(chunk)
        self.stats["mask_files"] += 1

    def capture_clean_v(self, *, layer: int, new_v: torch.Tensor, update_info: dict[str, Any]) -> None:
        if not self.probe.enabled or int(layer) not in self.probe.layers:
            return
        if str(self.runtime.get("phase") or "") != "clean_recache":
            return
        self.runtime["start_token"] = int(update_info.get("current_start", 0))
        soft, binary = self._ensure_current_mask()
        frames = int(update_info.get("num_new_frames", soft.shape[0]))
        if new_v.shape[1] != frames * self.frame_seqlen:
            raise RuntimeError(
                f"clean V length {new_v.shape[1]} != {frames}*{self.frame_seqlen}"
            )
        start_frame = int(update_info.get("current_start", 0)) // self.frame_seqlen
        token_indices: list[torch.Tensor] = []
        frame_indices: list[torch.Tensor] = []
        spatial_indices: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        for local_frame in range(frames):
            global_frame = start_frame + local_frame
            if global_frame % self.probe.frame_stride != 0:
                continue
            spatial = binary[local_frame].flatten().nonzero(as_tuple=False).flatten()
            if spatial.numel() == 0:
                continue
            token_indices.append(spatial + local_frame * self.frame_seqlen)
            frame_indices.append(torch.full_like(spatial, global_frame))
            spatial_indices.append(spatial)
            scores.append(soft[local_frame].flatten().index_select(0, spatial))
        if not token_indices:
            return
        local_tokens = torch.cat(token_indices).to(device=new_v.device)
        selected = new_v[0].index_select(0, local_tokens).detach().cpu()
        dtype = torch.float16 if self.probe.store_dtype in {"fp16", "float16"} else torch.bfloat16
        selected = selected.to(dtype=dtype)
        chunk = int(self.runtime.get("chunk_index", -1))
        shot = int(self.runtime.get("shot_index", -1))
        layer_dir = self.sample_dir / "v" / f"layer_{int(layer):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / f"chunk_{chunk:03d}_shot_{shot:02d}.pt"
        torch.save(
            {
                "v": selected,
                "frame_indices": torch.cat(frame_indices),
                "spatial_indices": torch.cat(spatial_indices),
                "attention_scores": torch.cat(scores).to(torch.float16),
                "layer": int(layer),
                "chunk_index": chunk,
                "shot_index": shot,
                "start_token": int(update_info.get("current_start", 0)),
                "frame_seqlen": self.frame_seqlen,
                "latent_token_grid": [self.grid_h, self.grid_w],
                "source": "clean_recache_self_attention_v",
            },
            path,
        )
        self.stats["dump_files"] += 1
        self.stats["dump_tokens"] += int(selected.shape[0])

    def mirror_cache_update(
        self,
        *,
        layer: int,
        update_info: dict[str, Any],
        cache_tokens: int,
    ) -> None:
        """Mirror V-cache rolling/insertion for the entity-token mask."""
        if not self.edit.enabled or int(layer) not in self._edit_layers:
            return
        if str(self.runtime.get("phase") or "") != "clean_recache":
            return
        _, binary = self._ensure_current_mask()
        inserted = binary.flatten().to(torch.bool)
        new_v = update_info.get("new_v")
        if new_v is None or inserted.numel() != new_v.shape[1]:
            raise RuntimeError(
                f"entity mask length {inserted.numel()} != inserted V length "
                f"{None if new_v is None else new_v.shape[1]}"
            )
        mask = self._cache_masks.get(int(layer))
        if mask is None or mask.numel() != int(cache_tokens):
            mask = torch.zeros(int(cache_tokens), dtype=torch.bool)
        if update_info["action"] == "roll_and_insert":
            sink = int(update_info["sink_tokens"])
            rolled = int(update_info["num_rolled_tokens"])
            evicted = int(update_info["num_evicted_tokens"])
            mask[sink : sink + rolled] = mask[
                sink + evicted : sink + evicted + rolled
            ].clone()
        start = int(update_info["local_start_index"])
        end = int(update_info["local_end_index"])
        mask[start:end] = inserted
        self._cache_masks[int(layer)] = mask

    def _artifact_on(
        self, device: torch.device, layer: int
    ) -> dict[str, torch.Tensor]:
        key = f"{device}:{int(layer)}"
        if key in self._artifact_device:
            return self._artifact_device[key]
        assert self._artifact_cpu is not None
        source = (
            self._artifact_cpu["per_layer"][str(int(layer))]
            if "per_layer" in self._artifact_cpu
            else self._artifact_cpu
        )
        fields = ("donor", "W", "W_inv", "U", "random_R")
        converted: dict[str, Any] = {}
        for field in fields:
            if field in source:
                converted[field] = torch.as_tensor(
                    source[field], device=device, dtype=torch.float32
                )
        self._artifact_device[key] = converted
        return converted

    def edit_active_window(
        self,
        kv_cache: list[dict[str, Any]],
        *,
        boundary_shot: int,
        local_attn_frames: int,
        global_sink_frames: int,
    ) -> dict[str, Any] | None:
        """In-place edit entity V rows already present in the attended cache."""
        if not self.edit.enabled or int(boundary_shot) not in self.edit.boundaries:
            return None
        records = [
            self._edit_one_layer(
                kv_cache,
                layer=layer,
                boundary_shot=boundary_shot,
                local_attn_frames=local_attn_frames,
                global_sink_frames=global_sink_frames,
            )
            for layer in self._edit_layers
        ]
        if len(records) == 1:
            return records[0]
        return {
            "boundary_shot": int(boundary_shot),
            "layers": [int(record["layer"]) for record in records],
            "edited_tokens_by_layer": {
                str(record["layer"]): int(record["edited_tokens"])
                for record in records
            },
            "records": records,
        }

    def _edit_one_layer(
        self,
        kv_cache: list[dict[str, Any]],
        *,
        layer: int,
        boundary_shot: int,
        local_attn_frames: int,
        global_sink_frames: int,
    ) -> dict[str, Any]:
        cache = kv_cache[layer]
        if cache.get("quantized", False):
            raise RuntimeError("capsule_edit does not support quantized KV caches")
        values = cache["v"]
        mask = self._cache_masks.get(layer)
        if mask is None:
            raise RuntimeError("no entity cache mask is available at edit boundary")
        local_end = int(cache["local_end_index"].item())
        local_start = max(0, local_end - int(local_attn_frames) * self.frame_seqlen)
        active = torch.zeros_like(mask)
        active[local_start:local_end] = True
        global_tokens = max(0, int(global_sink_frames) * self.frame_seqlen)
        active[: min(global_tokens, local_end)] = True
        pinned_start = int(cache.get("pinned_start", torch.tensor(-1)).item())
        pinned_len = int(cache.get("pinned_len", torch.tensor(0)).item())
        if pinned_start >= 0 and pinned_len > 0:
            active[pinned_start : min(mask.numel(), pinned_start + pinned_len)] = True
        indices = (mask & active).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("entity mask selected zero active-window tokens at boundary")
        device_indices = indices.to(device=values.device)
        target = values[0].index_select(0, device_indices).float()
        artifact = self._artifact_on(values.device, layer)
        donor = artifact["donor"]
        if donor.shape != target.shape[1:]:
            raise RuntimeError(
                f"donor shape {tuple(donor.shape)} != V head shape {tuple(target.shape[1:])}"
            )
        delta = donor.unsqueeze(0) - target
        if self.edit.arm == "full_vector":
            update = delta
        else:
            W = artifact["W"]
            W_inv = artifact["W_inv"]
            basis = artifact["U"] if self.edit.arm == "capsule" else artifact["random_R"]
            delta_w = torch.einsum("hij,nhj->nhi", W, delta)
            coefficient = torch.einsum("hdr,nhd->nhr", basis, delta_w)
            projected_w = torch.einsum("hdr,nhr->nhd", basis, coefficient)
            update = torch.einsum("hij,nhj->nhi", W_inv, projected_w)
        edited = target + float(self.edit.alpha) * update
        values[0].index_copy_(0, device_indices, edited.to(dtype=values.dtype))
        record = {
            "boundary_shot": int(boundary_shot),
            "layer": int(layer),
            "rank": int(self.edit.rank),
            "arm": self.edit.arm,
            "alpha": float(self.edit.alpha),
            "edited_tokens": int(indices.numel()),
            "mean_target_norm": float(target.norm(dim=-1).mean().item()),
            "mean_update_norm": float(update.norm(dim=-1).mean().item()),
            "active_cache_range": [int(local_start), int(local_end)],
        }
        self.stats.setdefault("edits", []).append(record)
        return record

    def handle_error(self, where: str, exc: Exception) -> None:
        message = f"{where}: {type(exc).__name__}: {exc}"
        self.stats.setdefault("errors", []).append(message)
        if not self.fail_open:
            raise exc
