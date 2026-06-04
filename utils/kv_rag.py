# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""KV-cache retrieval memory for long-video causal inference.

The memory stores compact K/V slices emitted by clean recache calls and returns
top-k historical slices for later attention calls. It is deliberately independent
from the model modules so the feature can be disabled without touching the
baseline cache flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"enabled": value}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True))
    except Exception:
        pass
    if isinstance(value, dict):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    if not name:
        return None
    normalized = str(name).lower().replace("torch.", "")
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported KV-RAG store_dtype={name!r}")
    return mapping[normalized]


@dataclass(frozen=True)
class KVRAGConfig:
    enabled: bool = False
    top_k: int = 2
    max_entries: int = 32
    max_tokens_per_entry: int = 1024
    layers: tuple[int, ...] | None = None
    layer_stride: int = 1
    min_frame_gap: int = 0
    retrieve_during_denoise: bool = True
    retrieve_during_recache: bool = False
    store_after_recache: bool = True
    token_policy: str = "uniform"
    similarity: str = "cosine"
    store_on_cpu: bool = False
    store_dtype: str | None = None
    fail_open: bool = True
    verbose: bool = False
    # --- Phase 1: retrieval-signal fixes ---
    # Match content, not RoPE phase: summarize the pre-RoPE query/key tensors so
    # the same content at a different time still scores as similar.
    summary_prerope: bool = True
    # Keep per-head structure when scoring instead of flattening all heads into
    # one vector (which mixes independent head subspaces).
    summary_per_head: bool = True
    # --- Phase 2: in-distribution injection ---
    # Re-RoPE retrieved keys into a virtual frame block placed just before the
    # local window so query<->memory relative positions stay in-distribution.
    # When False, the legacy (stale absolute-RoPE) injection path is used.
    reinject_rope: bool = True
    # Store/downsample whole frames (a prerequisite for clean re-RoPE) instead
    # of arbitrary uniform token subsets.
    frame_aligned_store: bool = True

    @classmethod
    def from_config(cls, value: Any) -> "KVRAGConfig":
        cfg = _to_plain_dict(value)
        if not cfg:
            return cls()

        raw_layers = cfg.get("layers", None)
        layers: tuple[int, ...] | None
        if raw_layers is None or str(raw_layers).lower() == "all":
            layers = None
        elif isinstance(raw_layers, int):
            layers = (raw_layers,)
        else:
            layers = tuple(int(v) for v in raw_layers)

        return cls(
            enabled=_as_bool(cfg.get("enabled", False), False),
            top_k=max(0, _as_int(cfg.get("top_k", 2), 2)),
            max_entries=max(1, _as_int(cfg.get("max_entries", 32), 32)),
            max_tokens_per_entry=max(0, _as_int(cfg.get("max_tokens_per_entry", 1024), 1024)),
            layers=layers,
            layer_stride=max(1, _as_int(cfg.get("layer_stride", 1), 1)),
            min_frame_gap=max(0, _as_int(cfg.get("min_frame_gap", 0), 0)),
            retrieve_during_denoise=_as_bool(cfg.get("retrieve_during_denoise", True), True),
            retrieve_during_recache=_as_bool(cfg.get("retrieve_during_recache", False), False),
            store_after_recache=_as_bool(cfg.get("store_after_recache", True), True),
            token_policy=str(cfg.get("token_policy", "uniform")).lower(),
            similarity=str(cfg.get("similarity", "cosine")).lower(),
            store_on_cpu=_as_bool(cfg.get("store_on_cpu", False), False),
            store_dtype=cfg.get("store_dtype", None),
            fail_open=_as_bool(cfg.get("fail_open", True), True),
            verbose=_as_bool(cfg.get("verbose", False), False),
            summary_prerope=_as_bool(cfg.get("summary_prerope", True), True),
            summary_per_head=_as_bool(cfg.get("summary_per_head", True), True),
            reinject_rope=_as_bool(cfg.get("reinject_rope", True), True),
            frame_aligned_store=_as_bool(cfg.get("frame_aligned_store", True), True),
        )

    def layer_enabled(self, layer: int) -> bool:
        if self.layers is not None:
            return layer in self.layers
        return layer % self.layer_stride == 0

    def describe(self) -> str:
        layers = "all" if self.layers is None else ",".join(str(v) for v in self.layers)
        return (
            f"enabled={self.enabled}, top_k={self.top_k}, max_entries={self.max_entries}, "
            f"max_tokens_per_entry={self.max_tokens_per_entry}, layers={layers}, "
            f"layer_stride={self.layer_stride}, min_frame_gap={self.min_frame_gap}, "
            f"retrieve_during_denoise={self.retrieve_during_denoise}, "
            f"retrieve_during_recache={self.retrieve_during_recache}, "
            f"store_after_recache={self.store_after_recache}, "
            f"summary_prerope={self.summary_prerope}, summary_per_head={self.summary_per_head}, "
            f"reinject_rope={self.reinject_rope}, frame_aligned_store={self.frame_aligned_store}"
        )


@dataclass
class KVRAGEntry:
    layer: int
    k: torch.Tensor
    v: torch.Tensor
    summary: torch.Tensor
    start_token: int
    end_token: int
    chunk_index: int | None = None
    phase: str | None = None
    # Frame geometry of the stored slice. Populated when frame_aligned_store is
    # on; required to re-RoPE the slice at retrieval time. frames == 0 means the
    # slice is not frame-aligned and re-RoPE must fall back to legacy injection.
    frames: int = 0
    h: int = 0
    w: int = 0
    frame_seqlen: int = 0

    @property
    def num_tokens(self) -> int:
        return int(self.k.shape[1])


class KVRAGMemory:
    """Per-sample in-memory KV retrieval bank.

    Entries are grouped by transformer layer. Retrieval uses cosine similarity
    between the current query summary and stored key summaries, then prepends
    the selected K/V slices to the attention window.
    """

    def __init__(self, config: KVRAGConfig):
        self.config = config
        self.entries_by_layer: dict[int, list[KVRAGEntry]] = {}
        self.stats = {
            "stored_entries": 0,
            "retrieval_calls": 0,
            "retrieval_hits": 0,
            "retrieved_tokens": 0,
            # Phase 0 diagnostics: post-softmax attention mass on retrieved
            # tokens (sum over the rag columns, averaged over heads/query/batch).
            "rag_attention_mass": 0.0,
            "rag_mass_calls": 0,
        }
        self._warnings: set[str] = set()
        self._store_dtype = _dtype_from_name(config.store_dtype)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def fail_open(self) -> bool:
        return self.config.fail_open

    @property
    def reinject_rope(self) -> bool:
        return self.config.reinject_rope

    @property
    def diag_enabled(self) -> bool:
        return self.config.verbose

    @property
    def diag_layer(self) -> int:
        """Layer the attention-mass diagnostic is computed on (cheapest single
        enabled layer), so the O(window) softmax runs once per forward."""
        if self.config.layers is not None and len(self.config.layers) > 0:
            return min(self.config.layers)
        return 0

    def clear(self) -> None:
        self.entries_by_layer.clear()
        for key in self.stats:
            self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0
        self._warnings.clear()

    def warn_once(self, key: str, message: str) -> None:
        if key in self._warnings:
            return
        self._warnings.add(key)
        print(message)

    def add(
        self,
        *,
        layer: int,
        k_pre: torch.Tensor | None,
        k_post: torch.Tensor | None,
        v: torch.Tensor,
        start_token: int,
        end_token: int,
        frame_seqlen: int = 0,
        h: int = 0,
        w: int = 0,
        frames: int = 0,
        chunk_index: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Store one chunk's K/V slice.

        ``k_pre`` is the pre-RoPE key (used for re-RoPE injection and, by
        default, the retrieval summary); ``k_post`` is the cached post-RoPE key
        (legacy injection / summary source). The caller passes both so this
        method can pick per-config without the model needing to know the flags.
        """
        if not self.enabled or not self.config.layer_enabled(layer):
            return
        if v.numel() == 0:
            return

        with torch.no_grad():
            k_pre = None if k_pre is None else k_pre.detach()
            k_post = None if k_post is None else k_post.detach()
            v = v.detach()

            inject_src = k_pre if self.config.reinject_rope else k_post
            if inject_src is None:
                inject_src = k_pre if k_pre is not None else k_post
            if inject_src is None:
                return
            summary_src = k_pre if self.config.summary_prerope else k_post
            if summary_src is None:
                summary_src = inject_src

            idx, kept_frames = self._select_index(
                num_tokens=int(inject_src.shape[1]),
                frame_seqlen=int(frame_seqlen),
                frames=int(frames),
                device=inject_src.device,
            )
            if idx is not None:
                inject_src = inject_src.index_select(1, idx)
                v = v.index_select(1, idx)
                summary_src = summary_src.index_select(1, idx)

            summary = self._summarize(summary_src)
            if summary is None:
                return

            k_store = inject_src
            v_store = v
            if self._store_dtype is not None:
                k_store = k_store.to(dtype=self._store_dtype)
                v_store = v_store.to(dtype=self._store_dtype)
            if self.config.store_on_cpu:
                k_store = k_store.cpu()
                v_store = v_store.cpu()
                summary = summary.cpu()

            entry = KVRAGEntry(
                layer=layer,
                k=k_store.contiguous(),
                v=v_store.contiguous(),
                summary=summary.contiguous(),
                start_token=int(start_token),
                end_token=int(end_token),
                chunk_index=chunk_index,
                phase=phase,
                frames=int(kept_frames),
                h=int(h),
                w=int(w),
                frame_seqlen=int(frame_seqlen),
            )
            layer_entries = self.entries_by_layer.setdefault(layer, [])
            layer_entries.append(entry)
            while len(layer_entries) > self.config.max_entries:
                layer_entries.pop(0)
            self.stats["stored_entries"] += 1

    def retrieve(
        self,
        *,
        layer: int,
        query: torch.Tensor,
        current_start: int,
        frame_seqlen: int,
        dtype: torch.dtype,
        device: torch.device,
        use_relative_rope: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[KVRAGEntry]] | None:
        if not self.enabled or self.config.top_k <= 0:
            return None
        if not self.config.layer_enabled(layer):
            return None
        if use_relative_rope:
            self.warn_once(
                "relative_rope_disabled",
                "[KV-RAG][warn] Retrieval is skipped when use_relative_rope=true; "
                "relative-RoPE memory re-injection is not implemented in this path.",
            )
            return None

        entries = self.entries_by_layer.get(layer)
        if not entries:
            return None

        self.stats["retrieval_calls"] += 1
        min_end = int(current_start) - self.config.min_frame_gap * int(frame_seqlen)
        candidates = [entry for entry in entries if entry.end_token <= min_end]
        if not candidates:
            return None

        query_summary = self._summarize(query.detach())
        if query_summary is None:
            return None

        scores = self._score_batch(query_summary, candidates)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        selected = [candidates[i] for i in order[: self.config.top_k]]
        k_parts = [self._match_batch(entry.k, query.shape[0]).to(device=device, dtype=dtype) for entry in selected]
        v_parts = [self._match_batch(entry.v, query.shape[0]).to(device=device, dtype=dtype) for entry in selected]
        rag_k = torch.cat(k_parts, dim=1).contiguous()
        rag_v = torch.cat(v_parts, dim=1).contiguous()
        self.stats["retrieval_hits"] += 1
        self.stats["retrieved_tokens"] += int(rag_k.shape[1])
        return rag_k, rag_v, selected

    def format_stats(self, prefix: str = "KV-RAG") -> str:
        layers = sum(1 for entries in self.entries_by_layer.values() if entries)
        live_entries = sum(len(entries) for entries in self.entries_by_layer.values())
        mass_calls = self.stats["rag_mass_calls"]
        mass = self.stats["rag_attention_mass"] / mass_calls if mass_calls else 0.0
        return (
            f"[{prefix}] layers={layers}, live_entries={live_entries}, "
            f"stored_entries={self.stats['stored_entries']}, "
            f"retrieval_calls={self.stats['retrieval_calls']}, "
            f"retrieval_hits={self.stats['retrieval_hits']}, "
            f"retrieved_tokens={self.stats['retrieved_tokens']}, "
            f"rag_attn_mass={mass:.4f} (n={mass_calls})"
        )

    def _select_index(
        self,
        *,
        num_tokens: int,
        frame_seqlen: int,
        frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int]:
        """Pick which tokens of a chunk to store.

        Returns ``(idx, kept_frames)``. ``idx is None`` means keep every token.
        ``kept_frames > 0`` means the kept tokens form whole frames and can be
        re-RoPE'd at retrieval time; ``0`` marks a legacy (non-frame-aligned)
        token subset that must fall back to the legacy injection path.
        """
        limit = self.config.max_tokens_per_entry
        aligned = (
            self.config.frame_aligned_store
            and frame_seqlen > 0
            and frames > 0
            and frames * frame_seqlen == num_tokens
        )
        if aligned:
            if limit <= 0 or num_tokens <= limit:
                return None, frames
            max_frames = max(1, limit // frame_seqlen)
            if frames <= max_frames:
                return None, frames
            fsel = torch.linspace(0, frames - 1, steps=max_frames, device=device)
            fsel = fsel.round().to(torch.long).unique(sorted=True)
            offsets = torch.arange(frame_seqlen, device=device)
            idx = (fsel.view(-1, 1) * frame_seqlen + offsets.view(1, -1)).reshape(-1)
            return idx, int(fsel.numel())

        # Legacy token-level selection (slice is not re-RoPE'able).
        if limit <= 0 or num_tokens <= limit:
            return None, 0
        policy = self.config.token_policy
        if policy == "tail":
            idx = torch.arange(num_tokens - limit, num_tokens, device=device)
        elif policy == "first":
            idx = torch.arange(0, limit, device=device)
        elif policy == "uniform":
            idx = torch.linspace(0, num_tokens - 1, steps=limit, device=device)
            idx = idx.round().to(torch.long).unique(sorted=True)
        else:
            raise ValueError(f"Unsupported KV-RAG token_policy={policy!r}")
        return idx, 0

    def _summarize(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Mean-pool over tokens into a normalized content summary.

        With ``summary_per_head`` the per-head structure ``[B, H, D]`` is kept
        and normalized per head, so scoring does not mix independent head
        subspaces. Otherwise the heads are flattened into one vector.
        """
        if tensor.numel() == 0:
            return None
        if tensor.dim() == 4:  # [B, T, H, D]
            pooled = tensor.float().mean(dim=1)  # [B, H, D]
            if not self.config.summary_per_head:
                pooled = pooled.flatten(1)  # [B, H*D]
        elif tensor.dim() == 3:  # [B, T, D]
            pooled = tensor.float().mean(dim=1)  # [B, D]
        else:
            pooled = tensor.float().reshape(tensor.shape[0], -1)
        return F.normalize(pooled, dim=-1, eps=1e-6)

    def _score_batch(
        self, query_summary: torch.Tensor, candidates: list[KVRAGEntry]
    ) -> list[float]:
        """Score every candidate against the query in one batched op.

        Falls back to a per-entry loop if the stacked shapes are inconsistent.
        """
        try:
            stacked = torch.stack(
                [c.summary for c in candidates], dim=0
            ).to(device=query_summary.device, dtype=query_summary.dtype)
            q = query_summary.unsqueeze(0)  # [1, B, ...]
            if self.config.similarity == "cosine":
                # sum over feature dim, mean over any head/batch dims -> [N]
                prod = (stacked * q).sum(dim=-1)
                reduce_dims = tuple(range(1, prod.dim()))
                scores = prod.mean(dim=reduce_dims) if reduce_dims else prod
            elif self.config.similarity == "l2":
                diff = (stacked - q).pow(2)
                reduce_dims = tuple(range(1, diff.dim()))
                scores = -diff.mean(dim=reduce_dims)
            else:
                raise ValueError(
                    f"Unsupported KV-RAG similarity={self.config.similarity!r}"
                )
            return scores.tolist()
        except RuntimeError:
            return [self._score(query_summary, c.summary) for c in candidates]

    def _score(self, query_summary: torch.Tensor, entry_summary: torch.Tensor) -> float:
        entry_summary = entry_summary.to(device=query_summary.device, dtype=query_summary.dtype)
        if self.config.similarity == "cosine":
            return float((query_summary * entry_summary).sum(dim=-1).mean().item())
        if self.config.similarity == "l2":
            return float(-(query_summary - entry_summary).pow(2).mean().item())
        raise ValueError(f"Unsupported KV-RAG similarity={self.config.similarity!r}")

    def record_attention_mass(
        self,
        query: torch.Tensor,
        window_k: torch.Tensor,
        rag_tokens: int,
        max_query_rows: int = 32,
    ) -> None:
        """Phase 0 diagnostic: post-softmax attention mass landing on the
        retrieved memory columns (the first ``rag_tokens`` columns of
        ``window_k``). Verbose-only and best-effort; never raises.

        Only a handful of query rows are sampled so the full ``[B,H,Lq,Lk]``
        score matrix (which flash-attention never materializes) is never built
        at real video token counts.
        """
        try:
            with torch.no_grad():
                if rag_tokens <= 0 or window_k.shape[1] <= rag_tokens:
                    return
                lq = query.shape[1]
                if lq > max_query_rows:
                    rows = torch.linspace(
                        0, lq - 1, steps=max_query_rows, device=query.device
                    ).round().to(torch.long)
                    q = query.index_select(1, rows).float()  # [B, r, H, D]
                else:
                    q = query.float()
                k = window_k.float()  # [B, Lk, H, D]
                scale = 1.0 / math.sqrt(q.shape[-1])
                logits = torch.einsum("blhd,bmhd->bhlm", q, k) * scale
                weights = torch.softmax(logits, dim=-1)
                mass = weights[..., :rag_tokens].sum(dim=-1).mean()
                self.stats["rag_attention_mass"] += float(mass.item())
                self.stats["rag_mass_calls"] += 1
        except Exception:
            pass

    @staticmethod
    def _match_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            repeat_dims = [batch_size] + [1] * (tensor.dim() - 1)
            return tensor.repeat(*repeat_dims)
        raise ValueError(
            f"KV-RAG entry batch size {tensor.shape[0]} does not match current batch size {batch_size}"
        )
