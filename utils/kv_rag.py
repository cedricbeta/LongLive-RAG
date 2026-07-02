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

import json
import math
from dataclasses import dataclass
from pathlib import Path
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


#: Retrieval KEY representations: how a stored chunk is indexed for matching.
#:   pooled/moment/multi_centroid/salient_set/positional -> computed from the
#:     stored K/V attention tensors (see _compute_key);
#:   subject_identity -> identity-aware prototype on the K/V tensors (subject
#:     tokens minus background), decoupled from appearance;
#:   attention_native -> Quest-style query-aware key: stored entries keep
#:     per-layer/head min/max key bounds, and live queries score those bounds
#:     by the same q.k quantity attention consumes;
#:   semantic -> a DECOUPLED external embedding (the perspective's caption-text
#:     vector) set per-perspective via set_context_key(); indexes the same scene
#:     across views by what the prompt DESCRIBES rather than by token statistics.
KEY_MODES = (
    "pooled", "moment", "multi_centroid", "salient_set", "positional",
    "subject_identity", "attention_native", "semantic",
)
#: KEY modes that ignore the K/V tensors and require an external context vector
#: (set via set_context_key); using one without its companion fails fast.
CONTEXT_KEY_MODES = ("semantic",)
#: Retrieval VALUE representations: what payload is injected back into attention.
#:   raw -> the stored frame-aligned K/V slice;
#:   mean_frame -> collapsed to one mean frame (bounded, re-RoPE'able);
#:   top_frame -> the single most subject-salient (highest-norm) frame kept
#:     (bounded, re-RoPE'able) -- a compressed, subject-masked payload.
#:   attention_mass -> the single frame whose tokens received the most attention
#:     during the clean recache store path (H2O/SnapKV-style received-attention
#:     importance, not key-norm saliency).
VALUE_MODES = ("raw", "mean_frame", "top_frame", "attention_mass")


def virtual_frame_start(
    *,
    current_end: int,
    frame_seqlen: int,
    window_tokens: int,
    prefix_tokens: int,
    rag_frames: int,
) -> int:
    """Frame index where a re-RoPE'd retrieved/scene block is placed.

    The retrieved (or force-injected scene) tokens are dropped into a virtual
    frame block immediately *before* the current local window so the
    query<->memory relative positions stay in the range the model was trained
    on. The block spans ``[rag_start, rag_start + rag_frames)`` and the local
    window starts at ``rag_start + rag_frames``; the two never overlap.

    Args:
        current_end: absolute end token of the current chunk in the cache.
        frame_seqlen: tokens per frame (h*w after patchify).
        window_tokens: total tokens in the attended window (incl. any prefix).
        prefix_tokens: sink/pinned tokens prepended ahead of the local frames.
        rag_frames: number of whole frames being injected.
    """
    if frame_seqlen <= 0:
        raise ValueError("virtual_frame_start requires frame_seqlen > 0")
    local_frames = (window_tokens - prefix_tokens) // frame_seqlen
    local_start_frame = (current_end // frame_seqlen) - local_frames
    return max(0, local_start_frame - rag_frames)


@dataclass(frozen=True)
class KVRAGConfig:
    enabled: bool = False
    top_k: int = 2
    max_entries: int = 32
    # Legacy token cap. Kept for byte-identical behavior when new configs do
    # not opt into the frame-level contract.
    max_tokens_per_entry: int = 1024
    # New frame-level cap. None means use the legacy token cap above; 0 means
    # keep all whole frames from the stored chunk.
    max_frames_per_entry: int | None = None
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
    # Gate-only contract: refuse to store/inject token subsets that cannot be
    # represented as whole latent frames. Violations are counted in diagnostics
    # so the rendered gate can fail closed with a blocked_reason.
    require_frame_aligned: bool = False
    # --- single-scene multi-perspective consistency ---
    # A persistent "scene" partition that survives shot boundaries, holding the
    # establishing anchors of the environment so a new camera angle reads as the
    # same place rather than a freshly hallucinated scene. Disabled by default:
    # only the per-shot partition exists and behavior matches the legacy
    # single-bucket memory exactly.
    scene_memory_enabled: bool = False
    # Capacity (stored chunks per layer) of the persistent partition. The
    # per-shot partition keeps using ``max_entries``.
    scene_memory_max_entries: int = 8
    # Additive similarity bonus applied to persistent-partition candidates so the
    # shared scene is preferred over incidental per-shot matches. 0.0 = no bias.
    scene_score_bonus: float = 0.0
    # Number of persistent anchors force-injected at a shot boundary regardless
    # of the content match, re-RoPE'd through the normal retrieval-injection
    # path. 0 = off. Requires scene_memory_enabled.
    boundary_inject_anchors: int = 0
    # Optional JSON plan written by the VLM optimizer. When present, boundary
    # force-injection uses the requested decoded source frame indices instead of
    # the default recency heuristic. This stays on the same whole-frame payload
    # path; it only changes which persistent entries are selected.
    manual_anchor_plan_path: str | None = None
    # When true, every completed shot contributes clean-recache anchors to the
    # persistent scene partition. When false, only the initial shot seeds it.
    scene_memory_rolling: bool = False
    # "boundary" preserves the incumbent shot-cut pulse. "every_chunk" pulses
    # persistent entries during every denoise chunk, exposing dose/structure
    # without changing the retrieval representation.
    scene_memory_injection_schedule: str = "boundary"
    # Structured diagnostic: per shot/layer post-softmax attention mass received
    # by injected persistent entries during denoise. Disabled by default.
    attention_diagnostic: bool = False
    attention_diag_max_query_rows: int = 16
    # One-shot verdict lever: additive attention-logit bias on injected
    # persistent frame columns. Default 0.0 keeps the normal attention path.
    persistent_logit_bias_lambda: float = 0.0
    # --- decoupled retrieval key/value representations ---
    # How an entry is INDEXED: the lookup signal matched against the query.
    #   "pooled"        -> mean-pooled (orderless) summary [baseline]
    #   "moment"        -> mean+std summary (richer orderless signature)
    #   "multi_centroid"-> several region centroids (subject vs background) so
    #                      the same subject matches across camera angles
    #   "salient_set"   -> bounded set of the most salient tokens matched by
    #                      symmetric mutual-best cosine (Chamfer); finer than
    #                      centroids and robust to which view a token appears in
    #   "positional"    -> position-weighted pooling (view-sensitive; used to
    #                      demonstrate the viewpoint-invariance probe rejects a
    #                      bad key -- not recommended for production)
    #   "subject_identity" -> identity-aware prototype: the top-M highest-norm
    #                      (subject) tokens minus the per-head background mean,
    #                      L2-normalized; matches the SAME subject across views
    #                      independent of background, decoupled from appearance
    #   "attention_native" -> Quest-style query-aware key: stored entries keep
    #                      per-head min/max key bounds and live queries score
    #                      relevance by q.k against those bounds, with no
    #                      external embedding or ad-hoc summary
    #   "semantic"      -> a decoupled EXTERNAL embedding (caption-text vector)
    #                      supplied per-perspective via set_context_key(); indexes
    #                      the same scene by what the prompt describes. Requires a
    #                      context-key provider (fails fast without one).
    retrieval_key_mode: str = "pooled"
    # Number of region centroids when retrieval_key_mode == "multi_centroid".
    retrieval_key_centroids: int = 4
    # Number of salient tokens kept when retrieval_key_mode in
    # {"salient_set", "subject_identity"}.
    retrieval_key_top_m: int = 8
    # What is INJECTED back into attention (the payload), independent of the key.
    #   "raw"        -> the stored frame-aligned K/V slice [baseline]
    #   "mean_frame" -> the slice collapsed to a single representative frame
    #                   (bounded injected length; stays frame-aligned/re-RoPE'able)
    #   "top_frame"  -> the single most subject-salient (highest mean token-norm)
    #                   frame kept (bounded, re-RoPE'able); requires
    #                   frame_aligned_store (fails fast otherwise)
    #   "attention_mass" -> the single frame whose tokens received the most
    #                   attention during the clean recache store path; requires
    #                   frame_aligned_store and a received-attention vector
    retrieval_value_mode: str = "raw"

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
            max_frames_per_entry=(
                None
                if cfg.get("max_frames_per_entry", None) is None
                else max(0, _as_int(cfg.get("max_frames_per_entry", 0), 0))
            ),
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
            require_frame_aligned=_as_bool(cfg.get("require_frame_aligned", False), False),
            scene_memory_enabled=_as_bool(cfg.get("scene_memory_enabled", False), False),
            scene_memory_max_entries=max(1, _as_int(cfg.get("scene_memory_max_entries", 8), 8)),
            scene_score_bonus=float(cfg.get("scene_score_bonus", 0.0) or 0.0),
            boundary_inject_anchors=max(0, _as_int(cfg.get("boundary_inject_anchors", 0), 0)),
            manual_anchor_plan_path=cfg.get("manual_anchor_plan_path", None),
            scene_memory_rolling=_as_bool(cfg.get("scene_memory_rolling", False), False),
            scene_memory_injection_schedule=str(
                cfg.get("scene_memory_injection_schedule", "boundary")
            ).lower().replace("-", "_"),
            attention_diagnostic=_as_bool(cfg.get("attention_diagnostic", False), False),
            attention_diag_max_query_rows=max(
                1, _as_int(cfg.get("attention_diag_max_query_rows", 16), 16)
            ),
            persistent_logit_bias_lambda=max(
                0.0, float(cfg.get("persistent_logit_bias_lambda", 0.0) or 0.0)
            ),
            retrieval_key_mode=str(cfg.get("retrieval_key_mode", "pooled")).lower(),
            retrieval_key_centroids=max(1, _as_int(cfg.get("retrieval_key_centroids", 4), 4)),
            retrieval_key_top_m=max(1, _as_int(cfg.get("retrieval_key_top_m", 8), 8)),
            retrieval_value_mode=str(cfg.get("retrieval_value_mode", "raw")).lower(),
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
            f"max_frames_per_entry={self.max_frames_per_entry}, "
            f"layer_stride={self.layer_stride}, min_frame_gap={self.min_frame_gap}, "
            f"retrieve_during_denoise={self.retrieve_during_denoise}, "
            f"retrieve_during_recache={self.retrieve_during_recache}, "
            f"store_after_recache={self.store_after_recache}, "
            f"summary_prerope={self.summary_prerope}, summary_per_head={self.summary_per_head}, "
            f"reinject_rope={self.reinject_rope}, frame_aligned_store={self.frame_aligned_store}, "
            f"require_frame_aligned={self.require_frame_aligned}, "
            f"scene_memory_enabled={self.scene_memory_enabled}, "
            f"scene_memory_max_entries={self.scene_memory_max_entries}, "
            f"scene_score_bonus={self.scene_score_bonus}, "
            f"boundary_inject_anchors={self.boundary_inject_anchors}, "
            f"manual_anchor_plan_path={self.manual_anchor_plan_path}, "
            f"scene_memory_rolling={self.scene_memory_rolling}, "
            f"scene_memory_injection_schedule={self.scene_memory_injection_schedule}, "
            f"attention_diagnostic={self.attention_diagnostic}, "
            f"persistent_logit_bias_lambda={self.persistent_logit_bias_lambda}, "
            f"retrieval_key_mode={self.retrieval_key_mode}, "
            f"retrieval_key_centroids={self.retrieval_key_centroids}, "
            f"retrieval_key_top_m={self.retrieval_key_top_m}, "
            f"retrieval_value_mode={self.retrieval_value_mode}"
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
    source_frame_start: int | None = None
    source_frame_indices: tuple[int, ...] = ()
    # True when the entry lives in the persistent scene partition (survives shot
    # boundaries); False for ordinary per-shot entries.
    persistent: bool = False

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
        self._validate_config(config)
        self.entries_by_layer: dict[int, list[KVRAGEntry]] = {}
        # Persistent partition: establishing anchors of the scene that survive
        # shot boundaries. Always present but only populated when
        # scene_memory_enabled, so the disabled path is byte-identical to legacy.
        self.scene_entries_by_layer: dict[int, list[KVRAGEntry]] = {}
        # Transient flag set by the scheduler on the first chunk of a new shot so
        # retrieval force-injects the scene anchors at that boundary.
        self._boundary_inject = False
        # External context key for retrieval_key_mode == "semantic": the current
        # perspective's caption-text embedding, set per inference() call. None
        # until provided; using semantic mode without it fails fast.
        self._context_key: torch.Tensor | None = None
        # True when the persistent scene partition was carried across a per-video
        # TOKEN-CLOCK restart (the cross-perspective path: each perspective is a
        # separate inference() starting at token 0). The live-window overlap dedup
        # must NOT apply to those anchors -- their [start,end) coincides with the
        # new perspective's window only by numeric coincidence across DIFFERENT
        # videos, not a real double-count. Set by _reset_kv_rag_keep_scene.
        self._scene_cross_clock = False
        self._manual_anchor_plan, self._manual_anchor_plan_scenes = self._load_manual_anchor_plan(
            config.manual_anchor_plan_path
        )
        self.manual_anchor_events: list[dict[str, object]] = []
        # Current sample/scene identifier so per-scene manual plans resolve to the
        # right scene; set from the driver right before each sample's inference().
        self._scene_name: str | None = None
        self._runtime_context = {
            "chunk_index": None,
            "shot_index": None,
            "phase": None,
        }
        self.attention_mass_by_shot_layer: dict[str, dict[str, dict[str, float | int]]] = {}
        self.attention_mass_by_shot_layer_frame: dict[
            str, dict[str, dict[str, dict[str, float | int]]]
        ] = {}
        self.frame_alignment_drop_events: list[dict[str, int | str | bool | None]] = []
        self.stats = {
            "stored_entries": 0,
            "stored_scene_entries": 0,
            "retrieval_calls": 0,
            "retrieval_hits": 0,
            "retrieved_tokens": 0,
            "injection_calls": 0,
            "injected_tokens": 0,
            "injected_frames": 0,
            "persistent_injected_tokens": 0,
            "persistent_injected_frames": 0,
            "outside_window_injection_calls": 0,
            "outside_window_injected_tokens": 0,
            "outside_window_injected_frames": 0,
            "direct_injection_calls": 0,
            "direct_injected_frames": 0,
            "reinject_rope_injection_calls": 0,
            "reinject_rope_injected_frames": 0,
            "persistent_logit_bias_calls": 0,
            "persistent_logit_bias_tokens": 0,
            "persistent_logit_bias_frames": 0,
            "persistent_logit_bias_lambda_sum": 0.0,
            "persistent_logit_bias_skipped_unaligned": 0,
            "boundary_injections": 0,
            "manual_anchor_injections": 0,
            "manual_anchor_requested_frames": 0,
            "manual_anchor_matched_frames": 0,
            "frame_alignment_store_drops": 0,
            "frame_alignment_inject_drops": 0,
            # Phase 0 diagnostics: post-softmax attention mass on retrieved
            # tokens (sum over the rag columns, averaged over heads/query/batch).
            "rag_attention_mass": 0.0,
            "rag_mass_calls": 0,
            "persistent_rag_attention_mass": 0.0,
            "persistent_rag_mass_calls": 0,
        }
        self._warnings: set[str] = set()
        self._store_dtype = _dtype_from_name(config.store_dtype)

    @staticmethod
    def _parse_boundary_map(raw: object) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        if not isinstance(raw, dict):
            return out
        for boundary, values in raw.items():
            try:
                boundary_idx = int(boundary)
            except Exception:
                continue
            frames: list[int] = []
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        item = item.get("frame_index", item.get("decoded_frame_index"))
                    try:
                        frames.append(int(item))
                    except Exception:
                        continue
            if frames:
                out[boundary_idx] = sorted(set(f for f in frames if f >= 0))
        return out

    @classmethod
    def _load_manual_anchor_plan(
        cls, path_value: str | None
    ) -> tuple[dict[int, list[int]], dict[str, dict[int, list[int]]]]:
        """Parse a manual anchor plan file.

        Returns ``(global_plan, per_scene_plans)``. Legacy files carry one
        boundary->frames map under ``boundaries`` (or at top level); v2 files add
        ``scenes: {scene_name: {boundary: [frames]}}`` so each scene gets the
        anchors chosen FOR it instead of a cross-scene union.
        """
        if not path_value:
            return {}, {}
        path = Path(str(path_value)).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}, {}
        if not isinstance(data, dict):
            return {}, {}
        scenes: dict[str, dict[int, list[int]]] = {}
        for name, raw in (data.get("scenes") or {}).items() if isinstance(data.get("scenes"), dict) else []:
            parsed = cls._parse_boundary_map(raw)
            if parsed:
                scenes[str(name)] = parsed
        global_plan = cls._parse_boundary_map(data.get("boundaries", data))
        return global_plan, scenes

    @staticmethod
    def _validate_config(config: KVRAGConfig) -> None:
        """Fail fast on unknown representations or flags missing a companion."""
        if config.retrieval_key_mode not in KEY_MODES:
            raise ValueError(
                f"Unsupported KV-RAG retrieval_key_mode={config.retrieval_key_mode!r}; "
                f"expected one of {KEY_MODES}"
            )
        if config.retrieval_value_mode not in VALUE_MODES:
            raise ValueError(
                f"Unsupported KV-RAG retrieval_value_mode={config.retrieval_value_mode!r}; "
                f"expected one of {VALUE_MODES}"
            )
        if config.retrieval_key_mode == "multi_centroid" and config.retrieval_key_centroids < 2:
            raise ValueError(
                "KV-RAG retrieval_key_mode='multi_centroid' requires "
                "retrieval_key_centroids >= 2"
            )
        if config.retrieval_value_mode in {"top_frame", "attention_mass"} and not config.frame_aligned_store:
            # These modes select one whole frame from a frame-aligned slice;
            # without frame_aligned_store there are no frames to choose from.
            # Fail fast rather than silently fall back to raw.
            raise ValueError(
                f"KV-RAG retrieval_value_mode={config.retrieval_value_mode!r} requires "
                "frame_aligned_store=true (there is no frame to select otherwise)"
            )
        if config.require_frame_aligned and not config.frame_aligned_store:
            raise ValueError(
                "KV-RAG require_frame_aligned=true requires frame_aligned_store=true."
            )
        if config.boundary_inject_anchors > 0 and not config.scene_memory_enabled:
            raise ValueError(
                "KV-RAG boundary_inject_anchors > 0 requires scene_memory_enabled=true; "
                "there is no persistent partition to inject from otherwise"
            )
        if config.scene_memory_injection_schedule not in {"boundary", "every_chunk"}:
            raise ValueError(
                "KV-RAG scene_memory_injection_schedule must be 'boundary' or "
                f"'every_chunk', got {config.scene_memory_injection_schedule!r}"
            )
        if (config.scene_memory_rolling or config.scene_memory_injection_schedule == "every_chunk") \
                and not config.scene_memory_enabled:
            raise ValueError(
                "KV-RAG scene_memory_rolling/every_chunk schedule requires "
                "scene_memory_enabled=true."
            )

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
    def scene_memory_enabled(self) -> bool:
        return self.config.scene_memory_enabled

    def set_boundary_inject(self, active: bool) -> None:
        """Toggle force-injection of scene anchors for the next retrieval(s).

        The scheduler calls this on the first chunk of a new shot so the
        persistent scene anchors are re-injected at the perspective change.
        """
        self._boundary_inject = bool(active) and self.config.boundary_inject_anchors > 0

    def set_runtime_context(
        self,
        *,
        chunk_index: int | None = None,
        shot_index: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Record lightweight inference context for diagnostics."""
        self._runtime_context = {
            "chunk_index": chunk_index,
            "shot_index": shot_index,
            "phase": phase,
        }

    def set_scene_name(self, name: str | None) -> None:
        """Identify the sample being generated so per-scene manual plans apply.

        Survives clear()/reset between shots; only the next set_scene_name call
        replaces it. A None or unmatched name falls back to the global plan.
        """
        self._scene_name = str(name) if name else None

    def _manual_plan_for_current_scene(self) -> dict[int, list[int]]:
        if self._manual_anchor_plan_scenes:
            name = self._scene_name or ""
            if name in self._manual_anchor_plan_scenes:
                return self._manual_anchor_plan_scenes[name]
            # Sample names embed the prompt-folder scene token between '-'
            # delimiters (e.g. "kv_only-rank0-african_savanna-seed0"), so match
            # the token delimiter-bounded — plain containment would let a scene
            # named "cat" claim "cathedral" samples. Longest key first so a
            # scene whose name is a substring of another cannot shadow it.
            bounded = f"-{name}-"
            for scene in sorted(self._manual_anchor_plan_scenes, key=len, reverse=True):
                if scene and f"-{scene}-" in bounded:
                    return self._manual_anchor_plan_scenes[scene]
        return self._manual_anchor_plan

    def mark_scene_cross_clock(self) -> None:
        """Flag that the scene partition now spans a per-video token-clock restart.

        Called when the persistent scene anchors are carried into a NEW perspective
        (a separate ``inference()`` call restarting at token 0). After this, the
        live-window overlap dedup is skipped for scene anchors during retrieval, so
        perspective-0's establishing anchors are not spuriously filtered just
        because their token range coincides with the new perspective's window.
        """
        self._scene_cross_clock = True

    @property
    def requires_context_key(self) -> bool:
        """True when the active key mode indexes by an external context vector."""
        return self.config.retrieval_key_mode in CONTEXT_KEY_MODES

    @property
    def wants_received_attention(self) -> bool:
        """True when the value mode needs per-token received-attention mass."""
        return self.config.retrieval_value_mode == "attention_mass"

    def set_context_key(self, vector: torch.Tensor | None) -> None:
        """Set the external semantic key for the current perspective.

        For ``retrieval_key_mode == "semantic"`` the pipeline supplies the
        perspective's caption-text embedding here (once per ``inference()`` call):
        entries stored during this perspective are indexed by it, and later
        perspectives match against THEIR own caption embedding, so the same scene
        is retrieved by what the prompt describes -- decoupled from the K/V
        payload (which can stay ``raw``). Accepts any ``[..., D]`` tensor (e.g. a
        pre-pooled ``[D]`` vector or T5 ``prompt_embeds`` of shape
        ``[blocks, seq_len, D]``); ALL leading dimensions are mean-pooled while the
        last embedding dim ``D`` is preserved, so the key is the shape-independent
        pooled ``[1, D]`` caption vector. ``None`` clears it.
        """
        if vector is None:
            self._context_key = None
            return
        with torch.no_grad():
            v = vector.detach().float()
            if v.dim() == 0:
                v = v.reshape(1)
            elif v.dim() > 1:
                # Pool every leading dim (blocks, tokens, ...) into one [D] vector;
                # do NOT flatten seq into D (that would make the key seq-dependent).
                v = v.reshape(-1, v.shape[-1]).mean(dim=0)
            self._context_key = F.normalize(v, dim=-1, eps=1e-6).reshape(1, -1)

    @property
    def diag_enabled(self) -> bool:
        return self.config.attention_diagnostic

    @property
    def diag_layer(self) -> int:
        """Layer the attention-mass diagnostic is computed on (cheapest single
        enabled layer), so the O(window) softmax runs once per forward."""
        if self.config.layers is not None and len(self.config.layers) > 0:
            return min(self.config.layers)
        return 0

    def clear(self) -> None:
        """Full reset: drop both partitions (use between independent samples)."""
        self.entries_by_layer.clear()
        self.scene_entries_by_layer.clear()
        self._boundary_inject = False
        self._context_key = None
        self._scene_cross_clock = False
        self._runtime_context = {"chunk_index": None, "shot_index": None, "phase": None}
        self.attention_mass_by_shot_layer.clear()
        self.attention_mass_by_shot_layer_frame.clear()
        self.frame_alignment_drop_events.clear()
        self.manual_anchor_events.clear()
        for key in self.stats:
            self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0
        self._warnings.clear()

    def reset_shot(self) -> None:
        """Reset only the per-shot partition at a shot boundary.

        The persistent scene partition is left untouched so the establishing
        anchors of the environment survive the cut; the transient per-shot
        composition/framing memory is cleared like the legacy reset.
        """
        self.entries_by_layer.clear()

    def export_diagnostics(self) -> dict[str, object]:
        """JSON-serializable diagnostic summary for gate sidecars."""
        stats = dict(self.stats)
        mass_calls = int(stats.get("rag_mass_calls", 0) or 0)
        persistent_calls = int(stats.get("persistent_rag_mass_calls", 0) or 0)
        stats["rag_attention_mass_mean"] = (
            float(stats.get("rag_attention_mass", 0.0)) / mass_calls if mass_calls else 0.0
        )
        stats["persistent_rag_attention_mass_mean"] = (
            float(stats.get("persistent_rag_attention_mass", 0.0)) / persistent_calls
            if persistent_calls else 0.0
        )
        by_shot: dict[str, dict[str, dict[str, float | int]]] = {}
        for shot, layers in sorted(self.attention_mass_by_shot_layer.items()):
            by_shot[shot] = {}
            for layer, rec in sorted(layers.items(), key=lambda kv: int(kv[0])):
                calls = int(rec.get("calls", 0) or 0)
                mass_sum = float(rec.get("mass_sum", 0.0) or 0.0)
                by_shot[shot][layer] = {
                    "mean_mass": mass_sum / calls if calls else 0.0,
                    "calls": calls,
                    "mass_sum": mass_sum,
                    "persistent_tokens_mean": (
                        float(rec.get("persistent_tokens_sum", 0.0) or 0.0) / calls
                        if calls else 0.0
                    ),
                }
        by_frame: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
        for shot, layers in sorted(self.attention_mass_by_shot_layer_frame.items()):
            by_frame[shot] = {}
            for layer, frames in sorted(layers.items(), key=lambda kv: int(kv[0])):
                by_frame[shot][layer] = {}
                for frame_key, rec in sorted(frames.items()):
                    calls = int(rec.get("calls", 0) or 0)
                    mass_sum = float(rec.get("mass_sum", 0.0) or 0.0)
                    by_frame[shot][layer][frame_key] = {
                        "mean_mass": mass_sum / calls if calls else 0.0,
                        "calls": calls,
                        "mass_sum": mass_sum,
                        "tokens_mean": (
                            float(rec.get("tokens_sum", 0.0) or 0.0) / calls
                            if calls else 0.0
                        ),
                        "source_start_token": int(rec.get("source_start_token", -1)),
                    }
        return {
            "stats": stats,
            "attention_mass_by_shot_layer": by_shot,
            "attention_mass_by_shot_layer_frame": by_frame,
            "frame_alignment_drop_events": list(self.frame_alignment_drop_events),
            "manual_anchor_plan_path": self.config.manual_anchor_plan_path,
            "manual_anchor_events": list(self.manual_anchor_events),
        }

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
        persistent: bool = False,
        received_attention: torch.Tensor | None = None,
    ) -> None:
        """Store one chunk's K/V slice.

        ``k_pre`` is the pre-RoPE key (used for re-RoPE injection and, by
        default, the retrieval summary); ``k_post`` is the cached post-RoPE key
        (legacy injection / summary source). The caller passes both so this
        method can pick per-config without the model needing to know the flags.

        ``persistent`` routes the entry into the scene partition (it survives
        shot boundaries) when scene memory is enabled; otherwise it is ignored
        and the entry lands in the per-shot partition exactly as before.
        """
        if not self.enabled or not self.config.layer_enabled(layer):
            return
        if v.numel() == 0:
            return
        persistent = bool(persistent) and self.config.scene_memory_enabled

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

            idx, kept_frames, frame_offsets = self._select_index(
                num_tokens=int(inject_src.shape[1]),
                frame_seqlen=int(frame_seqlen),
                frames=int(frames),
                device=inject_src.device,
            )
            if self.config.require_frame_aligned and int(kept_frames) <= 0:
                self.record_frame_alignment_drop(
                    kind="store",
                    tokens=int(inject_src.shape[1]),
                    frame_seqlen=int(frame_seqlen),
                    frames=int(frames),
                    persistent=persistent,
                    reason="stored payload is not whole-frame aligned",
                )
                return
            attention_src = None
            if received_attention is not None:
                attention_src = received_attention.detach().to(device=inject_src.device).float().reshape(-1)
                if attention_src.numel() != int(inject_src.shape[1]):
                    raise ValueError(
                        "KV-RAG received_attention length must match the source token count "
                        f"({attention_src.numel()} != {int(inject_src.shape[1])})"
                    )
            if idx is not None:
                inject_src = inject_src.index_select(1, idx)
                v = v.index_select(1, idx)
                summary_src = summary_src.index_select(1, idx)
                if attention_src is not None:
                    attention_src = attention_src.index_select(0, idx)

            # The retrieval KEY (index) is computed from the full content slice,
            # independent of the VALUE (payload) representation -- so a
            # viewpoint-robust key can index a faithful raw-K/V payload.
            summary = self._compute_key(summary_src)
            if summary is None:
                return

            k_store, v_store, stored_frames, source_offset = self._apply_value_mode(
                inject_src, v, int(kept_frames), int(frame_seqlen),
                frame_offsets=frame_offsets,
                received_attention=attention_src,
            )
            if self._store_dtype is not None:
                k_store = k_store.to(dtype=self._store_dtype)
                v_store = v_store.to(dtype=self._store_dtype)
            if self.config.store_on_cpu:
                k_store = k_store.cpu()
                v_store = v_store.cpu()
                summary = summary.cpu()

            entry_start = int(start_token)
            entry_end = int(end_token)
            if (
                int(stored_frames) == 1
                and int(frame_seqlen) > 0
                and source_offset is not None
                and self.config.retrieval_value_mode in {"top_frame", "attention_mass"}
            ):
                entry_start = int(start_token) + int(source_offset)
                entry_end = entry_start + int(frame_seqlen)

            entry = KVRAGEntry(
                layer=layer,
                k=k_store.contiguous(),
                v=v_store.contiguous(),
                summary=summary.contiguous(),
                start_token=entry_start,
                end_token=entry_end,
                chunk_index=chunk_index,
                phase=phase,
                frames=int(stored_frames),
                h=int(h),
                w=int(w),
                frame_seqlen=int(frame_seqlen),
                source_frame_start=(
                    int(entry_start) // int(frame_seqlen)
                    if int(frame_seqlen) > 0 else None
                ),
                source_frame_indices=self._source_frame_indices(
                    entry_start,
                    int(frame_seqlen),
                    int(stored_frames),
                ),
                persistent=persistent,
            )
            if persistent:
                layer_entries = self.scene_entries_by_layer.setdefault(layer, [])
                cap = self.config.scene_memory_max_entries
                stat_key = "stored_scene_entries"
            else:
                layer_entries = self.entries_by_layer.setdefault(layer, [])
                cap = self.config.max_entries
                stat_key = "stored_entries"
            layer_entries.append(entry)
            while len(layer_entries) > cap:
                layer_entries.pop(0)
            self.stats[stat_key] += 1

    @staticmethod
    def _overlaps_any(
        start: int, end: int, ranges: list[tuple[int, int]] | None
    ) -> bool:
        """True when ``[start, end)`` intersects any half-open range in ``ranges``."""
        if not ranges:
            return False
        for rs, re in ranges:
            if start < re and rs < end:
                return True
        return False

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
        exclude_token_ranges: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[KVRAGEntry]] | None:
        """Return re-injectable K/V for the current query, or None.

        ``exclude_token_ranges`` lists absolute ``[start, end)`` token spans that
        are already live in the attended sink / pinned / local window. Any stored
        entry whose ``[start_token, end_token)`` overlaps one of them is dropped
        from BOTH the force-injected scene anchors and the content-matched pool,
        so a boundary anchor (or a content hit) is never double-counted against
        the same frames already in the window.
        """
        force_n = self.config.boundary_inject_anchors if self._boundary_inject else 0
        if not self.enabled or (self.config.top_k <= 0 and force_n <= 0):
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

        shot_entries = self.entries_by_layer.get(layer) or []
        scene_entries = (
            self.scene_entries_by_layer.get(layer) or []
            if self.config.scene_memory_enabled
            else []
        )
        if not shot_entries and not scene_entries:
            return None

        self.stats["retrieval_calls"] += 1
        min_end = int(current_start) - self.config.min_frame_gap * int(frame_seqlen)

        def _eligible(entry: KVRAGEntry) -> bool:
            if entry.end_token > min_end:
                return False
            return not self._overlaps_any(
                entry.start_token, entry.end_token, exclude_token_ranges
            )

        # Persistent scene anchors are timeless establishing context: they survive
        # boundaries AND survive a per-video token-clock restart (cross-perspective
        # generation renders one video per viewpoint, each starting at token 0), so
        # they are exempt from the per-shot causality filter. The live-window
        # overlap dedup avoids double-counting a frame ALREADY in the window -- valid
        # WITHIN one rollout, but spurious across a token-clock restart, where a
        # perspective-0 anchor's [start,end) coincides with the new perspective's
        # window only by numeric coincidence (a DIFFERENT video's tokens). So once
        # the scene partition is cross-clock, scene anchors skip the overlap dedup
        # and the boundary force-inject can actually deliver them.
        if self._scene_cross_clock:
            scene_pool = list(scene_entries)
        else:
            scene_pool = [
                e for e in scene_entries
                if not self._overlaps_any(e.start_token, e.end_token, exclude_token_ranges)
            ]
        candidates = [e for e in shot_entries if _eligible(e)]
        candidates += scene_pool
        if not candidates:
            return None

        selected: list[KVRAGEntry] = []
        chosen_ids: set[int] = set()

        # Force-inject the persistent scene anchors at a shot boundary regardless
        # of content match, so a perspective change keeps the established scene.
        # Anchors overlapping the live window were already filtered out of
        # ``scene_pool``, so a frame still in the sink/window is not re-injected.
        if force_n > 0 and scene_pool:
            manual_entries = self._manual_anchor_entries(scene_pool, force_n)
            force_entries = manual_entries if manual_entries is not None else list(reversed(scene_pool))[:force_n]
            for entry in force_entries:
                if id(entry) not in chosen_ids:
                    selected.append(entry)
                    chosen_ids.add(id(entry))
            if selected:
                self.stats["boundary_injections"] += 1

        # Add up to ``top_k`` content-matched candidates ON TOP of the forced
        # anchors (additive by design: the boundary injects guaranteed scene
        # anchors *plus* the best content matches, so the total selected is
        # <= boundary_inject_anchors + top_k). Persistent entries receive an
        # additive score bonus so the shared scene is preferred when matching.
        if self.config.top_k > 0:
            remaining = [e for e in candidates if id(e) not in chosen_ids]
            if remaining:
                query_summary = self._compute_key(query.detach(), for_query=True)
                if query_summary is not None:
                    scores = self._score_candidates(query_summary, remaining)
                    if self.config.scene_score_bonus:
                        scores = [
                            s + (self.config.scene_score_bonus if e.persistent else 0.0)
                            for s, e in zip(scores, remaining)
                        ]
                    order = sorted(range(len(remaining)), key=lambda i: scores[i], reverse=True)
                    for i in order[: self.config.top_k]:
                        selected.append(remaining[i])
                        chosen_ids.add(id(remaining[i]))

        if not selected:
            return None

        k_parts = [self._match_batch(e.k, query.shape[0]).to(device=device, dtype=dtype) for e in selected]
        v_parts = [self._match_batch(e.v, query.shape[0]).to(device=device, dtype=dtype) for e in selected]
        rag_k = torch.cat(k_parts, dim=1).contiguous()
        rag_v = torch.cat(v_parts, dim=1).contiguous()
        self.stats["retrieval_hits"] += 1
        self.stats["retrieved_tokens"] += int(rag_k.shape[1])
        return rag_k, rag_v, selected

    @staticmethod
    def _source_frame_indices(
        start_token: int,
        frame_seqlen: int,
        frames: int,
    ) -> tuple[int, ...]:
        if frame_seqlen <= 0 or frames <= 0:
            return ()
        start_frame = int(start_token) // int(frame_seqlen)
        return tuple(range(start_frame, start_frame + int(frames)))

    def _entry_frame_distance(self, entry: KVRAGEntry, frame_index: int) -> int:
        frames = entry.source_frame_indices
        if frames:
            return min(abs(int(frame_index) - int(f)) for f in frames)
        if entry.source_frame_start is not None:
            return abs(int(frame_index) - int(entry.source_frame_start))
        if entry.frame_seqlen > 0:
            return abs(int(frame_index) - int(entry.start_token) // int(entry.frame_seqlen))
        return 10**9

    def _manual_anchor_entries(
        self,
        scene_pool: list[KVRAGEntry],
        force_n: int,
    ) -> list[KVRAGEntry] | None:
        shot_index = self._runtime_context.get("shot_index")
        try:
            shot = int(shot_index)
        except Exception:
            return None
        requested = list(self._manual_plan_for_current_scene().get(shot, []))
        if not requested:
            return None
        self.stats["manual_anchor_requested_frames"] += len(requested)
        selected: list[KVRAGEntry] = []
        selected_ids: set[int] = set()
        for frame in requested:
            candidates = [entry for entry in scene_pool if id(entry) not in selected_ids]
            if not candidates:
                break
            best = min(
                candidates,
                key=lambda entry: (
                    self._entry_frame_distance(entry, int(frame)),
                    -int(entry.end_token),
                ),
            )
            selected.append(best)
            selected_ids.add(id(best))
            if len(selected) >= int(force_n):
                break
        if selected:
            self.stats["manual_anchor_injections"] += 1
            self.stats["manual_anchor_matched_frames"] += len(selected)
            if len(self.manual_anchor_events) < 128:
                self.manual_anchor_events.append({
                    "shot_index": shot,
                    "scene_name": self._scene_name,
                    "plan_source": "per_scene" if (
                        self._manual_anchor_plan_scenes
                        and self._manual_plan_for_current_scene() is not self._manual_anchor_plan
                    ) else "global",
                    "requested_frames": requested,
                    "selected_source_frames": [
                        list(entry.source_frame_indices) for entry in selected
                    ],
                    "selected_start_tokens": [int(entry.start_token) for entry in selected],
                    "selected_chunk_indices": [entry.chunk_index for entry in selected],
                })
        return selected

    def format_stats(self, prefix: str = "KV-RAG") -> str:
        layers = sum(1 for entries in self.entries_by_layer.values() if entries)
        live_entries = sum(len(entries) for entries in self.entries_by_layer.values())
        live_scene = sum(len(entries) for entries in self.scene_entries_by_layer.values())
        mass_calls = self.stats["rag_mass_calls"]
        mass = self.stats["rag_attention_mass"] / mass_calls if mass_calls else 0.0
        return (
            f"[{prefix}] layers={layers}, live_entries={live_entries}, "
            f"live_scene_entries={live_scene}, "
            f"stored_entries={self.stats['stored_entries']}, "
            f"stored_scene_entries={self.stats['stored_scene_entries']}, "
            f"retrieval_calls={self.stats['retrieval_calls']}, "
            f"retrieval_hits={self.stats['retrieval_hits']}, "
            f"retrieved_tokens={self.stats['retrieved_tokens']}, "
            f"boundary_injections={self.stats['boundary_injections']}, "
            f"rag_attn_mass={mass:.4f} (n={mass_calls})"
        )

    def _select_index(
        self,
        *,
        num_tokens: int,
        frame_seqlen: int,
        frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int, torch.Tensor | None]:
        """Pick which tokens of a chunk to store.

        Returns ``(idx, kept_frames, frame_offsets)``. ``idx is None`` means keep every token.
        ``kept_frames > 0`` means the kept tokens form whole frames and can be
        re-RoPE'd at retrieval time, and ``frame_offsets`` gives each kept
        frame's starting token offset in the original source chunk. ``0`` marks
        a legacy (non-frame-aligned) token subset that must fall back to the
        legacy injection path.
        """
        aligned = (
            self.config.frame_aligned_store
            and frame_seqlen > 0
            and frames > 0
            and frames * frame_seqlen == num_tokens
        )
        if aligned:
            if self.config.max_frames_per_entry is not None:
                max_frames = int(self.config.max_frames_per_entry)
                if max_frames <= 0 or frames <= max_frames:
                    offsets = torch.arange(frames, device=device, dtype=torch.long) * int(frame_seqlen)
                    return None, frames, offsets
                fsel = torch.linspace(0, frames - 1, steps=max_frames, device=device)
                fsel = fsel.round().to(torch.long).unique(sorted=True)
                offsets = torch.arange(frame_seqlen, device=device)
                idx = (fsel.view(-1, 1) * frame_seqlen + offsets.view(1, -1)).reshape(-1)
                return idx, int(fsel.numel()), fsel * int(frame_seqlen)

            limit = self.config.max_tokens_per_entry
            if limit <= 0 or num_tokens <= limit:
                offsets = torch.arange(frames, device=device, dtype=torch.long) * int(frame_seqlen)
                return None, frames, offsets
            max_frames = max(1, limit // frame_seqlen)
            if frames <= max_frames:
                offsets = torch.arange(frames, device=device, dtype=torch.long) * int(frame_seqlen)
                return None, frames, offsets
            fsel = torch.linspace(0, frames - 1, steps=max_frames, device=device)
            fsel = fsel.round().to(torch.long).unique(sorted=True)
            offsets = torch.arange(frame_seqlen, device=device)
            idx = (fsel.view(-1, 1) * frame_seqlen + offsets.view(1, -1)).reshape(-1)
            return idx, int(fsel.numel()), fsel * int(frame_seqlen)

        # Legacy token-level selection (slice is not re-RoPE'able).
        limit = self.config.max_tokens_per_entry
        if limit <= 0 or num_tokens <= limit:
            return None, 0, None
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
        return idx, 0, None

    def record_frame_alignment_drop(
        self,
        *,
        kind: str,
        tokens: int,
        frame_seqlen: int,
        frames: int = 0,
        persistent: bool | None = None,
        reason: str,
    ) -> None:
        """Count a frame-level contract violation without crashing inference."""
        stat_key = "frame_alignment_inject_drops" if kind == "inject" else "frame_alignment_store_drops"
        if stat_key in self.stats:
            self.stats[stat_key] += 1
        if len(self.frame_alignment_drop_events) < 64:
            self.frame_alignment_drop_events.append({
                "kind": str(kind),
                "tokens": int(tokens),
                "frame_seqlen": int(frame_seqlen),
                "frames": int(frames),
                "persistent": None if persistent is None else bool(persistent),
                "chunk_index": self._runtime_context.get("chunk_index"),
                "shot_index": self._runtime_context.get("shot_index"),
                "phase": self._runtime_context.get("phase"),
                "reason": str(reason),
            })

    def record_injection(
        self,
        *,
        tokens: int,
        frame_seqlen: int,
        selected_entries: list[KVRAGEntry] | None,
        outside_live_window: bool = True,
        path: str | None = None,
    ) -> None:
        """Count successful injected payload length in whole-frame units."""
        tokens = int(tokens)
        frame_seqlen = int(frame_seqlen)
        frames = tokens // frame_seqlen if frame_seqlen > 0 and tokens % frame_seqlen == 0 else 0
        self.stats["injection_calls"] += 1
        self.stats["injected_tokens"] += tokens
        self.stats["injected_frames"] += frames
        persistent_tokens = 0
        persistent_frames = 0
        for entry in selected_entries or []:
            if not entry.persistent:
                continue
            n = int(entry.num_tokens)
            persistent_tokens += n
            fsl = int(entry.frame_seqlen or frame_seqlen)
            if fsl > 0 and n % fsl == 0:
                persistent_frames += n // fsl
        self.stats["persistent_injected_tokens"] += persistent_tokens
        self.stats["persistent_injected_frames"] += persistent_frames
        if outside_live_window:
            self.stats["outside_window_injection_calls"] += 1
            self.stats["outside_window_injected_tokens"] += tokens
            self.stats["outside_window_injected_frames"] += frames
        if path == "reinject_rope":
            self.stats["reinject_rope_injection_calls"] += 1
            self.stats["reinject_rope_injected_frames"] += frames
        elif path == "direct":
            self.stats["direct_injection_calls"] += 1
            self.stats["direct_injected_frames"] += frames

    def build_persistent_logit_bias(
        self,
        *,
        total_tokens: int,
        selected_entries: list[KVRAGEntry] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return additive bias for injected persistent frame columns.

        The vector is shaped for SDPA broadcasting as ``[1, 1, 1, total_tokens]``.
        Only persistent entries represented as whole frames are biased; unaligned
        persistent entries are counted and skipped so the lever cannot manipulate
        arbitrary token subsets.
        """
        lam = float(self.config.persistent_logit_bias_lambda or 0.0)
        if lam <= 0.0 or not selected_entries:
            return None
        total_tokens = int(total_tokens)
        if total_tokens <= 0:
            return None

        bias = torch.zeros(total_tokens, device=device, dtype=torch.float32)
        cursor = 0
        biased_tokens = 0
        biased_frames = 0
        skipped_unaligned = 0
        for entry in selected_entries:
            n = int(entry.num_tokens)
            if n <= 0:
                continue
            end = min(cursor + n, total_tokens)
            if entry.persistent and end > cursor:
                fsl = int(entry.frame_seqlen or 0)
                if fsl > 0 and (end - cursor) % fsl == 0:
                    bias[cursor:end] += lam
                    biased_tokens += end - cursor
                    biased_frames += (end - cursor) // fsl
                else:
                    skipped_unaligned += 1
            cursor += n
            if cursor >= total_tokens:
                break

        if skipped_unaligned:
            self.stats["persistent_logit_bias_skipped_unaligned"] += skipped_unaligned
        if biased_tokens <= 0:
            return None
        self.stats["persistent_logit_bias_calls"] += 1
        self.stats["persistent_logit_bias_tokens"] += biased_tokens
        self.stats["persistent_logit_bias_frames"] += biased_frames
        self.stats["persistent_logit_bias_lambda_sum"] += lam
        return bias.to(dtype=dtype).view(1, 1, 1, total_tokens)

    def _compute_key(self, tensor: torch.Tensor, *, for_query: bool = False) -> torch.Tensor | None:
        """Build the retrieval KEY (index summary) for the active key mode.

        Decoupled from the VALUE payload: the same content slice can be indexed
        by a viewpoint-robust key while a faithful raw-K/V slice is injected.
        """
        mode = self.config.retrieval_key_mode
        if mode == "semantic":
            # Decoupled external key: ignore the K/V tensor and index by the
            # perspective's caption-text embedding. Fail fast if the companion
            # context vector was never provided (AC-5 companion check).
            if self._context_key is None:
                raise RuntimeError(
                    "KV-RAG retrieval_key_mode='semantic' requires a context key; "
                    "call set_context_key(caption_embedding) per perspective before "
                    "storing/retrieving (no caption-text provider was wired)."
                )
            return self._context_key.to(device=tensor.device, dtype=torch.float32)
        if mode == "pooled":
            return self._summarize(tensor)
        if mode == "moment":
            return self._summarize_moment(tensor)
        if mode == "multi_centroid":
            return self._summarize_centroids(tensor, self.config.retrieval_key_centroids)
        if mode == "salient_set":
            return self._summarize_salient_set(tensor, self.config.retrieval_key_top_m)
        if mode == "subject_identity":
            return self._summarize_subject_identity(tensor, self.config.retrieval_key_top_m)
        if mode == "attention_native":
            return tensor.detach().float() if for_query else self._summarize_attention_bounds(tensor)
        if mode == "positional":
            return self._summarize_positional(tensor)
        raise ValueError(f"Unsupported KV-RAG retrieval_key_mode={mode!r}")

    def _summarize_attention_bounds(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Quest-style per-head key bounds used by the attention-native key.

        Stored entries keep min/max key values over their token dimension, per
        batch/head/channel. Live queries are not externally embedded or pooled;
        retrieval scores those query vectors directly against these bounds using
        an upper bound on q.k.
        """
        if tensor.numel() == 0:
            return None
        f = tensor.float()
        if f.dim() == 4:  # [B, T, H, D]
            lo = f.min(dim=1).values
            hi = f.max(dim=1).values
            return torch.stack([lo, hi], dim=1)  # [B, 2, H, D]
        if f.dim() == 3:  # [B, T, D]
            lo = f.min(dim=1).values.unsqueeze(1)
            hi = f.max(dim=1).values.unsqueeze(1)
            return torch.stack([lo, hi], dim=1)  # [B, 2, 1, D]
        flat = f.reshape(f.shape[0], -1).unsqueeze(1)
        return torch.stack([flat, flat], dim=1)

    def _summarize(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Mean-pool over tokens into a normalized content summary.

        With ``summary_per_head`` the per-head structure ``[B, H, D]`` is kept
        and normalized per head, so scoring does not mix independent head
        subspaces. Otherwise the heads are flattened into one vector.

        Mean-pooling is orderless, so a viewpoint change (which rearranges
        spatial tokens) leaves the summary unchanged -- the baseline key is
        permutation-invariant, just coarse (it blends subject and background).
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

    def _summarize_moment(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Orderless mean+std signature -- richer than the bare mean but still
        permutation-invariant (viewpoint-robust)."""
        if tensor.numel() == 0:
            return None
        f = tensor.float()
        if f.dim() == 4:  # [B, T, H, D]
            pooled = torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False)], dim=-1)
            if not self.config.summary_per_head:
                pooled = pooled.flatten(1)
        elif f.dim() == 3:  # [B, T, D]
            pooled = torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False)], dim=-1)
        else:
            pooled = f.reshape(f.shape[0], -1)
        return F.normalize(pooled, dim=-1, eps=1e-6)

    def _summarize_centroids(self, tensor: torch.Tensor, n: int) -> torch.Tensor | None:
        """Several region centroids per chunk (subject vs background).

        Tokens are ranked by activation magnitude and split into ``n`` contiguous
        buckets, each mean-pooled. Ranking by magnitude is orderless, so the
        centroids are permutation-invariant (viewpoint-robust) yet separate
        salient subject regions from background -- enabling the same subject to
        match across camera angles instead of being averaged away.
        """
        if tensor.numel() == 0:
            return None
        if tensor.dim() != 4:  # only the [B, T, H, D] attention-key layout is structured
            base = self._summarize(tensor)
            return None if base is None else base.unsqueeze(1)
        f = tensor.float()
        b, t, h, d = f.shape
        n = max(1, min(int(n), t))
        mag = f.reshape(b, t, h * d).norm(dim=-1)  # [B, T]
        order = torch.argsort(mag, dim=1, stable=True)  # ascending; order-independent
        bounds = torch.linspace(0, t, steps=n + 1).round().to(torch.long).tolist()
        centroids = []
        for i in range(n):
            s, e = bounds[i], bounds[i + 1]
            if e <= s:
                e = s + 1
            idxs = order[:, s:e]  # [B, bucket]
            gathered = torch.gather(
                f, 1, idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, d)
            )
            centroids.append(gathered.mean(dim=1))  # [B, H, D]
        pooled = torch.stack(centroids, dim=1)  # [B, n, H, D]
        return F.normalize(pooled, dim=-1, eps=1e-6)

    def _summarize_salient_set(self, tensor: torch.Tensor, m: int) -> torch.Tensor | None:
        """A bounded set of the most salient tokens, each L2-normalized.

        Keeps the top-``m`` tokens by activation magnitude (an orderless,
        viewpoint-robust selection) without averaging them, so individual scene
        elements stay matchable across views via symmetric mutual-best cosine
        (see ``_score_salient_set``) rather than being blended into one vector.
        """
        if tensor.numel() == 0:
            return None
        if tensor.dim() != 4:
            base = self._summarize(tensor)
            return None if base is None else base.unsqueeze(1)
        f = tensor.float()
        b, t, h, d = f.shape
        m = max(1, min(int(m), t))
        mag = f.reshape(b, t, h * d).norm(dim=-1)  # [B, T]
        idx = torch.topk(mag, k=m, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)  # deterministic, order-independent
        gathered = torch.gather(f, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, d))
        return F.normalize(gathered, dim=-1, eps=1e-6)  # [B, m, H, D]

    def _summarize_subject_identity(self, tensor: torch.Tensor, m: int) -> torch.Tensor | None:
        """Identity-aware prototype: the subject token mean MINUS the background.

        Background is the per-head mean over all tokens; the subject is the mean
        of the top-``m`` highest-norm tokens (an orderless, viewpoint-robust
        selection). Subtracting the background before normalizing isolates the
        subject's distinctive direction from the shared scene, so the SAME subject
        matches across camera angles (and across views that share a background but
        differ in subject) -- the failure mode plain ``pooled`` blends away.
        Returns one ``[B, H, D]`` prototype per head, scored by plain cosine.
        """
        if tensor.numel() == 0:
            return None
        if tensor.dim() != 4:  # only the [B, T, H, D] attention-key layout is structured
            return self._summarize(tensor)
        f = tensor.float()
        b, t, h, d = f.shape
        m = max(1, min(int(m), t))
        background = f.mean(dim=1, keepdim=True)  # [B, 1, H, D]
        mag = f.reshape(b, t, h * d).norm(dim=-1)  # [B, T]
        idx = torch.topk(mag, k=m, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)
        subject = torch.gather(f, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, d))
        proto = (subject - background).mean(dim=1)  # [B, H, D] subject minus background
        return F.normalize(proto, dim=-1, eps=1e-6)

    def _summarize_positional(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Position-weighted pooling -- intentionally view-SENSITIVE.

        A linear token-position weighting makes the summary depend on token
        order, so a spatial rearrangement (viewpoint change) shifts it. Used to
        demonstrate that the viewpoint-invariance probe rejects a bad key; not a
        production candidate.
        """
        if tensor.numel() == 0:
            return None
        f = tensor.float()
        if f.dim() == 4:  # [B, T, H, D]
            t = f.shape[1]
            w = torch.linspace(0.0, 1.0, steps=t, device=f.device, dtype=f.dtype).view(1, t, 1, 1)
            pooled = (f * w).sum(dim=1) / (w.sum() + 1e-6)  # [B, H, D]
            if not self.config.summary_per_head:
                pooled = pooled.flatten(1)
        elif f.dim() == 3:  # [B, T, D]
            t = f.shape[1]
            w = torch.linspace(0.0, 1.0, steps=t, device=f.device, dtype=f.dtype).view(1, t, 1)
            pooled = (f * w).sum(dim=1) / (w.sum() + 1e-6)
        else:
            pooled = f.reshape(f.shape[0], -1)
        return F.normalize(pooled, dim=-1, eps=1e-6)

    def _apply_value_mode(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kept_frames: int,
        frame_seqlen: int,
        *,
        frame_offsets: torch.Tensor | None = None,
        received_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int | None]:
        """Transform the stored payload per ``retrieval_value_mode``.

        ``raw`` keeps the frame-aligned slice; ``mean_frame`` collapses it to a
        single representative frame (bounded injected length) while preserving
        ``frame_seqlen`` tokens so the payload stays re-RoPE'able.
        """
        mode = self.config.retrieval_value_mode
        if mode == "raw":
            return k, v, kept_frames, None
        aligned = (
            kept_frames > 0
            and frame_seqlen > 0
            and k.dim() == 4
            and k.shape[1] == kept_frames * frame_seqlen
        )
        if mode == "mean_frame":
            if not aligned:
                return k, v, kept_frames, None  # cannot collapse safely -> keep raw

            def collapse(t: torch.Tensor) -> torch.Tensor:
                bsz, _, heads, dim = t.shape
                return t.view(bsz, kept_frames, frame_seqlen, heads, dim).mean(dim=1)

            return collapse(k), collapse(v), 1, None
        if mode in {"top_frame", "attention_mass"}:
            # Keep only the single most subject-salient frame (highest mean token
            # norm) OR the highest received-attention frame. Both stay whole
            # frames (re-RoPE'able). Falls back to raw if not frame-aligned.
            if not aligned:
                return k, v, kept_frames, None
            if frame_offsets is None or int(frame_offsets.numel()) != kept_frames:
                frame_offsets = torch.arange(
                    kept_frames, device=k.device, dtype=torch.long
                ) * int(frame_seqlen)

            def frame_view(t: torch.Tensor) -> torch.Tensor:
                bsz, _, heads, dim = t.shape
                return t.view(bsz, kept_frames, frame_seqlen, heads, dim)

            kf = frame_view(k)
            vf = frame_view(v)
            if mode == "attention_mass":
                if received_attention is None:
                    raise RuntimeError(
                        "KV-RAG retrieval_value_mode='attention_mass' requires "
                        "a per-token received_attention vector from the attention store path."
                    )
                mass = received_attention.to(device=k.device).float().reshape(-1)
                if mass.numel() != kept_frames * frame_seqlen:
                    raise ValueError(
                        "KV-RAG received_attention length must match the kept token count "
                        f"({mass.numel()} != {kept_frames * frame_seqlen})"
                    )
                frame_score = mass.view(kept_frames, frame_seqlen).mean(dim=1)
            else:
                # rank frames by mean token L2-norm of the key (subject saliency),
                # averaged over batch so a single frame is chosen for the whole slice.
                frame_score = kf.float().norm(dim=-1).mean(dim=(0, 2, 3))  # [kept_frames]
            top = int(torch.argmax(frame_score).item())
            source_offset = int(frame_offsets[top].item())
            return (kf[:, top].reshape(k.shape[0], frame_seqlen, k.shape[2], k.shape[3]),
                    vf[:, top].reshape(v.shape[0], frame_seqlen, v.shape[2], v.shape[3]),
                    1, source_offset)
        raise ValueError(f"Unsupported KV-RAG retrieval_value_mode={mode!r}")

    def _score_candidates(
        self, query_summary: torch.Tensor, candidates: list[KVRAGEntry]
    ) -> list[float]:
        """Score candidates against the query for the active key mode."""
        if self.config.retrieval_key_mode == "multi_centroid":
            return [self._score_centroid(query_summary, c.summary) for c in candidates]
        if self.config.retrieval_key_mode == "salient_set":
            return [self._score_salient_set(query_summary, c.summary) for c in candidates]
        if self.config.retrieval_key_mode == "attention_native":
            return [self._score_attention_native(query_summary, c.summary) for c in candidates]
        if self.config.retrieval_key_mode in ("subject_identity", "semantic"):
            # These keys are L2-normalized vectors whose contract is plain COSINE,
            # independent of config.similarity. Falling through to _score_batch under
            # similarity='l2' would put them on the L2 scale (near zero for
            # high-dim/normalized vectors), where scene_score_bonus can dominate
            # content matching. Force cosine for both.
            return [self._score_cosine(query_summary, c.summary) for c in candidates]
        return self._score_batch(query_summary, candidates)

    def _score_attention_native(
        self, query: torch.Tensor, entry_bounds: torch.Tensor
    ) -> float:
        """Quest-style relevance: live q.k against stored min/max key bounds.

        For each live query component, the upper-bound key value is ``max`` when
        q>=0 and ``min`` when q<0. Scores are q dot bound, max-pooled over live
        query tokens, then averaged over heads/batch. No external embedding is
        used.
        """
        q = query.to(dtype=torch.float32)
        if q.dim() == 3:  # [B, T, D] -> [B, T, 1, D]
            q = q.unsqueeze(2)
        elif q.dim() != 4:
            q = q.reshape(q.shape[0], 1, 1, -1)
        bounds = entry_bounds.to(device=q.device, dtype=torch.float32)
        if bounds.dim() == 3:  # [B, 2, D]
            bounds = bounds.unsqueeze(2)
        lo = bounds[:, 0].unsqueeze(1)  # [B, 1, H, D]
        hi = bounds[:, 1].unsqueeze(1)
        chosen = torch.where(q >= 0, hi, lo)
        dot = (q * chosen).sum(dim=-1) / math.sqrt(max(1, q.shape[-1]))  # [B, T, H]
        return float(dot.max(dim=1).values.mean().item())

    def _score_cosine(self, query_summary: torch.Tensor, entry_summary: torch.Tensor) -> float:
        """Plain cosine (dot of L2-normalized summaries), independent of
        ``config.similarity``; mean over any head/batch dims."""
        e = entry_summary.to(device=query_summary.device, dtype=query_summary.dtype)
        return float((query_summary * e).sum(dim=-1).mean().item())

    def _score_salient_set(
        self, query_summary: torch.Tensor, entry_summary: torch.Tensor
    ) -> float:
        """Symmetric mutual-best (Chamfer) cosine between two salient-token sets.

        Each query token is matched to its best entry token and vice versa; the
        two directions are averaged. A scene element present in both views scores
        high regardless of where it sits in the frame, while background-only
        overlap cannot inflate the score in both directions.
        """
        e = entry_summary.to(device=query_summary.device, dtype=query_summary.dtype)
        q = query_summary
        if q.dim() != 4 or e.dim() != 4:
            return self._score(q, e)
        sim = torch.einsum("bqhd,bkhd->bqkh", q, e)  # [B, Mq, Me, H]
        q_to_e = sim.max(dim=2).values.mean()  # each query token's best match
        e_to_q = sim.max(dim=1).values.mean()  # each entry token's best match
        return float((0.5 * (q_to_e + e_to_q)).item())

    def _score_centroid(
        self, query_summary: torch.Tensor, entry_summary: torch.Tensor
    ) -> float:
        """Best-matching-centroid similarity between two centroid sets.

        ``query_summary``/``entry_summary``: ``[B, Nc, H, D]`` (L2-normalized per
        head). For each query centroid take its best match among the entry
        centroids, then average over query centroids, heads and batch -- so the
        same subject region scores high even when the rest of the frame differs.
        """
        e = entry_summary.to(device=query_summary.device, dtype=query_summary.dtype)
        q = query_summary
        if q.dim() != 4 or e.dim() != 4:
            return self._score(q, e)
        # cosine (already normalized) between every (query, entry) centroid pair
        sim = torch.einsum("bqhd,bkhd->bqkh", q, e)  # [B, Nq, Ne, H]
        best = sim.max(dim=2).values  # [B, Nq, H]
        return float(best.mean().item())

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

    def record_injected_attention_mass(
        self,
        query: torch.Tensor,
        window_k: torch.Tensor,
        selected_entries: list[KVRAGEntry],
        *,
        layer: int,
        logit_bias: torch.Tensor | None = None,
    ) -> None:
        """Record denoise attention mass on injected PERSISTENT entries.

        ``window_k`` must be the actual attention key matrix after injected
        entries were prepended. ``selected_entries`` is the retrieval order used
        to build that prefix, so persistent token columns can be isolated even
        when forced scene entries and content-matched transient entries are mixed.
        """
        if not self.config.attention_diagnostic:
            return
        phase = self._runtime_context.get("phase")
        if phase is not None and "denoise" not in str(phase):
            return
        try:
            persistent_spans: list[tuple[int, int]] = []
            cursor = 0
            for entry in selected_entries:
                n = int(entry.num_tokens)
                if n <= 0:
                    continue
                if entry.persistent:
                    persistent_spans.append((cursor, cursor + n))
                cursor += n
            if not persistent_spans or cursor <= 0 or window_k.shape[1] <= cursor:
                return
            with torch.no_grad():
                lq = query.shape[1]
                max_rows = int(self.config.attention_diag_max_query_rows)
                if lq > max_rows:
                    rows = torch.linspace(
                        0, lq - 1, steps=max_rows, device=query.device
                    ).round().to(torch.long)
                    q = query.index_select(1, rows).float()
                else:
                    q = query.float()
                k = window_k.float()
                scale = 1.0 / math.sqrt(max(1, q.shape[-1]))
                logits = torch.einsum("blhd,bmhd->bhlm", q, k) * scale
                if logit_bias is not None:
                    logits = logits + logit_bias.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                weights = torch.softmax(logits, dim=-1)
                mass = None
                persistent_tokens = 0
                persistent_frame_mass: list[tuple[str, float, int, int]] = []
                for start, end in persistent_spans:
                    persistent_tokens += end - start
                    part = weights[..., start:end].sum(dim=-1)
                    mass = part if mass is None else mass + part
                if mass is None:
                    return
                cursor = 0
                entry_index = 0
                for entry in selected_entries:
                    n = int(entry.num_tokens)
                    if n <= 0:
                        continue
                    if entry.persistent:
                        fsl = int(entry.frame_seqlen)
                        if fsl > 0 and n % fsl == 0:
                            for local_frame in range(n // fsl):
                                fs = cursor + local_frame * fsl
                                fe = fs + fsl
                                frame_part = weights[..., fs:fe].sum(dim=-1)
                                source_start = int(entry.start_token) + local_frame * fsl
                                frame_key = f"src_frame_{source_start // fsl}"
                                persistent_frame_mass.append(
                                    (frame_key, float(frame_part.mean().item()), fsl, source_start)
                                )
                        else:
                            source_start = int(entry.start_token)
                            frame_key = f"entry_{entry_index}_unaligned"
                            frame_part = weights[..., cursor:cursor + n].sum(dim=-1)
                            persistent_frame_mass.append(
                                (frame_key, float(frame_part.mean().item()), n, source_start)
                            )
                    cursor += n
                    entry_index += 1
                mean_mass = float(mass.mean().item())
            self.stats["persistent_rag_attention_mass"] += mean_mass
            self.stats["persistent_rag_mass_calls"] += 1
            shot_key = str(self._runtime_context.get("shot_index"))
            layer_key = str(int(layer))
            layers = self.attention_mass_by_shot_layer.setdefault(shot_key, {})
            rec = layers.setdefault(
                layer_key,
                {"mass_sum": 0.0, "calls": 0, "persistent_tokens_sum": 0.0},
            )
            rec["mass_sum"] = float(rec.get("mass_sum", 0.0)) + mean_mass
            rec["calls"] = int(rec.get("calls", 0)) + 1
            rec["persistent_tokens_sum"] = (
                float(rec.get("persistent_tokens_sum", 0.0)) + float(persistent_tokens)
            )
            frame_layers = self.attention_mass_by_shot_layer_frame.setdefault(shot_key, {})
            frame_recs = frame_layers.setdefault(layer_key, {})
            for frame_key, frame_mass, token_count, source_start in persistent_frame_mass:
                frec = frame_recs.setdefault(
                    frame_key,
                    {
                        "mass_sum": 0.0,
                        "calls": 0,
                        "tokens_sum": 0.0,
                        "source_start_token": int(source_start),
                    },
                )
                frec["mass_sum"] = float(frec.get("mass_sum", 0.0)) + float(frame_mass)
                frec["calls"] = int(frec.get("calls", 0)) + 1
                frec["tokens_sum"] = float(frec.get("tokens_sum", 0.0)) + float(token_count)
        except Exception:
            pass

    def estimate_received_attention(
        self,
        query: torch.Tensor,
        window_k: torch.Tensor,
        source_tokens: int,
        max_query_rows: int = 32,
    ) -> torch.Tensor | None:
        """Estimate received attention for the newest source tokens.

        The newest clean-recache tokens are the last ``source_tokens`` columns of
        ``window_k``. We sample a bounded number of live query rows, compute the
        normal attention softmax over the attended window, and return the mean
        post-softmax mass received by each newest token. This is flag-gated by
        ``retrieval_value_mode='attention_mass'`` in the caller, so the disabled
        path performs no extra work.
        """
        if source_tokens <= 0 or window_k.shape[1] < source_tokens or query.numel() == 0:
            return None
        with torch.no_grad():
            lq = query.shape[1]
            if lq > max_query_rows:
                rows = torch.linspace(
                    0, lq - 1, steps=max_query_rows, device=query.device
                ).round().to(torch.long)
                q = query.index_select(1, rows).float()
            else:
                q = query.float()
            k = window_k.float()
            scale = 1.0 / math.sqrt(max(1, q.shape[-1]))
            logits = torch.einsum("blhd,bmhd->bhlm", q, k) * scale
            weights = torch.softmax(logits, dim=-1)
            mass = weights[..., -source_tokens:].mean(dim=(0, 1, 2))
            return mass.detach().cpu()

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
