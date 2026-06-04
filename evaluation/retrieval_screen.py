# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free offline PRE-FILTER for the retrieval key/value candidate sweep (AC-3).

This is a *labeled proxy*, NOT the selector. The prior round established
(`BL-20260604-rendered-metric-over-probe`, `docs/retrieval-key-value-study.md`)
that this synthetic viewpoint-invariance probe MIS-RANKED the keys -- plain
`pooled` beat the probe's favorite `salient_set` ~2.5x on rendered 5B video. So
the screen here only PRUNES obviously-broken cells (position-sensitive keys,
missing-context keys, non-frame-aligned compressed values) and ranks the rest as
a shortlist; the winner is chosen ONLY on the rendered AC-2 suite
(`evaluation/vbench_consistency.py`).

What the probe CAN tell us: whether a key recognizes the same scene across a pure
token rearrangement (permutation/viewpoint invariance) and rejects a different
scene; structural facts about a value (frame-aligned / re-RoPE'able / token
count). What it CANNOT tell us: how a representation shifts the model's actual
attention, denoising trajectory, identity, adherence, diversity, or rendered
pixels; and it under-represents real perspective changes where the subject
changes scale/appearance and only part of the background persists.
"""

from __future__ import annotations

import torch

from utils.kv_rag import KEY_MODES, VALUE_MODES, CONTEXT_KEY_MODES, KVRAGConfig, KVRAGMemory

#: Minimum non-degenerate pass on the proxy (a candidate below this is pruned,
#: NOT promoted; passing only earns a render slot).
MIN_SAME_SCORE = 0.6
MIN_MARGIN = 0.1

#: Default per-key hyperparameters used when screening.
_KEY_KW = {
    "multi_centroid": dict(retrieval_key_centroids=4),
    "salient_set": dict(retrieval_key_top_m=6),
    "subject_identity": dict(retrieval_key_top_m=6),
}


def build_probe_tensors(*, seed: int = 0, heads: int = 2, dim: int = 4,
                        frames: int = 3, frame_seqlen: int = 4):
    """Frame-aligned synthetic K/V for the viewpoint-invariance probe.

    ``view_a`` and ``view_b`` are the SAME scene elements rearranged in the frame
    (a camera move); ``diff`` is a different scene. Returns ``(view_a, view_b,
    diff, frames, frame_seqlen)``.
    """
    g = torch.Generator().manual_seed(seed)
    t = frames * frame_seqlen
    a = torch.randn(1, 1, heads, dim, generator=g) * 3.0
    b = torch.randn(1, 1, heads, dim, generator=g) * 3.0
    c = torch.randn(1, 1, heads, dim, generator=g) * 3.0
    d = torch.randn(1, 1, heads, dim, generator=g) * 3.0

    def bg():
        return 0.05 * torch.randn(1, t, heads, dim, generator=g)

    view_a = bg(); view_a[:, 0:3] = a; view_a[:, t - 3:t] = b
    view_b = bg(); view_b[:, 0:3] = b; view_b[:, t - 3:t] = a  # same scene, new framing
    diff = bg(); diff[:, 0:3] = c; diff[:, t - 3:t] = d         # different scene
    return view_a, view_b, diff, frames, frame_seqlen


def build_probe_captions(*, seed: int = 0, dim: int = 16):
    """Synthetic caption-text embeddings for the ``semantic`` key.

    ``emb_a``/``emb_b`` are two phrasings of the SAME scene (a shared base
    direction + small noise); ``emb_d`` describes a different scene. Returns
    ``(emb_a, emb_b, emb_d)``.
    """
    g = torch.Generator().manual_seed(seed + 101)
    base = torch.randn(dim, generator=g)
    emb_a = base + 0.10 * torch.randn(dim, generator=g)
    emb_b = base + 0.10 * torch.randn(dim, generator=g)
    emb_d = torch.randn(dim, generator=g)  # different scene
    return emb_a, emb_b, emb_d


def _key_summary(mem: KVRAGMemory, mode: str, tensor, caption):
    if mode in CONTEXT_KEY_MODES:
        mem.set_context_key(caption)
        return mem._compute_key(tensor)  # tensor ignored; uses the context key
    return mem._compute_key(tensor)


def screen_key(mode: str, *, seed: int = 0) -> dict:
    """Score one KEY mode on the viewpoint-invariance proxy.

    Returns same-scene score, different-scene score, their margin, and whether
    the cell passes the (non-degenerate) pre-filter. ``positional`` is the
    deliberate negative control and is expected to FAIL -- that keeps the proxy
    non-vacuous.
    """
    view_a, view_b, diff, _frames, _fsl = build_probe_tensors(seed=seed)
    emb_a, emb_b, emb_d = build_probe_captions(seed=seed)
    kw = _KEY_KW.get(mode, {})
    mem = KVRAGMemory(KVRAGConfig(enabled=True, retrieval_key_mode=mode, **kw))

    qs = _key_summary(mem, mode, view_a, emb_a)
    cand_same = _key_summary(mem, mode, view_b, emb_b)
    cand_diff = _key_summary(mem, mode, diff, emb_d)

    def _fake(summary):
        return type("E", (), {"summary": summary, "persistent": False})()

    same = float(mem._score_candidates(qs, [_fake(cand_same)])[0])
    different = float(mem._score_candidates(qs, [_fake(cand_diff)])[0])
    margin = same - different
    is_control = mode == "positional"
    passes = (same >= MIN_SAME_SCORE) and (margin >= MIN_MARGIN) and not is_control
    return {
        "key": mode,
        "same_scene": same,
        "different_scene": different,
        "margin": margin,
        "is_negative_control": is_control,
        "passes_prefilter": bool(passes),
    }


def screen_value(mode: str) -> dict:
    """Record STRUCTURAL facts about one VALUE mode (the proxy does not rank value
    quality -- only whether the payload is frame-aligned / re-RoPE'able)."""
    frames, fsl, heads, dim = 3, 4, 2, 4
    mem = KVRAGMemory(KVRAGConfig(enabled=True, layers=(0,), retrieval_value_mode=mode,
                                  frame_aligned_store=True))
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, frames * fsl, heads, dim, generator=g)
    v = torch.randn(1, frames * fsl, heads, dim, generator=g)
    mem.add(layer=0, k_pre=k, k_post=k, v=v, start_token=0, end_token=frames * fsl,
            frame_seqlen=fsl, h=2, w=2, frames=frames)
    entry = mem.entries_by_layer[0][0]
    return {
        "value": mode,
        "stored_frames": int(entry.frames),
        "stored_tokens": int(entry.num_tokens),
        "input_tokens": frames * fsl,
        "frame_aligned": bool(entry.frames > 0),
        "reropeable": bool(entry.frames > 0),
        "bounded": bool(entry.frames > 0 and entry.frames < frames),
    }


def run_screen(*, key_modes=KEY_MODES, value_modes=VALUE_MODES, seed: int = 0) -> dict:
    """Rank all (key, value) candidate cells on the GPU-free pre-filter.

    Returns a dict with the per-key ranking, the per-value structural facts, the
    full (key x value) cell matrix, and an explicit ``is_prefilter`` flag + note.
    The key margin drives the ranking; the value contributes only structural
    eligibility (decoupled, AC-3.1). NOTHING here selects a winner -- finalists
    graduate to the rendered AC-2 gate.
    """
    keys = [screen_key(k, seed=seed) for k in key_modes]
    keys_ranked = sorted(keys, key=lambda r: r["margin"], reverse=True)
    values = [screen_value(v) for v in value_modes]
    value_by_name = {v["value"]: v for v in values}

    cells = []
    for kr in keys:
        for v in value_modes:
            vf = value_by_name[v]
            cells.append({
                "key": kr["key"],
                "value": v,
                "key_margin": kr["margin"],
                "key_passes": kr["passes_prefilter"],
                "value_reropeable": vf["reropeable"],
                # A cell is a render candidate only if the key passes and the value
                # is structurally usable; positional cells are screen-only controls.
                "render_candidate": bool(kr["passes_prefilter"] and vf["reropeable"]),
            })

    passing = [k for k in keys_ranked if k["passes_prefilter"]]
    return {
        "is_prefilter": True,
        "selector": "rendered AC-2 suite (evaluation/vbench_consistency.py)",
        "note": (
            "PROXY ONLY (BL-20260604-rendered-metric-over-probe): this ranking "
            "PRUNES broken cells and orders a shortlist; it MUST NOT pick a "
            "winner. Promotion requires a rendered AC-2 confirmation."
        ),
        "thresholds": {"min_same_scene": MIN_SAME_SCORE, "min_margin": MIN_MARGIN},
        "keys_ranked": keys_ranked,
        "values": values,
        "cells": cells,
        "shortlist_keys": [k["key"] for k in passing],
    }
