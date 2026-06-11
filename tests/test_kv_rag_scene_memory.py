# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free, checkpoint-free unit tests for single-scene multi-perspective KV-RAG.

Run with the standard library (no pytest needed):

    python -m unittest discover -s tests -p 'test_*.py'

Everything here runs on CPU tensors of tiny synthetic shapes; no model
checkpoint or GPU is touched.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kv_rag import KVRAGConfig, KVRAGMemory, virtual_frame_start

CPU = torch.device("cpu")


def _entry_tensor(b=1, frames=2, fsl=4, heads=2, dim=3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(b, frames * fsl, heads, dim)


def _store(memory, *, layer=0, frames=2, fsl=4, heads=2, dim=3, start=0,
           persistent=False, k=None, v=None):
    t = frames * fsl
    if k is None:
        k = torch.randn(1, t, heads, dim)
    if v is None:
        v = torch.randn(1, t, heads, dim)
    memory.add(
        layer=layer, k_pre=k, k_post=k, v=v,
        start_token=start, end_token=start + t,
        frame_seqlen=fsl, h=heads, w=2, frames=frames,
        persistent=persistent,
    )
    return k, v


# ---------------------------------------------------------------------------
# AC-1: persistent scene partition, eviction, reset isolation
# ---------------------------------------------------------------------------
class TestScenePartition(unittest.TestCase):
    def _scene_cfg(self, **kw):
        base = dict(enabled=True, top_k=2, scene_memory_enabled=True,
                    scene_memory_max_entries=4)
        base.update(kw)
        return KVRAGConfig(**base)

    def test_scene_survives_shot_reset_while_per_shot_clears(self):
        m = KVRAGMemory(self._scene_cfg())
        _store(m, persistent=True, start=0)   # shot-0 anchor -> scene
        _store(m, persistent=True, start=8)
        _store(m, persistent=False, start=100)  # later per-shot entry
        self.assertEqual(len(m.scene_entries_by_layer[0]), 2)
        self.assertEqual(len(m.entries_by_layer[0]), 1)

        m.reset_shot()  # boundary: clear per-shot, keep scene
        self.assertEqual(len(m.scene_entries_by_layer[0]), 2)
        self.assertEqual(len(m.entries_by_layer.get(0, [])), 0)

    def test_clear_drops_both_partitions(self):
        m = KVRAGMemory(self._scene_cfg())
        _store(m, persistent=True)
        _store(m, persistent=False, start=100)
        m.clear()
        self.assertEqual(len(m.scene_entries_by_layer), 0)
        self.assertEqual(len(m.entries_by_layer), 0)

    def test_scene_eviction_enforced(self):
        m = KVRAGMemory(self._scene_cfg(scene_memory_max_entries=3))
        for i in range(10):
            _store(m, persistent=True, start=i * 8)
        self.assertEqual(len(m.scene_entries_by_layer[0]), 3)  # not unbounded

    def test_disabled_flag_is_regression_safe(self):
        # AC-1/AC-4 negative: with the flag OFF, persistent=True is ignored and
        # the entry is byte-identical to a plain store (no scene partition).
        torch.manual_seed(7)
        k = torch.randn(1, 8, 2, 3)
        v = torch.randn(1, 8, 2, 3)

        off = KVRAGMemory(KVRAGConfig(enabled=True, top_k=2))  # no scene flag
        ref = KVRAGMemory(KVRAGConfig(enabled=True, top_k=2))
        _store(off, persistent=True, k=k.clone(), v=v.clone())   # asks for scene
        _store(ref, persistent=False, k=k.clone(), v=v.clone())  # plain

        self.assertEqual(len(off.scene_entries_by_layer), 0)  # nothing persisted
        self.assertEqual(len(off.entries_by_layer[0]), 1)
        e_off, e_ref = off.entries_by_layer[0][0], ref.entries_by_layer[0][0]
        self.assertTrue(torch.equal(e_off.k, e_ref.k))
        self.assertTrue(torch.equal(e_off.summary, e_ref.summary))
        self.assertFalse(e_off.persistent)


# ---------------------------------------------------------------------------
# AC-1: per-perspective seed / force-inject protocol (Round 1 substrate fix)
# ---------------------------------------------------------------------------
class TestPerPerspectiveProtocol(unittest.TestCase):
    """Perspective 0 SEEDS the persistent anchors; later perspectives FORCE-INJECT
    them at their first chunk and store only TRANSIENT per-shot entries."""

    def _cfg(self, **kw):
        base = dict(enabled=True, top_k=0, scene_memory_enabled=True,
                    boundary_inject_anchors=2, scene_memory_max_entries=8,
                    frame_aligned_store=True)
        base.update(kw)
        return KVRAGConfig(**base)

    def test_persp0_seeds_then_persp1_forceinjects_and_stores_transient(self):
        m = KVRAGMemory(self._cfg())
        # Perspective 0 (seed_scene_memory=True): store persistent anchors.
        for i in range(3):
            _store(m, persistent=True, start=i * 8)
        self.assertEqual(len(m.scene_entries_by_layer[0]), 3)
        self.assertEqual(len(m.entries_by_layer.get(0, [])), 0)

        # Boundary into perspective 1: keep scene, clear per-shot (mirrors
        # _reset_kv_rag_keep_scene), then pulse the boundary force-inject.
        m.reset_shot()
        m.set_boundary_inject(True)
        out = m.retrieve(layer=0, query=torch.randn(1, 4, 2, 3), current_start=0,
                         frame_seqlen=4, dtype=torch.float32, device=CPU)
        self.assertIsNotNone(out)
        self.assertGreaterEqual(m.stats["boundary_injections"], 1)  # B1 fix
        _, _, selected = out
        self.assertTrue(selected and all(e.persistent for e in selected))

        # Perspective 1 (seed_scene_memory=False): store TRANSIENT, do not reseed.
        _store(m, persistent=False, start=200)
        self.assertEqual(len(m.scene_entries_by_layer[0]), 3)   # B2 fix: not reseeded
        self.assertEqual(len(m.entries_by_layer[0]), 1)         # transient stored

    def test_cross_clock_anchors_survive_window_overlap(self):
        # A later perspective restarts at token 0, so a perspective-0 anchor stored
        # at [0,8) numerically overlaps the new perspective's live window. Without
        # the cross-clock mark the overlap dedup filters it (force-inject gets
        # nothing); with the mark it is exempt and force-injected. (P1 fix.)
        m = KVRAGMemory(self._cfg())
        _store(m, persistent=True, start=0)   # perspective-0 anchor at [0, 8)
        m.reset_shot()
        exclude = [(0, 8)]                     # the new perspective's window range
        m.set_boundary_inject(True)
        self.assertIsNone(m.retrieve(
            layer=0, query=torch.randn(1, 4, 2, 3), current_start=0, frame_seqlen=4,
            dtype=torch.float32, device=CPU, exclude_token_ranges=exclude))

        m.mark_scene_cross_clock()             # perspective boundary -> token clock restarted
        m.set_boundary_inject(True)
        out = m.retrieve(
            layer=0, query=torch.randn(1, 4, 2, 3), current_start=0, frame_seqlen=4,
            dtype=torch.float32, device=CPU, exclude_token_ranges=exclude)
        self.assertIsNotNone(out)
        self.assertGreaterEqual(m.stats["boundary_injections"], 1)

    def test_clear_resets_cross_clock(self):
        m = KVRAGMemory(self._cfg())
        m.mark_scene_cross_clock()
        self.assertTrue(m._scene_cross_clock)
        m.clear()
        self.assertFalse(m._scene_cross_clock)

    def test_no_boundary_inject_without_pulse(self):
        # Without the perspective-boundary pulse, no force-injection happens
        # (this is the pre-fix behavior the new force_scene_memory_boundary cures).
        m = KVRAGMemory(self._cfg())
        for i in range(2):
            _store(m, persistent=True, start=i * 8)
        m.reset_shot()
        out = m.retrieve(layer=0, query=torch.randn(1, 4, 2, 3), current_start=0,
                         frame_seqlen=4, dtype=torch.float32, device=CPU)
        self.assertIsNone(out)  # top_k=0 and no pulse -> nothing retrieved
        self.assertEqual(m.stats["boundary_injections"], 0)


class TestPipelineSceneMemoryPredicates(unittest.TestCase):
    """The pure boundary/seed predicates the per-perspective inference loop uses."""

    def setUp(self):
        try:
            from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
        except Exception as exc:  # pragma: no cover - heavy deps absent
            self.skipTest(f"pipeline import unavailable: {exc}")
        self.P = CausalDiffusionInferencePipeline

    def test_memory_boundary_active(self):
        P = self.P
        self.assertTrue(P._memory_boundary_active(True, False, 5))    # intra-video shot cut
        self.assertTrue(P._memory_boundary_active(False, True, 0))    # new perspective chunk 0
        self.assertFalse(P._memory_boundary_active(False, True, 1))   # later chunk, not boundary
        self.assertFalse(P._memory_boundary_active(False, False, 0))  # non-multiview chunk 0
        self.assertTrue(P._memory_boundary_active(False, False, 3, "every_chunk"))

    def test_should_store_persistent(self):
        P = self.P
        self.assertTrue(P._should_store_persistent(True, True, 0))    # persp0, shot0, scene mem
        self.assertFalse(P._should_store_persistent(True, False, 0))  # later perspective
        self.assertFalse(P._should_store_persistent(True, True, 1))   # later shot
        self.assertFalse(P._should_store_persistent(False, True, 0))  # scene memory off

    def test_rolling_store_only_on_shot_end(self):
        P = self.P
        self.assertFalse(P._should_store_persistent(
            True, True, 1, scene_memory_rolling=True, is_shot_end=False))
        self.assertTrue(P._should_store_persistent(
            True, True, 1, scene_memory_rolling=True, is_shot_end=True))
        self.assertFalse(P._should_store_persistent(
            True, False, 1, scene_memory_rolling=True, is_shot_end=True))


class TestSemanticContextKeyCFG(unittest.TestCase):
    """The semantic key must index the SCENE each perspective describes -- the
    conditional caption -- for BOTH banks, not the scene-independent negative
    prompt (which is identical across perspectives and would be scene-blind)."""

    def setUp(self):
        try:
            from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
        except Exception as exc:  # pragma: no cover - heavy deps absent
            self.skipTest(f"pipeline import unavailable: {exc}")
        self.P = CausalDiffusionInferencePipeline

    def _fake_pipe(self, key_mode):
        fake = type("FakePipe", (), {})()
        cfg = KVRAGConfig(enabled=True, layers=(0,), retrieval_key_mode=key_mode)
        fake.kv_rag_enabled = True
        fake.kv_rag_config = cfg
        fake.kv_rag_pos = KVRAGMemory(cfg)
        fake.kv_rag_neg = KVRAGMemory(cfg)
        return fake

    def test_both_banks_keyed_on_conditional_scene_caption(self):
        import torch.nn.functional as F
        p = self._fake_pipe("semantic")
        cond = {"prompt_embeds": torch.tensor([[1.0, 0.0, 0.0, 0.0]])}  # scene caption
        self.P._set_kv_rag_context_key(p, cond)
        self.assertIsNotNone(p.kv_rag_pos._context_key)
        self.assertIsNotNone(p.kv_rag_neg._context_key)
        # both banks share the SAME conditional-derived key (negative bank is scene-aware)
        self.assertTrue(torch.allclose(p.kv_rag_pos._context_key, p.kv_rag_neg._context_key))
        cond_key = F.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=-1).reshape(1, -1)
        uncond_key = F.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=-1).reshape(1, -1)
        self.assertTrue(torch.allclose(p.kv_rag_neg._context_key, cond_key, atol=1e-5))
        # the negative bank is NOT keyed on a (different) negative-prompt embedding
        self.assertFalse(torch.allclose(p.kv_rag_neg._context_key, uncond_key, atol=1e-3))

    def test_inert_for_non_semantic_mode(self):
        p = self._fake_pipe("pooled")
        self.P._set_kv_rag_context_key(p, {"prompt_embeds": torch.tensor([[1.0, 0.0, 0.0, 0.0]])})
        self.assertIsNone(p.kv_rag_pos._context_key)
        self.assertIsNone(p.kv_rag_neg._context_key)


# ---------------------------------------------------------------------------
# AC-2: boundary force-injection, dedup, flag isolation
# ---------------------------------------------------------------------------
class TestBoundaryInjection(unittest.TestCase):
    def _mem(self, **kw):
        params = dict(enabled=True, top_k=1, scene_memory_enabled=True,
                      scene_memory_max_entries=8, boundary_inject_anchors=2)
        params.update(kw)
        return KVRAGMemory(KVRAGConfig(**params))

    def _populate(self, m):
        for i in range(3):
            _store(m, persistent=True, start=i * 8)

    def test_force_inject_adds_scene_anchors_at_boundary(self):
        m = self._mem()
        self._populate(m)
        q = torch.randn(1, 4, 2, 3)
        m.set_boundary_inject(True)
        res = m.retrieve(layer=0, query=q, current_start=1000,
                         frame_seqlen=4, dtype=torch.float32, device=CPU)
        self.assertIsNotNone(res)
        _, _, selected = res
        forced = [e for e in selected if e.persistent]
        self.assertGreaterEqual(len(forced), 2)  # the 2 forced anchors are present
        self.assertEqual(m.stats["boundary_injections"], 1)

    def test_no_duplicate_tokens(self):
        # The forced anchors and the content-matched fill must not double-count
        # the same entry.
        m = self._mem(top_k=3)
        self._populate(m)
        q = torch.randn(1, 4, 2, 3)
        m.set_boundary_inject(True)
        _, _, selected = m.retrieve(layer=0, query=q, current_start=1000,
                                    frame_seqlen=4, dtype=torch.float32, device=CPU)
        ids = [id(e) for e in selected]
        self.assertEqual(len(ids), len(set(ids)))

    def test_flag_isolation_no_injection_when_zero(self):
        # boundary_inject_anchors == 0 -> no forced tokens even if asked.
        m = KVRAGMemory(KVRAGConfig(enabled=True, top_k=1, scene_memory_enabled=True,
                                    boundary_inject_anchors=0))
        self._populate(m)
        m.set_boundary_inject(True)  # no-op since anchors == 0
        self.assertFalse(m._boundary_inject)
        q = torch.randn(1, 4, 2, 3)
        res = m.retrieve(layer=0, query=q, current_start=1000,
                         frame_seqlen=4, dtype=torch.float32, device=CPU)
        # only content-matched (top_k=1), no forced anchors
        self.assertEqual(len(res[2]), 1)
        self.assertEqual(m.stats["boundary_injections"], 0)


# ---------------------------------------------------------------------------
# AC-2 / AC-5: virtual-frame placement math (and a broken-offset guard)
# ---------------------------------------------------------------------------
class TestVirtualFrameStart(unittest.TestCase):
    def test_block_sits_immediately_before_window_without_overlap(self):
        fsl = 4
        current_end = 80          # 20 frames of history end here
        window_tokens = 40        # window has 10 frames
        prefix_tokens = 8         # 2 sink frames
        rag_frames = 3
        local_frames = (window_tokens - prefix_tokens) // fsl
        local_start_frame = (current_end // fsl) - local_frames

        rag_start = virtual_frame_start(
            current_end=current_end, frame_seqlen=fsl,
            window_tokens=window_tokens, prefix_tokens=prefix_tokens,
            rag_frames=rag_frames,
        )
        self.assertGreaterEqual(rag_start, 0)
        # rag block [rag_start, rag_start+rag_frames) must end at/ before the
        # local window's first frame -> no overlap / no duplicate frames.
        self.assertLessEqual(rag_start + rag_frames, local_start_frame)

    def test_broken_offset_would_overlap_window(self):
        # A deliberately broken offset (forgetting to subtract rag_frames) lands
        # the block inside the window -> the no-overlap assertion catches it.
        fsl = 4
        current_end, window_tokens, prefix_tokens, rag_frames = 80, 40, 8, 3
        local_frames = (window_tokens - prefix_tokens) // fsl
        local_start_frame = (current_end // fsl) - local_frames
        broken_start = max(0, local_start_frame)  # missing "- rag_frames"
        self.assertGreater(broken_start + rag_frames, local_start_frame)

    def test_clamps_to_zero(self):
        rag_start = virtual_frame_start(
            current_end=8, frame_seqlen=4, window_tokens=8,
            prefix_tokens=0, rag_frames=5,
        )
        self.assertEqual(rag_start, 0)

    def test_requires_frame_seqlen(self):
        with self.assertRaises(ValueError):
            virtual_frame_start(current_end=8, frame_seqlen=0, window_tokens=8,
                                prefix_tokens=0, rag_frames=1)


# ---------------------------------------------------------------------------
# AC-3 / AC-3.1: decoupled key/value, fail-fast, determinism
# ---------------------------------------------------------------------------
class TestKeyValueInterface(unittest.TestCase):
    def test_decoupled_key_and_value(self):
        # AC-3.1: a viewpoint-robust key with a faithful raw-K/V value works.
        m = KVRAGMemory(KVRAGConfig(enabled=True, top_k=1,
                                    retrieval_key_mode="multi_centroid",
                                    retrieval_key_centroids=4,
                                    retrieval_value_mode="raw"))
        k, v = _store(m, frames=3, fsl=4)
        e = m.entries_by_layer[0][0]
        self.assertTrue(torch.equal(e.v, v))         # value untouched (raw)
        self.assertEqual(e.summary.dim(), 4)          # key is centroid-structured

    def test_unknown_modes_fail_fast(self):
        for bad in (dict(retrieval_key_mode="nope"),
                    dict(retrieval_value_mode="nope"),
                    dict(retrieval_key_mode="multi_centroid", retrieval_key_centroids=1),
                    dict(boundary_inject_anchors=1, scene_memory_enabled=False),
                    dict(scene_memory_injection_schedule="always"),
                    dict(scene_memory_rolling=True, scene_memory_enabled=False),
                    dict(scene_memory_injection_schedule="every_chunk", scene_memory_enabled=False)):
            with self.assertRaises(ValueError):
                KVRAGMemory(KVRAGConfig(enabled=True, **bad))

    def test_value_mean_frame_is_bounded_and_frame_aligned(self):
        m = KVRAGMemory(KVRAGConfig(enabled=True, retrieval_value_mode="mean_frame"))
        fsl, frames = 4, 3
        _store(m, frames=frames, fsl=fsl)
        e = m.entries_by_layer[0][0]
        self.assertEqual(e.num_tokens, fsl)  # collapsed to one frame
        self.assertEqual(e.frames, 1)        # still frame-aligned / re-RoPE'able

    def test_value_raw_unchanged(self):
        m = KVRAGMemory(KVRAGConfig(enabled=True, retrieval_value_mode="raw"))
        fsl, frames = 4, 3
        _store(m, frames=frames, fsl=fsl)
        e = m.entries_by_layer[0][0]
        self.assertEqual(e.num_tokens, fsl * frames)

    def test_retrieval_deterministic(self):
        cfg = KVRAGConfig(enabled=True, top_k=2)
        m = KVRAGMemory(cfg)
        torch.manual_seed(3)
        for i in range(5):
            _store(m, start=i * 8)
        q = torch.randn(1, 4, 2, 3)
        r1 = m.retrieve(layer=0, query=q.clone(), current_start=1000,
                        frame_seqlen=4, dtype=torch.float32, device=CPU)
        r2 = m.retrieve(layer=0, query=q.clone(), current_start=1000,
                        frame_seqlen=4, dtype=torch.float32, device=CPU)
        self.assertEqual([id(e) for e in r1[2]], [id(e) for e in r2[2]])
        self.assertTrue(torch.equal(r1[0], r2[0]))

    def test_pooled_key_matches_legacy_summary(self):
        # AC-4: default key == the legacy mean-pool, byte-identical.
        m = KVRAGMemory(KVRAGConfig(enabled=True))  # default pooled
        t = torch.randn(1, 8, 2, 3)
        self.assertTrue(torch.equal(m._compute_key(t), m._summarize(t)))

    def test_attention_diagnostic_records_persistent_mass_by_shot_layer(self):
        cfg = KVRAGConfig(
            enabled=True,
            scene_memory_enabled=True,
            attention_diagnostic=True,
            attention_diag_max_query_rows=2,
        )
        m = KVRAGMemory(cfg)
        persistent_k, persistent_v = _store(m, persistent=True, start=0)
        transient_k, transient_v = _store(m, persistent=False, start=100)
        selected = [m.scene_entries_by_layer[0][0], m.entries_by_layer[0][0]]
        local_k = torch.randn(1, 4, 2, 3)
        window_k = torch.cat([persistent_k, transient_k, local_k], dim=1)
        query = persistent_k[:, :4].clone()

        m.set_runtime_context(chunk_index=3, shot_index=2, phase="denoise")
        m.record_injected_attention_mass(query, window_k, selected, layer=0)
        diag = m.export_diagnostics()

        self.assertEqual(diag["stats"]["persistent_rag_mass_calls"], 1)
        self.assertIn("2", diag["attention_mass_by_shot_layer"])
        self.assertIn("0", diag["attention_mass_by_shot_layer"]["2"])
        self.assertGreater(
            diag["attention_mass_by_shot_layer"]["2"]["0"]["mean_mass"], 0.0
        )


# ---------------------------------------------------------------------------
# AC-3: viewpoint-invariance probe + cross-scene contamination
# ---------------------------------------------------------------------------
class TestViewpointInvarianceProbe(unittest.TestCase):
    """Same scene = same set of scene elements, rearranged in the frame
    (a camera move). A good key recognizes it across views; a position-sensitive
    key does not."""

    def setUp(self):
        torch.manual_seed(0)
        B, H, D, T = 1, 2, 4, 12
        a = torch.randn(B, 1, H, D) * 3.0
        b = torch.randn(B, 1, H, D) * 3.0
        c = torch.randn(B, 1, H, D) * 3.0
        d = torch.randn(B, 1, H, D) * 3.0
        bg = lambda: 0.05 * torch.randn(B, T, H, D)
        self.view_a = bg(); self.view_a[:, 0:3] = a; self.view_a[:, T - 3:T] = b
        self.view_b = bg(); self.view_b[:, 0:3] = b; self.view_b[:, T - 3:T] = a  # same scene, new framing
        self.diff = bg(); self.diff[:, 0:3] = c; self.diff[:, T - 3:T] = d        # different scene

    def _score(self, mode, ref, cand, **kw):
        m = KVRAGMemory(KVRAGConfig(enabled=True, retrieval_key_mode=mode, **kw))
        qs = m._compute_key(ref)
        fake = type("E", (), {"summary": m._compute_key(cand), "persistent": False})()
        return m._score_candidates(qs, [fake])[0]

    def test_invariant_keys_recognize_same_scene(self):
        for mode, kw in (("pooled", {}), ("moment", {}),
                         ("multi_centroid", dict(retrieval_key_centroids=4)),
                         ("salient_set", dict(retrieval_key_top_m=6))):
            same = self._score(mode, self.view_a, self.view_b, **kw)
            diff = self._score(mode, self.view_a, self.diff, **kw)
            self.assertGreater(same, diff + 0.1, f"{mode}: same-scene not preferred")
            self.assertGreater(same, 0.6, f"{mode}: same-scene similarity too low")

    def test_view_sensitive_key_fails_probe(self):
        # The 'positional' key cannot tell it is the same scene across views;
        # its same-scene score collapses well below the invariant keys'.
        same_pos = self._score("positional", self.view_a, self.view_b)
        same_pooled = self._score("pooled", self.view_a, self.view_b)
        self.assertLess(same_pos, 0.5)
        self.assertLess(same_pos, same_pooled - 0.3)

    def test_cross_scene_contamination_rejected(self):
        # AC-3 negative: at equal raw-token distance, the same scene (rearranged)
        # is preferred over a genuinely different scene. Retrieval must pick the
        # same-scene anchor.
        m = KVRAGMemory(KVRAGConfig(enabled=True, top_k=1,
                                    retrieval_key_mode="multi_centroid",
                                    retrieval_key_centroids=4))
        fsl = 12

        def add(t, start):
            m.add(layer=0, k_pre=t, k_post=t, v=t, start_token=start,
                  end_token=start + fsl, frame_seqlen=fsl, h=2, w=2, frames=1)

        add(self.view_b, 0)    # same scene, different view
        add(self.diff, 12)     # different scene
        res = m.retrieve(layer=0, query=self.view_a, current_start=1000,
                         frame_seqlen=fsl, dtype=torch.float32, device=CPU)
        chosen = res[2][0]
        self.assertTrue(torch.equal(chosen.k, self.view_b),
                        "retrieval picked the different scene over the same scene")


# ---------------------------------------------------------------------------
# AC-3 scene-aware scoring
# ---------------------------------------------------------------------------
class TestSceneAwareScoring(unittest.TestCase):
    def test_scene_bonus_breaks_ties_toward_persistent(self):
        # Two candidates with identical content (same score); the persistent one
        # wins once a positive scene bonus is applied.
        torch.manual_seed(1)
        shared = torch.randn(1, 8, 2, 3)
        cfg = KVRAGConfig(enabled=True, top_k=1, scene_memory_enabled=True,
                          scene_score_bonus=0.5)
        m = KVRAGMemory(cfg)
        # identical content in both partitions
        m.add(layer=0, k_pre=shared, k_post=shared, v=shared, start_token=0,
              end_token=8, frame_seqlen=4, h=2, w=2, frames=2, persistent=True)
        m.add(layer=0, k_pre=shared, k_post=shared, v=shared, start_token=8,
              end_token=16, frame_seqlen=4, h=2, w=2, frames=2, persistent=False)
        q = torch.randn(1, 4, 2, 3)
        res = m.retrieve(layer=0, query=q, current_start=1000, frame_seqlen=4,
                         dtype=torch.float32, device=CPU)
        self.assertTrue(res[2][0].persistent)


if __name__ == "__main__":
    unittest.main()
