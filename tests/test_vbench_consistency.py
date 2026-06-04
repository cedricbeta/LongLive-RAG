# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the VBench-style cross-perspective aggregation.

Covers the pure cross-video math (mean embedding, pairwise / to-first cosine),
the optical-flow dynamic-degree proxy, and the per-scene perspective grouping of
generated filenames. The DINO/CLIP backbones are milestone-only and not exercised
here (no GPU, no downloads).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.vbench_consistency import (
    aggregate_consistency,
    appearance_style_signature,
    appearance_style_video,
    cross_video_consistency,
    evaluate_multiview_vbench_gate,
    group_perspectives_by_scene,
    identity_consistency,
    layout_signature_video,
    mean_pairwise_cosine,
    mean_to_first_cosine,
    motion_smoothness,
    optical_flow_dynamics,
    select_subject_kind,
    temporal_dynamics_signature,
    temporal_style_video,
    video_mean_embedding,
)


def _vbench_record(scene, b_agg, m_agg, *, b_div=0.3, m_div=0.3, b_adh=0.3, m_adh=0.3,
                   b_dyn=2.0, m_dyn=2.0):
    return {
        "scene": scene,
        "baseline_metrics": {"aggregate_consistency": b_agg, "inter_video_diversity": b_div,
                             "dynamic_degree": b_dyn,
                             "prompt_adherence_mean": b_adh, "prompt_adherence_min": b_adh},
        "modified_metrics": {"aggregate_consistency": m_agg, "inter_video_diversity": m_div,
                             "dynamic_degree": m_dyn,
                             "prompt_adherence_mean": m_adh, "prompt_adherence_min": m_adh},
    }


class TestAggregation(unittest.TestCase):
    def test_video_mean_embedding_is_unit_norm(self):
        feats = np.random.RandomState(0).randn(7, 16) * 3.0
        emb = video_mean_embedding(feats)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=6)

    def test_identical_videos_consistency_one(self):
        e = np.array([1.0, 0.0, 0.0, 0.0])
        emb = np.stack([e, e, e])
        out = cross_video_consistency(emb)
        self.assertAlmostEqual(out["pairwise"], 1.0, places=6)
        self.assertAlmostEqual(out["to_first"], 1.0, places=6)
        self.assertAlmostEqual(out["score"], 1.0, places=6)

    def test_orthogonal_videos_consistency_zero(self):
        out = cross_video_consistency(np.eye(4))
        self.assertAlmostEqual(out["pairwise"], 0.0, places=6)
        self.assertAlmostEqual(out["to_first"], 0.0, places=6)

    def test_same_scene_more_consistent_than_unrelated(self):
        rng = np.random.RandomState(1)
        anchor = rng.randn(32)
        same = np.stack([video_mean_embedding((anchor + 0.05 * rng.randn(4, 32))) for _ in range(3)])
        diff = np.stack([video_mean_embedding(rng.randn(4, 32)) for _ in range(3)])
        self.assertGreater(
            cross_video_consistency(same)["score"], cross_video_consistency(diff)["score"]
        )

    def test_to_first_only_compares_against_index_zero(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # to-first: cos(v0,v1)=1, cos(v0,v2)=0 -> mean 0.5
        self.assertAlmostEqual(mean_to_first_cosine(emb), 0.5, places=6)
        # pairwise includes cos(v1,v2)=0 too -> (1+0+0)/3
        self.assertAlmostEqual(mean_pairwise_cosine(emb), 1.0 / 3.0, places=6)


class TestDynamics(unittest.TestCase):
    def test_static_video_has_low_dynamics(self):
        frame = (np.random.RandomState(0).rand(32, 32, 3) * 255).astype(np.uint8)
        static = np.stack([frame] * 6)
        self.assertLess(optical_flow_dynamics(static), 0.5)

    def test_moving_content_has_more_dynamics_than_static(self):
        H = W = 48
        base = np.zeros((H, W, 3), dtype=np.uint8)
        moving = []
        for i in range(6):
            f = base.copy()
            x = 4 + i * 5
            f[10:25, x:x + 10] = 255
            moving.append(f)
        moving = np.stack(moving)
        static = np.stack([moving[0]] * 6)
        self.assertGreater(optical_flow_dynamics(moving), optical_flow_dynamics(static))


class TestPerspectiveGrouping(unittest.TestCase):
    def test_group_by_scene_ordered_by_perspective(self):
        tmp = Path(tempfile.mkdtemp())
        names = [
            "kv_rag-rank0-frying_egg_closeup-p0-seed0_regular.mp4",
            "kv_rag-rank0-frying_egg_closeup-p2-seed0_regular.mp4",
            "kv_rag-rank0-frying_egg_closeup-p1-seed0_regular.mp4",
            "kv_rag-rank0-african_savanna-p0-seed0_regular.mp4",
            "kv_rag-rank0-african_savanna-p1-seed0_regular.mp4",
            "not_a_perspective_video.mp4",  # ignored
        ]
        for n in names:
            (tmp / n).write_bytes(b"")
        groups = group_perspectives_by_scene(tmp)
        self.assertEqual(set(groups), {"frying_egg_closeup", "african_savanna"})
        # frying_egg perspectives are ordered p0,p1,p2 despite file order
        order = [p.stem for p in groups["frying_egg_closeup"]]
        self.assertEqual(order, [
            "kv_rag-rank0-frying_egg_closeup-p0-seed0_regular",
            "kv_rag-rank0-frying_egg_closeup-p1-seed0_regular",
            "kv_rag-rank0-frying_egg_closeup-p2-seed0_regular",
        ])
        self.assertEqual(len(groups["african_savanna"]), 2)

    def test_duplicate_perspective_across_seeds_raises(self):
        # p0 from two seeds must not be silently grouped as two perspectives
        # (that would measure seed-to-seed, not cross-view, consistency). (P2 fix.)
        tmp = Path(tempfile.mkdtemp())
        for n in ["kv_rag-rank0-scene_a-p0-seed0_regular.mp4",
                  "kv_rag-rank0-scene_a-p1-seed0_regular.mp4",
                  "kv_rag-rank0-scene_a-p0-seed1_regular.mp4"]:  # p0 from seed0 AND seed1
            (tmp / n).write_bytes(b"")
        with self.assertRaises(ValueError):
            group_perspectives_by_scene(tmp)

    def test_mixed_seed_distinct_perspectives_raises(self):
        # Distinct perspective indices from DIFFERENT seeds (p0-seed0 + p1-seed1)
        # must also be rejected -- a scene's perspectives must all share one seed.
        tmp = Path(tempfile.mkdtemp())
        for n in ["kv_rag-rank0-scene_a-p0-seed0_regular.mp4",
                  "kv_rag-rank0-scene_a-p1-seed1_regular.mp4"]:  # different seeds
            (tmp / n).write_bytes(b"")
        with self.assertRaises(ValueError):
            group_perspectives_by_scene(tmp)

    def test_single_seed_grouping_unaffected(self):
        tmp = Path(tempfile.mkdtemp())
        for n in ["kv_rag-rank0-scene_a-p0-seed0_regular.mp4",
                  "kv_rag-rank0-scene_a-p1-seed0_regular.mp4"]:
            (tmp / n).write_bytes(b"")
        groups = group_perspectives_by_scene(tmp)
        self.assertEqual(set(groups), {"scene_a"})
        self.assertEqual(len(groups["scene_a"]), 2)


class TestNormalizationGuard(unittest.TestCase):
    """AC-5 negative: a broken (missing) per-frame normalization is caught."""

    def test_high_norm_frame_does_not_dominate(self):
        # One frame has a huge norm; per-frame L2-normalization (the correct
        # aggregation) keeps the video embedding aligned with the BULK direction,
        # not the outlier. A broken aggregation that skipped renormalization would
        # be dragged toward the outlier and fail this assertion.
        bulk_dir = np.array([1.0, 0.0, 0.0, 0.0])
        outlier_dir = np.array([0.0, 1.0, 0.0, 0.0])
        feats = np.stack([bulk_dir, bulk_dir, bulk_dir, 1000.0 * outlier_dir])
        emb = video_mean_embedding(feats)
        # cosine with the bulk direction stays high despite the giant outlier frame.
        self.assertGreater(float(np.dot(emb, bulk_dir)), 0.7)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=6)


class TestIdentityConsistency(unittest.TestCase):
    def test_identity_beats_similarity_at_equal_global_distance(self):
        # AC-2 negative (non-vacuous): with the GLOBAL feature held identical
        # across two configs (so subject_consistency cannot tell them apart), the
        # discriminative IDENTITY embedding still scores same-subject strictly
        # above different-subject. Identity != mere similarity.
        rng = np.random.RandomState(0)

        def unit(x):
            return x / np.linalg.norm(x)

        global_vec = unit(rng.randn(16))
        id_a, id_b, id_c = (unit(rng.randn(16)) for _ in range(3))

        global_same = np.stack([global_vec, global_vec, global_vec])
        global_diff = np.stack([global_vec, global_vec, global_vec])
        # Equal global-feature distance (both are the same global config):
        self.assertAlmostEqual(
            cross_video_consistency(global_same)["score"],
            cross_video_consistency(global_diff)["score"], places=9,
        )

        same_subject = np.stack([id_a, id_a, id_a])
        diff_subject = np.stack([id_a, id_b, id_c])
        s_same = identity_consistency(same_subject)["identity"]
        s_diff = identity_consistency(diff_subject)["identity"]
        self.assertGreater(s_same, s_diff)
        self.assertAlmostEqual(s_same, 1.0, places=6)

    def test_identity_keys_match_expected_names(self):
        out = identity_consistency(np.eye(3))
        self.assertIn("identity", out)
        self.assertIn("identity_pairwise", out)


class TestSubjectKindSelector(unittest.TestCase):
    def test_default_is_auto(self):
        self.assertEqual(select_subject_kind("any_scene"), "auto")

    def test_explicit_map(self):
        kinds = {"indoor_tender_moment": "human", "frying_egg_closeup": "object"}
        self.assertEqual(select_subject_kind("indoor_tender_moment", scene_subject_kinds=kinds), "human")
        self.assertEqual(select_subject_kind("frying_egg_closeup", scene_subject_kinds=kinds), "object")

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            select_subject_kind("s", scene_subject_kinds={"s": "alien"})


class TestMotionSmoothness(unittest.TestCase):
    def test_static_is_smooth(self):
        frame = (np.random.RandomState(0).rand(48, 48, 3) * 255).astype(np.uint8)
        static = np.stack([frame] * 8)
        self.assertGreater(motion_smoothness(static), 0.95)

    def test_linear_pan_is_smoother_than_jitter(self):
        H = W = 48
        pan, jitter = [], []
        rng = np.random.RandomState(1)
        for i in range(8):
            f = np.zeros((H, W, 3), dtype=np.uint8)
            f[10:30, 4 + i * 3: 14 + i * 3] = 255  # steady linear motion
            pan.append(f)
            g = np.zeros((H, W, 3), dtype=np.uint8)
            x = int(rng.randint(0, 30))  # teleporting / jittery motion
            g[10:30, x: x + 10] = 255
            jitter.append(g)
        self.assertGreater(motion_smoothness(np.stack(pan)), motion_smoothness(np.stack(jitter)))

    def test_short_video_is_smooth(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(motion_smoothness(np.stack([frame, frame])), 1.0)


class TestStyleSignatures(unittest.TestCase):
    def _gradient_frame(self, shift):
        x = (np.linspace(0, 255, 32).astype(np.uint8) + shift) % 256
        return np.tile(x, (32, 1))[..., None].repeat(3, axis=2).astype(np.uint8)

    def test_appearance_style_same_palette_more_consistent(self):
        # Two videos with the same palette/texture are more appearance-consistent
        # than one of them vs a very different palette.
        warm = np.zeros((6, 32, 32, 3), dtype=np.uint8); warm[..., 0] = 200
        warm2 = np.zeros((6, 32, 32, 3), dtype=np.uint8); warm2[..., 0] = 190
        cool = np.zeros((6, 32, 32, 3), dtype=np.uint8); cool[..., 2] = 200
        same = np.stack([appearance_style_video(warm), appearance_style_video(warm2)])
        diff = np.stack([appearance_style_video(warm), appearance_style_video(cool)])
        self.assertGreater(mean_pairwise_cosine(same), mean_pairwise_cosine(diff))

    def test_temporal_style_signature_shape_and_norm(self):
        frames = (np.random.RandomState(0).rand(6, 32, 32, 3) * 255).astype(np.uint8)
        sig = temporal_dynamics_signature(frames, bins=8)
        self.assertEqual(sig.shape[0], 10)  # 8 hist bins + mean + std
        emb = temporal_style_video(frames)
        self.assertTrue(np.isclose(np.linalg.norm(emb), 1.0) or np.allclose(emb, 0.0))


class TestAggregate(unittest.TestCase):
    def test_aggregate_averages_available_dims(self):
        m = {"subject_consistency": 0.8, "background_consistency": 0.6,
             "temporal_style": 0.4, "irrelevant": 999.0}
        self.assertAlmostEqual(aggregate_consistency(m), (0.8 + 0.6 + 0.4) / 3, places=6)

    def test_aggregate_ignores_nan_and_absent(self):
        m = {"subject_consistency": 0.8, "background_consistency": float("nan")}
        self.assertAlmostEqual(aggregate_consistency(m), 0.8, places=6)

    def test_aggregate_nan_when_no_dim(self):
        self.assertTrue(np.isnan(aggregate_consistency({"dynamic_degree": 0.5})))


class TestIdentityAutoLazyDino(unittest.TestCase):
    """subject_kind='auto' must NOT build the DINO fallback when ArcFace is
    available -- an env with working InsightFace but no DINO should still score."""

    def test_auto_with_arcface_does_not_build_dino(self):
        import types
        import evaluation.vbench_consistency as vc

        fake_face = types.SimpleNamespace(
            bbox=[0.0, 0.0, 10.0, 10.0], normed_embedding=np.ones(8, dtype=np.float32))

        class _FakeApp:
            def __init__(self, *a, **k):
                pass

            def prepare(self, *a, **k):
                pass

            def get(self, img):
                return [fake_face]

        fake_insight = types.ModuleType("insightface")
        fake_app = types.ModuleType("insightface.app")
        fake_app.FaceAnalysis = _FakeApp
        fake_insight.app = fake_app

        orig_dino = vc._build_dino_patch_encoder

        def _boom(*a, **k):  # DINO unavailable -> would raise if built
            raise RuntimeError("DINO unavailable")

        sys.modules["insightface"] = fake_insight
        sys.modules["insightface.app"] = fake_app
        vc._build_dino_patch_encoder = _boom
        try:
            # auto + working ArcFace + NO DINO must not raise (DINO is lazy here).
            enc = vc.build_identity_encoder(subject_kind="auto", device="cpu")
            out = enc([np.zeros((16, 16, 3), dtype=np.uint8)])  # the (fake) face is detected
            self.assertEqual(tuple(np.asarray(out).shape), (1, 8))
        finally:
            vc._build_dino_patch_encoder = orig_dino
            sys.modules.pop("insightface", None)
            sys.modules.pop("insightface.app", None)


class TestCollapseVisible(unittest.TestCase):
    def test_identical_perspectives_have_zero_diversity(self):
        # AC-2 negative anti-cheat: identical (copy-collapsed) perspectives max out
        # raw consistency but inter_video_diversity collapses toward 0.
        frame = (np.random.RandomState(0).rand(40, 40, 3) * 255).astype(np.uint8)
        clip = np.stack([frame] * 6)
        layouts = np.stack([layout_signature_video(clip) for _ in range(3)])
        diversity = 1.0 - mean_pairwise_cosine(layouts)
        self.assertLess(diversity, 0.02)


class TestMultiviewVbenchGate(unittest.TestCase):
    def test_pass_on_majority_aggregate_wins(self):
        result = {"records": [_vbench_record("a", 0.5, 0.6), _vbench_record("b", 0.5, 0.7)]}
        gate = evaluate_multiview_vbench_gate(result)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["scene_wins"], 2)
        self.assertEqual(gate["min_scene_wins"], 1)  # ceil(2/2)

    def test_null_result_when_no_win(self):
        result = {"records": [_vbench_record("a", 0.6, 0.5), _vbench_record("b", 0.7, 0.6)]}
        gate = evaluate_multiview_vbench_gate(result)
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["is_null_result"])
        self.assertEqual(gate["scene_wins"], 0)

    def test_diversity_collapse_blocks_pass(self):
        # Consistency wins on both scenes, but inter_video_diversity collapses.
        result = {"records": [
            _vbench_record("a", 0.5, 0.7, b_div=0.30, m_div=0.05),
            _vbench_record("b", 0.5, 0.7, b_div=0.30, m_div=0.04),
        ]}
        gate = evaluate_multiview_vbench_gate(result, diversity_tolerance=0.05)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["diversity_ok"])
        self.assertIn("a", gate["diversity_failures"])

    def test_adherence_regression_blocks_pass(self):
        result = {"records": [
            _vbench_record("a", 0.5, 0.7, b_adh=0.30, m_adh=0.10),
            _vbench_record("b", 0.5, 0.7, b_adh=0.30, m_adh=0.31),
        ]}
        gate = evaluate_multiview_vbench_gate(result, adherence_tolerance=0.05)
        self.assertFalse(gate["passed"])
        self.assertIn("a", gate["adherence_failures"])

    def test_three_scenes_need_two_wins(self):
        result = {"records": [_vbench_record("a", 0.5, 0.6), _vbench_record("b", 0.5, 0.6),
                              _vbench_record("c", 0.6, 0.5)]}
        gate = evaluate_multiview_vbench_gate(result)
        self.assertEqual(gate["min_scene_wins"], 2)  # ceil(3/2)
        self.assertTrue(gate["passed"])

    def test_motion_guard_off_by_default(self):
        # A consistency win with a big dynamic_degree drop passes when the motion
        # guard is OFF (default) -- the drop is reported, not silently failed.
        result = {"records": [_vbench_record("a", 0.5, 0.7, b_dyn=3.0, m_dyn=2.2),
                              _vbench_record("b", 0.5, 0.7, b_dyn=3.0, m_dyn=2.3)]}
        gate = evaluate_multiview_vbench_gate(result)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["motion_ok"])  # not enforced -> ok
        self.assertEqual(gate["per_scene"][0]["dynamic_degree_modified"], 2.2)

    def test_motion_guard_blocks_collapse_when_enabled(self):
        result = {"records": [_vbench_record("a", 0.5, 0.7, b_dyn=3.0, m_dyn=2.2),
                              _vbench_record("b", 0.5, 0.7, b_dyn=3.0, m_dyn=2.95)]}
        gate = evaluate_multiview_vbench_gate(result, motion_tolerance=0.3)
        self.assertFalse(gate["passed"])
        self.assertIn("a", gate["motion_failures"])   # 3.0 -> 2.2 exceeds 0.3 tol
        self.assertNotIn("b", gate["motion_failures"])  # 3.0 -> 2.95 within tol


if __name__ == "__main__":
    unittest.main()
