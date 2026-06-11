# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free unit tests for the offline cross-perspective consistency metric.

The metric must (a) score same-scene multi-viewpoint clips higher than
independent scenes and (b) expose a degenerate "copy shot 0" collapse via the
anti-cheating companion metrics. Tested on tiny synthetic frame arrays -- no
video files, GPU, or checkpoint required.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.video_consistency import (
    cross_perspective_metrics,
    cross_perspective_consistency,
    evaluate_cross_perspective_gate,
    farneback_dynamic_degree,
    invariant_probe_for_video,
    prompt_text_similarity_lint,
    shot_anchor_centroid_consistency,
    shot_ranges,
)

H = W = 48


def _scene(base_color, n, rng, layout_shift=0, jitter=8):
    """n frames sharing a palette (base_color) with a structured block whose
    position shifts -> a different framing of the same scene."""
    frames = []
    for i in range(n):
        img = np.clip(base_color + rng.randint(-jitter, jitter, (H, W, 3)), 0, 255).astype(np.uint8)
        x = (10 + layout_shift + i) % (W - 10)
        img[10:20, x:x + 10] = 240
        frames.append(img)
    return np.stack(frames)


class TestShotRanges(unittest.TestCase):
    def test_equal_split(self):
        self.assertEqual(shot_ranges(20, num_shots=4), [(0, 5), (5, 10), (10, 15), (15, 20)])

    def test_explicit_boundaries(self):
        self.assertEqual(shot_ranges(20, boundaries=[5, 12]), [(0, 5), (5, 12), (12, 20)])

    def test_uneven_split(self):
        ranges = shot_ranges(10, num_shots=3)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 10)
        self.assertEqual(len(ranges), 3)


class TestCrossPerspectiveMetric(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def _genuine(self):
        return np.concatenate(
            [_scene(np.array([60, 120, 200]), 5, self.rng, layout_shift=s * 8) for s in range(4)], 0
        )

    def _collapse(self):
        shot0 = _scene(np.array([60, 120, 200]), 5, self.rng, layout_shift=0)
        return np.concatenate([shot0, shot0, shot0, shot0], 0)

    def _independent(self):
        colors = [[60, 120, 200], [200, 60, 60], [60, 200, 90], [200, 200, 40]]
        return np.concatenate(
            [_scene(np.array(c), 5, self.rng, layout_shift=i * 8) for i, c in enumerate(colors)], 0
        )

    def test_genuine_more_consistent_than_independent(self):
        g = cross_perspective_metrics(self._genuine(), num_shots=4)
        i = cross_perspective_metrics(self._independent(), num_shots=4)
        self.assertGreater(
            g["palette_agreement"], i["palette_agreement"]
        )

    def test_copy_collapse_is_visible(self):
        # Collapse maxes scene consistency but kills viewpoint variation -> the
        # companion metric makes the cheat visible instead of rewarding it.
        g = cross_perspective_metrics(self._genuine(), num_shots=4)
        c = cross_perspective_metrics(self._collapse(), num_shots=4)
        self.assertGreaterEqual(c["palette_agreement"], 0.99)
        self.assertLess(
            c["inter_shot_composition_diversity"], g["inter_shot_composition_diversity"]
        )
        self.assertLess(c["inter_shot_composition_diversity"], 0.05)

    def test_deterministic(self):
        frames = self._genuine()
        a = cross_perspective_metrics(frames, num_shots=4)
        b = cross_perspective_metrics(frames, num_shots=4)
        self.assertEqual(a, b)

    def test_requires_two_shots(self):
        with self.assertRaises(ValueError):
            cross_perspective_metrics(self._genuine(), num_shots=1)


class TestPureScoring(unittest.TestCase):
    def test_identical_shots_score_one(self):
        scene = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1))
        comp = np.tile(np.array([[0.0, 1.0, 0.0, 0.0]]), (4, 1))
        out = cross_perspective_consistency(scene, comp)
        self.assertAlmostEqual(out["palette_agreement"], 1.0, places=5)
        self.assertAlmostEqual(out["inter_shot_composition_diversity"], 0.0, places=5)

    def test_orthogonal_scenes_score_zero(self):
        scene = np.eye(4)  # four mutually orthogonal shot signatures
        comp = np.eye(4)
        out = cross_perspective_consistency(scene, comp)
        self.assertAlmostEqual(out["palette_agreement"], 0.0, places=5)


def _guard_record(
    *,
    b_cons=0.5,
    m_cons=0.6,
    b_dyn=1.0,
    m_dyn=1.0,
    b_div=0.3,
    m_div=0.3,
    b_adh=0.4,
    m_adh=0.4,
    b_inv=0.2,
    m_inv=0.2,
    negative=False,
):
    return {
        "stem": "rank0-scene-seed0_regular",
        "theme": "scene",
        "negative_control": negative,
        "baseline_metrics": {
            "anchor_centroid_consistency": b_cons,
            "subject_anchor_consistency": b_cons,
            "background_anchor_consistency": b_cons,
            "dynamic_degree": b_dyn,
            "inter_shot_composition_diversity": b_div,
            "prompt_adherence_mean": b_adh,
            "prompt_adherence_min": b_adh,
            "invariant_margin_mean": b_inv,
            "invariant_margin_min": b_inv,
        },
        "modified_metrics": {
            "anchor_centroid_consistency": m_cons,
            "subject_anchor_consistency": m_cons,
            "background_anchor_consistency": m_cons,
            "dynamic_degree": m_dyn,
            "inter_shot_composition_diversity": m_div,
            "prompt_adherence_mean": m_adh,
            "prompt_adherence_min": m_adh,
            "invariant_margin_mean": m_inv,
            "invariant_margin_min": m_inv,
        },
    }


class TestAnchorCentroidMetric(unittest.TestCase):
    def test_centroid_score_uses_subject_and_background_mean(self):
        subject = np.array([[1.0, 0.0], [1.0, 0.0]])
        background = np.array([[0.0, 1.0], [0.0, 1.0]])
        out = shot_anchor_centroid_consistency(subject, background)
        self.assertAlmostEqual(out["subject_anchor_consistency"], 1.0, places=6)
        self.assertAlmostEqual(out["background_anchor_consistency"], 1.0, places=6)
        self.assertAlmostEqual(out["anchor_centroid_consistency"], 1.0, places=6)


class TestGuardGate(unittest.TestCase):
    def _gate(self, rec):
        return evaluate_cross_perspective_gate(
            {"records": [rec]},
            min_consistency_wins=1,
            require_adherence=True,
            adherence_tolerance=0.05,
            diversity_tolerance=0.1,
            motion_tolerance=0.2,
            invariant_tolerance=0.05,
        )

    def test_motion_freeze_rejected_by_relative_tolerance(self):
        gate = self._gate(_guard_record(b_dyn=2.0, m_dyn=1.5))  # needs >= 1.6
        self.assertFalse(gate["passed"])
        self.assertIn("rank0-scene-seed0_regular", gate["motion_failures"])

    def test_copied_shots_rejected_by_diversity(self):
        gate = self._gate(_guard_record(b_div=0.4, m_div=0.2))
        self.assertFalse(gate["passed"])
        self.assertIn("rank0-scene-seed0_regular", gate["diversity_failures"])

    def test_prompt_ignoring_rejected_by_mean_or_min_adherence(self):
        gate = self._gate(_guard_record(b_adh=0.5, m_adh=0.3))
        self.assertFalse(gate["passed"])
        self.assertIn("rank0-scene-seed0_regular", gate["adherence_failures"])

    def test_object_state_overwrite_rejected_by_invariant_margin(self):
        gate = self._gate(_guard_record(b_inv=0.4, m_inv=0.1))
        self.assertFalse(gate["passed"])
        self.assertIn("rank0-scene-seed0_regular", gate["invariant_failures"])

    def test_unscorable_input_fails_closed_with_blocked_reason(self):
        rec = _guard_record()
        rec["modified_metrics"]["anchor_centroid_consistency"] = float("nan")
        gate = self._gate(rec)
        self.assertFalse(gate["passed"])
        self.assertIn("blocked_reason", gate)
        self.assertIn("anchor_centroid_consistency", gate["blocked_reason"])

    def test_negative_control_win_is_not_a_pass(self):
        gate = self._gate(_guard_record(negative=True, m_cons=0.8, m_adh=0.6))
        self.assertFalse(gate["passed"])
        self.assertIn("negative control", gate["blocked_reason"])


class TestInvariantProbe(unittest.TestCase):
    def test_margin_prefers_invariant_over_contrast(self):
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        ranges = [(0, 2), (2, 4)]

        def scorer(sample, caption):
            return 1.0 if "intact" in caption else 0.2

        out = invariant_probe_for_video(frames, ranges, "intact egg", "scrambled egg", scorer)
        self.assertGreater(out["invariant_margin_min"], 0.0)


class TestPromptLint(unittest.TestCase):
    def test_pairwise_text_similarity_floor_rejects_ill_posed_scene(self):
        def encoder(captions):
            return np.eye(len(captions), dtype=np.float64)

        out = prompt_text_similarity_lint(["macro", "wide"], encoder, floor=0.5)
        self.assertFalse(out["passed"])
        self.assertEqual(len(out["matrix"]), 2)


class TestDynamicDegree(unittest.TestCase):
    def test_farneback_detects_frozen_cheat(self):
        moving = _scene(np.array([60, 120, 200]), 6, np.random.RandomState(2), layout_shift=0)
        frozen = np.stack([moving[0]] * moving.shape[0])
        self.assertGreater(farneback_dynamic_degree(moving), farneback_dynamic_degree(frozen))


if __name__ == "__main__":
    unittest.main()
