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
            g["cross_shot_scene_consistency"], i["cross_shot_scene_consistency"]
        )

    def test_copy_collapse_is_visible(self):
        # Collapse maxes scene consistency but kills viewpoint variation -> the
        # companion metric makes the cheat visible instead of rewarding it.
        g = cross_perspective_metrics(self._genuine(), num_shots=4)
        c = cross_perspective_metrics(self._collapse(), num_shots=4)
        self.assertGreaterEqual(c["cross_shot_scene_consistency"], 0.99)
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
        self.assertAlmostEqual(out["cross_shot_scene_consistency"], 1.0, places=5)
        self.assertAlmostEqual(out["inter_shot_composition_diversity"], 0.0, places=5)

    def test_orthogonal_scenes_score_zero(self):
        scene = np.eye(4)  # four mutually orthogonal shot signatures
        comp = np.eye(4)
        out = cross_perspective_consistency(scene, comp)
        self.assertAlmostEqual(out["cross_shot_scene_consistency"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
