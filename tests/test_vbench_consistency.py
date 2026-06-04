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
    cross_video_consistency,
    group_perspectives_by_scene,
    mean_pairwise_cosine,
    mean_to_first_cosine,
    optical_flow_dynamics,
    video_mean_embedding,
)


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


if __name__ == "__main__":
    unittest.main()
