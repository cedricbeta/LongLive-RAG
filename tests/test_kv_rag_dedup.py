# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for KV-RAG sink/window token-range dedup (AC-2 no-duplication).

Verifies that a scene anchor (or content match) whose absolute token range is
already live in the attended sink/pinned/local window is suppressed at retrieval
-- not re-injected -- while a non-overlapping anchor is still injected.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kv_rag import KVRAGConfig, KVRAGMemory

CPU = torch.device("cpu")


def _store(memory, *, layer=0, frames=2, fsl=4, heads=2, dim=3, start=0, persistent=False):
    t = frames * fsl
    k = torch.randn(1, t, heads, dim)
    v = torch.randn(1, t, heads, dim)
    memory.add(
        layer=layer, k_pre=k, k_post=k, v=v,
        start_token=start, end_token=start + t,
        frame_seqlen=fsl, h=heads, w=2, frames=frames,
        persistent=persistent,
    )


class TestOverlapHelper(unittest.TestCase):
    def test_half_open_overlap(self):
        f = KVRAGMemory._overlaps_any
        self.assertTrue(f(0, 8, [(4, 12)]))     # partial overlap
        self.assertTrue(f(4, 12, [(0, 8)]))     # partial overlap (other side)
        self.assertTrue(f(2, 6, [(0, 100)]))    # contained
        self.assertFalse(f(8, 16, [(0, 8)]))    # touching at boundary, half-open
        self.assertFalse(f(0, 8, [(8, 16)]))    # touching at boundary, half-open
        self.assertFalse(f(0, 8, None))         # no ranges
        self.assertFalse(f(0, 8, []))           # empty ranges


class TestExcludeTokenRanges(unittest.TestCase):
    def _mem(self, **kw):
        params = dict(
            enabled=True, top_k=1, scene_memory_enabled=True,
            scene_memory_max_entries=8, boundary_inject_anchors=2,
        )
        params.update(kw)
        return KVRAGMemory(KVRAGConfig(**params))

    def _populate_anchors(self, m):
        # Three persistent anchors at absolute token ranges [0,8) [8,16) [16,24).
        for i in range(3):
            _store(m, persistent=True, start=i * 8)

    def test_overlapping_anchor_suppressed_nonoverlapping_injected(self):
        m = self._mem()
        self._populate_anchors(m)
        q = torch.randn(1, 4, 2, 3)
        m.set_boundary_inject(True)
        # Exclude the middle anchor's range; it must not be injected, while the
        # other two anchors remain available.
        _, _, selected = m.retrieve(
            layer=0, query=q, current_start=1000, frame_seqlen=4,
            dtype=torch.float32, device=CPU, exclude_token_ranges=[(8, 16)],
        )
        starts = {e.start_token for e in selected}
        self.assertNotIn(8, starts)          # overlapping anchor suppressed
        self.assertTrue(starts.issubset({0, 16}))
        self.assertTrue(starts)              # non-overlapping anchors still injected

    def test_probe_is_non_vacuous_without_exclusion(self):
        # Without the exclusion the same middle anchor IS selectable -> proves the
        # suppression above is caused by the exclusion, not by chance.
        m = self._mem()
        self._populate_anchors(m)
        q = torch.randn(1, 4, 2, 3)
        m.set_boundary_inject(True)
        _, _, selected = m.retrieve(
            layer=0, query=q, current_start=1000, frame_seqlen=4,
            dtype=torch.float32, device=CPU,
        )
        starts = {e.start_token for e in selected}
        self.assertIn(8, starts)  # forced anchors are the two most-recent (8 and 16)

    def test_content_match_overlapping_window_excluded(self):
        # A per-shot (content) entry whose range is live in the local window is
        # not retrieved even though end_token <= current_start.
        m = KVRAGMemory(KVRAGConfig(enabled=True, top_k=4))
        for i in range(3):
            _store(m, persistent=False, start=i * 8)  # [0,8) [8,16) [16,24)
        q = torch.randn(1, 4, 2, 3)
        # current_start=24; pretend the local window covers [16, 24).
        _, _, selected = m.retrieve(
            layer=0, query=q, current_start=24, frame_seqlen=4,
            dtype=torch.float32, device=CPU, exclude_token_ranges=[(16, 24)],
        )
        starts = {e.start_token for e in selected}
        self.assertNotIn(16, starts)
        self.assertTrue(starts.issubset({0, 8}))

    def test_all_excluded_returns_none(self):
        m = KVRAGMemory(KVRAGConfig(enabled=True, top_k=4))
        for i in range(2):
            _store(m, persistent=False, start=i * 8)
        q = torch.randn(1, 4, 2, 3)
        res = m.retrieve(
            layer=0, query=q, current_start=1000, frame_seqlen=4,
            dtype=torch.float32, device=CPU, exclude_token_ranges=[(0, 1000)],
        )
        self.assertIsNone(res)

    def test_no_exclude_arg_is_backward_compatible(self):
        # Omitting exclude_token_ranges reproduces the prior retrieval (no filter).
        m = KVRAGMemory(KVRAGConfig(enabled=True, top_k=4))
        for i in range(3):
            _store(m, persistent=False, start=i * 8)
        q = torch.randn(1, 4, 2, 3)
        res = m.retrieve(
            layer=0, query=q, current_start=1000, frame_seqlen=4,
            dtype=torch.float32, device=CPU,
        )
        self.assertIsNotNone(res)
        self.assertEqual(len(res[2]), 3)


if __name__ == "__main__":
    unittest.main()
