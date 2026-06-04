# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the consolidated finalist-gate decision (AC-3/AC-6 fail-closed).

A rendered winner must NOT be selected on a reduced AC-2 suite: if any requested
backbone (DINO subject / CLIP background / identity) fails to load, the gate must
return a null with a blocked_reason, even if a finalist would otherwise pass.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# scripts/ is not a package; load the module by path.
_spec = importlib.util.spec_from_file_location(
    "run_kv_rag_ablation", os.path.join(ROOT, "scripts", "run_kv_rag_ablation.py")
)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)


def _records():
    return [
        {"key": "semantic", "value": "raw", "passed": True, "mean_aggregate_delta": 0.010,
         "scene_wins": 2, "num_scenes": 2},
        {"key": "pooled", "value": "raw", "passed": True, "mean_aggregate_delta": 0.006,
         "scene_wins": 2, "num_scenes": 2},
        {"key": "subject_identity", "value": "raw", "passed": False, "mean_aggregate_delta": 0.004,
         "scene_wins": 1, "num_scenes": 2},
    ]


class TestMissingBackboneDetection(unittest.TestCase):
    def test_extracts_requested_but_unloaded(self):
        backbones = {
            "subject_dino": {"requested": True, "loaded": False},
            "background_clip": {"requested": True, "loaded": True},
            "identity": {"requested": False, "loaded": False},
            "subject_kind": "auto",  # non-dict entry must be ignored
        }
        self.assertEqual(abl._missing_requested_backbones(backbones), ["subject_dino"])

    def test_none_missing_when_all_loaded(self):
        backbones = {
            "subject_dino": {"requested": True, "loaded": True},
            "background_clip": {"requested": True, "loaded": True},
            "identity": {"requested": True, "loaded": True},
            "subject_kind": "auto",
        }
        self.assertEqual(abl._missing_requested_backbones(backbones), [])


class TestFailClosed(unittest.TestCase):
    def test_missing_backbone_forces_null(self):
        ranked, winner, reason = abl._finalize_finalist_ranking(_records(), ["subject_dino"])
        self.assertIsNone(winner)                       # no winner on a reduced suite
        self.assertIsNotNone(reason)
        self.assertTrue(all(not r["passed"] for r in ranked))  # every finalist forced False
        self.assertTrue(all("blocked_reason" in r for r in ranked))

    def test_no_missing_allows_best_winner(self):
        ranked, winner, reason = abl._finalize_finalist_ranking(_records(), [])
        self.assertIsNone(reason)
        self.assertIsNotNone(winner)
        self.assertEqual(winner["key"], "semantic")     # best passing by aggregate delta
        # ranking: passing finalists first, ordered by mean_aggregate_delta desc.
        self.assertEqual([r["key"] for r in ranked][:2], ["semantic", "pooled"])

    def test_no_passing_finalist_is_null_without_blocked(self):
        recs = _records()
        for r in recs:
            r["passed"] = False
        ranked, winner, reason = abl._finalize_finalist_ranking(recs, [])
        self.assertIsNone(winner)
        self.assertIsNone(reason)  # honest null (guards), NOT a backbone block


if __name__ == "__main__":
    unittest.main()
