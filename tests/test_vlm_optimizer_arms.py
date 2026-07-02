# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free unit tests for the VLM-arm upgrades: per-scene manual anchor plans,
per-boundary KV selection sanitization, and the prompt-addition review gate.

Run with the standard library (no pytest needed):

    python -m unittest tests.test_vlm_optimizer_arms
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kv_rag import KVRAGConfig, KVRAGMemory
from scripts.run_vlm_closed_judge_ablation import (
    apply_review_verdicts,
    per_scene_anchor_plan,
    qwen_call_fingerprint,
    sanitize_boundary_selection,
    subsample_candidates,
    universal_anchor_plan,
)


def _write_plan(tmpdir: str, payload: dict) -> str:
    path = Path(tmpdir) / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _bank(plan_path: str) -> KVRAGMemory:
    cfg = KVRAGConfig.from_config({
        "enabled": True,
        "top_k": 0,
        "scene_memory_enabled": True,
        "manual_anchor_plan_path": plan_path,
    })
    return KVRAGMemory(cfg)


class PerScenePlanLoading(unittest.TestCase):
    def test_v2_scene_plan_resolves_by_exact_name_and_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_plan(tmp, {
                "version": 2,
                "scenes": {
                    "african_savanna": {"1": [{"frame_index": 0}, {"frame_index": 48}]},
                    "tattooed_noodle_chef": {"2": [96]},
                },
                "boundaries": {"1": [{"frame_index": 7}]},
            })
            bank = _bank(path)
            bank.set_scene_name("african_savanna")
            self.assertEqual(bank._manual_plan_for_current_scene(), {1: [0, 48]})
            # Sample names embed the scene token.
            bank.set_scene_name("kv_only-rank0-tattooed_noodle_chef-seed0")
            self.assertEqual(bank._manual_plan_for_current_scene(), {2: [96]})

    def test_containment_prefers_longest_scene_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_plan(tmp, {
                "scenes": {
                    "frying_egg": {"1": [1]},
                    "frying_egg_closeup": {"1": [2]},
                },
            })
            bank = _bank(path)
            bank.set_scene_name("kv_only-rank0-frying_egg_closeup-seed0")
            self.assertEqual(bank._manual_plan_for_current_scene(), {1: [2]})

    def test_unmatched_or_missing_scene_falls_back_to_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_plan(tmp, {
                "scenes": {"african_savanna": {"1": [0]}},
                "boundaries": {"3": [12, 5]},
            })
            bank = _bank(path)
            bank.set_scene_name("brand_new_scene")
            self.assertEqual(bank._manual_plan_for_current_scene(), {3: [5, 12]})
            bank.set_scene_name(None)
            self.assertEqual(bank._manual_plan_for_current_scene(), {3: [5, 12]})

    def test_legacy_flat_plan_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_plan(tmp, {"boundaries": {"2": [{"frame_index": 4}, {"frame_index": 4}, -1]}})
            bank = _bank(path)
            self.assertEqual(bank._manual_anchor_plan, {2: [4]})
            self.assertEqual(bank._manual_anchor_plan_scenes, {})


class BoundarySelectionSanitization(unittest.TestCase):
    def test_snaps_dedups_and_caps(self):
        item, notes = sanitize_boundary_selection(
            {"selected_frame_indices": [50, 48, 47, 0, 96], "skip": False, "rationale": "r"},
            boundary=3,
            allowed=[0, 48, 96],
            anchor_cap=2,
        )
        # 50 and 47 both snap to 48; dedup keeps one; cap stops at 2.
        self.assertEqual(item["decoded_frame_indices"], [48, 0])
        self.assertTrue(any("snapped" in n for n in notes))

    def test_skip_and_empty_return_none(self):
        item, notes = sanitize_boundary_selection(
            {"selected_frame_indices": [1], "skip": True, "rationale": "no relevant subject"},
            boundary=2, allowed=[0, 48], anchor_cap=4,
        )
        self.assertIsNone(item)
        self.assertTrue(any("skipped" in n for n in notes))
        item, notes = sanitize_boundary_selection(
            {"selected_frame_indices": [], "skip": False, "rationale": ""},
            boundary=2, allowed=[0, 48], anchor_cap=4,
        )
        self.assertIsNone(item)
        self.assertTrue(any("empty" in n for n in notes))


class ReviewGate(unittest.TestCase):
    DECISION = {
        "which_cut_broke": {"cut_index": 1, "how": "x"},
        "global_invariant_additions": ["The host wears a dark blue T-shirt with a green logo."],
        "per_shot_additions": [
            {"shot_index": 0, "text": "The host wears a dark blue T-shirt.", "rationale": "seen"},
            {"shot_index": 1, "text": "Same wood wall panels persist.", "rationale": "env"},
        ],
        "kv_anchor_plan": [],
        "stop_rule": "one bounded refinement round",
    }

    def test_rewrite_and_drop_apply(self):
        verdicts = [
            {"index": 0, "target": "global", "verdict": "rewrite",
             "text": "The host wears a white V-neck T-shirt.", "reason": "contradicts authored invariant"},
            {"index": 0, "target": "per_shot", "verdict": "drop", "text": "", "reason": "contradicts"},
            {"index": 1, "target": "per_shot", "verdict": "keep", "text": "Same wood wall panels persist.",
             "reason": "reinforces"},
        ]
        reviewed, record = apply_review_verdicts(self.DECISION, verdicts)
        self.assertEqual(reviewed["global_invariant_additions"],
                         ["The host wears a white V-neck T-shirt."])
        self.assertEqual([p["text"] for p in reviewed["per_shot_additions"]],
                         ["Same wood wall panels persist."])
        self.assertEqual(len(record["applied"]), 3)
        self.assertEqual(record["invalid"], [])
        # The input decision is not mutated.
        self.assertEqual(len(self.DECISION["per_shot_additions"]), 2)

    def test_out_of_range_and_bad_verdicts_are_recorded_not_applied(self):
        verdicts = [
            {"index": 9, "target": "global", "verdict": "drop", "text": "", "reason": "no such index"},
            {"index": 0, "target": "nowhere", "verdict": "keep", "text": "", "reason": "bad target"},
        ]
        reviewed, record = apply_review_verdicts(self.DECISION, verdicts)
        self.assertEqual(reviewed["global_invariant_additions"], self.DECISION["global_invariant_additions"])
        self.assertEqual(len(record["invalid"]), 2)


class CandidateSubsampling(unittest.TestCase):
    def test_under_cap_is_untouched(self):
        kept, dropped = subsample_candidates([5, 1, 9], 24)
        self.assertEqual(kept, [5, 1, 9])
        self.assertEqual(dropped, [])

    def test_over_cap_keeps_temporal_spread_and_endpoints(self):
        allowed = list(range(0, 480, 10))  # 48 candidates
        kept, dropped = subsample_candidates(allowed, 12)
        self.assertLessEqual(len(kept), 12)
        self.assertIn(0, kept)
        self.assertIn(470, kept)
        self.assertEqual(sorted(set(kept) | set(dropped)), sorted(allowed))
        gaps = [b - a for a, b in zip(kept, kept[1:])]
        self.assertLessEqual(max(gaps), 60)  # no giant temporal hole


class CacheFingerprint(unittest.TestCase):
    def test_prompt_schema_and_image_order_all_change_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.png"
            b = Path(tmp) / "b.png"
            a.write_bytes(b"aaa")
            b.write_bytes(b"bbb")
            base = dict(model="m", prompt="p", image_paths=[a, b], json_schema=None)
            fp = qwen_call_fingerprint(**base)
            self.assertEqual(fp, qwen_call_fingerprint(**base))
            self.assertNotEqual(fp, qwen_call_fingerprint(**{**base, "prompt": "p2"}))
            self.assertNotEqual(fp, qwen_call_fingerprint(**{**base, "json_schema": {"type": "object"}}))
            # Image ORDER matters: Q0/C3-style labels refer to positions.
            self.assertNotEqual(fp, qwen_call_fingerprint(**{**base, "image_paths": [b, a]}))
            # Content change with same path also changes it.
            a.write_bytes(b"mutated")
            self.assertNotEqual(fp, qwen_call_fingerprint(**base))


class AnchorPlanWriters(unittest.TestCase):
    DECISIONS = {
        "scene_a": {"kv_anchor_plan": [
            {"boundary_shot_index": 1, "decoded_frame_indices": [0, 48], "rationale": "r"},
            {"boundary_shot_index": 2, "decoded_frame_indices": [140], "rationale": "r"},
        ]},
        "scene_b": {"kv_anchor_plan": [
            {"boundary_shot_index": 1, "decoded_frame_indices": [96], "rationale": "r"},
        ]},
    }

    def test_per_scene_plan_keeps_scene_choices_separate(self):
        plan = per_scene_anchor_plan(self.DECISIONS, anchor_cap=4)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(
            plan["scenes"]["scene_a"]["1"],
            [{"frame_index": 0}, {"frame_index": 48}],
        )
        self.assertEqual(plan["scenes"]["scene_b"]["1"], [{"frame_index": 96}])
        # scene_b boundary 1 is NOT polluted by scene_a's frames.
        self.assertNotIn({"frame_index": 0}, plan["scenes"]["scene_b"]["1"])
        # Legacy union fallback still exists for unmatched samples.
        self.assertEqual([f["frame_index"] for f in plan["boundaries"]["1"]], [0, 48, 96])

    def test_universal_plan_still_pools_across_scenes(self):
        plan = universal_anchor_plan(self.DECISIONS, anchor_cap=4)
        self.assertEqual([f["frame_index"] for f in plan["boundaries"]["1"]], [0, 48, 96])


if __name__ == "__main__":
    unittest.main()
