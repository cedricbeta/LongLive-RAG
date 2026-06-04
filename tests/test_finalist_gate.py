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
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

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


class TestBackboneGuard(unittest.TestCase):
    def test_try_build_backbone_returns_none_on_failure(self):
        def boom():
            raise RuntimeError("backbone unavailable")
        self.assertIsNone(abl._try_build_backbone("x", boom))      # missing -> None (no crash)
        self.assertEqual(abl._try_build_backbone("y", lambda: 42), 42)  # available -> value

    def test_missing_requested_backbone_flags_fail_closed(self):
        # The non-finalist gate marks a requested-but-unloaded backbone for fail-closed.
        backbones = {
            "subject_dino": {"requested": True, "loaded": False},
            "background_clip": {"requested": False, "loaded": False},
            "identity": {"requested": True, "loaded": True},
            "subject_kind": "auto",
        }
        self.assertEqual(abl._missing_requested_backbones(backbones), ["subject_dino"])


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

    def test_dry_run_forces_null(self):
        # A --dry_run_without_adherence run (gate_evaluated=False) cannot select a
        # winner even if a finalist's gate reported passed=True (guard was skipped).
        ranked, winner, reason = abl._finalize_finalist_ranking(_records(), [], gate_evaluated=False)
        self.assertIsNone(winner)
        self.assertIsNotNone(reason)
        self.assertIn("dry_run", reason)
        self.assertTrue(all(not r["passed"] for r in ranked))

    def test_missing_backbone_and_dry_run_both_reported(self):
        ranked, winner, reason = abl._finalize_finalist_ranking(
            _records(), ["subject_dino"], gate_evaluated=False)
        self.assertIsNone(winner)
        self.assertIn("subject_dino", reason)
        self.assertIn("dry_run", reason)


class TestFinalistKvRagMerge(unittest.TestCase):
    def test_base_kv_rag_honored_then_settings_win(self):
        # The requested config's KV-RAG block (layers/limits/hyperparams) must be
        # merged before the finalist key/value settings, which take precedence. (P2 fix.)
        base = {"layers": [0, 5, 10], "top_k": 4, "max_tokens_per_entry": 512,
                "retrieval_key_mode": "pooled"}
        settings = {"retrieval_key_mode": "semantic", "retrieval_value_mode": "raw",
                    "scene_memory_enabled": True}
        merged = abl._finalist_kv_rag(base, settings)
        self.assertEqual(merged["layers"], [0, 5, 10])             # base honored
        self.assertEqual(merged["top_k"], 4)
        self.assertEqual(merged["max_tokens_per_entry"], 512)
        self.assertTrue(merged["enabled"])
        self.assertEqual(merged["retrieval_key_mode"], "semantic")  # finalist setting wins
        self.assertTrue(merged["scene_memory_enabled"])

    def test_empty_base_falls_back_to_defaults(self):
        merged = abl._finalist_kv_rag({}, {"retrieval_key_mode": "pooled",
                                           "retrieval_value_mode": "raw"})
        self.assertTrue(merged["enabled"])
        self.assertIn("layers", merged)  # from DEFAULT_KV_RAG
        self.assertEqual(merged["layers"], abl.DEFAULT_KV_RAG["layers"])

    def test_base_kv_rag_block_handles_boolean_shorthand(self):
        # `inference.kv_rag: false/true` is a supported shorthand and must NOT crash
        # OmegaConf.to_container. (P2 fix.)
        from omegaconf import OmegaConf
        self.assertEqual(
            abl._base_kv_rag_block(OmegaConf.create({"inference": {"kv_rag": False}})),
            {"enabled": False})
        self.assertEqual(
            abl._base_kv_rag_block(OmegaConf.create({"inference": {"kv_rag": True}})),
            {"enabled": True})
        self.assertEqual(
            abl._base_kv_rag_block(OmegaConf.create({"inference": {"kv_rag": {"layers": [1, 2]}}})),
            {"layers": [1, 2]})
        self.assertEqual(abl._base_kv_rag_block(OmegaConf.create({"inference": {}})), {})

    def test_finalist_enables_kv_rag_over_boolean_false_base(self):
        from omegaconf import OmegaConf
        base = abl._base_kv_rag_block(OmegaConf.create({"inference": {"kv_rag": False}}))
        merged = abl._finalist_kv_rag(base, {"retrieval_key_mode": "semantic",
                                             "retrieval_value_mode": "raw"})
        self.assertTrue(merged["enabled"])  # the finalist overlay re-enables KV-RAG


def _write_scene(root: Path, name: str, n_perspectives: int):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_perspectives):
        (folder / f"{i}.json").write_text(json.dumps({"caption": f"{name} perspective {i}"}))


class TestMultiviewSubset(unittest.TestCase):
    def test_single_scene_with_two_perspectives_is_allowed(self):
        src = Path(tempfile.mkdtemp())
        _write_scene(src, "lone_scene", 4)
        dest = Path(tempfile.mkdtemp()) / "subset"
        chosen, coverage = abl.build_multiview_subset(str(src), ["lone_scene"], dest)
        self.assertEqual(chosen, ["lone_scene"])           # one scene is fine
        self.assertEqual(coverage["lone_scene"], 4)
        self.assertTrue((dest / "lone_scene" / "0.json").exists())

    def test_scene_with_one_perspective_fails(self):
        src = Path(tempfile.mkdtemp())
        _write_scene(src, "too_few", 1)
        dest = Path(tempfile.mkdtemp()) / "subset"
        with self.assertRaises(ValueError):
            abl.build_multiview_subset(str(src), ["too_few"], dest)

    def test_max_perspectives_one_fails(self):
        # Capping a multi-perspective scene to 1 leaves < 2 to score -> fail fast.
        src = Path(tempfile.mkdtemp())
        _write_scene(src, "plenty", 6)
        dest = Path(tempfile.mkdtemp()) / "subset"
        with self.assertRaises(ValueError):
            abl.build_multiview_subset(str(src), ["plenty"], dest, max_perspectives=1)


if __name__ == "__main__":
    unittest.main()
