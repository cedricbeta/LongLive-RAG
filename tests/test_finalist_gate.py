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


class TestLongRegimeProof(unittest.TestCase):
    def _write_prompt_scene(self, root: Path, name: str, shots: int, durations):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(shots):
            (folder / f"{i}.json").write_text(
                json.dumps({"caption": f"{name} shot {i}"}),
                encoding="utf-8",
            )
        (folder / "shot_durations.txt").write_text(
            "\n".join(str(d) for d in durations) + "\n",
            encoding="utf-8",
        )

    def _cfg(self, frames=480):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "num_output_frames": frames,
            "model_kwargs": {"num_frame_per_block": 8, "local_attn_size": 32},
            "data": {"image_or_video_shape": [1, frames, 48, 44, 80]},
        })

    def test_balanced_durations_fill_budget_in_range(self):
        durations = abl._balanced_shot_durations(
            5, 60, fallback_blocks_per_shot=6,
        )
        self.assertEqual(sum(durations), 60)
        self.assertTrue(all(6 <= d <= 12 for d in durations))

    def test_long_regime_report_passes_valid_plan(self):
        root = Path(tempfile.mkdtemp())
        self._write_prompt_scene(root, "scene", 5, [12, 12, 12, 12, 12])
        report = abl._long_regime_report(self._cfg(), root, num_blocks=60)
        self.assertTrue(report["passed"], report.get("blocked_reason"))
        self.assertEqual(report["planned_frames"], 480)

    def test_long_regime_report_blocks_short_plan(self):
        root = Path(tempfile.mkdtemp())
        self._write_prompt_scene(root, "scene", 4, [2, 2, 2, 2])
        report = abl._long_regime_report(self._cfg(frames=128), root, num_blocks=16)
        self.assertFalse(report["passed"])
        self.assertIn("planned frames 128 < 480", report["blocked_reason"])

    def test_long_regime_report_blocks_local_window_cheat(self):
        root = Path(tempfile.mkdtemp())
        self._write_prompt_scene(root, "scene", 6, [2, 2, 2, 2, 2, 6])
        report = abl._long_regime_report(self._cfg(frames=480), root, num_blocks=18)
        self.assertFalse(report["passed"])
        self.assertIn("shot_durations outside 6-12", report["scene_regime"]["scene"]["blocked_reason"])


class TestFrameContractGateHelpers(unittest.TestCase):
    def test_frame_contract_blocks_any_drop(self):
        reason = abl._frame_contract_blocked_reason({
            "scene": {"diagnostic": {
                "pos": {"stats": {
                    "frame_alignment_store_drops": 0,
                    "frame_alignment_inject_drops": 1,
                    "outside_window_injected_frames": 2,
                }},
                "neg": {"stats": {}},
            }}
        })
        self.assertIn("frame-contract drop", reason)

    def test_frame_contract_blocks_zero_outside_window_injection(self):
        reason = abl._frame_contract_blocked_reason({
            "scene": {"diagnostic": {
                "pos": {"stats": {
                    "frame_alignment_store_drops": 0,
                    "frame_alignment_inject_drops": 0,
                    "outside_window_injected_frames": 0,
                }},
                "neg": {"stats": {}},
            }}
        })
        self.assertIn("zero outside-window frame injections", reason)

    def test_kv_rag_contract_summary_requires_frame_cap(self):
        summary = abl._kv_rag_contract_summary({
            "enabled": True,
            "max_frames_per_entry": None,
            "max_tokens_per_entry": 1024,
            "frame_aligned_store": True,
            "require_frame_aligned": True,
            "reinject_rope": True,
        })
        self.assertFalse(summary["frame_level_contract_ok"])
        self.assertIn("max_frames_per_entry missing", summary["blocked_reason"])

    def test_kv_rag_contract_summary_logs_legacy_cap_without_using_it(self):
        summary = abl._kv_rag_contract_summary({
            "enabled": True,
            "max_frames_per_entry": 1,
            "max_tokens_per_entry": 1024,
            "frame_aligned_store": True,
            "require_frame_aligned": True,
            "reinject_rope": True,
            "retrieval_key_mode": "pooled",
            "retrieval_value_mode": "raw",
        })
        self.assertTrue(summary["frame_level_contract_ok"])
        self.assertEqual(summary["max_frames_per_entry"], 1)
        self.assertEqual(summary["legacy_max_tokens_per_entry"], 1024)


class TestStrategyStageCandidates(unittest.TestCase):
    def test_b1_screens_all_keys_with_raw_value(self):
        ns = type("Args", (), {"strategy_stage": "B1", "top_keys": None, "top_combos": None})()
        combos = abl._strategy_stage_candidates(ns)
        self.assertEqual([value for _, value in combos], ["raw"] * len(abl.KEY_MODES))
        self.assertEqual([key for key, _ in combos], list(abl.KEY_MODES))

    def test_b2_uses_top_two_keys_across_all_values(self):
        ns = type("Args", (), {
            "strategy_stage": "B2",
            "top_keys": "attention_native,subject_identity",
            "top_combos": None,
        })()
        combos = abl._strategy_stage_candidates(ns)
        self.assertEqual(len(combos), 2 * len(abl.VALUE_MODES))
        self.assertEqual(combos[:len(abl.VALUE_MODES)],
                         [("attention_native", value) for value in abl.VALUE_MODES])

    def test_verdict_uses_top_two_combos_from_previous_ranking(self):
        ns = type("Args", (), {"strategy_stage": "verdict", "top_keys": None, "top_combos": None})()
        previous = {"ranking": [
            {"key": "attention_native", "value": "raw"},
            {"key": "subject_identity", "value": "attention_mass"},
            {"key": "pooled", "value": "raw"},
        ]}
        self.assertEqual(
            abl._strategy_stage_candidates(ns, previous),
            [("attention_native", "raw"), ("subject_identity", "attention_mass")],
        )

    def test_lever_uses_best_prior_combo_with_requested_lambdas(self):
        ns = type("Args", (), {
            "strategy_stage": "lever",
            "top_keys": None,
            "top_combos": None,
            "logit_bias_lambdas": "1,2",
        })()
        previous = {"ranking": [
            {"key": "attention_native", "value": "raw"},
            {"key": "subject_identity", "value": "attention_mass"},
        ]}
        self.assertEqual(
            abl._strategy_lever_arms(ns, previous),
            [
                {"key": "attention_native", "value": "raw", "persistent_logit_bias_lambda": 1.0},
                {"key": "attention_native", "value": "raw", "persistent_logit_bias_lambda": 2.0},
            ],
        )


class TestStrategyStageHelpers(unittest.TestCase):
    def _previous(self, root: Path):
        prompt_subset = root / "prompts"
        prompt_subset.mkdir(parents=True)
        baseline_dirs = {}
        for seed in (0, 1, 2):
            d = root / f"baseline_seed{seed}"
            d.mkdir()
            baseline_dirs[str(seed)] = str(d)
        noise = {
            "scene_a": {"threshold": 0.02},
            "scene_b": {"threshold": 0.02},
            "scene_c": {"threshold": 0.02},
            "negative": {"threshold": 0.02},
        }
        return {
            "prompt_subset_dir": str(prompt_subset),
            "baseline_seed_dirs": baseline_dirs,
            "baseline_seeds": [0, 1, 2],
            "admitted_main_scenes": ["scene_a", "scene_b", "scene_c"],
            "negative_controls": ["negative"],
            "noise_floor": noise,
            "regime_proof": {"passed": True},
        }

    def test_strategy_context_blocks_missing_previous_before_render(self):
        ns = type("Args", (), {"strategy_stage": "B1", "previous_stage_json": "missing.json"})()
        result = abl._strategy_stage_render_eval(
            ns,
            Path(tempfile.mkdtemp()),
            previous=None,
            candidates=[("pooled", "raw")],
        )
        self.assertFalse(result["render_attempted"])
        self.assertIn("requires --previous_stage_json", result["blocked_reason"])

    def test_strategy_context_requires_three_seed_dirs_for_verdict(self):
        root = Path(tempfile.mkdtemp())
        previous = self._previous(root)
        del previous["baseline_seed_dirs"]["2"]
        context, reason = abl._strategy_context_from_previous(previous, stage="verdict")
        self.assertIsNone(context)
        self.assertIn("baseline seed 2 dir missing", reason)

    def test_strategy_blocked_json_carries_admitted_main_scenes(self):
        result = abl._strategy_blocked_json(
            stage="B1",
            blocked_reason="render incomplete",
            render_attempted=True,
            admitted_main=["scene_a", "scene_b"],
            negative_controls=["negative"],
        )
        self.assertEqual(result["admitted_main_scenes"], ["scene_a", "scene_b"])
        self.assertEqual(result["negative_controls"], ["negative"])
        self.assertTrue(result["render_attempted"])

    def test_strategy_summary_counts_scene_wins_and_frame_mass(self):
        seed_records = [{
            "seed": 0,
            "gate": {
                "adherence_ok": True,
                "diversity_ok": True,
                "motion_ok": True,
                "invariant_ok": True,
                "scorable_ok": True,
                "negative_control_ok": True,
                "frame_contract_ok": True,
            },
            "frontier": [
                {
                    "scene": "scene_a",
                    "negative_control": False,
                    "consistency_delta": 0.03,
                    "attention_mass_mean": 0.2,
                    "per_frame_attention_mass_mean": 0.4,
                },
                {
                    "scene": "scene_b",
                    "negative_control": False,
                    "consistency_delta": 0.01,
                    "attention_mass_mean": 0.1,
                    "per_frame_attention_mass_mean": 0.2,
                },
                {
                    "scene": "negative",
                    "negative_control": True,
                    "consistency_delta": 0.0,
                    "attention_mass_mean": 0.3,
                    "per_frame_attention_mass_mean": 0.6,
                },
            ],
        }]
        rec = abl._strategy_summarize_candidate(
            stage="B1",
            key="pooled",
            value="raw",
            settings={"retrieval_key_mode": "pooled", "retrieval_value_mode": "raw"},
            seed_records=seed_records,
            admitted_main=["scene_a", "scene_b"],
            negative_controls=["negative"],
            noise_floor={
                "scene_a": {"threshold": 0.02},
                "scene_b": {"threshold": 0.02},
                "negative": {"threshold": 0.02},
            },
            min_wins=1,
        )
        self.assertTrue(rec["passed"])
        self.assertEqual(rec["scene_wins"], 1)
        self.assertAlmostEqual(rec["per_frame_attention_mass_mean"], 0.4)

    def test_strategy_summary_fails_on_frame_contract_guard(self):
        rec = abl._strategy_summarize_candidate(
            stage="B1",
            key="pooled",
            value="raw",
            settings={},
            seed_records=[{
                "seed": 0,
                "gate": {"frame_contract_ok": False},
                "frontier": [{"scene": "scene_a", "negative_control": False, "consistency_delta": 0.5}],
            }],
            admitted_main=["scene_a"],
            negative_controls=[],
            noise_floor={"scene_a": {"threshold": 0.01}},
            min_wins=1,
        )
        self.assertFalse(rec["passed"])
        self.assertEqual(rec["guard_status"], "failed")
        self.assertTrue(any("frame_contract_ok=false" in r for r in rec["guard_failures"]))

    def test_lever_summary_requires_rerope_and_bias_manipulation(self):
        rec = abl._strategy_summarize_candidate(
            stage="lever",
            key="pooled",
            value="raw",
            settings={"persistent_logit_bias_lambda": 1.0},
            seed_records=[{
                "seed": 0,
                "gate": {"frame_contract_ok": True},
                "frame_contract": {"scene_a": {
                    "reinject_rope_injected_frames": 0,
                    "persistent_logit_bias_frames": 0,
                    "persistent_logit_bias_calls": 0,
                }},
                "frontier": [{"scene": "scene_a", "negative_control": False, "consistency_delta": 0.5}],
            }],
            admitted_main=["scene_a"],
            negative_controls=[],
            noise_floor={"scene_a": {"threshold": 0.01}},
            min_wins=1,
        )
        self.assertFalse(rec["passed"])
        self.assertTrue(any("re-RoPE" in r for r in rec["guard_failures"]))
        self.assertTrue(any("logit bias" in r for r in rec["guard_failures"]))

    def test_lever_summary_passes_with_rerope_bias_and_noise_win(self):
        rec = abl._strategy_summarize_candidate(
            stage="lever",
            key="pooled",
            value="raw",
            settings={"persistent_logit_bias_lambda": 1.0},
            seed_records=[{
                "seed": 0,
                "gate": {"frame_contract_ok": True},
                "frame_contract": {"scene_a": {
                    "reinject_rope_injected_frames": 2,
                    "persistent_logit_bias_frames": 2,
                    "persistent_logit_bias_calls": 1,
                }},
                "frontier": [{"scene": "scene_a", "negative_control": False, "consistency_delta": 0.5}],
            }],
            admitted_main=["scene_a"],
            negative_controls=[],
            noise_floor={"scene_a": {"threshold": 0.01}},
            min_wins=1,
        )
        self.assertTrue(rec["passed"])
        self.assertEqual(rec["reinject_rope_injected_frames"], 2)
        self.assertEqual(rec["persistent_logit_bias_frames"], 2)

    def test_strategy_screen_selection_uses_guard_clean_candidates(self):
        ranked = [
            {"key": "bad", "value": "raw", "guard_failures": ["frame contract"]},
            {"key": "pooled", "value": "raw", "guard_failures": []},
            {"key": "subject_identity", "value": "raw", "guard_failures": []},
        ]
        selection, reason = abl._strategy_screen_selection("B1", ranked)
        self.assertIsNone(reason)
        self.assertEqual(selection["selected_keys"], ["pooled", "subject_identity"])


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
