# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the closed judge fail-closed harness."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.closed_judge import (
    build_fail_closed_ledger,
    build_judge_fixtures,
    evaluate_fixture_expectation,
    load_openai_api_key,
    run_judge_validation,
    validate_verdict_shape,
)


def _verdict(subject=0.9, background=0.9, *, copy=False, freeze=False, collapse=False, bias=False):
    return {
        "subject_identity_preserved": {
            "score": subject,
            "verdict": "high" if subject >= 0.65 else "low",
            "rationale": "test",
        },
        "background_coherent": {
            "score": background,
            "verdict": "high" if background >= 0.65 else "low",
            "rationale": "test",
        },
        "per_shot_adherence": [
            {"shot_index": 0, "score": 0.8, "issue": ""},
            {"shot_index": 1, "score": 0.8, "issue": ""},
        ],
        "inter_shot_diversity": {
            "score": 0.8,
            "verdict": "healthy",
            "rationale": "test",
        },
        "motion_continuity": {
            "score": 0.8,
            "verdict": "active",
            "rationale": "test",
        },
        "which_cut_broke": {"cut_index": None, "confidence": 0.0, "how": "none"},
        "cheat_flags": {
            "copy_cheat": copy,
            "freeze_cheat": freeze,
            "prompt_collapse": collapse,
            "position_bias_suspected": bias,
            "rationale": "test",
        },
        "overall_consistency_score": min(subject, background),
    }


class TestAuth(unittest.TestCase):
    def test_null_auth_key_is_missing(self):
        path = Path(tempfile.mkdtemp()) / "auth.json"
        path.write_text(json.dumps({"OPENAI_API_KEY": None, "tokens": {"access_token": "x"}}))
        with mock.patch.dict(os.environ, {}, clear=True):
            key, meta = load_openai_api_key(path)
        self.assertIsNone(key)
        self.assertFalse(meta["present"])
        self.assertIn("null", meta["reason"])

    def test_env_key_wins(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            key, meta = load_openai_api_key("/does/not/matter")
        self.assertEqual(key, "sk-test")
        self.assertEqual(meta["source"], "env:OPENAI_API_KEY")


class TestVerdictShape(unittest.TestCase):
    def test_accepts_complete_shape(self):
        validate_verdict_shape(_verdict())

    def test_rejects_missing_required_field(self):
        v = _verdict()
        del v["cheat_flags"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_verdict_shape(v)

    def test_rejects_extra_field(self):
        v = _verdict()
        v["extra"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_verdict_shape(v)


class TestFixtureExpectations(unittest.TestCase):
    def setUp(self):
        self.fixtures = {f.name: f for f in build_judge_fixtures(tempfile.mkdtemp())}

    def test_subject_swap_must_score_low(self):
        ok, _ = evaluate_fixture_expectation(self.fixtures["subject_swap"], _verdict(subject=0.2))
        self.assertTrue(ok)
        ok, msg = evaluate_fixture_expectation(self.fixtures["subject_swap"], _verdict(subject=0.8))
        self.assertFalse(ok)
        self.assertIn("expected low", msg)

    def test_copy_cheat_must_be_flagged(self):
        ok, _ = evaluate_fixture_expectation(self.fixtures["copy_cheat"], _verdict(copy=True))
        self.assertTrue(ok)
        ok, _ = evaluate_fixture_expectation(self.fixtures["copy_cheat"], _verdict())
        self.assertFalse(ok)

    def test_viewpoint_only_not_penalized(self):
        ok, _ = evaluate_fixture_expectation(self.fixtures["viewpoint_only"], _verdict(subject=0.8))
        self.assertTrue(ok)
        ok, _ = evaluate_fixture_expectation(self.fixtures["viewpoint_only"], _verdict(subject=0.3))
        self.assertFalse(ok)


class TestFailClosedValidation(unittest.TestCase):
    def test_missing_api_key_blocks_before_any_client(self):
        root = Path(tempfile.mkdtemp())
        auth = root / "auth.json"
        auth.write_text(json.dumps({"OPENAI_API_KEY": None}))
        with mock.patch.dict(os.environ, {}, clear=True):
            result = run_judge_validation(output_dir=root, auth_path=auth)
        self.assertFalse(result["passed"])
        self.assertIn("OpenAI API key unavailable", result["blocked_reason"])
        self.assertEqual(result["fixtures"], [])

    def test_ledger_never_claims_pass_on_failed_validation(self):
        validation = {
            "model": "gpt-5.5",
            "passed": False,
            "blocked_reason": "OpenAI API key unavailable",
            "api_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        with mock.patch("evaluation.closed_judge.gpu_inventory", return_value={"available": False}), \
                mock.patch("evaluation.closed_judge.probe_local_vlm", return_value={"selected_optimizer": None}):
            ledger = build_fail_closed_ledger(validation=validation, output_dir="out")
        self.assertFalse(ledger["pass"])
        self.assertTrue(ledger["is_null_result"])
        self.assertFalse(ledger["ablation"]["evaluated"])
        self.assertFalse(ledger["separation"]["optimizer_used"])

    def test_success_path_writes_transcripts_and_cache_with_fake_client(self):
        class FakeClient:
            def create(self, payload):
                first = payload["input"][0]["content"][0]["text"]
                if "subject_swap" in first:
                    verdict = _verdict(subject=0.15)
                elif "copy_cheat" in first:
                    verdict = _verdict(copy=True)
                elif "viewpoint_only_shuffled" in first:
                    verdict = _verdict(subject=0.8, bias=False)
                else:
                    verdict = _verdict(subject=0.9, background=0.9)
                return {
                    "output_text": json.dumps(verdict),
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                }

        root = Path(tempfile.mkdtemp())
        result = run_judge_validation(output_dir=root, client=FakeClient())
        self.assertTrue(result["passed"], result.get("blocked_reason"))
        self.assertEqual(len(result["fixtures"]), 5)  # viewpoint fixture is run twice
        self.assertTrue(result["repeat_run_stability"]["passed"])
        self.assertEqual(result["api_usage"]["total_tokens"], 25)
        for item in result["fixtures"]:
            self.assertTrue(Path(item["transcript"]).exists())
            self.assertTrue(Path(item["cache"]).exists())


if __name__ == "__main__":
    unittest.main()
