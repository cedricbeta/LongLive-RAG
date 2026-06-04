# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the offline retrieval-candidate screen (AC-3 pre-filter).

The screen is a labeled PROXY: it must prune the position-sensitive negative
control, keep the viewpoint-invariant keys (including the two new ones), expose
value structural facts, and never present itself as the selector.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.retrieval_screen import run_screen, screen_key, screen_value
from utils.kv_rag import KEY_MODES, VALUE_MODES


class TestScreenKeys(unittest.TestCase):
    def test_positional_is_the_only_failing_control(self):
        results = {k: screen_key(k) for k in KEY_MODES}
        self.assertFalse(results["positional"]["passes_prefilter"])
        self.assertTrue(results["positional"]["is_negative_control"])
        for k in ("pooled", "moment", "multi_centroid", "salient_set",
                  "subject_identity", "semantic"):
            self.assertTrue(results[k]["passes_prefilter"], f"{k} should pass the proxy")

    def test_new_keys_are_non_degenerate(self):
        for k in ("subject_identity", "semantic"):
            r = screen_key(k)
            self.assertGreater(r["same_scene"], 0.6)
            self.assertGreater(r["margin"], 0.1)

    def test_margin_is_same_minus_diff(self):
        r = screen_key("pooled")
        self.assertAlmostEqual(r["margin"], r["same_scene"] - r["different_scene"], places=6)


class TestScreenValues(unittest.TestCase):
    def test_value_structural_facts(self):
        facts = {v["value"]: v for v in (screen_value(m) for m in VALUE_MODES)}
        self.assertFalse(facts["raw"]["bounded"])      # raw keeps all frames
        self.assertEqual(facts["raw"]["stored_frames"], 3)
        self.assertTrue(facts["mean_frame"]["bounded"])
        self.assertEqual(facts["mean_frame"]["stored_frames"], 1)
        self.assertTrue(facts["top_frame"]["bounded"])
        self.assertEqual(facts["top_frame"]["stored_frames"], 1)
        for v in facts.values():
            self.assertTrue(v["reropeable"])


class TestRunScreen(unittest.TestCase):
    def test_is_labeled_prefilter_not_selector(self):
        r = run_screen()
        self.assertTrue(r["is_prefilter"])
        self.assertIn("AC-2", r["selector"])
        self.assertIn("PROXY", r["note"])

    def test_full_matrix_and_shortlist(self):
        r = run_screen()
        self.assertEqual(len(r["cells"]), len(KEY_MODES) * len(VALUE_MODES))  # 21
        self.assertNotIn("positional", r["shortlist_keys"])
        self.assertIn("subject_identity", r["shortlist_keys"])
        self.assertIn("semantic", r["shortlist_keys"])
        # positional cells are screen-only controls, never render candidates.
        for c in r["cells"]:
            if c["key"] == "positional":
                self.assertFalse(c["render_candidate"])


if __name__ == "__main__":
    unittest.main()
