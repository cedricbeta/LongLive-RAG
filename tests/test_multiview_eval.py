# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the cross-perspective evaluation harness (AC-6/AC-7).

Covers the prompt/video resolver (uneven shot_durations, generated-stem
matching, clamping to the rendered chunk budget, captions), the chunk->frame
boundary conversion, the prompt-adherence guard reduction, and the milestone
gate pass/fail decision. No video decode, GPU, CLIP checkpoint, or network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.multiview_prompts import (
    build_spec_resolver,
    clamp_chunk_durations,
    load_shot_specs,
    match_theme,
)
from evaluation.video_consistency import (
    chunk_durations_to_boundaries,
    cross_perspective_pair_key,
    evaluate_cross_perspective_gate,
    pair_cross_perspective_dirs,
    prompt_adherence_for_video,
    shot_ranges,
)


def _write_theme(root: Path, name: str, durations, *, with_txt=True, num_blocks=None):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(durations):
        meta = {"caption": f"{name} shot {i}"}
        if num_blocks is not None:
            meta["num_blocks"] = d
        with open(folder / f"{i}.json", "w") as fh:
            json.dump(meta, fh)
    if with_txt:
        (folder / "shot_durations.txt").write_text("\n".join(str(d) for d in durations))


class TestResolver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp)
        # frying_egg-like uneven folder: 7 shots, last one shorter.
        _write_theme(root, "frying_egg_closeup", [7, 7, 7, 7, 7, 7, 6])
        # even folder.
        _write_theme(root, "african_savanna", [6, 6, 6, 6])
        # durations only from per-JSON num_blocks (no shot_durations.txt).
        _write_theme(root, "indoor_tender_moment", [3, 5], with_txt=False, num_blocks=True)

    def test_uneven_durations_and_captions(self):
        specs = load_shot_specs(self.tmp)
        self.assertEqual(specs["frying_egg_closeup"]["chunk_durations"], [7, 7, 7, 7, 7, 7, 6])
        self.assertEqual(len(specs["frying_egg_closeup"]["captions"]), 7)
        self.assertEqual(specs["frying_egg_closeup"]["captions"][0], "frying_egg_closeup shot 0")

    def test_durations_from_json_fallback(self):
        specs = load_shot_specs(self.tmp)
        self.assertEqual(specs["indoor_tender_moment"]["chunk_durations"], [3, 5])

    def test_generated_stem_matching(self):
        themes = ["frying_egg_closeup", "african_savanna", "indoor_tender_moment"]
        self.assertEqual(
            match_theme("kv_rag-rank0-frying_egg_closeup-seed0_regular", themes),
            "frying_egg_closeup",
        )
        self.assertEqual(
            match_theme("baseline-rank3-african_savanna-seed0_lora", themes),
            "african_savanna",
        )
        self.assertEqual(match_theme("african_savanna", themes), "african_savanna")
        self.assertIsNone(match_theme("rank0-unknown_theme-seed0_regular", themes))

    def test_resolver_clamps_to_num_blocks(self):
        resolve = build_spec_resolver(self.tmp, with_captions=True, max_chunks=16)
        spec = resolve("kv_rag-rank0-frying_egg_closeup-seed0_regular")
        # [7,7,7,7,7,7,6] clamped to 16 -> [7,7,2]; captions truncated to match.
        self.assertEqual(spec["chunk_durations"], [7, 7, 2])
        self.assertEqual(len(spec["captions"]), 3)
        self.assertEqual(spec["theme"], "frying_egg_closeup")

    def test_resolver_none_for_unmatched(self):
        resolve = build_spec_resolver(self.tmp)
        self.assertIsNone(resolve("kv_rag-rank0-not_a_theme-seed0_regular"))


class TestClampAndBoundaries(unittest.TestCase):
    def test_clamp_mirrors_dataset(self):
        self.assertEqual(clamp_chunk_durations([7, 7, 7, 7, 7, 7, 6], 16), [7, 7, 2])
        self.assertEqual(clamp_chunk_durations([6, 6, 6, 6], 16), [6, 6, 4])
        self.assertEqual(clamp_chunk_durations([6, 6], 0), [6, 6])  # no clamp

    def test_chunk_durations_to_boundaries(self):
        # 3 shots [6,6,4] over 64 frames -> starts at frame 24 and 48.
        self.assertEqual(chunk_durations_to_boundaries(64, [6, 6, 4]), [24, 48])

    def test_uneven_boundaries_partition_all_frames(self):
        durations = [7, 7, 2]
        boundaries = chunk_durations_to_boundaries(50, durations)
        ranges = shot_ranges(50, boundaries=boundaries)
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 50)

    def test_too_few_shots_returns_none(self):
        self.assertIsNone(chunk_durations_to_boundaries(64, [16]))
        self.assertIsNone(chunk_durations_to_boundaries(0, [6, 6]))


class TestAdherenceReduction(unittest.TestCase):
    def test_mean_and_min_over_shots(self):
        frames = np.zeros((9, 4, 4, 3), dtype=np.uint8)
        ranges = [(0, 3), (3, 6), (6, 9)]
        captions = ["a", "b", "c"]
        # Fake scorer: shot 1 (caption "b") is the weak shot.
        scores = {"a": 0.4, "b": 0.1, "c": 0.5}

        def scorer(frame_block, caption):
            return scores[caption]

        out = prompt_adherence_for_video(frames, ranges, captions, scorer)
        self.assertAlmostEqual(out["prompt_adherence_mean"], (0.4 + 0.1 + 0.5) / 3, places=6)
        self.assertAlmostEqual(out["prompt_adherence_min"], 0.1, places=6)

    def test_no_captions_is_nan(self):
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        out = prompt_adherence_for_video(frames, [(0, 4)], [""], lambda f, c: 1.0)
        self.assertTrue(np.isnan(out["prompt_adherence_mean"]))


def _record(stem, b_cons, m_cons, b_adh=None, m_adh=None):
    bm = {"cross_shot_scene_consistency": b_cons}
    mm = {"cross_shot_scene_consistency": m_cons}
    if b_adh is not None:
        bm["prompt_adherence_mean"] = b_adh
        bm["prompt_adherence_min"] = b_adh
        mm["prompt_adherence_mean"] = m_adh
        mm["prompt_adherence_min"] = m_adh
    return {"stem": stem, "baseline_metrics": bm, "modified_metrics": mm}


class TestGateDecision(unittest.TestCase):
    def test_pass_on_two_consistency_wins(self):
        result = {"records": [_record("a", 0.5, 0.6), _record("b", 0.5, 0.7)]}
        gate = evaluate_cross_perspective_gate(result)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["consistency_wins"], 2)

    def test_fail_with_only_one_win(self):
        result = {"records": [_record("a", 0.5, 0.6), _record("b", 0.7, 0.6)]}
        gate = evaluate_cross_perspective_gate(result)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["consistency_wins"], 1)

    def test_adherence_regression_fails_gate(self):
        # Both consistency wins, but one prompt's adherence drops beyond tolerance.
        result = {"records": [
            _record("a", 0.5, 0.6, b_adh=0.30, m_adh=0.29),
            _record("b", 0.5, 0.7, b_adh=0.30, m_adh=0.10),
        ]}
        gate = evaluate_cross_perspective_gate(
            result, adherence_tolerance=0.05, require_adherence=True
        )
        self.assertFalse(gate["passed"])
        self.assertIn("b", gate["adherence_failures"])
        self.assertNotIn("a", gate["adherence_failures"])

    def test_adherence_within_tolerance_passes(self):
        result = {"records": [
            _record("a", 0.5, 0.6, b_adh=0.30, m_adh=0.28),
            _record("b", 0.5, 0.7, b_adh=0.30, m_adh=0.31),
        ]}
        gate = evaluate_cross_perspective_gate(
            result, adherence_tolerance=0.05, require_adherence=True
        )
        self.assertTrue(gate["passed"])

    def test_nan_adherence_fails_when_required(self):
        result = {"records": [_record("a", 0.5, 0.6), _record("b", 0.5, 0.7)]}
        gate = evaluate_cross_perspective_gate(result, require_adherence=True)
        self.assertFalse(gate["passed"])  # unscored adherence cannot certify


class TestPairing(unittest.TestCase):
    def test_pair_key_strips_variant_prefix(self):
        b = cross_perspective_pair_key("baseline-rank0-frying_egg_closeup-seed0_regular")
        m = cross_perspective_pair_key("kv_rag-rank0-frying_egg_closeup-seed0_regular")
        self.assertEqual(b, m)
        self.assertEqual(b, "rank0-frying_egg_closeup-seed0_regular")

    def test_pair_key_handles_no_prefix(self):
        self.assertEqual(
            cross_perspective_pair_key("rank0-african_savanna-seed0_regular"),
            "rank0-african_savanna-seed0_regular",
        )

    def test_pair_key_non_generated_falls_back_to_stem(self):
        self.assertEqual(cross_perspective_pair_key("some_random_name"), "some_random_name")

    def _touch(self, d, names):
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"")

    def test_pairs_official_filenames(self):
        tmp = Path(tempfile.mkdtemp())
        self._touch(tmp / "baseline", [
            "baseline-rank0-frying_egg_closeup-seed0_regular.mp4",
            "baseline-rank0-african_savanna-seed0_regular.mp4",
        ])
        self._touch(tmp / "kv_rag", [
            "kv_rag-rank0-frying_egg_closeup-seed0_regular.mp4",
            "kv_rag-rank0-african_savanna-seed0_regular.mp4",
        ])
        pairs = pair_cross_perspective_dirs(tmp / "baseline", tmp / "kv_rag")
        self.assertEqual(len(pairs), 2)
        for b, m in pairs:  # paired by prompt, never zip-mismatched
            self.assertEqual(
                cross_perspective_pair_key(b.stem), cross_perspective_pair_key(m.stem)
            )

    def test_missing_render_raises(self):
        tmp = Path(tempfile.mkdtemp())
        self._touch(tmp / "baseline", [
            "baseline-rank0-frying_egg_closeup-seed0_regular.mp4",
            "baseline-rank0-african_savanna-seed0_regular.mp4",
        ])
        self._touch(tmp / "kv_rag", ["kv_rag-rank0-frying_egg_closeup-seed0_regular.mp4"])
        with self.assertRaises(ValueError):
            pair_cross_perspective_dirs(tmp / "baseline", tmp / "kv_rag")

    def test_duplicate_key_raises(self):
        tmp = Path(tempfile.mkdtemp())
        # Two baseline files collapse to the same pair key (different prefixes) -> ambiguous.
        self._touch(tmp / "baseline", [
            "baseline-rank0-african_savanna-seed0_regular.mp4",
            "other-rank0-african_savanna-seed0_regular.mp4",
        ])
        self._touch(tmp / "kv_rag", ["kv_rag-rank0-african_savanna-seed0_regular.mp4"])
        with self.assertRaises(ValueError):
            pair_cross_perspective_dirs(tmp / "baseline", tmp / "kv_rag")


if __name__ == "__main__":
    unittest.main()
