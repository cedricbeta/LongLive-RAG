#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Evaluate baseline videos against KV-RAG videos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.video_consistency import (
    compare_cross_perspective_dirs,
    compare_video_dirs,
    save_metrics_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True, help="Directory containing baseline mp4 outputs.")
    parser.add_argument("--kv_rag_dir", required=True, help="Directory containing KV-RAG mp4 outputs.")
    parser.add_argument("--output_json", required=True, help="Where to write metric details and aggregates.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on decoded frames per video.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for faster evaluation.")
    parser.add_argument(
        "--mode",
        choices=("temporal", "cross_perspective"),
        default="temporal",
        help="temporal: adjacent/optical-flow consistency. cross_perspective: "
        "single-scene multi-viewpoint scene consistency + anti-cheating companions.",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=None,
        help="cross_perspective: number of shots per video (equal split).",
    )
    parser.add_argument(
        "--prompts_dir",
        default=None,
        help="cross_perspective: directory of <theme>/{0..N}.json prompt sets; "
        "per-video shot count is inferred from the matching theme (overrides --num_shots).",
    )
    return parser.parse_args()


def _shots_from_prompts(prompts_dir: str):
    """Return a stem->num_shots resolver built from a multi-view prompts dir.

    Each theme subfolder holds ``0.json .. N.json`` (one per shot). Videos are
    matched by stem == theme name (e.g. ``african_savanna``).
    """
    from pathlib import Path

    root = Path(prompts_dir)
    counts: dict[str, int] = {}
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        n = len(sorted(sub.glob("[0-9]*.json")))
        if n >= 2:
            counts[sub.name] = n

    def resolve(stem: str):
        return counts.get(stem)

    return resolve


def main() -> None:
    args = parse_args()
    if args.mode == "cross_perspective":
        shots_for = (
            _shots_from_prompts(args.prompts_dir)
            if args.prompts_dir
            else args.num_shots
        )
        if shots_for is None:
            raise SystemExit("cross_perspective mode requires --num_shots or --prompts_dir")
        result = compare_cross_perspective_dirs(
            args.baseline_dir,
            args.kv_rag_dir,
            shots_for=shots_for,
            max_frames=args.max_frames,
            stride=max(1, args.stride),
        )
    else:
        result = compare_video_dirs(
            args.baseline_dir,
            args.kv_rag_dir,
            max_frames=args.max_frames,
            stride=max(1, args.stride),
        )
    save_metrics_json(result, args.output_json)
    print(f"Wrote metrics: {os.path.abspath(args.output_json)}")
    print(f"Compared pairs: {result['num_pairs']}")
    print("Delta means (KV-RAG - baseline):")
    for name, stats in result["delta_summary"].items():
        print(f"  {name}: {stats['mean']:.6f}")


if __name__ == "__main__":
    main()
