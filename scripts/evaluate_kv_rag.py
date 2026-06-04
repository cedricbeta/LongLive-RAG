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

from evaluation.multiview_prompts import build_spec_resolver
from evaluation.video_consistency import (
    build_clip_adherence_scorer,
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
        "exact per-shot boundaries (shot_durations.txt) and shot captions are "
        "resolved from the theme matching each generated video stem "
        "(overrides --num_shots).",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="cross_perspective: rendered generation-chunk budget (num_output_frames"
        " // num_frame_per_block). Clamps per-shot durations so boundaries match "
        "the shots actually rendered. Omit to assume full shot coverage.",
    )
    parser.add_argument(
        "--score_adherence",
        action="store_true",
        help="cross_perspective: also score per-shot prompt adherence with a "
        "frozen CLIP backbone (milestone-only; needs --prompts_dir for captions "
        "and a CLIP checkpoint). Off by default so the metric stays GPU-free.",
    )
    parser.add_argument(
        "--clip_model", default="ViT-B-32", help="open_clip model name for adherence scoring."
    )
    parser.add_argument(
        "--clip_pretrained", default="openai", help="open_clip pretrained tag for adherence."
    )
    parser.add_argument(
        "--clip_device", default=None, help="Device for the CLIP adherence scorer (auto if unset)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cross_perspective":
        if args.prompts_dir:
            shots_for = build_spec_resolver(
                args.prompts_dir,
                with_captions=args.score_adherence,
                max_chunks=args.num_blocks,
            )
        elif args.num_shots is not None:
            shots_for = args.num_shots
        else:
            raise SystemExit("cross_perspective mode requires --num_shots or --prompts_dir")
        adherence_scorer = None
        if args.score_adherence:
            if not args.prompts_dir:
                raise SystemExit("--score_adherence requires --prompts_dir for shot captions")
            adherence_scorer = build_clip_adherence_scorer(
                model_name=args.clip_model,
                pretrained=args.clip_pretrained,
                device=args.clip_device,
            )
        result = compare_cross_perspective_dirs(
            args.baseline_dir,
            args.kv_rag_dir,
            shots_for=shots_for,
            adherence_scorer=adherence_scorer,
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
