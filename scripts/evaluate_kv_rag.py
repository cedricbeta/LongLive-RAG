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

from evaluation.video_consistency import compare_video_dirs, save_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True, help="Directory containing baseline mp4 outputs.")
    parser.add_argument("--kv_rag_dir", required=True, help="Directory containing KV-RAG mp4 outputs.")
    parser.add_argument("--output_json", required=True, help="Where to write metric details and aggregates.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on decoded frames per video.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for faster evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
