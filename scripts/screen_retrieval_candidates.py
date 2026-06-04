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
"""GPU-free offline PRE-FILTER over the retrieval key/value candidate matrix (AC-3).

This ranks candidates on the synthetic viewpoint-invariance proxy and prints the
shortlist. It is NOT the selector: per `BL-20260604-rendered-metric-over-probe`
the proxy mis-ranked keys on rendered video, so finalists must graduate to the
rendered AC-2 gate (`run_kv_rag_ablation.py --mode multiview_vbench`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_screen import run_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", default=None, help="Optional path to write the full ranking.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_screen(seed=args.seed)
    print("OFFLINE PRE-FILTER (proxy only -- NOT the selector).")
    print(f"Selector: {result['selector']}\n")
    print(f"{'key':16s} {'same':>7s} {'diff':>7s} {'margin':>7s}  pass  note")
    for k in result["keys_ranked"]:
        note = "negative control" if k["is_negative_control"] else ""
        print(f"{k['key']:16s} {k['same_scene']:7.3f} {k['different_scene']:7.3f} "
              f"{k['margin']:7.3f}  {'Y' if k['passes_prefilter'] else 'N':>4s}  {note}")
    print(f"\nShortlist (render candidates): {result['shortlist_keys']}")
    print("Values:")
    for v in result["values"]:
        print(f"  {v['value']:11s} frames={v['stored_frames']} tokens={v['stored_tokens']}/{v['input_tokens']} "
              f"bounded={v['bounded']} re-RoPE'able={v['reropeable']}")
    print(f"\n{result['note']}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"\nWrote ranking: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
