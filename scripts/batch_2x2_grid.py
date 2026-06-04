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
"""Batch-pair two ablation folders into 2x2 comparison grids.

For every video in the first folder, find the matching video in the second
folder and render a 2x2 grid (first-frame reference row + live row) via
scripts/make_2x2_grid.py.

Pairing key = the file's basename with a leading ``<dirlabel>-`` prefix removed
(e.g. ``baseline-rank0-000_man...mp4`` and ``kv_rag-rank0-000_man...mp4`` both
reduce to ``rank0-000_man...``), which also matches files that already share an
identical name across the two folders.

Example:
    python scripts/batch_2x2_grid.py \
        videos/kv_rag_3stage_ablation/baseline \
        videos/kv_rag_3stage_ablation/kv_rag \
        -o videos/kv_rag_3stage_ablation/grids \
        --titles "baseline,kv_rag" --cell-height 352
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GRID_SCRIPT = os.path.join(_HERE, "make_2x2_grid.py")


def pair_key(filename: str, prefixes: tuple[str, ...]) -> str:
    """Basename without extension and without a leading '<prefix>-' tag."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    for p in prefixes:
        tag = f"{p}-"
        if stem.startswith(tag):
            return stem[len(tag):]
    return stem


def slugify(key: str, max_len: int = 70) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")
    return slug[:max_len] or "grid"


def list_videos(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        sys.exit(f"error: not a directory: {folder}")
    exts = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 2x2 grid comparison of two video folders.")
    parser.add_argument("dir_a", help="First folder (left column).")
    parser.add_argument("dir_b", help="Second folder (right column).")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output folder for grids. Default: <dir_a>/../grids")
    parser.add_argument("--titles", default=None,
                        help="Comma-separated column titles. Default: the two folder names.")
    parser.add_argument("--cell-height", type=int, default=None, help="Per-cell height (px).")
    parser.add_argument("--fps", type=float, default=None, help="Output fps.")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF.")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    label_a = os.path.basename(os.path.normpath(args.dir_a))
    label_b = os.path.basename(os.path.normpath(args.dir_b))
    if args.titles:
        parts = [t.strip() for t in args.titles.split(",")]
        if len(parts) != 2:
            sys.exit("error: --titles needs exactly two comma-separated values.")
        label_a, label_b = parts

    prefixes = (label_a, label_b, "baseline", "kv_rag")

    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.normpath(args.dir_a)), "grids")
    os.makedirs(out_dir, exist_ok=True)

    files_a = list_videos(args.dir_a)
    b_by_key = {pair_key(f, prefixes): f for f in list_videos(args.dir_b)}

    pairs, unmatched = [], []
    for fa in files_a:
        key = pair_key(fa, prefixes)
        fb = b_by_key.get(key)
        if fb is None:
            unmatched.append(fa)
            continue
        pairs.append((key, fa, fb))

    if not pairs:
        sys.exit("error: no matching pairs found between the two folders.")

    print(f"[batch] {len(pairs)} pair(s) -> {out_dir}")
    if unmatched:
        print(f"[batch] {len(unmatched)} unmatched in '{label_a}': "
              + ", ".join(unmatched[:5]) + (" ..." if len(unmatched) > 5 else ""))

    ok, failed = 0, []
    for key, fa, fb in pairs:
        out_path = os.path.join(out_dir, f"{slugify(key)}.mp4")
        cmd = [
            sys.executable, _GRID_SCRIPT,
            os.path.join(args.dir_a, fa),
            os.path.join(args.dir_b, fb),
            "-o", out_path,
            "--titles", f"{label_a},{label_b}",
            "--crf", str(args.crf),
        ]
        if args.cell_height:
            cmd += ["--cell-height", str(args.cell_height)]
        if args.fps:
            cmd += ["--fps", str(args.fps)]
        if args.no_labels:
            cmd.append("--no-labels")
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"\n[batch] {key}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            ok += 1
        else:
            failed.append(key)

    print(f"\n[batch] done: {ok}/{len(pairs)} grids written to {out_dir}")
    if failed:
        print(f"[batch] failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
