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
"""Compose two videos into a single 2x2 comparison grid.

Layout (columns = the two input videos):

    +-------------------------+-------------------------+
    |  video A  first frame   |  video B  first frame   |   <- reference row (frozen)
    +-------------------------+-------------------------+
    |  video A  playing       |  video B  playing       |   <- live row
    +-------------------------+-------------------------+

The top row holds each video's first frame, frozen for the whole duration, as a
reference anchor; the bottom row plays the videos. Handy for eyeballing temporal
drift / identity consistency (e.g. baseline vs KV-RAG) against the start frame.

Example:
    python scripts/make_2x2_grid.py \
        videos/kv_rag_ablation/baseline/foo.mp4 \
        videos/kv_rag_ablation/kv_rag/foo.mp4 \
        -o videos/compare_foo.mp4 --titles "baseline,kv_rag"

Requires ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

# Common font locations for the drawtext labels; first hit wins.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        sys.exit(f"error: '{binary}' not found on PATH. Install ffmpeg first.")


def _even(value: float) -> int:
    """Round to the nearest even integer (H.264 requires even dimensions)."""
    n = int(round(value))
    return n - (n % 2)


def probe_video(path: str) -> dict:
    """Return {width, height, fps} for the first video stream."""
    if not os.path.isfile(path):
        sys.exit(f"error: input not found: {path}")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(out).get("streams", [])
    if not streams:
        sys.exit(f"error: no video stream in {path}")
    s = streams[0]
    rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/0"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps}


def find_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def escape_drawtext(text: str) -> str:
    # Escape characters that are special inside an ffmpeg filtergraph drawtext.
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def label_filter(text: str, font: str, cell_w: int, position: str) -> str:
    """A drawtext clause with a translucent box. position: 'top' or 'bottom'."""
    fontsize = max(14, _even(cell_w / 22))
    y = "10" if position == "top" else f"h-text_h-10"
    return (
        f"drawtext=fontfile='{font}':text='{escape_drawtext(text)}':"
        f"fontcolor=white:fontsize={fontsize}:x=(w-text_w)/2:y={y}:"
        f"box=1:boxcolor=black@0.5:boxborderw=6"
    )


def cell_chain(
    input_idx: int,
    out_label: str,
    *,
    cell_w: int,
    cell_h: int,
    fps: float,
    frozen: bool,
    text: str | None,
    font: str | None,
) -> str:
    """Build one filtergraph branch that produces a cell.

    Scales+pads the source into cell_w x cell_h, optionally freezes it on its
    first frame, and optionally stamps a label.
    """
    steps = [
        f"fps={fps:.6f}",
        f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease",
        f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]
    if frozen:
        # Keep only the first frame, then loop it forever; the stack filters cap
        # the duration to the (finite) playing row via shortest=1.
        steps += [
            "select=eq(n\\,0)",
            "loop=loop=-1:size=1:start=0",
            f"setpts=N/{fps:.6f}/TB",
        ]
    if text and font:
        steps.append(label_filter(text, font, cell_w, "top" if frozen else "bottom"))
    return f"[{input_idx}:v]" + ",".join(steps) + f"[{out_label}]"


def build_filtergraph(
    cell_w: int, cell_h: int, fps: float, titles: tuple[str, str], font: str | None
) -> str:
    title_a, title_b = titles
    ref_a = f"{title_a} (first frame)" if font else None
    ref_b = f"{title_b} (first frame)" if font else None
    live_a = title_a if font else None
    live_b = title_b if font else None

    chains = [
        cell_chain(0, "ar", cell_w=cell_w, cell_h=cell_h, fps=fps, frozen=True, text=ref_a, font=font),
        cell_chain(1, "br", cell_w=cell_w, cell_h=cell_h, fps=fps, frozen=True, text=ref_b, font=font),
        cell_chain(0, "al", cell_w=cell_w, cell_h=cell_h, fps=fps, frozen=False, text=live_a, font=font),
        cell_chain(1, "bl", cell_w=cell_w, cell_h=cell_h, fps=fps, frozen=False, text=live_b, font=font),
        # Reference row (both inputs are infinite loops -> shortest=1 keeps it
        # well-defined); live row ends at the shorter of the two videos.
        "[ar][br]hstack=inputs=2:shortest=1[top]",
        "[al][bl]hstack=inputs=2:shortest=1[bottom]",
        # Final stack ends when the (finite) live row ends.
        "[top][bottom]vstack=inputs=2:shortest=1[out]",
    ]
    return ";".join(chains)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose two videos into a 2x2 grid (first-frame reference row + live row).",
    )
    parser.add_argument("video_a", help="First video (left column).")
    parser.add_argument("video_b", help="Second video (right column).")
    parser.add_argument("-o", "--output", default="grid_2x2.mp4", help="Output mp4 path.")
    parser.add_argument(
        "--titles", default=None,
        help="Comma-separated column titles, e.g. 'baseline,kv_rag'. "
             "Defaults to the input file stems.",
    )
    parser.add_argument(
        "--cell-height", type=int, default=None,
        help="Per-cell height in px (width auto from video A aspect). "
             "Default: video A's native height.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output fps. Default: video A's fps.")
    parser.add_argument("--no-labels", action="store_true", help="Disable text labels.")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF quality (lower=better). Default 18.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = parser.parse_args()

    _require("ffmpeg")
    _require("ffprobe")

    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"error: {args.output} exists (pass --overwrite to replace).")

    info_a = probe_video(args.video_a)
    info_b = probe_video(args.video_b)

    fps = args.fps or info_a["fps"] or info_b["fps"] or 24.0

    # Cell geometry from video A's aspect ratio; both inputs are fit+padded into it.
    if args.cell_height:
        cell_h = _even(args.cell_height)
        cell_w = _even(info_a["width"] * cell_h / info_a["height"])
    else:
        cell_w = _even(info_a["width"])
        cell_h = _even(info_a["height"])
    cell_w = max(cell_w, 2)
    cell_h = max(cell_h, 2)

    if args.titles:
        parts = [t.strip() for t in args.titles.split(",")]
        if len(parts) != 2:
            sys.exit("error: --titles needs exactly two comma-separated values.")
        titles = (parts[0], parts[1])
    else:
        titles = (
            os.path.splitext(os.path.basename(args.video_a))[0],
            os.path.splitext(os.path.basename(args.video_b))[0],
        )

    font = None if args.no_labels else find_font()
    if not args.no_labels and font is None:
        print("[warn] no usable font found; rendering grid without labels.", file=sys.stderr)

    filtergraph = build_filtergraph(cell_w, cell_h, fps, titles, font)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y" if args.overwrite else "-n",
        "-i", args.video_a,
        "-i", args.video_b,
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
        "-pix_fmt", "yuv420p",
        args.output,
    ]

    print(f"[grid] {cell_w*2}x{cell_h*2} @ {fps:.2f}fps  ->  {args.output}")
    print(f"[grid] columns: '{titles[0]}' | '{titles[1]}'")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: ffmpeg failed ({exc.returncode}).")
    print(f"[grid] done: {args.output}")


if __name__ == "__main__":
    main()
