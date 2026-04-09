"""
Stitch baseline and RAG videos side-by-side with first frame as reference.

Layout per output frame:
  ┌─────────────┬─────────────┐
  │ Baseline    │ RAG         │
  │ Ref (F0)    │ Ref (F0)    │
  ├─────────────┼─────────────┤
  │ Baseline    │ RAG         │
  │ Current (Ft)│ Current (Ft)│
  └─────────────┴─────────────┘
"""
import argparse
import os
import numpy as np
from pathlib import Path
from torchvision.io import read_video, write_video
import torch
import cv2


def load_frames(path):
    """Load video frames as numpy [T, H, W, C] uint8."""
    frames, _, info = read_video(str(path), pts_unit='sec')
    fps = info.get('video_fps', 24)
    return frames.numpy(), fps


def add_label(frame, text, position='top-left', font_scale=0.7, thickness=2):
    """Burn a text label onto a frame (in-place)."""
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    if position == 'top-left':
        x, y = pad, th + pad
    elif position == 'top-right':
        x, y = img.shape[1] - tw - pad, th + pad
    else:
        x, y = pad, th + pad
    # Background rectangle
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def stitch_video(baseline_path, rag_path, output_path):
    """Create a 2x2 comparison video: ref frames on top, current frames on bottom."""
    b_frames, b_fps = load_frames(baseline_path)
    r_frames, r_fps = load_frames(rag_path)
    n = min(len(b_frames), len(r_frames))
    fps = b_fps

    h, w = b_frames.shape[1], b_frames.shape[2]

    # Reference frames (frame 0)
    b_ref = add_label(b_frames[0], 'Baseline - Ref (F0)')
    r_ref = add_label(r_frames[0], 'RAG - Ref (F0)')

    # Separator line (2px white horizontal)
    sep_h = 2
    sep_line = np.ones((sep_h, w, 3), dtype=np.uint8) * 255
    sep_v = np.ones((h, sep_h, 3), dtype=np.uint8) * 255
    sep_v_double = np.ones((h * 2 + sep_h, sep_h, 3), dtype=np.uint8) * 255

    stitched = []
    for t in range(n):
        b_cur = add_label(b_frames[t], f'Baseline - F{t}')
        r_cur = add_label(r_frames[t], f'RAG - F{t}')

        # Top row: refs
        top = np.concatenate([b_ref, sep_v, r_ref], axis=1)
        # Bottom row: current
        bot = np.concatenate([b_cur, sep_v, r_cur], axis=1)
        # Horizontal separator
        full_w = top.shape[1]
        h_sep = np.ones((sep_h, full_w, 3), dtype=np.uint8) * 255
        # Stack
        frame = np.concatenate([top, h_sep, bot], axis=0)
        stitched.append(frame)

    stitched = np.stack(stitched)
    write_video(output_path, torch.from_numpy(stitched), fps=round(float(fps)))
    print(f"  Saved: {output_path}  ({n} frames, {fps:.0f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Stitch baseline vs RAG videos side-by-side with reference frames")
    parser.add_argument("--baseline", default="videos/reappear-baseline-v2", help="Baseline video folder")
    parser.add_argument("--rag", default="videos/reappear-rag-v2", help="RAG video folder")
    parser.add_argument("--output", default="videos/comparison-v2", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    baseline_videos = sorted(Path(args.baseline).glob("rank0-*_lora.mp4"))
    rag_videos = sorted(Path(args.rag).glob("rank0-*_lora.mp4"))

    assert len(baseline_videos) > 0, f"No videos found in {args.baseline}"
    assert len(baseline_videos) == len(rag_videos), \
        f"Mismatch: {len(baseline_videos)} baseline vs {len(rag_videos)} RAG videos"

    print(f"Stitching {len(baseline_videos)} video pairs...")
    for i, (bv, rv) in enumerate(zip(baseline_videos, rag_videos)):
        out_path = os.path.join(args.output, f"comparison_{i}.mp4")
        print(f"\nPrompt {i}: {bv.name} vs {rv.name}")
        stitch_video(str(bv), str(rv), out_path)

    print(f"\nAll comparison videos saved to {args.output}/")


if __name__ == "__main__":
    main()
