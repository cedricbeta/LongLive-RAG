#!/usr/bin/env python
"""CLIPSeg subject masks for the SDEdit renoise-repair arms.

For each requested chunk (8 video frames = 8 latent frames, 1:1 mapping
verified on the attn_probe renders), run CLIPSeg on every frame with the
subject phrase and store soft masks resampled to the 22x40 latent token grid.
Output: {"masks": {chunk: FloatTensor [8, 22, 40]}, meta...} plus a QC
overlay PNG per chunk so the localization can be eyeballed before rendering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "CIDAS/clipseg-rd64-refined"
FRAMES_PER_CHUNK = 8
GRID = (22, 40)


def read_frames(video: Path, start: int, count: int) -> list[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(bgr[:, :, ::-1]))
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--chunks", required=True,
                    help="inclusive range, e.g. 30-35")
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)
    model.to(args.device).eval()

    c0, c1 = (int(x) for x in args.chunks.split("-"))
    masks: dict[int, torch.Tensor] = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc_dir = out_path.parent / (out_path.stem + "_qc")
    qc_dir.mkdir(exist_ok=True)

    for chunk in range(c0, c1 + 1):
        frames = read_frames(Path(args.video), chunk * FRAMES_PER_CHUNK,
                             FRAMES_PER_CHUNK)
        if len(frames) < FRAMES_PER_CHUNK:
            raise SystemExit(f"chunk {chunk}: only {len(frames)} frames read")
        with torch.no_grad():
            inputs = processor(
                text=[args.phrase] * len(frames), images=frames,
                return_tensors="pt", padding=True,
            ).to(args.device)
            logits = model(**inputs).logits  # [F, h, w]
            soft = torch.sigmoid(logits)
            soft = torch.nn.functional.interpolate(
                soft.unsqueeze(1), size=GRID, mode="bilinear",
                align_corners=False,
            ).squeeze(1).cpu()
        masks[chunk] = soft

        fig, axes = plt.subplots(2, 4, figsize=(13, 4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(frames[i])
            m = torch.nn.functional.interpolate(
                soft[i][None, None], size=frames[i].size[::-1], mode="bilinear",
                align_corners=False,
            )[0, 0]
            ax.imshow(m.numpy(), alpha=0.35, cmap="spring")
            ax.set_title(f"c{chunk} f{chunk*8+i}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"CLIPSeg '{args.phrase}' chunk {chunk}", fontsize=10)
        fig.tight_layout()
        fig.savefig(qc_dir / f"chunk_{chunk:03d}.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
        cov = float((soft >= 0.35).float().mean())
        print(f"chunk {chunk}: mask coverage@0.35 = {cov:.3f}")

    torch.save(
        {"masks": masks, "phrase": args.phrase, "video": str(args.video),
         "grid": list(GRID), "frames_per_chunk": FRAMES_PER_CHUNK},
        out_path,
    )
    print(f"wrote {out_path} + QC at {qc_dir}")


if __name__ == "__main__":
    main()
