#!/usr/bin/env python
"""Scheme B evaluation: does a repaired FIRST chunk propagate through free
within-shot continuation?

Per-chunk subject-identity trajectory (DINO cosine of CLIPSeg crops vs the
golden reference centroid) for: original / full-shot repair (scheme A) /
first-chunk-only repair + free continuation (scheme B). Also crosses the next
shot boundary to show carry. One line figure + frame strip per scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.vbench_consistency import build_dino_encoder
from scripts.evaluate_renoise_edit import (  # reuse helpers
    ORIG, PHRASE, REF_FRAMES, FPC, SubjectCropper, read_frames, find_arm_video,
)

SCENES = {
    "sandy_beach_driftwood": [
        ("original", None),
        ("A full shot (e1)", "drift3_e1"),
        ("B chunk30-only s0.94", "drift3_B1"),
        ("B chunk30-only s0.83", "drift3_B2"),
    ],
    "bearded_watchmaker_workshop": [
        ("original", None),
        ("A full shot (e1)", "watch3_e1"),
        ("B chunk30-only s0.94", "watch3_B1"),
    ],
}
CHUNKS = list(range(30, 48))  # edited shot (30-35) + next two shots' worth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir",
                    default=str(ROOT / "videos/renoise_edit_20260821/eval"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    dino = build_dino_encoder(device=args.device)
    cropper = SubjectCropper(args.device)

    ref_centroid = {}
    for scene, idxs in REF_FRAMES.items():
        emb = dino(cropper.crops(read_frames(ORIG[scene], idxs), PHRASE[scene]))
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        ref_centroid[scene] = emb.mean(0) / np.linalg.norm(emb.mean(0))

    summary = {}
    for scene, arms in SCENES.items():
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        rows = {}
        for label, arm in arms:
            vid = ORIG[scene] if arm is None else find_arm_video(arm, scene)
            ys = []
            for c in CHUNKS:
                idxs = [c * FPC + 1, c * FPC + 4, c * FPC + 7]
                if idxs[-1] >= 480:
                    break
                emb = dino(cropper.crops(read_frames(vid, idxs), PHRASE[scene]))
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                ys.append(float((emb @ ref_centroid[scene]).mean()))
            rows[label] = ys
            ax.plot(CHUNKS[:len(ys)], ys, marker="o", ms=3, lw=1.4, label=label)
        for b in (36, 42):
            ax.axvline(b - 0.5, color="gray", ls=":", lw=1)
        ax.axvspan(29.5, 30.5, color="red", alpha=0.08)
        ax.set_xlabel("chunk index (red band = edited chunk, dotted = shot cuts)")
        ax.set_ylabel("subject identity vs golden reference (DINO cos)")
        ax.set_title(f"{scene}: identity trajectory — does the repaired first "
                     "chunk carry?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_scheme_b_trajectory.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        summary[scene] = rows

        # strip: chunk 30 / 32 / 35 midframes per arm
        sample = [30 * FPC + 4, 32 * FPC + 4, 35 * FPC + 4]
        fig, axes = plt.subplots(len(arms), 3,
                                 figsize=(3 * 2.9, len(arms) * 1.8))
        axes = np.atleast_2d(axes)
        for r, (label, arm) in enumerate(arms):
            vid = ORIG[scene] if arm is None else find_arm_video(arm, scene)
            frames = read_frames(vid, sample)
            for c in range(3):
                axes[r][c].imshow(frames[c])
                axes[r][c].axis("off")
                if r == 0:
                    axes[r][c].set_title(f"chunk {sample[c] // FPC}", fontsize=8)
            axes[r][0].set_ylabel(label, fontsize=7)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        fig.suptitle(f"{scene}: A vs B across the edited shot", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_scheme_b_strip.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    with open(out_dir / "scheme_b_results.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", out_dir / "scheme_b_results.json")


if __name__ == "__main__":
    main()
