#!/usr/bin/env python
"""Which injected anchor frames actually get read? (zero new renders)

From the Round 18 kv_only diag JSONs, build per-scene heatmaps of
[injected source frame x boundary shot] mean attention mass, averaged over
the recorded layers. If consumption were indifferent to which anchor a
column came from, rows would be flat; structure here tells us whether the
model prefers early/recent/particular anchors inside the injected block.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scene_name(path: Path) -> str:
    return path.name.split("-rank0-")[-1].split("-seed")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir",
                    default="videos/round18_bench_20260701/iter01/kv_only")
    ap.add_argument("--out-dir", default="docs/figures/attention_downweighting")
    args = ap.parse_args()

    paths = sorted(Path(args.diag_dir).glob("*_kv_rag_diag.json"))
    scenes = {}
    for p in paths:
        d = json.load(open(p))["pos"]["attention_mass_by_shot_layer_frame"]
        scenes[scene_name(p)] = d

    n = len(scenes)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (name, per_shot) in enumerate(sorted(scenes.items())):
        ax = axes[idx // ncols][idx % ncols]
        shots = sorted(per_shot, key=int)
        frames = sorted({f for sh in per_shot.values()
                         for layer in sh.values() for f in layer},
                        key=lambda s: int(s.split("_")[-1]))
        mat = np.full((len(frames), len(shots)), np.nan)
        for j, s in enumerate(shots):
            layer_maps = per_shot[s].values()
            for i, f in enumerate(frames):
                vals = [lm[f]["mean_mass"] for lm in layer_maps if f in lm]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        im = ax.imshow(mat, aspect="auto", cmap="magma",
                       interpolation="nearest")
        ax.set_xticks(range(len(shots)), [f"s{s}" for s in shots], fontsize=7)
        ax.set_yticks(range(len(frames)),
                      [f.replace("src_frame_", "f") for f in frames],
                      fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("boundary shot", fontsize=8)
        ax.set_ylabel("injected source frame", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Round 18 kv_only: per-anchor-frame attention mass "
                 "(mean over layers 0/7/14/21/29, lambda=1 biased)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "injected_frame_consumption.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
