#!/usr/bin/env python
"""Figure 3 of docs/attention-downweighting-study.zh.md (step 0, no new renders).

Reads the Round 18 per-shot/layer/frame injected-column attention-mass records
(`*_kv_rag_diag.json`, written by record_injected_attention_mass) and plots

  (a) per-scene timelines: mean mass on injected persistent columns vs shot
      index, one line per recorded layer, against the uniform-share baseline
      implied by the injected token count and the attended-window size;
  (b) a per-layer down-weighting summary (uniform share / measured mass)
      pooled over scenes and shots.

Round 18 ran with persistent logit bias lambda=1.0, so the measured mass is
the *biased* consumption; the unbiased down-weighting is worse by roughly e^1
in odds (Round 18 manipulation check: 0.017 -> 0.041 mean mass).
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

WINDOW_TOKENS = 32 * 880  # local_attn_size=32 frames incl. 8 sink frames


def load_diag(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def scene_name(path: Path) -> str:
    # kv_only-rank0-<scene>-seed0_regular_kv_rag_diag.json
    stem = path.name.split("-rank0-")[-1]
    return stem.split("-seed")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--diag-dir",
        default="videos/round18_bench_20260701/iter01/kv_only",
        help="directory holding *_kv_rag_diag.json files",
    )
    ap.add_argument("--out-dir", default="docs/figures/attention_downweighting")
    ap.add_argument("--branch", default="pos", choices=("pos", "neg"))
    args = ap.parse_args()

    diag_paths = sorted(Path(args.diag_dir).glob("*_kv_rag_diag.json"))
    if not diag_paths:
        raise SystemExit(f"no diag files under {args.diag_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes: dict[str, dict] = {}
    for p in diag_paths:
        scenes[scene_name(p)] = load_diag(p)[args.branch]["attention_mass_by_shot_layer"]

    layers = sorted({int(l) for per_shot in scenes.values() for l in
                     {ll for sh in per_shot.values() for ll in sh}})
    cmap = plt.get_cmap("viridis")
    layer_color = {l: cmap(i / max(1, len(layers) - 1)) for i, l in enumerate(layers)}

    n = len(scenes)
    ncols = 3
    nrows = math.ceil(n / ncols) + 1  # last row = summary panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)

    # down-weight factors pooled for the summary panel: layer -> [uniform/mass]
    pooled: dict[int, list[float]] = {l: [] for l in layers}

    for idx, (name, per_shot) in enumerate(sorted(scenes.items())):
        ax = axes[idx // ncols][idx % ncols]
        shots = sorted(per_shot, key=int)
        uniform_line = []
        for l in layers:
            xs, ys = [], []
            for s in shots:
                rec = per_shot[s].get(str(l))
                if rec is None or rec.get("calls", 0) == 0:
                    continue
                toks = float(rec.get("persistent_tokens_mean", 0.0))
                if toks <= 0:
                    continue
                uniform = toks / (WINDOW_TOKENS + toks)
                mass = float(rec["mean_mass"])
                xs.append(int(s))
                ys.append(mass)
                if mass > 0:
                    pooled[l].append(uniform / mass)
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.2,
                        color=layer_color[l], label=f"L{l}")
        # uniform baseline per shot (same for all layers)
        for s in shots:
            any_rec = next((r for r in per_shot[s].values()
                            if r.get("persistent_tokens_mean", 0) > 0), None)
            if any_rec:
                toks = float(any_rec["persistent_tokens_mean"])
                uniform_line.append((int(s), toks / (WINDOW_TOKENS + toks)))
        if uniform_line:
            ux, uy = zip(*sorted(uniform_line))
            ax.plot(ux, uy, ls="--", color="k", lw=1.0, label="uniform")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("shot", fontsize=8)
        ax.set_ylabel("mass on injected cols", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.25, lw=0.4)

    # hide unused per-scene axes
    for j in range(n, (nrows - 1) * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # summary panel: per-layer down-weight factor distribution
    ax = axes[-1][0]
    data = [pooled[l] for l in layers]
    bp = ax.boxplot(data, labels=[f"L{l}" for l in layers], showfliers=False,
                    patch_artist=True)
    for patch, l in zip(bp["boxes"], layers):
        patch.set_facecolor(layer_color[l])
        patch.set_alpha(0.6)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("down-weight = uniform / mass", fontsize=8)
    ax.set_title(f"per-layer down-weighting, pooled over scenes x shots "
                 f"(lambda=1 biased)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25, lw=0.4)
    for j in range(1, ncols):
        axes[-1][j].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "Round 18 kv_only: attention mass on injected persistent columns "
        "(reinject_rope placement, logit bias lambda=1.0)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = out_dir / f"injected_mass_timeline_{args.branch}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # console summary
    print("\nper-layer median down-weight (uniform/mass, lambda=1 biased):")
    for l in layers:
        arr = np.array(pooled[l])
        if arr.size:
            print(f"  L{l:02d}: median {np.median(arr):6.1f}x   "
                  f"IQR [{np.percentile(arr,25):.1f}, {np.percentile(arr,75):.1f}]  "
                  f"n={arr.size}")


if __name__ == "__main__":
    main()
