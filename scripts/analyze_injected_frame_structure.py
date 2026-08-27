#!/usr/bin/env python
"""Token-level structure of attention onto injected frames.

Answers three questions from the attn_probe dumps (GPU, full queries):

  1. WHY is mass nearly equal across injected anchor frames? Discriminator:
     if the anchors' keys are near-duplicates of each other, a content-driven
     reader would also give them equal mass -- so we compare (a) pairwise
     pooled-key cosine among anchors vs among current-chunk frames, and
     (b) the spread of per-frame mean logits among anchors vs across the
     whole window.
  2. WITHIN one injected frame, which tokens get read? Received attention
     per key token, reshaped to the 22x40 latent grid, next to the actual
     video frame.
  3. Which QUERY tokens do the reading? Per-query mass onto the injected
     block, reshaped to the query grid.

"Weight"/"mass" throughout = post-softmax attention, averaged over all
query tokens (and heads unless stated), exactly what the model uses to mix
values.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def attn_step0_files(root: Path, layer: int) -> list[Path]:
    return sorted((root / "attn" / f"layer_{layer:02d}").glob("chunk_*_step0.pt"))


def rag_frame_spans(meta: dict) -> list[dict]:
    """[{label, lo, hi, src_frame}] for each injected anchor frame."""
    fsl = int(meta["frame_seqlen"])
    out, cursor = [], 0
    for e in meta["entries"]:
        n = int(e["num_tokens"])
        for f in range(n // fsl):
            src = int(e["start_token"]) // fsl + f
            out.append({"label": f"f{src}", "lo": cursor + f * fsl,
                        "hi": cursor + (f + 1) * fsl, "src_frame": src})
        cursor += n
    return out


def window_frame_groups(meta: dict) -> list[dict]:
    """Every whole frame in the window: rag anchors, sink, pinned, chunk."""
    fsl = int(meta["frame_seqlen"])
    rag_n = int(meta["rag_tokens"])
    sink_n = int(meta["effective_sink"])
    pin_n = int(meta["pinned_len"])
    total = int(meta["window_tokens"])
    chunk_n = int(meta["current_end"]) - int(meta["current_start"])
    groups = [dict(g, group="rag") for g in rag_frame_spans(meta)]
    for i in range(sink_n // fsl):
        lo = rag_n + i * fsl
        groups.append({"label": f"sink{i}", "lo": lo, "hi": lo + fsl,
                       "group": "sink"})
    for i in range(pin_n // fsl):
        lo = rag_n + sink_n + i * fsl
        groups.append({"label": f"pin{i}", "lo": lo, "hi": lo + fsl,
                       "group": "pinned"})
    for i in range(chunk_n // fsl):
        lo = total - chunk_n + i * fsl
        groups.append({"label": f"cur{i}", "lo": lo, "hi": lo + fsl,
                       "group": "chunk"})
    return groups


def compute_boundary(meta: dict, device: str):
    """Full-query attention -> received mass/logit per key column."""
    q = meta["q_post"].to(device).float()          # [Lq, H, D]
    k = meta["window_k_post"].to(device).float()   # [Lk, H, D]
    logits = torch.einsum("lhd,mhd->hlm", q, k) * SCALE
    col_logit = logits.mean(dim=1)                 # [H, Lk]
    w = torch.softmax(logits, dim=-1)
    recv = w.mean(dim=1)                           # [H, Lk] received mass/col
    rag_n = int(meta["rag_tokens"])
    q_rag = w[..., :rag_n].sum(-1)                 # [H, Lq] per-query rag mass
    del logits, w
    return recv.cpu(), col_logit.cpu(), q_rag.cpu()


def pooled_cosine(frames: list[torch.Tensor]) -> np.ndarray:
    """Pairwise cosine of per-frame pooled (token-mean) key vectors."""
    pooled = torch.stack([f.mean(dim=0).flatten() for f in frames])
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return (pooled @ pooled.T).numpy()


def extract_video_frame(video: Path, idx: int, out_png: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video),
           "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", str(out_png)]
    return subprocess.run(cmd).returncode == 0 and out_png.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="sandy_beach_driftwood")
    ap.add_argument("--probe-root", default="videos/attn_probe_20260820")
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 7, 14, 21, 29])
    ap.add_argument("--fig-layers", type=int, nargs="*", default=[0, 14, 29])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="docs/figures/attention_downweighting")
    ap.add_argument("--tmp-dir", default=None)
    args = ap.parse_args()

    root = Path(args.probe_root) / args.scene / "dumps"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir) if args.tmp_dir else out_dir / "_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    videos = list((Path(args.probe_root) / args.scene / "renders").glob("*.mp4"))
    video = videos[0] if videos else None

    # pass 1: compute everything, keep per-layer per-boundary results
    results: dict[int, dict[int, dict]] = defaultdict(dict)
    for layer in args.layers:
        for f in attn_step0_files(root, layer):
            meta = load(f)
            if "window_k_post" not in meta:
                continue
            recv, col_logit, q_rag = compute_boundary(meta, args.device)
            gh, gw = int(meta["grid_h"]), int(meta["grid_w"])
            fsl = int(meta["frame_seqlen"])
            groups = window_frame_groups(meta)
            per_frame = []
            for g in groups:
                m = recv[:, g["lo"]:g["hi"]].sum(-1)      # [H] mass on frame
                lg = col_logit[:, g["lo"]:g["hi"]].mean(-1)
                per_frame.append({**{k: v for k, v in g.items()
                                     if k != "lo" and k != "hi"},
                                  "mass_mean": float(m.mean()),
                                  "mass_per_head": m.numpy(),
                                  "logit_mean": float(lg.mean())})
            rag_maps = {g["label"]:
                        recv[:, g["lo"]:g["hi"]].mean(0).reshape(gh, gw).numpy()
                        for g in groups if g["group"] == "rag"}
            qmap = q_rag.mean(0).reshape(-1, gh, gw).mean(0).numpy()
            # key cosine: anchors (pre-RoPE) vs current chunk frames (pre-RoPE)
            cos_rag = cos_chunk = None
            if "rag_k_pre" in meta:
                spans = rag_frame_spans(meta)
                rk = meta["rag_k_pre"].float()
                cos_rag = pooled_cosine([rk[s["lo"]:s["hi"]] for s in spans])
            sp = root / "store_k" / f"layer_{layer:02d}" / \
                f"chunk_{int(meta['chunk_index']):03d}.pt"
            if sp.exists():
                ck = load(sp)["k_pre"].float()
                fr = ck.shape[0] // fsl
                cos_chunk = pooled_cosine(
                    [ck[i * fsl:(i + 1) * fsl] for i in range(fr)])
            results[layer][int(meta["chunk_index"])] = {
                "per_frame": per_frame, "rag_maps": rag_maps, "qmap": qmap,
                "cos_rag": cos_rag, "cos_chunk": cos_chunk,
            }
        print(f"layer {layer} done")

    # boundary with the most anchors -> showcase figure
    any_layer = args.fig_layers[0]
    show_chunk = max(results[any_layer],
                     key=lambda c: len(results[any_layer][c]["rag_maps"]))

    # ---- Fig A: video frame vs within-frame received attention -------------
    maps0 = results[args.fig_layers[0]][show_chunk]["rag_maps"]
    anchors = sorted(maps0, key=lambda s: int(s[1:]))
    ncol = 1 + len(args.fig_layers)
    fig, axes = plt.subplots(len(anchors), ncol,
                             figsize=(3.1 * ncol, 1.9 * len(anchors)))
    axes = np.atleast_2d(axes)
    for r, label in enumerate(anchors):
        src = int(label[1:])
        png = tmp / f"{args.scene}_{label}.png"
        if video is not None and (png.exists() or
                                  extract_video_frame(video, src, png)):
            axes[r][0].imshow(plt.imread(png))
        axes[r][0].set_ylabel(f"{label}", fontsize=9)
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        if r == 0:
            axes[r][0].set_title("video frame", fontsize=9)
        for c, layer in enumerate(args.fig_layers, start=1):
            m = results[layer][show_chunk]["rag_maps"][label]
            im = axes[r][c].imshow(m, cmap="magma")
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(f"received attn L{layer}", fontsize=9)
            fig.colorbar(im, ax=axes[r][c], fraction=0.03, pad=0.02)
    fig.suptitle(f"{args.scene} boundary chunk {show_chunk}: which parts of "
                 "each injected frame get read", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.scene}_framestructure_maps.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Fig B: per-frame mass across the whole window + key cosines ------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    colors = {"rag": "#e76f51", "sink": "#999999", "pinned": "#4c72b0",
              "chunk": "#c9a227"}
    for layer in args.fig_layers:
        pf = results[layer][show_chunk]["per_frame"]
        xs = np.arange(len(pf))
        ys = [p["mass_mean"] for p in pf]
        axes[0].plot(xs, ys, lw=1.0, alpha=0.7, label=f"L{layer}")
        if layer == args.fig_layers[-1]:
            for x, p in zip(xs, pf):
                axes[0].axvspan(x - 0.5, x + 0.5,
                                color=colors[p["group"]], alpha=0.08)
    pf = results[args.fig_layers[-1]][show_chunk]["per_frame"]
    axes[0].set_xticks(range(len(pf)), [p["label"] for p in pf],
                       rotation=90, fontsize=6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mass per frame (mean over heads)")
    axes[0].set_title(f"chunk {show_chunk}: mass on every window frame "
                      "(shaded: rag/sink/pinned/chunk)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, lw=0.4)
    # cosine summary over all boundaries, one layer
    cr, cc = [], []
    for c, res in results[args.fig_layers[1]].items():
        if res["cos_rag"] is not None and res["cos_rag"].shape[0] > 1:
            m = res["cos_rag"]
            cr.extend(m[np.triu_indices_from(m, k=1)])
        if res["cos_chunk"] is not None:
            m = res["cos_chunk"]
            cc.extend(m[np.triu_indices_from(m, k=1)])
    axes[1].boxplot([cr, cc], tick_labels=["anchor pairs", "chunk-frame pairs"])
    axes[1].set_ylabel("pooled pre-RoPE key cosine")
    axes[1].set_title(f"key similarity (L{args.fig_layers[1]}, all boundaries)")
    axes[1].grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.scene}_framestructure_flatness.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Fig C: query-side map --------------------------------------------
    fig, axes = plt.subplots(1, len(args.fig_layers) + 1,
                             figsize=(3.4 * (len(args.fig_layers) + 1), 2.6))
    cur_frame = None
    meta0 = load(attn_step0_files(root, args.fig_layers[0])
                 [[int(load(f)["chunk_index"]) for f in
                   attn_step0_files(root, args.fig_layers[0])].index(show_chunk)])
    src_cur = int(meta0["current_start"]) // int(meta0["frame_seqlen"]) + 4
    png = tmp / f"{args.scene}_cur{src_cur}.png"
    if video is not None and (png.exists() or
                              extract_video_frame(video, src_cur, png)):
        axes[0].imshow(plt.imread(png))
    axes[0].set_title(f"current chunk (frame {src_cur})", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for i, layer in enumerate(args.fig_layers, start=1):
        im = axes[i].imshow(results[layer][show_chunk]["qmap"], cmap="viridis")
        axes[i].set_title(f"query-side rag mass L{layer}", fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        fig.colorbar(im, ax=axes[i], fraction=0.03, pad=0.02)
    fig.suptitle(f"{args.scene} chunk {show_chunk}: WHICH query tokens read "
                 "the injected block", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.scene}_framestructure_queryside.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- JSON summary ------------------------------------------------------
    summary = {"scene": args.scene, "showcase_chunk": int(show_chunk)}
    for layer in args.layers:
        rows = []
        logit_spread_rag, logit_spread_win = [], []
        for c, res in results[layer].items():
            pf = res["per_frame"]
            rag = [p for p in pf if p["group"] == "rag"]
            if len(rag) > 1:
                lg = [p["logit_mean"] for p in rag]
                logit_spread_rag.append(max(lg) - min(lg))
            lg_all = [p["logit_mean"] for p in pf]
            logit_spread_win.append(max(lg_all) - min(lg_all))
            rows.append({
                "chunk": c,
                "rag": {p["label"]: round(p["mass_mean"], 6) for p in rag},
            })
        summary[f"L{layer}"] = {
            "per_boundary_rag_mass": rows,
            "anchor_logit_spread_nats_mean": (
                float(np.mean(logit_spread_rag)) if logit_spread_rag else None),
            "window_logit_spread_nats_mean": float(np.mean(logit_spread_win)),
        }
    cr_m = float(np.mean(cr)) if cr else None
    cc_m = float(np.mean(cc)) if cc else None
    summary["key_cosine_mean"] = {"anchor_pairs": cr_m, "chunk_frame_pairs": cc_m}
    out_json = out_dir / f"{args.scene}_framestructure_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
