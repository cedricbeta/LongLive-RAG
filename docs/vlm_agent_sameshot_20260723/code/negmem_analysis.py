"""Offline validation of the negative-memory margin signal.

Replays observe-mode runs (VLM labels every block, no intervention) and
simulates the online pools: at block t, positive pool = admitted blocks < t,
negative pool = rejected blocks < t. Tests whether the margin
(sim_to_pos - sim_to_neg) predicts the VLM's verdict at block t better than
positive-pool-only signals.
"""
import argparse
import json
import os

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="negmem_obs/latentmem")
parser.add_argument("--prompt_indices", default="0,1,2,4,7,8")
parser.add_argument("--out", default="negmem_obs/analysis")
parser.add_argument("--label_mode", choices=["gate", "drift"], default="gate",
                    help="gate: absolute judge admit/reject; drift: comparative "
                         "relabel (mild-or-worse counts as positive)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

def block_desc(descs, frames):
    d = descs[frames].mean(0)
    return d / (d.norm() + 1e-8)

def sim_to_pool(d, pool):
    if not pool:
        return None
    return max(float(d @ p) for p in pool)

def auc(labels, scores):
    """AUC for predicting label=1 (reject) with higher score."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))

all_rows = []
per_prompt = {}
for pi in [int(i) for i in args.prompt_indices.split(",")]:
    log = json.load(open(os.path.join(args.root, f"prompt{pi:03d}_log.json")))
    descs = torch.load(os.path.join(args.root, f"prompt{pi:03d}_descriptors.pt"), map_location="cpu").float()
    descs = descs / (descs.norm(dim=-1, keepdim=True) + 1e-8)
    blocks = [e for e in log["vlm_log"] if "frames" in e]
    drift_by_block = None
    if args.label_mode == "drift":
        dl = json.load(open(os.path.join(args.root, f"prompt{pi:03d}_driftlabels.json")))
        drift_by_block = {r["block"]: r for r in dl}
    pos_pool, neg_pool = [], []
    rows = []
    for e in blocks:
        d = block_desc(descs, e["frames"])
        if drift_by_block is not None:
            r = drift_by_block.get(e["block_start_frame"] // 3, {})
            reject = int(r.get("subject_drift", "none") != "none"
                         or r.get("scene_drift", "none") != "none")
        else:
            reject = 0 if e.get("admit", True) else 1
        sp = sim_to_pool(d, pos_pool)
        sn = sim_to_pool(d, neg_pool)
        novelty = None
        if pos_pool or neg_pool:
            novelty = max(s for s in [sp, sn] if s is not None)
        rows.append(dict(block=e["block_start_frame"] // 3, reject=reject,
                         sim_pos=sp, sim_neg=sn, novelty=novelty,
                         reason=e.get("reason", "")))
        (neg_pool if reject else pos_pool).append(d)
    per_prompt[pi] = rows
    n_rej = sum(r["reject"] for r in rows)
    print(f"prompt {pi}: {len(rows)} blocks, {n_rej} rejected")
    all_rows += [dict(prompt=pi, **r) for r in rows]

# --- AUC on blocks where BOTH pools are non-empty (negative memory usable) ---
usable = [r for r in all_rows if r["sim_pos"] is not None and r["sim_neg"] is not None]
mixed_prompts = {p for p in per_prompt
                 if 0 < sum(r["reject"] for r in per_prompt[p]) < len(per_prompt[p])}
usable = [r for r in usable if r["prompt"] in mixed_prompts]
print(f"\nusable blocks (both pools non-empty, mixed-label prompts {sorted(mixed_prompts)}): {len(usable)}")
if usable:
    labels = [r["reject"] for r in usable]
    print(f"reject rate among usable: {np.mean(labels):.2f}")
    signals = {
        "margin (pos - neg)  [neg-memory]": [-(r["sim_pos"] - r["sim_neg"]) for r in usable],
        "sim_neg only        [neg-memory]": [r["sim_neg"] for r in usable],
        "1 - sim_pos         [pos-only]  ": [-r["sim_pos"] for r in usable],
    }
    for name, sc in signals.items():
        print(f"AUC {name}: {auc(labels, sc):.3f}")

# --- figures ---
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt

n = len(per_prompt)
fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 2.3 * ((n + 1) // 2)), sharex=True)
axes = np.array(axes).flatten()
for ax, (pi, rows) in zip(axes, sorted(per_prompt.items())):
    xs = [r["block"] for r in rows]
    margin = [None if (r["sim_pos"] is None or r["sim_neg"] is None)
              else r["sim_pos"] - r["sim_neg"] for r in rows]
    ax.plot(xs, [m if m is not None else np.nan for m in margin],
            color="#3a6ea5", lw=1.5, label="margin = sim(正池) − sim(负池)")
    rej = [r["block"] for r in rows if r["reject"]]
    for b in rej:
        ax.axvline(b, color="#c0392b", alpha=0.25, lw=1)
    ax.axhline(0, color="#888", ls=":", lw=1)
    ax.set_title(f"prompt {pi}  (拒绝 {len(rej)}/{len(rows)} 块, 红线=VLM 拒绝)", fontsize=10, loc="left")
axes[0].legend(fontsize=8)
for ax in axes[len(per_prompt):]:
    ax.axis("off")
axes[-2].set_xlabel("block"); axes[-1].set_xlabel("block")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "fig_margin_timeline.png"), dpi=140)
print("\nsaved", os.path.join(args.out, "fig_margin_timeline.png"))

# --- mammoth saturation + intra-negative tightness (retry pre-filter feasibility) ---
sat_p = 1
if sat_p in per_prompt:
    rows = per_prompt[sat_p]
    descs = torch.load(os.path.join(args.root, f"prompt{sat_p:03d}_descriptors.pt"), map_location="cpu").float()
    descs = descs / (descs.norm(dim=-1, keepdim=True) + 1e-8)
    neg_ds = [block_desc(descs, [3 * r["block"], 3 * r["block"] + 1, 3 * r["block"] + 2])
              for r in rows if r["reject"]]
    if len(neg_ds) > 2:
        N = torch.stack(neg_ds)
        S = (N @ N.T).numpy()
        iu = np.triu_indices(len(N), 1)
        print(f"\nprompt1 negative cluster: {len(N)} blocks, "
              f"intra-neg cos mean={S[iu].mean():.3f} p10={np.percentile(S[iu],10):.3f} "
              f"-> retry pre-filter threshold headroom")
