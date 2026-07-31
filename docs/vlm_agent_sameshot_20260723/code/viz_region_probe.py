"""Visualize the region-split attention probe.

Figure 1: per-layer subject vs background per-token attention density
          (relative to the uniform per-token share), averaged over prompts.
Figure 2: attention heatmaps over the token grid of the most-injected
          frames (golden anchors), overlaid on the actual frame thumbnails
          with the VLM subject bbox.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else "region_probe_out"
prompts = [int(p) for p in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "2", "6"])]
GH, GW = 30, 52

data = {}
for pi in prompts:
    with open(os.path.join(root, f"prompt{pi:03d}_probe.json")) as f:
        data[pi] = json.load(f)

# ---------- Figure 1: per-layer subject/background density ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), width_ratios=[3, 1])
ax = axes[0]
n_layers = 30
subj_d = np.zeros(n_layers); bg_d = np.zeros(n_layers); cnt = np.zeros(n_layers)
overall = {"subj": [], "bg": []}
for pi in prompts:
    for r in data[pi]["records"]:
        if r.get("mass_mem_subj") is None or r["n_known_tokens"] in (None, 0):
            continue
        ns, nk, tot = r["n_subj_tokens"], r["n_known_tokens"], r["total_len"]
        nb = nk - ns
        if ns == 0 or nb == 0:
            continue
        uni = 1.0 / tot
        ds = (r["mass_mem_subj"] / ns) / uni
        db = (r["mass_mem_bg"] / nb) / uni
        l = r["layer"]
        subj_d[l] += ds; bg_d[l] += db; cnt[l] += 1
        overall["subj"].append(ds); overall["bg"].append(db)
cnt[cnt == 0] = 1
x = np.arange(n_layers)
ax.bar(x - 0.2, subj_d / cnt, width=0.4, label="主体 token", color="#3a6ea5")
ax.bar(x + 0.2, bg_d / cnt, width=0.4, label="背景 token", color="#c07d1a")
ax.axhline(1.0, color="#888", ls=":", lw=1, label="均匀基线")
ax.set_xlabel("层"); ax.set_ylabel("每 token 注意力密度 (×uniform)")
ax.set_title("注入记忆帧内部:主体 vs 背景 token 的注意力密度(逐层)")
ax.legend(fontsize=9)

ax2 = axes[1]
ms, mb = np.mean(overall["subj"]), np.mean(overall["bg"])
ax2.bar([0, 1], [ms, mb], color=["#3a6ea5", "#c07d1a"], width=0.6)
ax2.axhline(1.0, color="#888", ls=":", lw=1)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["主体", "背景"])
ax2.set_title(f"全局均值\n主体/背景 = {ms/mb:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(root, "fig_region_density.png"), dpi=140)
plt.close()
print(f"overall: subj {ms:.2f}x uniform, bg {mb:.2f}x uniform, ratio {ms/mb:.2f}")

# ---------- Figure 2: heatmaps on golden frames ----------
rows = len(prompts)
top_n = 3
fig, axes = plt.subplots(rows, top_n, figsize=(4.2 * top_n, 2.75 * rows))
if rows == 1:
    axes = axes[None, :]
for ri, pi in enumerate(prompts):
    d = data[pi]
    counts = d["frame_col_count"]
    top = sorted(counts, key=lambda k: -counts[k])[:top_n]
    for ci, fid in enumerate(top):
        ax = axes[ri, ci]
        col = np.array(d["frame_col_mean"][fid]).reshape(GH, GW)
        col = col / max(col.mean(), 1e-9)  # relative to frame-average
        thumb_fid = (int(fid) // 3) * 3 + 1  # thumbs exist for block-mid frames
        thumb_path = os.path.join(root, f"prompt{pi:03d}_thumbs", f"frame{thumb_fid:03d}.jpg")
        if os.path.exists(thumb_path):
            img = Image.open(thumb_path)
            ax.imshow(img, extent=[0, GW, GH, 0])
        hm = ax.imshow(col, cmap="inferno", alpha=0.55, extent=[0, GW, GH, 0],
                       vmin=0.3, vmax=3.0, interpolation="bilinear")
        bbox = d["frame_bboxes"].get(str(fid)) or d["frame_bboxes"].get(int(fid))
        if bbox:
            x1, y1, x2, y2 = bbox
            ax.add_patch(plt.Rectangle((x1 / 1000 * GW, y1 / 1000 * GH),
                                       (x2 - x1) / 1000 * GW, (y2 - y1) / 1000 * GH,
                                       fill=False, ec="#00e5ff", lw=2))
        gold = "金帧" if int(fid) in d["golden_frames"] else ""
        ax.set_title(f"prompt{pi} f{fid} {gold} (被注入 {counts[fid]} 次)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(hm, ax=axes, shrink=0.75, label="相对该帧平均的注意力密度")
fig.suptitle("模型从注入记忆帧里读什么:注意力热力图(青框 = VLM 主体 bbox)", fontsize=12)
plt.savefig(os.path.join(root, "fig_region_heatmap.png"), dpi=140, bbox_inches="tight")
print("figures saved")
