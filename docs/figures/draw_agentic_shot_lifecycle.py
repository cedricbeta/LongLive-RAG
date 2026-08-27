"""Shot-lifecycle flowchart v2 — clean labels, no glyph gaps, no label collisions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

names = {f.name for f in fm.fontManager.ttflist}
for cand in ["Noto Sans CJK SC", "Noto Sans CJK JP"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15.6, 8.6), dpi=170)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor("white")

INK = "#1f2937"; MUT = "#5f6b7a"
BOX = dict(fc="#f3f6fb", ec="#7b93b8")
GRN = dict(fc="#e8f5ec", ec="#3e9d5c")
ORG = dict(fc="#fdf3e3", ec="#d9903a")
RED = dict(fc="#fdecec", ec="#cf5f5f")


def box(cx, cy, w, h, title, lines, style, title_c=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.6,rounding_size=1.2",
                 lw=1.7, fc=style["fc"], ec=style["ec"], zorder=2))
    ax.text(cx, cy + h / 2 - 4.0, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=title_c, zorder=3)
    ax.text(cx, cy - 2.8, "\n".join(lines), ha="center", va="center",
            fontsize=10.2, color=INK, zorder=3, linespacing=1.55)


def arrow(p, q, color=MUT, ls="-", lw=2.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=17,
                 lw=lw, color=color, linestyle=ls, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))


def lab(x, y, s, c, fs=10.8):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=c,
            fontweight="bold", zorder=4,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.4))


# ---------- top pipeline ----------
TY, TH = 80, 21
box(11.5, TY, 19, TH, "1. 生成前",
    ["读取规范记忆与", "可信实体记忆；遥测有", "漂移风险则预防性干预"], BOX)
box(35.5, TY, 17, TH, "2. 边界快照",
    ["保存全部生成状态", "latent · KV · RNG · VAE", "（回滚的凭据）"], BOX)
box(58.0, TY, 14, TH, "3. 生成镜头 k",
    ["暂定状态", "尚未交付"], BOX)
box(84.0, TY, 21, TH, "4. Verifier 验证",
    ["实体在场 · 外观不变量", "授权状态变化", "运动 / 多样性 / adherence"], BOX)

arrow((21.5, TY), (26.5, TY)); arrow((44.5, TY), (50.5, TY)); arrow((65.5, TY), (73.0, TY))

# ---------- outcome row ----------
BY, BH = 24, 30
box(16.5, BY, 27, BH, "出口 A：通过，commit",
    ["镜头正式交付", "", "实体素材与对应 KV 索引", "写入可信记忆，", "供后续镜头参照"], GRN, "#256b3d")
box(50.0, BY, 27, BH, "出口 B：失败，修复重试",
    ["恢复边界快照，", "附带针对性修复动作", "（V 编辑 / KV 注入 / prompt 修正）", "重新生成本镜头；", "每镜头至多一次"], ORG, "#9a641f")
box(84.0, BY, 27, BH, "出口 C：重试仍失败，隔离",
    ["镜头照常交付（预算有限），", "但素材进入隔离区，", "不写入可信记忆：", "错误不向后续镜头传播", "（视频越长此效应越大）"], RED, "#8f3c3c")

# ---------- verdict arrows (distinct lanes, explicit labels) ----------
# A: leftmost lane, deep arc passing low
arrow((74.5, TY - TH / 2 - 1.0), (17.5, BY + BH / 2 + 1.0), color="#3e9d5c", rad=0.24)
lab(34.5, 51.5, "通过", "#256b3d")
# B: middle lane
arrow((81.0, TY - TH / 2 - 1.0), (51.0, BY + BH / 2 + 1.0), color="#d9903a", rad=0.10)
lab(63.0, 52.5, "失败，重试预算未用", "#9a641f")
# C: straight down
arrow((88.5, TY - TH / 2 - 1.0), (85.5, BY + BH / 2 + 1.0), color="#cf5f5f")
lab(92.0, 52.5, "重试后仍失败", "#8f3c3c")

# retry loop: orange box top-left → generation box bottom, dashed, high arc on the left
arrow((38.5, BY + BH / 2 + 1.0), (55.0, TY - TH / 2 - 1.0), color="#d9903a",
      ls=(0, (5, 3)), rad=0.30)
lab(33.5, 61.0, "带修复动作重试", "#9a641f", fs=10.2)

# commit → next shot
arrow((16.5, BY - BH / 2 - 1.2), (16.5, 5.5), color="#256b3d")
lab(27.0, 5.0, "进入镜头 k+1", "#256b3d")

ax.text(50, 97.4, "一个镜头的处理流程（shot-buffered 协议）", ha="center",
        fontsize=15, fontweight="bold", color=INK)

import os
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/agentic_shot_lifecycle.png", bbox_inches="tight", facecolor="white")
print("saved")
