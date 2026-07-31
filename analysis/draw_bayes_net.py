"""Draw the proposed Bayesian network for LingBot-VLA-2 failure localization.

Two decision slices are drawn out explicitly instead of being hidden behind a
plate. That costs space but makes the two things that are easy to misread
visible: which nodes are per-episode vs per-decision, and why the undirected
skeleton has cycles even though the DAG obviously has none.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

FG = "#1a1a1a"
BLUE = "#2b6cb0"
GREY = "#9aa0a6"
RED = "#c53030"
GREEN = "#2f855a"
PURP = "#8a5fa8"
CYC = "#d97706"

fig, ax = plt.subplots(figsize=(18.5, 12.4))
ax.set_xlim(0, 18.5)
ax.set_ylim(0, 12.4)
ax.axis("off")


def rbox(x, y, w, h, label, color=FG, lw=1.4, fs=11, face="white", weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=lw, edgecolor=color, facecolor=face, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color,
            zorder=4, fontweight=weight)


def sbox(x, y, w, h, label, color=FG, lw=1.4, fs=11, face="white"):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, linewidth=lw,
                           edgecolor=color, facecolor=face, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def circ(x, y, r, label="", color=BLUE, lw=1.6, fs=10, face="white", sub=""):
    ax.add_patch(Circle((x, y), r, linewidth=lw, edgecolor=color, facecolor=face, zorder=3))
    if label:
        ax.text(x, y + (0.10 if sub else 0), label, ha="center", va="center",
                fontsize=fs, color=color, zorder=4)
    if sub:
        ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=7.3, color=color, zorder=4)


def arrow(x1, y1, x2, y2, color=FG, ls="-", lw=1.4, rad=0.0, alpha=1.0, z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", alpha=alpha,
                                 shrinkA=1, shrinkB=1))


def plate(x, y, w, h, label, color=GREY, ls=(0, (6, 4)), fs=11, lw=1.7, lx=None, ly=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.12",
                                linewidth=lw, edgecolor=color, facecolor="none",
                                linestyle=ls, zorder=1))
    ax.text(lx if lx else x + w - 0.15, ly if ly else y + 0.18, label,
            ha="right", va="bottom", fontsize=fs, color=color, style="italic",
            fontweight="bold")


ax.text(0.15, 12.05, "LingBot-VLA-2 失败定位贝叶斯网络", fontsize=17, fontweight="bold", color=FG)
ax.text(0.15, 11.68, "决策 k 展开成两片，以显示 plate 归属与无向环的来源", fontsize=10.5, color=GREY)

# ═════════ STATIC LAYER ═════════
ax.add_patch(FancyBboxPatch((0.35, 9.85), 17.8, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                            linewidth=1.9, edgecolor=BLUE, facecolor="#f0f6fc", zorder=0))
ax.text(0.62, 11.1, "静态层 — 全局共享，不在任何 plate 内。这是你要定位并优化的对象",
        fontsize=11.5, color=BLUE, fontweight="bold")

theta_x = [3.0, 6.0, 9.0, 12.0, 15.0]
theta_lab = ["θ 感知", "θ 语言落地", "θ 联合前向\n(MoE 路由)", "θ 去噪", "θ 归一化"]
for x, lab in zip(theta_x, theta_lab):
    rbox(x, 10.35, 2.5, 0.66, lab, color=BLUE, lw=1.8, fs=11, face="white")
ax.text(0.95, 10.35, "θ_c ∈ [0,1]\n缺陷倾向", fontsize=9, color=BLUE, ha="center", va="center")

# knobs
sbox(16.9, 10.35, 1.9, 0.66, "旋钮\nnum_steps / exec_horizon", color=GREEN, fs=8.4, face="#f0fff4")

# ═════════ EPISODE PLATE ═════════
plate(0.35, 0.95, 17.8, 8.55, "样本  i = 1 … N   （动态：每个 episode 重画一次）",
      color=GREY, lx=17.9, ly=1.08)

# exogenous per-episode
circ(1.25, 7.05, 0.44, "场景", color=FG, fs=9.5, face="#f7f7f7", sub="s_i")
circ(1.25, 5.55, 0.44, "指令", color=FG, fs=9.5, face="#f7f7f7", sub="g_i")
ax.text(1.25, 4.72, "混淆因子\n场景难度", fontsize=8.4, color=RED, ha="center", va="top")

vx = [3.4, 5.55, 7.7, 9.85, 12.0]
vnames = ["v 感知", "v 语言", "v 前向", "v 去噪", "v 动作"]
vsub = ["查询位输出", "落地一致性", "路由熵/专家集", "‖v_t‖ 收敛度", "chunk 统计"]


def decision_row(y, kname, color=PURP, faded=False):
    a = 0.45 if faded else 1.0
    plate(2.55, y - 0.95, 11.3, 1.9, kname, color=color, ls=(0, (4, 3)), fs=10, lw=1.4,
          lx=13.75, ly=y - 0.85)
    for x, n, s in zip(vx, vnames, vsub):
        circ(x, y, 0.5, n, color=BLUE, fs=9.5, sub=s, face="white")
    for i in range(len(vx) - 1):
        arrow(vx[i] + 0.5, y, vx[i + 1] - 0.5, y, color=FG, lw=1.6, alpha=a)
    # noise into denoise
    circ(9.85, y + 1.28, 0.34, "噪声", color=RED, fs=8, face="#fff5f5", sub="ε")
    arrow(9.85, y + 0.94, 9.85, y + 0.5, color=RED, lw=1.4, alpha=a)


Y1, Y2 = 7.55, 4.55
decision_row(Y1, "决策 k = 1")
decision_row(Y2, "决策 k = 2")
ax.text(8.2, 2.95, "...   决策 k = 3 … K_i", fontsize=11, color=PURP, ha="center", style="italic")

# scene/instruction feed both rows
for y in (Y1, Y2):
    arrow(1.69, 7.05, vx[0] - 0.44, y + 0.22, color=FG, lw=1.2, rad=-0.1)
    arrow(1.69, 5.55, vx[1] - 0.5, y - 0.28, color=FG, lw=1.2, rad=-0.12)

# θ → v  for both rows
for tx, xx in zip(theta_x, vx):
    arrow(tx, 10.02, xx, Y1 + 0.52, color=BLUE, ls=(0, (3, 2.5)), lw=1.3, alpha=0.9)
    arrow(tx, 10.02, xx, Y2 + 0.52, color=BLUE, ls=(0, (3, 2.5)), lw=1.0, alpha=0.35, rad=0.12)

# ═════════ OUTCOME — inside episode plate, outside decision plates ═════════
circ(16.15, 6.05, 0.78, "结局", color=FG, fs=13, face="#fffbe6", sub="成功 / 失败")
ax.text(16.15, 7.35, "动态：每个 episode 一个", fontsize=9.5, color=FG, ha="center",
        fontweight="bold")
ax.text(16.15, 7.08, "（在样本 plate 内，决策 plate 外）", fontsize=8.5, color=GREY, ha="center")
ax.text(16.15, 4.95, "noisy-OR 聚合\n任一次决策出问题\n即足以导致失败", fontsize=8.6,
        color=GREY, ha="center", va="top")

# every decision row contributes to the single outcome
arrow(vx[-1] + 0.5, Y1, 15.5, 6.28, color=CYC, lw=2.0, rad=-0.1, z=5)
arrow(vx[-1] + 0.5, Y2, 15.5, 5.78, color=CYC, lw=2.0, rad=0.1, z=5)
arrow(13.9, 2.95, 15.55, 5.35, color=PURP, lw=1.3, rad=-0.15, alpha=0.5)

# ═════════ CYCLE CALLOUT ═════════
ax.add_patch(FancyBboxPatch((0.55, 1.25), 6.4, 1.5, boxstyle="round,pad=0.03,rounding_size=0.1",
                            linewidth=1.8, edgecolor=CYC, facecolor="#fffbeb", zorder=6))
ax.text(0.8, 2.5, "无向环从哪来（DAG 无有向环，但骨架有环）", fontsize=10.5,
        color=CYC, fontweight="bold", zorder=7)
ax.text(0.8, 2.13,
        "① 同一个 θ_c 同时喂 k=1 和 k=2，两条链又都汇到同一个结局：\n"
        "     θ_c — v_{c,1} — 结局 — v_{c,2} — θ_c   ← 长度 4 的无向环\n"
        "② 即使只有一次决策：θ_感知 既经 v感知 直达结局，又经 v前向 到结局",
        fontsize=8.8, color=FG, va="top", zorder=7)

ax.text(9.6, 2.05,
        "结论：骨架有环 → LBP 无收敛保证、且系统性低估方差；\n"
        "而组件只有 5 个、样本几百，精确推断/NUTS 都跑得动 → 不需要用 LBP 的近似",
        fontsize=9.6, color=CYC, ha="center", va="center", fontweight="bold")

# ═════════ legend ═════════
lx, ly = 14.45, 2.6
ax.text(lx, ly + 0.45, "图例", fontsize=9.5, color=FG, fontweight="bold")
rbox(lx + 0.3, ly + 0.1, 0.45, 0.24, "", color=BLUE)
ax.text(lx + 0.62, ly + 0.1, "组件（静态）", fontsize=8.4, color=FG, va="center")
circ(lx + 0.3, ly - 0.28, 0.17, "", color=BLUE)
ax.text(lx + 0.62, ly - 0.28, "值节点（动态，须可观测）", fontsize=8.4, color=FG, va="center")
ax.plot([lx + 0.1, lx + 0.5], [ly - 0.66, ly - 0.66], color=BLUE, ls=(0, (3, 2.5)), lw=1.3)
ax.text(lx + 0.62, ly - 0.66, "静态 → 动态", fontsize=8.4, color=FG, va="center")
ax.plot([lx + 0.1, lx + 0.5], [ly - 1.0, ly - 1.0], color=CYC, lw=2.0)
ax.text(lx + 0.62, ly - 1.0, "构成环的边", fontsize=8.4, color=CYC, va="center")

plt.tight_layout()
out = "/data/whn/robotwin_eval/bayes_net_proposed.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", out)
