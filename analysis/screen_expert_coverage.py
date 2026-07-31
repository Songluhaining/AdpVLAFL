"""Is there signal at individual-expert granularity, rather than aggregated routing?

The five architectural components turned out to be task-specific (tau_routing's
94% HDI excludes zero), so a coarse component-level attribution does not
generalize. MoE routing offers a much finer decomposition -- 36 layers x 32
experts = 1152 units, each with a per-episode measurement -- which is the
granularity spectrum-based fault localization is designed for.

The literal coverage analogy has to be checked, not assumed. Program fault
localization works without intervention because coverage is exact and a
statement that never ran cannot have caused the failure. That only transfers
here if experts are actually sparsely activated; with top-4 of 32 over ~51
tokens x 10 denoising steps per decision, every expert may well fire in every
episode, which would make binary coverage vacuous. So this measures the
activation structure first and only then scores discriminativeness.
"""

import glob
import json

import numpy as np

W = 4  # same matched leading window as the component-level extraction


def episode_load(z, w):
    """Per (layer, expert) share of routed tokens, averaged over the window."""
    rc = z["intro_router_counts"][:w].astype(np.float32)   # (n, steps, layers, experts)
    load = rc.mean(axis=(0, 1))                            # (layers, experts)
    return load / (load.sum(axis=-1, keepdims=True) + 1e-8)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


loads, ys, tasks_of = [], [], []
for f in sorted(glob.glob("/data/whn/robotwin_eval/rollouts/*/episodes/*.npz")):
    meta = json.load(open(f.replace(".npz", ".json")))
    if meta["num_decisions"] < W:
        continue
    with np.load(f) as z:
        if "intro_router_counts" not in z.files:
            continue
        loads.append(episode_load(z, W))
    ys.append(0 if meta["success"] else 1)
    tasks_of.append(meta["task_name"])

L = np.stack(loads)                       # (episodes, layers, experts)
y = np.asarray(ys)
tasks = np.asarray(tasks_of)
n_ep, n_layer, n_exp = L.shape
print(f"{n_ep} episodes | {y.sum()} failures | {n_layer} layers x {n_exp} experts "
      f"= {n_layer * n_exp} units | window={W}\n")

# ---- 1. is activation sparse enough for a coverage analogy? ---------------
active = L > 1e-6
print("=== 激活结构（决定能否套用覆盖谱） ===")
print(f"  每 episode 平均激活专家数/层: {active.sum(axis=2).mean():.1f} / {n_exp}")
print(f"  从未激活的 (层,专家) 单元占比: {(~active).all(axis=0).mean():.3f}")
print(f"  在所有 episode 都激活的占比 : {active.all(axis=0).mean():.3f}")
share = L.mean(axis=0)
print(f"  负载份额: 最小 {share.min():.4f} 最大 {share.max():.4f} "
      f"(均匀=1/{n_exp}={1/n_exp:.4f})")

# ---- 2. within-task AUC per unit ------------------------------------------
uniq = sorted(set(tasks_of))
flat = L.reshape(n_ep, -1)
scores = np.full(flat.shape[1], np.nan)
for u in range(flat.shape[1]):
    aa, ww = [], []
    for t in uniq:
        m = tasks == t
        pos, neg = flat[m & (y == 1), u], flat[m & (y == 0), u]
        if len(pos) and len(neg):
            v = auc(pos, neg)
            if np.isfinite(v):
                aa.append(v); ww.append(len(pos))
    if aa:
        scores[u] = np.average(aa, weights=ww)

sep = np.abs(scores - 0.5)
ok = np.isfinite(sep)
print(f"\n=== 单元判别力（任务内 AUC） ===")
print(f"  可评分单元: {ok.sum()} / {len(sep)}")
for thr in (0.05, 0.10, 0.15, 0.20):
    print(f"  |AUC-0.5| > {thr:.2f} 的单元数: {(sep[ok] > thr).sum():>5}  "
          f"（随机期望下应接近 0）")
print(f"  最大 |AUC-0.5| = {np.nanmax(sep):.3f}")

# A permutation null: shuffle outcomes within task and re-score a subsample, so
# "how many units look discriminative" can be read against chance rather than
# against zero.
rng = np.random.default_rng(0)
y_perm = y.copy()
for t in uniq:
    m = tasks == t
    y_perm[m] = rng.permutation(y[m])
sub = rng.choice(flat.shape[1], size=200, replace=False)
null = []
for u in sub:
    aa, ww = [], []
    for t in uniq:
        m = tasks == t
        pos, neg = flat[m & (y_perm == 1), u], flat[m & (y_perm == 0), u]
        if len(pos) and len(neg):
            v = auc(pos, neg)
            if np.isfinite(v):
                aa.append(v); ww.append(len(pos))
    if aa:
        null.append(abs(np.average(aa, weights=ww) - 0.5))
null = np.asarray(null)
print(f"\n  置换零假设 (200 单元抽样): |AUC-0.5| 均值 {null.mean():.3f}, "
      f"95 分位 {np.percentile(null, 95):.3f}")
print(f"  实测超过该 95 分位的单元占比: {(sep[ok] > np.percentile(null, 95)).mean():.3f} "
      f"(零假设下应为 0.05)")

# ---- 3. where does the signal live? ---------------------------------------
per_layer = np.nanmax(sep.reshape(n_layer, n_exp), axis=1)
print(f"\n=== 信号在哪些层 (每层最强单元的 |AUC-0.5|) ===")
top = np.argsort(-per_layer)[:8]
for l in sorted(top):
    e = int(np.nanargmax(sep.reshape(n_layer, n_exp)[l]))
    print(f"  层 {l:>2}  专家 {e:>2}   |AUC-0.5| = {per_layer[l]:.3f}   "
          f"AUC = {scores.reshape(n_layer, n_exp)[l, e]:.3f}")
print(f"\n  浅层(0-11) 均值 {per_layer[:12].mean():.3f} | "
      f"中层(12-23) {per_layer[12:24].mean():.3f} | 深层(24-35) {per_layer[24:].mean():.3f}")

np.savez_compressed("/data/whn/robotwin_eval/expert_scores.npz",
                    auc=scores.reshape(n_layer, n_exp), load=L, y=y, tasks=tasks)
print("\n-> /data/whn/robotwin_eval/expert_scores.npz")
