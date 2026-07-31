"""Do the discriminative experts agree across tasks, or does each task have its own?

Expert-level routing carries real signal (52.9% of the 1152 units beat a
within-task permutation null, against 5% expected). But that says nothing about
whether it is the *same* experts everywhere. The component-level model already
failed on exactly this point -- tau_routing's 94% HDI excluded zero, meaning the
routing effect was strongly task-specific -- so the finer granularity is only
worth pursuing if the signature transfers.

Three tests, from weakest to strongest evidence:
  1. Rank correlation of the per-unit AUC between disjoint halves of the tasks.
  2. Overlap of each task's own top-k units against a permutation baseline.
  3. Held-out-task AUC of a signature fitted only on the other tasks, which is
     the number that actually matters for a general method.
"""

import numpy as np
from scipy.stats import spearmanr

D = np.load("/data/whn/robotwin_eval/expert_scores.npz", allow_pickle=True)
L, y, tasks = D["load"], D["y"], D["tasks"]
n_ep, n_layer, n_exp = L.shape
flat = L.reshape(n_ep, -1)
uniq = sorted(set(tasks.tolist()))
# Only tasks with both outcomes can score anything.
usable = [t for t in uniq if 0 < y[tasks == t].sum() < (tasks == t).sum()]
print(f"{n_ep} episodes | {y.sum()} failures | {len(usable)} 个任务同时有成功和失败样本\n")


def auc_vec(mask_pos, mask_neg):
    """AUC of every unit at once, from rank sums."""
    pos, neg = flat[mask_pos], flat[mask_neg]
    if len(pos) == 0 or len(neg) == 0:
        return np.full(flat.shape[1], np.nan)
    allv = np.vstack([pos, neg])
    r = np.argsort(np.argsort(allv, axis=0), axis=0).astype(float) + 1.0
    return (r[: len(pos)].sum(0) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def signature(task_list):
    """Failure-direction signature pooled over a set of tasks, task-standardised."""
    acc, wts = [], []
    for t in task_list:
        m = tasks == t
        a = auc_vec(m & (y == 1), m & (y == 0))
        if np.isfinite(a).all():
            acc.append(a - 0.5)
            wts.append(int((m & (y == 1)).sum()))
    return np.average(acc, axis=0, weights=wts) if acc else None


rng = np.random.default_rng(0)

# ---- 1. split-half agreement ---------------------------------------------
print("=== 检验1: 任务分半，两半的单元判别力是否一致 ===")
rhos = []
for rep in range(20):
    perm = rng.permutation(usable)
    h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
    s1, s2 = signature(h1), signature(h2)
    if s1 is not None and s2 is not None:
        rhos.append(spearmanr(s1, s2).statistic)
print(f"  20 次随机分半的 Spearman rho: 均值 {np.mean(rhos):+.3f}  "
      f"范围 [{np.min(rhos):+.3f}, {np.max(rhos):+.3f}]")
print("  (接近 0 = 两半各自认出的专家没有共识；接近 1 = 高度一致)")

# ---- 2. top-k overlap vs a label-permutation baseline ---------------------
print("\n=== 检验2: 各任务的 top-k 单元重叠度 vs 置换基线 ===")
per_task = {}
for t in usable:
    m = tasks == t
    a = auc_vec(m & (y == 1), m & (y == 0))
    if np.isfinite(a).all():
        per_task[t] = a - 0.5
keys = list(per_task)
for k in (25, 100):
    tops = {t: set(np.argsort(-np.abs(v))[:k].tolist()) for t, v in per_task.items()}
    obs = [len(tops[a] & tops[b]) / k
           for i, a in enumerate(keys) for b in keys[i + 1:]]
    # Null: shuffle outcomes within each task, redo everything.
    null = []
    for rep in range(5):
        yp = y.copy()
        for t in usable:
            m = tasks == t
            yp[m] = rng.permutation(y[m])
        tp = {}
        for t in usable:
            m = tasks == t
            pos, neg = flat[m & (yp == 1)], flat[m & (yp == 0)]
            if len(pos) and len(neg):
                allv = np.vstack([pos, neg])
                r = np.argsort(np.argsort(allv, axis=0), axis=0).astype(float) + 1.0
                a = (r[: len(pos)].sum(0) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
                tp[t] = set(np.argsort(-np.abs(a - 0.5))[:k].tolist())
        kk = list(tp)
        null += [len(tp[a] & tp[b]) / k for i, a in enumerate(kk) for b in kk[i + 1:]]
    print(f"  top-{k:>3}: 实测任务两两重叠 {np.mean(obs):.3f}   "
          f"置换基线 {np.mean(null):.3f}   比值 {np.mean(obs)/max(np.mean(null),1e-9):.2f}x")

# ---- 3. held-out-task transfer -------------------------------------------
print("\n=== 检验3: 留出任务迁移（真正决定通用性的指标）===")
folds = np.array_split(rng.permutation(usable), 4)
aucs = []
for f_i, held in enumerate(folds):
    train = [t for t in usable if t not in set(held)]
    sig = signature(train)
    if sig is None:
        continue
    # Score a held-out episode by projecting its load onto the trained
    # signature, standardised within its own task so the comparison is about
    # the pattern rather than that task's absolute load level.
    scores, labels = [], []
    for t in held:
        m = tasks == t
        X = flat[m]
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        scores.append(Xs @ sig)
        labels.append(y[m])
    s = np.concatenate(scores); lb = np.concatenate(labels)
    if 0 < lb.sum() < len(lb):
        allv = np.concatenate([s[lb == 1], s[lb == 0]])
        r = np.argsort(np.argsort(allv)).astype(float) + 1.0
        npos = int(lb.sum())
        aucs.append((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * (len(lb) - npos)))
        print(f"  fold {f_i}: 留出 {len(held)} 任务 / {len(lb)} episode / "
              f"{npos} 失败 -> AUC {aucs[-1]:.3f}")
print(f"  平均留出任务 AUC = {np.mean(aucs):.3f}")
print(f"  对照: 5 组件分层模型的留出任务 AUC = 0.542")
