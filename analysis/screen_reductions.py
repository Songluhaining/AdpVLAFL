"""Screen candidate reductions for the routing and denoising value nodes.

The first definitions of both carried no signal (routing -0.005, denoise -0.025
in mean percentile rank between successes and failures). Rather than guess again,
this enumerates plausible reductions of the same recorded tensors and ranks them
by how well they separate outcomes.

Scored with within-task AUC, pooled across tasks. AUC because it is
threshold-free and invariant to any monotone rescaling, which is what the
percentile-rank step downstream applies anyway; within-task because task
difficulty is the dominant confound and must not be what gets measured.
"""

import glob
import json

import numpy as np

W = 4  # same matched leading window as extract_node_values.py


def reductions(z, n):
    """Candidate scalars per decision, from the recorded internals."""
    out = {}

    rc = z["intro_router_counts"][:n].astype(np.float32)   # (n, steps, layers, experts)
    re_ = z["intro_router_entropy"][:n].astype(np.float32)  # (n, steps, layers)
    load = rc / (rc.sum(-1, keepdims=True) + 1e-8)

    out["route_ent_mean"] = re_.mean(axis=(1, 2))
    out["route_ent_min"] = re_.min(axis=(1, 2))
    out["route_ent_max"] = re_.max(axis=(1, 2))
    out["route_ent_std_layer"] = re_.mean(axis=1).std(axis=1)
    out["route_ent_deep"] = re_[:, :, 24:].mean(axis=(1, 2))
    out["route_ent_shallow"] = re_[:, :, :12].mean(axis=(1, 2))
    # How much the expert choice moves as denoising proceeds. Routing is
    # recomputed at every denoising step, so this drift is only visible because
    # the capture keeps the steps separate.
    out["route_drift"] = np.abs(load[:, -1] - load[:, 0]).sum(-1).mean(-1)
    out["route_load_max"] = load.mean(axis=1).max(-1).mean(-1)
    # Concentration of load: 1 = one expert does everything.
    p = load.mean(axis=1)
    out["route_gini"] = (p ** 2).sum(-1).mean(-1)
    out["route_dead"] = (p < 1e-6).sum(-1).mean(-1).astype(np.float32)

    dx = z["intro_denoise_x"][:n].astype(np.float32)       # (n, steps, 50, 55)
    vn = z["intro_denoise_v_norm"][:n].astype(np.float32)  # (n, steps)
    flat = dx.reshape(dx.shape[0], dx.shape[1], -1)
    seg = np.linalg.norm(np.diff(flat, axis=1), axis=2)    # (n, steps-1)
    chord = np.linalg.norm(flat[:, -1] - flat[:, 0], axis=1)

    out["dn_straight"] = seg.sum(1) / (chord + 1e-8)
    out["dn_chord"] = chord
    out["dn_last_frac"] = seg[:, -1] / (seg.sum(1) + 1e-8)
    out["dn_first_frac"] = seg[:, 0] / (seg.sum(1) + 1e-8)
    out["dn_seg_std"] = seg.std(1) / (seg.mean(1) + 1e-8)
    out["dn_vnorm_mean"] = vn.mean(1)
    out["dn_vnorm_slope"] = (vn[:, -1] - vn[:, 0]) / (vn[:, 0] + 1e-8)
    out["dn_final_absmax"] = np.abs(dx[:, -1]).max(axis=(1, 2))
    out["dn_final_absmean"] = np.abs(dx[:, -1]).mean(axis=(1, 2))

    hq = z["intro_h_query"][:n].astype(np.float32)
    out["perc_qnorm"] = np.linalg.norm(hq, axis=1)
    qt = z["intro_h_query_tokens"][:n].astype(np.float32)  # (n, 8, D)
    # Spread across the 8 distillation queries: they are asked to encode
    # different aspects of the scene, so collapse between them is meaningful.
    out["perc_qspread"] = qt.std(axis=1).mean(axis=1)

    return out


def auc(pos, neg):
    """Mann-Whitney AUC, ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    rp = r[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rows = []
for f in sorted(glob.glob("rollouts/*/episodes/*.npz")):
    meta = json.load(open(f.replace(".npz", ".json")))
    if meta["num_decisions"] < W:
        continue
    with np.load(f) as z:
        if "intro_router_counts" not in z.files:
            continue
        r = reductions(z, min(W, z["intro_router_counts"].shape[0]))
    rows.append((meta["task_name"], meta["success"], {k: float(np.nanmean(v)) for k, v in r.items()}))

keys = sorted(rows[0][2])
tasks = sorted({t for t, _, _ in rows})
n_f = sum(1 for _, s, _ in rows if not s)
print(f"{len(rows)} episodes | {n_f} failures | {len(tasks)} tasks | window={W}\n")

scored = []
for k in keys:
    aucs, wts = [], []
    for t in tasks:
        pos = np.array([d[k] for tt, s, d in rows if tt == t and not s])
        neg = np.array([d[k] for tt, s, d in rows if tt == t and s])
        if len(pos) and len(neg):
            a = auc(pos, neg)
            if np.isfinite(a):
                aucs.append(a)
                wts.append(len(pos))
    if aucs:
        pooled = float(np.average(aucs, weights=wts))
        scored.append((abs(pooled - 0.5), pooled, k, len(aucs), sum(wts)))

scored.sort(reverse=True)
print(f"{'reduction':<22}{'within-task AUC':>17}{'|AUC-.5|':>11}{'tasks':>7}{'fails':>7}")
print("-" * 66)
for sep, a, k, nt, nf in scored:
    mark = "  <-- 有信号" if sep >= 0.10 else ("  ~" if sep >= 0.06 else "")
    print(f"{k:<22}{a:>17.3f}{sep:>11.3f}{nt:>7}{nf:>7}{mark}")
print("\nAUC 0.5 = 无区分度；>0.5 表示失败样本该量更大，<0.5 表示更小")
