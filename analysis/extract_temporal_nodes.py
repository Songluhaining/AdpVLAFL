"""Value nodes built from *when* each component goes out-of-distribution, not how much.

The magnitude-based nodes collapsed each episode into one number over a fixed
leading window, discarding the ordering entirely. Ordering is the one piece of
causal information available without intervening: a downstream effect cannot
precede its cause. So if perception drifts first and routing only afterwards,
routing is the weaker candidate -- and if execution's chunk-seam disagreement
consistently arrives last, it is a symptom of the failure rather than its cause,
which would explain why shortening exec_horizon flipped nothing.

Two design choices carry the analysis:

1. The node is the *relative order between components*, not the absolute
   changepoint index. Failures run to the step limit (median 16 decisions)
   while successes end early (median 6), so any absolute time is confounded by
   episode length. Which component moved first is an intra-episode comparison,
   where every component faces the same length.

2. Thresholding stays label-free: a decision counts as out-of-distribution when
   its score exceeds the task's own 90th percentile, pooled over every episode
   of that task regardless of outcome. Nothing about success or failure enters
   the feature, so the downstream fit is not circular.
"""

import argparse
import glob
import json

import numpy as np

COMPONENTS = ["perception", "language", "routing", "denoise", "execution"]


def per_decision_series(z, centroids):
    """Full-length per-decision raw score for every component (no window)."""
    n = z["predicted_chunks"].shape[0]
    out = {}
    for comp, key in (("perception", "intro_h_query"), ("language", "intro_h_lang")):
        v = z[key][:n].astype(np.float32)
        out[comp] = np.linalg.norm(v - centroids[comp][None, :], axis=1)

    out["routing"] = z["intro_router_entropy"][:n].astype(np.float32).min(axis=(1, 2))
    dx = z["intro_denoise_x"][:n].astype(np.float32)
    out["denoise"] = np.abs(dx[:, -1]).max(axis=(1, 2))

    chunks = z["predicted_chunks"].astype(np.float32)
    steps = z["decision_steps"]
    h = int(np.median(np.diff(steps))) if len(steps) > 2 else 25
    overlap = chunks.shape[1] - h
    seam = np.full(n, np.nan, dtype=np.float32)
    if overlap > 0:
        for k in range(n - 1):
            seam[k] = np.abs(chunks[k, h:h + overlap] - chunks[k + 1, :overlap]).mean()
    out["execution"] = seam
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/data/whn/robotwin_eval/rollouts")
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="within-task percentile above which a decision counts as OOD")
    ap.add_argument("--persist", type=int, default=2,
                    help="consecutive OOD decisions required, to reject single-decision noise")
    ap.add_argument("--out", default="/data/whn/robotwin_eval/node_values_temporal.npz")
    cfg = ap.parse_args()

    files = []
    for run in sorted(glob.glob(f"{cfg.rollouts}/*/")):
        for f in sorted(glob.glob(run + "episodes/*.npz")):
            meta = json.load(open(f.replace(".npz", ".json")))
            if meta["num_decisions"] >= cfg.persist + 1:
                files.append((f, meta))
    print(f"{len(files)} episodes")

    # Pass 1: per-task centroids for the two hidden-state components. Streamed
    # rather than held in memory -- the raw prefix readouts are ~900MB in fp32.
    acc = {}
    for f, meta in files:
        t = meta["task_name"]
        with np.load(f) as z:
            if "intro_h_query" not in z.files:
                continue
            for comp, key in (("perception", "intro_h_query"), ("language", "intro_h_lang")):
                v = z[key].astype(np.float64)
                s, c = acc.setdefault((t, comp), [np.zeros(v.shape[1]), 0])
                acc[(t, comp)] = [s + v.sum(0), c + v.shape[0]]
    centroids = {k: (s / max(c, 1)).astype(np.float32) for k, (s, c) in acc.items()}
    print(f"centroids for {len(centroids)} (task, component) pairs")

    # Pass 2: per-decision series per episode.
    episodes = []
    for f, meta in files:
        t = meta["task_name"]
        with np.load(f) as z:
            if "intro_h_query" not in z.files:
                continue
            cen = {c: centroids[(t, c)] for c in ("perception", "language")}
            episodes.append(dict(meta=meta, series=per_decision_series(z, cen)))
    tasks = sorted({e["meta"]["task_name"] for e in episodes})
    n_fail = sum(1 for e in episodes if not e["meta"]["success"])
    print(f"{len(episodes)} usable | {n_fail} failures | {len(tasks)} tasks")

    # Within-task OOD thresholds, pooled over all decisions of all episodes.
    thr = {}
    for t in tasks:
        idx = [i for i, e in enumerate(episodes) if e["meta"]["task_name"] == t]
        for c in COMPONENTS:
            allv = np.concatenate([episodes[i]["series"][c] for i in idx])
            allv = allv[np.isfinite(allv)]
            thr[(t, c)] = np.quantile(allv, cfg.quantile) if len(allv) else np.inf

    # Changepoint = first index starting a run of `persist` OOD decisions.
    onset = np.full((len(episodes), len(COMPONENTS)), np.inf)
    for i, e in enumerate(episodes):
        t = e["meta"]["task_name"]
        for j, c in enumerate(COMPONENTS):
            s = e["series"][c]
            hot = np.isfinite(s) & (s > thr[(t, c)])
            run = 0
            for k, h in enumerate(hot):
                run = run + 1 if h else 0
                if run >= cfg.persist:
                    onset[i, j] = k - cfg.persist + 1
                    break

    # Node: was this component the first to go OOD in this episode. Ties share.
    a = np.zeros_like(onset)
    for i in range(len(episodes)):
        m = onset[i].min()
        if np.isfinite(m):
            a[i] = (onset[i] == m).astype(float)

    y = np.array([0 if e["meta"]["success"] else 1 for e in episodes], dtype=np.int32)
    task_id = np.array([tasks.index(e["meta"]["task_name"]) for e in episodes], dtype=np.int32)
    np.savez_compressed(cfg.out, a=a, y=y, task_id=task_id, onset=onset,
                        components=np.array(COMPONENTS), tasks=np.array(tasks))
    print(f"wrote {cfg.out}\n")

    print("=== 谁最先变异常（占该组样本的比例）===")
    print(f"{'component':<12}{'成功':>10}{'失败':>10}{'差':>9}")
    print("-" * 42)
    for j, c in enumerate(COMPONENTS):
        s, f_ = a[y == 0, j].mean(), a[y == 1, j].mean()
        flag = "  <--" if abs(f_ - s) > 0.05 else ""
        print(f"{c:<12}{s:>10.3f}{f_:>10.3f}{f_ - s:>+9.3f}{flag}")

    print("\n=== 变点时刻（决策序号，仅计有变点的样本）===")
    print(f"{'component':<12}{'成功中位':>10}{'失败中位':>10}{'失败样本有变点占比':>20}")
    print("-" * 54)
    for j, c in enumerate(COMPONENTS):
        so = onset[(y == 0), j]; fo = onset[(y == 1), j]
        sm = np.median(so[np.isfinite(so)]) if np.isfinite(so).any() else np.nan
        fm = np.median(fo[np.isfinite(fo)]) if np.isfinite(fo).any() else np.nan
        print(f"{c:<12}{sm:>10.1f}{fm:>10.1f}{np.isfinite(fo).mean():>20.2f}")
    print("\n  若 execution 的变点系统性晚于其他组件，即为下游症状而非原因")


if __name__ == "__main__":
    main()
