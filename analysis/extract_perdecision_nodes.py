"""Per-decision value nodes: is component c out-of-distribution at decision k.

Model A asked only *which* component drifted first and threw away when. This
keeps the decision axis so the responsibility posterior can name a moment as
well as a component.

The trap this has to avoid: a failure runs to the step limit and makes ~60
decisions, a success ends after ~6. If every decision contributed its own chance
of dooming the episode, longer episodes would look more failure-prone -- but the
length is a *consequence* of failing, so that reasoning is circular. Every
episode is therefore cut to the same fixed number of decisions, and an episode
that finished early is padded with "not out-of-distribution", which is what
finishing early actually means.

Thresholds stay label-free: a decision counts as out-of-distribution when its
score exceeds that task's own 90th percentile, pooled over all of the task's
episodes regardless of how they ended.
"""

import argparse
import glob
import json

import numpy as np

COMPONENTS = ["perception", "language", "routing", "denoise", "execution"]


def per_decision_series(z, centroids):
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
    ap.add_argument("--kmax", type=int, default=8,
                    help="decisions kept per episode; shorter episodes are padded as normal")
    ap.add_argument("--quantile", type=float, default=0.90)
    ap.add_argument("--out", default="/data/whn/robotwin_eval/node_values_perdecision.npz")
    cfg = ap.parse_args()

    files = []
    for run in sorted(glob.glob(f"{cfg.rollouts}/*/")):
        for f in sorted(glob.glob(run + "episodes/*.npz")):
            meta = json.load(open(f.replace(".npz", ".json")))
            if meta["num_decisions"] >= 3:
                files.append((f, meta))

    acc = {}
    for f, meta in files:
        with np.load(f) as z:
            if "intro_h_query" not in z.files:
                continue
            for comp, key in (("perception", "intro_h_query"), ("language", "intro_h_lang")):
                v = z[key].astype(np.float64)
                s, c = acc.setdefault((meta["task_name"], comp), [np.zeros(v.shape[1]), 0])
                acc[(meta["task_name"], comp)] = [s + v.sum(0), c + v.shape[0]]
    centroids = {k: (s / max(c, 1)).astype(np.float32) for k, (s, c) in acc.items()}

    episodes = []
    for f, meta in files:
        with np.load(f) as z:
            if "intro_h_query" not in z.files:
                continue
            cen = {c: centroids[(meta["task_name"], c)] for c in ("perception", "language")}
            episodes.append(dict(meta=meta, series=per_decision_series(z, cen)))

    tasks = sorted({e["meta"]["task_name"] for e in episodes})
    print(f"{len(episodes)} 次尝试 | {sum(1 for e in episodes if not e['meta']['success'])} 次失败 "
          f"| {len(tasks)} 个任务 | 每次尝试只看前 {cfg.kmax} 次决策")

    thr = {}
    for t in tasks:
        idx = [i for i, e in enumerate(episodes) if e["meta"]["task_name"] == t]
        for c in COMPONENTS:
            allv = np.concatenate([episodes[i]["series"][c] for i in idx])
            allv = allv[np.isfinite(allv)]
            thr[(t, c)] = np.quantile(allv, cfg.quantile) if len(allv) else np.inf

    a = np.zeros((len(episodes), cfg.kmax, len(COMPONENTS)), dtype=np.float32)
    for i, e in enumerate(episodes):
        t = e["meta"]["task_name"]
        for j, c in enumerate(COMPONENTS):
            s = e["series"][c][:cfg.kmax]
            hot = np.isfinite(s) & (s > thr[(t, c)])
            a[i, :len(hot), j] = hot.astype(np.float32)

    y = np.array([0 if e["meta"]["success"] else 1 for e in episodes], dtype=np.int32)
    task_id = np.array([tasks.index(e["meta"]["task_name"]) for e in episodes], dtype=np.int32)
    n_dec = np.array([min(e["meta"]["num_decisions"], cfg.kmax) for e in episodes], dtype=np.int32)
    np.savez_compressed(cfg.out, a=a, y=y, task_id=task_id, n_dec=n_dec,
                        components=np.array(COMPONENTS), tasks=np.array(tasks))
    print(f"wrote {cfg.out}\n")

    print("每次决策上「被判为异常」的比例（越靠后的决策，成功组因为已结束而补为正常）")
    print(f"{'决策序号':<10}" + "".join(f"{c[:9]:>11}" for c in COMPONENTS))
    for k in range(cfg.kmax):
        print(f"  第{k}次    " + "".join(f"{a[:, k, j].mean():>11.3f}" for j in range(len(COMPONENTS))))
    print(f"\n成功组平均决策次数 {n_dec[y == 0].mean():.1f}，失败组 {n_dec[y == 1].mean():.1f}"
          f"（上限 {cfg.kmax}，差距已被截断压到最小）")


if __name__ == "__main__":
    main()
