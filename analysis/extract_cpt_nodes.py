"""Per-component CPT inputs, with the arbitration left to the Bayesian network.

Three faults in the previous extraction, each verified against the data:

1. The node was "was this component the FIRST to drift", produced by an argmin.
   That made the observation matrix effectively one-hot (59% all-zero, 36% one,
   6% two), so which component takes the blame was already settled before the
   network saw anything. Explaining away -- the whole reason to use a Bayesian
   network rather than a rule -- had nothing left to arbitrate. The node is now
   "did this component drift at all", decided independently per component.

2. `execution` was measuring chunk-seam disagreement, i.e. how far the policy
   revised its own plan. That is not execution: arm tracking error is exactly
   zero here because the simulator sets joint positions directly. Tracking error
   is restored as `execution` (it stays near zero in simulation, which is the
   correct null, and becomes informative on real hardware), and plan revision
   gets its own node instead of squatting in execution's slot.

3. At a 90th-percentile cut, 29% of failures had every component reading normal
   and could only be explained by task difficulty. The cut is now looser.

Timing survives as a second indicator rather than as a winner-take-all: whether
the drift began in the opening third of the episode. Encoding timing at all is
what lifted held-out-task discrimination from 0.54 to 0.80; whether that came
from the timing or from the one-hot structure was never separated, and this
version is what separates them.
"""

import argparse
import glob
import json

import numpy as np

COMPONENTS = ["perception", "language", "routing", "denoise", "execution", "plan_revision"]
ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP = np.array([6, 13])


def series(z, cen):
    """Per-decision raw score for every component, full episode length."""
    n = z["predicted_chunks"].shape[0]
    out = {}
    for c, k in (("perception", "intro_h_query"), ("language", "intro_h_lang")):
        out[c] = np.linalg.norm(z[k][:n].astype(np.float32) - cen[c][None, :], axis=1)
    out["routing"] = z["intro_router_entropy"][:n].astype(np.float32).min(axis=(1, 2))
    out["denoise"] = np.abs(z["intro_denoise_x"][:n].astype(np.float32)[:, -1]).max(axis=(1, 2))

    st = z["decision_steps"]
    h = int(np.median(np.diff(st))) if len(st) > 2 else 25

    # execution: what the simulator actually reached vs what was commanded.
    # states[k+1] is the configuration produced by executing executed[k].
    states = z["states"].astype(np.float32)
    ex = z["executed_actions"].astype(np.float32)
    T = min(len(states) - 1, len(ex))
    step_err = np.abs(states[1:T + 1] - ex[:T]).mean(axis=1) if T > 0 else np.zeros(1)
    dec_err = np.full(n, np.nan, np.float32)
    for k in range(n):
        lo, hi = k * h, min((k + 1) * h, T)
        if hi > lo:
            dec_err[k] = step_err[lo:hi].mean()
    out["execution"] = dec_err

    # plan_revision: how far decision k+1 walks back decision k's plan for the
    # same future timesteps.
    ch = z["predicted_chunks"].astype(np.float32)
    ov = ch.shape[1] - h
    seam = np.full(n, np.nan, np.float32)
    if ov > 0:
        for k in range(n - 1):
            seam[k] = np.abs(ch[k, h:h + ov] - ch[k + 1, :ov]).mean()
    out["plan_revision"] = seam
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/data/whn/robotwin_eval/rollouts")
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--early-frac", type=float, default=0.34,
                    help="a drift counts as early if it begins in this leading fraction")
    ap.add_argument("--out", default="/data/whn/robotwin_eval/node_values_cpt.npz")
    cfg = ap.parse_args()

    files = []
    for run in sorted(glob.glob(f"{cfg.rollouts}/*/")):
        for f in sorted(glob.glob(run + "episodes/*.npz")):
            meta = json.load(open(f.replace(".npz", ".json")))
            if meta["num_decisions"] >= cfg.persist + 1:
                files.append((f, meta))

    acc = {}
    for f, meta in files:
        with np.load(f) as z:
            if "intro_h_query" not in z.files:
                continue
            for c, k in (("perception", "intro_h_query"), ("language", "intro_h_lang")):
                v = z[k].astype(np.float64)
                s, n = acc.setdefault((meta["task_name"], c), [np.zeros(v.shape[1]), 0])
                acc[(meta["task_name"], c)] = [s + v.sum(0), n + v.shape[0]]
    cen_all = {k: (s / max(n, 1)).astype(np.float32) for k, (s, n) in acc.items()}

    eps = []
    for f, meta in files:
        with np.load(f) as z:
            if "intro_h_query" not in z.files:
                continue
            cen = {c: cen_all[(meta["task_name"], c)] for c in ("perception", "language")}
            eps.append(dict(meta=meta, s=series(z, cen)))

    tasks = sorted({e["meta"]["task_name"] for e in eps})
    y = np.array([0 if e["meta"]["success"] else 1 for e in eps], dtype=np.int32)
    print(f"{len(eps)} 次尝试 | {int(y.sum())} 次失败 | {len(tasks)} 个任务 | "
          f"异常判定阈值 = 任务内 {cfg.quantile:.0%} 分位")

    thr = {}
    for t in tasks:
        idx = [i for i, e in enumerate(eps) if e["meta"]["task_name"] == t]
        for c in COMPONENTS:
            v = np.concatenate([eps[i]["s"][c] for i in idx])
            v = v[np.isfinite(v)]
            thr[(t, c)] = np.quantile(v, cfg.quantile) if len(v) else np.inf

    ood = np.zeros((len(eps), len(COMPONENTS)), np.float32)
    early = np.zeros_like(ood)
    for i, e in enumerate(eps):
        t = e["meta"]["task_name"]
        for j, c in enumerate(COMPONENTS):
            v = e["s"][c]
            hot = np.isfinite(v) & (v > thr[(t, c)])
            run, at = 0, np.inf
            for k, h in enumerate(hot):
                run = run + 1 if h else 0
                if run >= cfg.persist:
                    at = k - cfg.persist + 1
                    break
            if np.isfinite(at):
                ood[i, j] = 1.0
                if at <= max(1, int(cfg.early_frac * len(v))):
                    early[i, j] = 1.0

    task_id = np.array([tasks.index(e["meta"]["task_name"]) for e in eps], dtype=np.int32)
    np.savez_compressed(cfg.out, ood=ood, early=early, y=y, task_id=task_id,
                        components=np.array(COMPONENTS), tasks=np.array(tasks),
                        seeds=np.array([e["meta"]["seed"] for e in eps]))
    print(f"wrote {cfg.out}\n")

    print("=== 每个环节被判为「偏离过」的比例 ===")
    print(f"{'环节':<15}{'成功样本':>10}{'失败样本':>10}{'差':>9}{'其中偏离得早':>14}")
    print("-" * 60)
    for j, c in enumerate(COMPONENTS):
        s_, f_ = ood[y == 0, j].mean(), ood[y == 1, j].mean()
        e_ = early[y == 1, j].mean()
        print(f"{c:<15}{s_:>10.3f}{f_:>10.3f}{f_ - s_:>+9.3f}{e_:>14.3f}")

    n_on = ood.sum(1)
    print(f"\n=== 仲裁权是否交还给了网络 ===")
    print(f"  每个失败样本平均有 {n_on[y == 1].mean():.2f} 个环节同时异常"
          f"（旧编码恒为 ≤1，网络无从仲裁）")
    print(f"  全部正常的失败样本: {(n_on[y == 1] == 0).mean():.0%}（旧编码 29%）")
    print(f"  同时 2 个及以上异常的失败样本: {(n_on[y == 1] >= 2).mean():.0%}")


if __name__ == "__main__":
    main()
