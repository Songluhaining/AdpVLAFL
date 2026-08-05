"""Prepare the routing-correction experiment from the diagnosis.

For every admitted failure the diagnosis produced a responsibility split; here
each failing scene gets (a) a group label -- A if routing carries the largest
responsibility, B otherwise -- and (b) a routing-correction target: the average
expert-load profile of the success decisions it was matched to, minus its own.
Pushing the router scores by that delta steers selection toward the experts its
counterfactual used, per layer.

The validation logic is the A/B contrast. Both groups receive the *same*
correction; if the diagnosis tracks reality, scenes it blamed on routing should
be rescued more often than scenes it blamed elsewhere. Numerical (bf16)
run-to-run noise hits both groups alike, so the contrast survives it.

Matching is imported from build_pairs, not re-implemented, so the pairs used to
build the correction are byte-identical to the ones the diagnosis saw.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from build_pairs import COMPONENTS, KMAX, load_task, match_task

SLOW_TASKS = {"open_microwave", "hanging_mug"}   # ~50s+/scene vs ~15s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/data/whn/robotwin_eval/rollouts")
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock",
                    "open_microwave", "hanging_mug", "place_can_basket", "stamp_seal"])
    ap.add_argument("--img-weight", type=float, default=0.5)
    ap.add_argument("--cf-quantile", type=float, default=0.50)
    ap.add_argument("--slow-cap", type=int, default=12,
                    help="per group, per slow task: cap on scenes so the run stays hours not days")
    ap.add_argument("--resp", default="/data/whn/robotwin_eval/pair_responsibility.npz")
    ap.add_argument("--outdir", default="/data/whn/robotwin_eval/correction")
    cfg = ap.parse_args()

    R = np.load(cfg.resp, allow_pickle=True)
    resp, y_ep, ep_ids = R["resp"], R["y"], R["episode"]
    comps = [str(c) for c in R["components"]]
    j_route = comps.index("routing")
    top = resp.argmax(axis=1)
    blamed = {int(e): int(t) for e, t, yy in zip(ep_ids, top, y_ep) if yy == 1}

    out = Path(cfg.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ep_base = 0
    plan = {}

    for task in cfg.tasks:
        rows = load_task(task, cfg.rollouts)
        if len(rows) < 6:
            continue
        mt = match_task(rows, cfg.img_weight)
        ep, f_idx = mt["ep"], mt["f_idx"]
        m_f, dist_f = mt["m_f"], mt["dist_f"]
        thr = np.quantile(mt["dist_s"], cfg.cf_quantile)

        starts = np.cumsum([0] + [r["n"] for r in rows])   # decision row offsets

        # per-decision normalized expert-load profiles, read lazily per episode
        files = sorted(glob.glob(f"{cfg.rollouts}/{task}_*/episodes/*.npz"))
        usable = []
        for f in files:
            with np.load(f) as z:
                if all(k in z.files for k in
                       ("decision_images_head_camera", "intro_h_query", "intro_noise")):
                    usable.append(f)
        assert len(usable) == len(rows), f"{task}: file/row mismatch"
        cache = {}

        def load_p(local_ep):
            if local_ep not in cache:
                with np.load(usable[local_ep]) as z:
                    rc = z["intro_router_counts"][:KMAX].astype(np.float32)
                p = rc.mean(axis=1)
                cache[local_ep] = p / (p.sum(-1, keepdims=True) + 1e-8)   # (n, 36, 32)
            return cache[local_ep]

        groups = {"A": [], "B": []}
        biases = {}
        for r in rows:
            if not r["fail"]:
                continue
            g_ep = r["ep"] + ep_base
            if g_ep not in blamed:      # no admitted counterfactual -> left alone
                continue
            pos = [i for i in range(len(f_idx)) if ep[f_idx[i]] == r["ep"] and dist_f[i] <= thr]
            if not pos:
                continue
            p_fail = load_p(r["ep"])[:r["n"]].mean(0)
            succ_profiles = []
            for i in pos:
                m_dec = m_f[i]
                m_ep = int(ep[m_dec])
                succ_profiles.append(load_p(m_ep)[m_dec - starts[m_ep]])
            delta = np.mean(succ_profiles, axis=0) - p_fail       # (36, 32)
            grp = "A" if blamed[g_ep] == j_route else "B"
            groups[grp].append(int(r["seed"]))
            biases[str(r["seed"])] = delta.astype(np.float32)

        if task in SLOW_TASKS:
            for grp in groups:
                if len(groups[grp]) > cfg.slow_cap:
                    groups[grp] = sorted(rng.choice(groups[grp], cfg.slow_cap,
                                                    replace=False).tolist())

        keep = set(groups["A"]) | set(groups["B"])
        biases = {k: v for k, v in biases.items() if int(k) in keep}
        scenes = sorted(keep)
        json.dump({"A": sorted(groups["A"]), "B": sorted(groups["B"]),
                   "scenes": scenes}, open(out / f"{task}_scenes.json", "w"))
        np.savez_compressed(out / f"{task}_bias.npz", **biases)

        mags = np.concatenate([np.abs(v).ravel() for v in biases.values()])
        plan[task] = (len(groups["A"]), len(groups["B"]))
        print(f"{task:<20} A(路由担责) {len(groups['A']):>3}  B(其他担责) {len(groups['B']):>3}"
              f"   |bias| 中位 {np.median(mags):.4f}  95分位 {np.percentile(mags, 95):.4f}")
        ep_base += len(rows)

    n_scenes = sum(a + b for a, b in plan.values())
    slow = sum(plan[t][0] + plan[t][1] for t in plan if t in SLOW_TASKS)
    est_h = (slow * 60 + (n_scenes - slow) * 16) * 2 / 3600   # two arms
    print(f"\n共 {n_scenes} 个场景 × 2 臂，估计 {est_h:.1f} 小时")
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
