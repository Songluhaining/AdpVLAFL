"""Reduce recorded model internals to one value-node scalar per component per decision.

Three choices here are load-bearing for whether the causal claim survives:

1. Matched leading window. Failures run to the step limit and so have ~3x more
   decisions than successes, and their late decisions are spent stuck and
   retrying. Averaging over a whole episode would therefore measure failure
   causing weird internals, not the reverse. Only the first W decisions are used.

2. Label-free standardization. Deviations are turned into percentile ranks
   *within a task, pooling successes and failures alike*. Nothing about the
   outcome enters the feature, so the later noisy-OR fit is not circular. It
   also removes task difficulty, which is the main confounder.

3. Model-internal quantities only. No simulator privileged state is touched, so
   the same extraction runs unchanged on a real robot.
"""

import argparse
import glob
import json
import os

import numpy as np

W_DEFAULT = 4
COMPONENTS = ["perception", "language", "routing", "denoise", "execution"]


def per_decision_raw(z, w):
    """Raw (unstandardized) per-decision quantities for one episode."""
    n = min(w, z["predicted_chunks"].shape[0])
    out = {}

    out["h_query"] = z["intro_h_query"][:n].astype(np.float32)
    out["h_lang"] = z["intro_h_lang"][:n].astype(np.float32)

    # Routing: the entropy of the router distribution, not the expert-load
    # histogram. Load distance to the task centroid was the first choice and
    # screened at AUC 0.50 -- no signal at all -- while entropy screens at 0.64.
    # The minimum over layers is the strongest single reduction, i.e. failures
    # are marked by even the most decisive layer becoming hesitant.
    re_ = z["intro_router_entropy"][:n].astype(np.float32)      # (n, steps, layers)
    out["route_ent"] = re_.min(axis=(1, 2))

    # Denoising: the magnitude of the final action. Trajectory straightness and
    # the velocity norm both screened flat (0.44-0.48); this was the only
    # denoising reduction above chance, and only barely (0.59).
    dx = z["intro_denoise_x"][:n].astype(np.float32)
    out["dn_final"] = np.abs(dx[:, -1]).max(axis=(1, 2))

    # Execution: how far decision k+1 walks back what decision k had planned for
    # the same future timesteps. This is the direct signature of committing to
    # exec_horizon steps of blind execution.
    chunks = z["predicted_chunks"].astype(np.float32)
    steps = z["decision_steps"]
    h = int(np.median(np.diff(steps))) if len(steps) > 2 else 25
    overlap = chunks.shape[1] - h
    seam = np.full(n, np.nan, dtype=np.float32)
    if overlap > 0:
        for k in range(n):
            if k + 1 < chunks.shape[0]:
                seam[k] = np.abs(chunks[k, h : h + overlap] - chunks[k + 1, :overlap]).mean()
    out["seam"] = seam
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/data/whn/robotwin_eval/rollouts")
    ap.add_argument("--window", type=int, default=W_DEFAULT)
    ap.add_argument("--out", default="/data/whn/robotwin_eval/node_values.npz")
    cfg = ap.parse_args()

    episodes = []
    for run in sorted(glob.glob(f"{cfg.rollouts}/*/")):
        for f in sorted(glob.glob(run + "episodes/*.npz")):
            meta = json.load(open(f.replace(".npz", ".json")))
            if meta["num_decisions"] < cfg.window:
                continue
            with np.load(f) as z:
                if "intro_h_query" not in z.files:
                    continue
                raw = per_decision_raw(z, cfg.window)
            episodes.append(dict(meta=meta, raw=raw))

    if not episodes:
        raise SystemExit(f"no instrumented episodes with >= {cfg.window} decisions under {cfg.rollouts}")

    tasks = sorted({e["meta"]["task_name"] for e in episodes})
    print(f"{len(episodes)} episodes over {len(tasks)} tasks "
          f"({sum(1 for e in episodes if not e['meta']['success'])} failures), window={cfg.window}")

    # ---- per-task deviation, then per-task percentile rank -----------------
    dev = {c: np.full(len(episodes), np.nan) for c in COMPONENTS}
    for t in tasks:
        idx = [i for i, e in enumerate(episodes) if e["meta"]["task_name"] == t]

        def stack(key):
            return np.concatenate([episodes[i]["raw"][key] for i in idx], axis=0)

        # Centroid of a task's own decisions; distance to it is "how atypical
        # this decision was for this task", with task identity divided out.
        for comp, key in (("perception", "h_query"), ("language", "h_lang")):
            allv = stack(key)
            mu = allv.mean(axis=0)
            for i in idx:
                d = np.linalg.norm(episodes[i]["raw"][key] - mu, axis=1)
                dev[comp][i] = float(np.nanmean(d))

        # Signed, not absolute: screening established a consistent direction for
        # both (failures show higher values), and folding that into |x - median|
        # would throw the direction away.
        for comp, key in (("routing", "route_ent"), ("denoise", "dn_final")):
            for i in idx:
                dev[comp][i] = float(np.nanmean(episodes[i]["raw"][key]))

        for comp, key in (("execution", "seam"),):
            med = np.nanmedian(stack(key))
            for i in idx:
                dev[comp][i] = float(np.nanmean(np.abs(episodes[i]["raw"][key] - med)))

    a = {}
    for c in COMPONENTS:
        a[c] = np.zeros(len(episodes))
        for t in tasks:
            idx = np.array([i for i, e in enumerate(episodes) if e["meta"]["task_name"] == t])
            v = dev[c][idx]
            ok = ~np.isnan(v)
            r = np.zeros(len(v))
            if ok.sum() > 1:
                order = np.argsort(np.argsort(v[ok]))
                r[ok] = order / max(ok.sum() - 1, 1)
            r[~ok] = 0.5  # seam is undefined for a trailing decision; stay neutral
            a[c][idx] = r

    y = np.array([0 if e["meta"]["success"] else 1 for e in episodes], dtype=np.int32)
    task_id = np.array([tasks.index(e["meta"]["task_name"]) for e in episodes], dtype=np.int32)

    np.savez_compressed(
        cfg.out,
        a=np.stack([a[c] for c in COMPONENTS], axis=1),  # (episodes, components)
        y=y, task_id=task_id,
        components=np.array(COMPONENTS), tasks=np.array(tasks),
        seeds=np.array([e["meta"]["seed"] for e in episodes]),
    )
    print(f"wrote {cfg.out}")

    print(f"\n{'component':<12}{'mean a | success':>18}{'mean a | fail':>16}{'diff':>9}")
    print("-" * 55)
    for j, c in enumerate(COMPONENTS):
        s, f = a[c][y == 0].mean(), a[c][y == 1].mean()
        print(f"{c:<12}{s:>18.4f}{f:>16.4f}{f - s:>+9.4f}")
    print("\n(a is a within-task percentile rank, so 0.5 is the task average; "
          "a difference near 0 means the component carries no signal)")


if __name__ == "__main__":
    main()
