"""Attribution-conditioned scoring of the top-N selection A/B.

The pass criterion is NOT the aggregate success delta. Scenes carry
pre-registered labels from the multi-draw replicate record:

  noise_resc    rep0 failed, some other draw succeeded  -> selection SHOULD rescue
  scene_dec     every recorded draw failed              -> selection should do ~nothing
  stable_succ   rep0 succeeded                          -> selection should not break
  fail_unknown  rep0 failed, not enough draws to label

Top-N is judged by rescues CONCENTRATING in noise_resc (exact binomial against
the scene_dec rescue rate as the floor), plus the break rate on stable_succ.
Labels are rebuilt here from ALL replicate dirs (including top-up reps run by
run_topn_ab.sh) so late draws refine the classes before scoring.
"""

import glob
import json
from math import comb

ROOT = "/data/whn/robotwin_eval"
TASKS = ["click_bell", "click_alarmclock"]


def outcomes(pattern):
    o = {}
    for f in glob.glob(pattern):
        for line in open(f):
            m = json.loads(line)
            o[int(m["seed"])] = bool(m["success"])
    return o


def build_labels(task):
    by_rep = {}
    for d in sorted(glob.glob(f"{ROOT}/rollouts_replicate/{task}_rep*")):
        by_rep[d.split("_rep")[-1]] = outcomes(f"{d}/*/summary.jsonl")
    lab = {}
    for s in sorted(by_rep.get("0", {})):
        outs = {r: by_rep[r][s] for r in by_rep if s in by_rep[r]}
        if outs["0"]:
            lab[s] = "stable_succ"
        elif any(v for r, v in outs.items() if r != "0"):
            lab[s] = "noise_resc"
        elif len(outs) >= 4:
            lab[s] = "scene_dec"
        else:
            lab[s] = "fail_unknown"
    return lab


def binom_tail(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


for task in TASKS:
    C = outcomes(f"{ROOT}/rollouts_topn/{task}_C/*/summary.jsonl")
    T = outcomes(f"{ROOT}/rollouts_topn/{task}_T/*/summary.jsonl")
    lab = build_labels(task)
    common = sorted(set(C) & set(T) & set(lab))
    if not common:
        continue
    print(f"\n=== {task}: {len(common)} 配对场景 | "
          f"C {sum(C[s] for s in common)} vs T {sum(T[s] for s in common)} ===")
    print(f"{'归因类':<14}{'n':>4}{'C失败':>6}{'救回':>5}{'打破':>5}")
    stats = {}
    for cls in ("noise_resc", "scene_dec", "fail_unknown", "stable_succ"):
        ss = [s for s in common if lab[s] == cls]
        cf = sum(not C[s] for s in ss)
        up = sum(not C[s] and T[s] for s in ss)
        dn = sum(C[s] and not T[s] for s in ss)
        stats[cls] = (len(ss), cf, up, dn)
        print(f"{cls:<14}{len(ss):>4}{cf:>6}{up:>5}{dn:>5}")

    n_nr, cf_nr, up_nr, _ = stats["noise_resc"]
    n_sd, cf_sd, up_sd, _ = stats["scene_dec"]
    if cf_nr:
        # floor: the scene_dec rescue rate (bf16 wobble on doomed scenes)
        floor = up_sd / cf_sd if cf_sd else 0.0
        p = binom_tail(up_nr, cf_nr, max(floor, 0.15))
        print(f"判定: 噪声可救类救回 {up_nr}/{cf_nr}, 场景决定类救回 {up_sd}/{cf_sd}"
              f" | 若救回率不超过底噪({max(floor, 0.15):.0%}), 出现这么多救回的概率 = {p:.3f}")
        print("  (显著小于 0.05 且场景决定类≈0 => 收益确来自换噪声机制)")
