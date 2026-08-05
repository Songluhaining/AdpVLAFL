"""Score the routing-correction contrast.

Per scene: outcome without correction (arm C), outcome with it (arm T), and
which group the diagnosis put it in -- A if routing carried the largest
responsibility, B otherwise. The diagnosis is supported only if A's rescue rate
beats B's; both groups got the identical treatment, so the contrast nets out
the bf16 flip floor and any generic effect of perturbing the router.

Also reported: scenes broken by the correction (C success -> T failure), which
an aligned-but-overshooting bias would produce, and the C-arm failure rate per
group against the corpus outcome those scenes were selected by -- the noise
stream differs from the original collection, so some selected failures succeed
in C and drop out of the rescue denominator.
"""

import glob
import json
from collections import defaultdict

import numpy as np

ROOT = "/data/whn/robotwin_eval"
TASKS = ["click_bell", "click_alarmclock", "stamp_seal",
         "place_can_basket", "open_microwave", "hanging_mug"]


def outcomes(task, arm):
    out = {}
    for f in glob.glob(f"{ROOT}/rollouts_correction/{task}_{arm}/*/summary.jsonl"):
        for line in open(f):
            m = json.loads(line)
            out[int(m["seed"])] = bool(m["success"])
    return out


tot = defaultdict(lambda: defaultdict(int))
rows = []
for task in TASKS:
    scenes = json.load(open(f"{ROOT}/correction/{task}_scenes.json"))
    grp = {s: "A" for s in scenes["A"]}
    grp.update({s: "B" for s in scenes["B"]})
    C, T = outcomes(task, "C"), outcomes(task, "T")
    common = sorted(set(C) & set(T) & set(grp))
    if not common:
        continue
    for s in common:
        g = grp[s]
        tot[g]["n"] += 1
        tot[g]["c_fail"] += (not C[s])
        tot[g]["rescued"] += (not C[s]) and T[s]
        tot[g]["broken"] += C[s] and (not T[s])
        tot[g]["c_succ"] += C[s]
    a = [s for s in common if grp[s] == "A"]
    b = [s for s in common if grp[s] == "B"]
    rows.append((task, len(a), sum(not C[s] for s in a),
                 sum((not C[s]) and T[s] for s in a),
                 len(b), sum(not C[s] for s in b),
                 sum((not C[s]) and T[s] for s in b)))

print(f"{'任务':<20}{'A场景':>6}{'A中C失败':>9}{'A被救回':>8}"
      f"{'B场景':>7}{'B中C失败':>9}{'B被救回':>8}")
print("-" * 70)
for r in rows:
    print(f"{r[0]:<20}{r[1]:>6}{r[2]:>9}{r[3]:>8}{r[4]:>7}{r[5]:>9}{r[6]:>8}")

print("\n=== 总对比 ===")
for g, lab in (("A", "A: 诊断归因于路由"), ("B", "B: 诊断归因于其他")):
    d = tot[g]
    rr = d["rescued"] / d["c_fail"] if d["c_fail"] else float("nan")
    br = d["broken"] / d["c_succ"] if d["c_succ"] else float("nan")
    print(f"{lab}: 场景 {d['n']}, C臂失败 {d['c_fail']}, "
          f"救回 {d['rescued']} ({rr:.0%}), 打破 {d['broken']}/{d['c_succ']} ({br:.0%})")

ca, cb = tot["A"], tot["B"]
if ca["c_fail"] and cb["c_fail"]:
    ra = ca["rescued"] / ca["c_fail"]; rb = cb["rescued"] / cb["c_fail"]
    print(f"\n救回率对比: A {ra:.0%} vs B {rb:.0%}  (差 {ra - rb:+.0%})")
    # Fisher exact via hypergeometric tail, no scipy dependence on p-hacking knobs
    from math import comb
    N = ca["c_fail"] + cb["c_fail"]; K = ca["rescued"] + cb["rescued"]
    n = ca["c_fail"]
    p = sum(comb(n, k) * comb(N - n, K - k) for k in range(ca["rescued"], min(n, K) + 1)) / comb(N, K)
    print(f"若两组救回率其实相同，看到 A 至少这么高的概率(单侧): {p:.3f}")
    print("  (小于 0.05 = A 组确实更容易被路由矫正救回 → 支持诊断)")
