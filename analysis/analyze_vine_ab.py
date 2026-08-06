"""Score the euler-vs-VINE sampler contrast.

Paired per-scene comparison on identical scenes with an identical pinned noise
stream. The question is not whether outcomes flip -- bf16 wobble alone flips
them at a known floor -- but whether the flips are *asymmetric*: a genuinely
better sampler rescues (C fail -> T success) more scenes than it breaks
(C success -> T fail). The euler-vs-euler replicate pairs on the same scenes
(rollouts_replicate rep0 vs rep0b, plus this session's C arm) calibrate how
much asymmetry same-sampler reruns produce.

Test: exact binomial (McNemar) on the discordant pairs, one-sided toward VINE.
"""

import glob
import json
from math import comb

ROOT = "/data/whn/robotwin_eval"
TASKS = ["click_bell", "click_alarmclock", "stamp_seal", "place_can_basket"]


def outcomes(pattern):
    out = {}
    for f in glob.glob(pattern):
        for line in open(f):
            m = json.loads(line)
            out[int(m["seed"])] = bool(m["success"])
    return out


def pair_stats(a, b):
    """Flips between outcome dicts a -> b on common scenes."""
    common = sorted(set(a) & set(b))
    up = sum((not a[s]) and b[s] for s in common)     # a-fail -> b-success
    down = sum(a[s] and (not b[s]) for s in common)   # a-success -> b-fail
    return len(common), up, down


def binom_one_sided(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return float("nan")
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


print(f"{'任务':<20}{'C(欧拉)':>10}{'T(VINE)':>10}{'救回':>6}{'打破':>6}")
print("-" * 56)
tot = dict(n=0, c=0, t=0, up=0, down=0)
null_pairs = []
for task in TASKS:
    C = outcomes(f"{ROOT}/rollouts_vine_ab/{task}_C/*/summary.jsonl")
    T = outcomes(f"{ROOT}/rollouts_vine_ab/{task}_T/*/summary.jsonl")
    n, up, down = pair_stats(C, T)
    if not n:
        continue
    common = sorted(set(C) & set(T))
    c_succ = sum(C[s] for s in common)
    t_succ = sum(T[s] for s in common)
    print(f"{task:<20}{c_succ:>7}/{n:<3}{t_succ:>7}/{n:<3}{up:>6}{down:>6}")
    tot["n"] += n; tot["c"] += c_succ; tot["t"] += t_succ
    tot["up"] += up; tot["down"] += down

    # euler-vs-euler nulls on the same scenes: historical rep0/rep0b + this C arm
    R0 = outcomes(f"{ROOT}/rollouts_replicate/{task}_rep0/*/summary.jsonl")
    R0b = outcomes(f"{ROOT}/rollouts_replicate/{task}_rep0b/*/summary.jsonl")
    for a, b, lab in ((R0, R0b, "rep0->rep0b"), (R0, C, "rep0->C"), (R0b, C, "rep0b->C")):
        nn, uu, dd = pair_stats(a, b)
        if nn:
            null_pairs.append((task, lab, nn, uu, dd))

print("\n=== 总计 ===")
print(f"欧拉 {tot['c']}/{tot['n']} ({tot['c']/tot['n']:.0%})  vs  "
      f"VINE {tot['t']}/{tot['n']} ({tot['t']/tot['n']:.0%})   "
      f"净变化 {tot['t']-tot['c']:+d}")
print(f"救回 (欧拉失败→VINE成功): {tot['up']}")
print(f"打破 (欧拉成功→VINE失败): {tot['down']}")
disc = tot["up"] + tot["down"]
p = binom_one_sided(tot["up"], disc)
print(f"翻盘不对称性检验: {disc} 个结局不同的场景里 {tot['up']} 个偏向 VINE, "
      f"若采样器其实无差别, 出现至少这么偏的概率(单侧) = {p:.3f}")

print("\n=== 底噪对照: 欧拉 vs 欧拉 重跑在同一批场景上的翻盘 ===")
print(f"{'任务':<20}{'对':>14}{'场景':>6}{'上翻':>6}{'下翻':>6}")
for task, lab, nn, uu, dd in null_pairs:
    print(f"{task:<20}{lab:>14}{nn:>6}{uu:>6}{dd:>6}")
nu = sum(x[3] for x in null_pairs); nd = sum(x[4] for x in null_pairs)
print(f"合计 上翻 {nu} / 下翻 {nd}  (同采样器重跑的不对称性基准; "
      f"VINE 的 {tot['up']}↑/{tot['down']}↓ 需要明显偏离这个基准才算真效应)")
