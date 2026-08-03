"""Per-scene analysis of the noise-replicate experiment.

Aggregate success rates are uninformative here -- every replicate lands in the
same band because the same scenes are being run. The question is about the
*individual* scene: run it eight times with a different flow-matching draw each
time, does the outcome hold or does it swing?

  outcomes spread in between      -> the draw decides, and draw selection has
                                     genuine headroom
  outcomes hold at 0/8            -> the draw does NOT decide. That rules out
                                     one remedy, not all of them: instruction
                                     wording, routing, denoising steps and the
                                     observation itself were all held fixed
                                     here and remain untested.

Replicate "0b" repeats replicate 0 with an identical noise stream, so any
disagreement between them is pure bf16 rounding. That is the floor a real
noise effect has to clear.
"""

import glob
import json
from collections import defaultdict

import numpy as np

ROOT = "/data/whn/robotwin_eval/rollouts_replicate"

# task -> replicate -> {scene seed: success}
data = defaultdict(lambda: defaultdict(dict))
for d in sorted(glob.glob(f"{ROOT}/*_rep*/")):
    name = d.rstrip("/").split("/")[-1]
    task, rep = name.rsplit("_rep", 1)
    for f in glob.glob(d + "*/summary.jsonl"):
        for line in open(f):
            m = json.loads(line)
            data[task][rep][m["seed"]] = bool(m["success"])

for task in sorted(data):
    reps = data[task]
    real = sorted([r for r in reps if r not in ("0", "0b")], key=int)
    seeds = sorted(set(reps["0"]) & set.intersection(*[set(reps[r]) for r in real]))
    if not seeds:
        continue

    # numerical floor: same noise, run twice
    both = [s for s in seeds if s in reps.get("0b", {})]
    flip_noise = sum(reps["0"][s] != reps["0b"][s] for s in both)

    M = np.array([[reps[r][s] for r in real] for s in seeds])  # (scenes, replicates)
    n_rep = M.shape[1]
    wins = M.sum(axis=1)

    print(f"=== {task} ===  {len(seeds)} 个场景 × {n_rep} 次不同噪声")
    print(f"  同噪声重跑的翻转（数值误差造成的底噪）: {flip_noise} / {len(both)} 个场景")
    print(f"  {n_rep} 次全成功: {int((wins == n_rep).sum()):>3} 个场景")
    print(f"  {n_rep} 次全失败: {int((wins == 0).sum()):>3} 个场景   <- 换噪声无效，原因在别处（尚未确定）")
    print(f"  有成有败    : {int(((wins > 0) & (wins < n_rep)).sum()):>3} 个场景   <- 成败取决于抽到什么噪声")

    base = M[:, 0]  # treat the first real replicate as the baseline run
    lost = ~base
    if lost.any():
        rescued = (M[lost, 1:].any(axis=1)).sum()
        print(f"  基线失败的 {int(lost.sum())} 个场景里，另外 {n_rep-1} 次噪声中至少成功过一次的: "
              f"{int(rescued)} 个 ({rescued/lost.sum():.0%})")
    won = base
    if won.any():
        broken = (~M[won, 1:]).any(axis=1).sum()
        print(f"  基线成功的 {int(won.sum())} 个场景里，换噪声后曾失败过的: "
              f"{int(broken)} 个 ({broken/won.sum():.0%})")
    print(f"  成功次数分布 0..{n_rep}: {np.bincount(wins, minlength=n_rep+1).tolist()}")
    print()

print("解读：「有成有败」= 开局噪声决定成败，挑噪声这条路有空间。")
print("      「全失败」= 换噪声这一种手段无效，失败原因在别处 —— 这正是需要贝叶斯定位")
print("      去查明的部分，不等于无解（指令措辞、路由、去噪步数等均未试过）。")
