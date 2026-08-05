"""Are there close enough counterfactual matches to compare against?

The whole matched-pair design rests on a premise that has to be checked before
anything is built on it: for a decision taken in a failing attempt, does a
near-identical decision exist among the successful attempts of the same task?
If the nearest success is far away, "same or similar state" never holds and the
comparison has no basis.

Matching uses the true inputs -- the 14 joint values and the three camera frames.
Deliberately not the model's own image embedding: that is already the perception
output, and matching on it would silently condition perception out of suspicion.

Read against two baselines computed the same way:
  success -> success   how far apart two successful decisions typically are,
                       i.e. the normal spread of this task
  failure -> failure   whether failures cluster among themselves

If failure->success matches are no worse than success->success, matching is
usable. If they are systematically farther, the failing scenes are themselves
off-distribution and the divergence is upstream of anything in the forward pass.
"""

import glob
import json
from collections import defaultdict

import numpy as np

TASKS = ["click_bell", "click_alarmclock", "open_microwave",
         "hanging_mug", "place_can_basket", "stamp_seal"]
KMAX = 8          # decisions per episode to consider
IMG_DS = 8        # downsample factor for the image part of the distance


def load(task):
    """Per-decision (state, image) descriptors for one task."""
    S, I, Y, EP = [], [], [], []
    for f in sorted(glob.glob(f"/data/whn/robotwin_eval/rollouts/{task}_*/episodes/*.npz")):
        meta = json.load(open(f.replace(".npz", ".json")))
        with np.load(f) as z:
            if "decision_images_head_camera" not in z.files:
                continue
            n = min(KMAX, z["decision_states"].shape[0])
            S.append(z["decision_states"][:n].astype(np.float32))
            im = [z[f"decision_images_{c}"][:n, ::IMG_DS, ::IMG_DS, :].astype(np.float32) / 255.0
                  for c in ("head_camera", "left_camera", "right_camera")]
            I.append(np.concatenate([x.reshape(n, -1) for x in im], axis=1))
        Y.append(np.full(n, 0 if meta["success"] else 1))
        EP.append(np.full(n, len(EP)))
    if not S:
        return None
    return (np.concatenate(S), np.concatenate(I), np.concatenate(Y), np.concatenate(EP))


def nn_dist(query, bank, bank_ep, query_ep):
    """Nearest-neighbour distance from each query row to the bank, excluding
    rows from the query's own episode so a decision cannot match itself."""
    out = np.empty(len(query))
    for i in range(len(query)):
        d = np.linalg.norm(bank - query[i], axis=1)
        d[bank_ep == query_ep[i]] = np.inf
        out[i] = d.min()
    return out


print(f"{'任务':<20}{'配对':<14}{'最近邻距离中位数':>18}{'25分位':>10}{'75分位':>10}")
print("-" * 74)
summary = defaultdict(dict)
for task in TASKS:
    got = load(task)
    if got is None:
        continue
    S, I, Y, EP = got
    # State and image are on different scales; standardise each block over this
    # task so neither dominates, then concatenate.
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-6)
    Iz = (I - I.mean(0)) / (I.std(0) + 1e-6)
    X = np.concatenate([Sz / np.sqrt(Sz.shape[1]), Iz / np.sqrt(Iz.shape[1])], axis=1)

    f, s = Y == 1, Y == 0
    if f.sum() == 0 or s.sum() == 0:
        continue
    d_fs = nn_dist(X[f], X[s], EP[s], EP[f])
    d_ss = nn_dist(X[s], X[s], EP[s], EP[s])
    d_ff = nn_dist(X[f], X[f], EP[f], EP[f])
    for name, d in (("失败→成功", d_fs), ("成功→成功", d_ss), ("失败→失败", d_ff)):
        print(f"{task if name=='失败→成功' else '':<20}{name:<14}"
              f"{np.median(d):>18.3f}{np.percentile(d,25):>10.3f}{np.percentile(d,75):>10.3f}")
        summary[task][name] = np.median(d)
    print()

print("=== 判读 ===")
ratios = []
for t, v in summary.items():
    if "失败→成功" in v and "成功→成功" in v:
        r = v["失败→成功"] / v["成功→成功"]
        ratios.append(r)
        print(f"  {t:<20} 失败→成功 是 成功→成功 的 {r:.2f} 倍")
print(f"\n  平均 {np.mean(ratios):.2f} 倍")
print("  接近 1.0 = 失败样本能找到和成功样本之间同样近的匹配，反事实配对成立")
print("  明显大于 1 = 失败场景本身就偏离了成功场景的分布，前向计算之外就已经不同")
