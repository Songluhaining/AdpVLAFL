"""Check the CPT localization against a label the model never saw.

Every other check scores the model against the same outcomes it was fitted to.
The replicate experiment supplies something external: for each scene, whether the
outcome actually swings when only the flow-matching draw changes, established by
rerunning rather than inferred.

`plan_revision` measures how far decision k+1 walks back decision k's plan, and
the plan comes out of a sampled denoising trajectory, so it is the component most
directly downstream of the draw. If the localization tracks something real,
failures it blames on plan_revision should be the ones a different draw rescues,
and failures blamed elsewhere should be the ones that hold.

Thresholds are recomputed from the labelled set itself rather than borrowed from
the training corpus, which is the situation on a fresh deployment.
"""

import glob
import json
import subprocess
import sys
from collections import defaultdict

import arviz as az
import numpy as np

COMPONENTS = ["perception", "language", "routing", "denoise", "execution", "plan_revision"]
NODES = "/data/whn/robotwin_eval/node_values_labelled.npz"

# ---- external label -------------------------------------------------------
reps = defaultdict(lambda: defaultdict(dict))
for d in sorted(glob.glob("/data/whn/robotwin_eval/rollouts_replicate/*_rep*/")):
    task, rep = d.rstrip("/").split("/")[-1].rsplit("_rep", 1)
    for f in glob.glob(d + "*/summary.jsonl"):
        for line in open(f):
            m = json.loads(line)
            reps[task][rep][m["seed"]] = bool(m["success"])

label = {}
for task, r in reps.items():
    real = sorted([k for k in r if k not in ("0", "0b")], key=int)
    for s in set.intersection(*[set(r[k]) for k in real]):
        w = sum(r[k][s] for k in real)
        label[(task, s)] = "stable" if w == len(real) else ("holds" if w == 0 else "noise")

# ---- nodes for the labelled scenes ----------------------------------------
subprocess.run([sys.executable, "extract_cpt_nodes.py",
                "--rollouts", "/data/whn/robotwin_eval/rollouts_labelled",
                "--out", NODES], check=True, capture_output=True)
d = np.load(NODES, allow_pickle=True)
ood, early, y = d["ood"], d["early"], d["y"]
tasks = [str(t) for t in d["tasks"]]
keys = [(tasks[t], int(s)) for t, s in zip(d["task_id"], d["seeds"])]

# ---- apply the fitted model ----------------------------------------------
post = az.from_netcdf("/data/whn/robotwin_eval/post_cpt.nc").posterior
b0 = post["b0"].values.reshape(-1, len(COMPONENTS))
b2 = post["b2"].values.reshape(-1, len(COMPONENTS))
mu_b1 = post["mu_b1"].values.reshape(-1, len(COMPONENTS))
leak = 1 / (1 + np.exp(-post["mu_leak"].values.reshape(-1)))

pz = 1 / (1 + np.exp(-(b0[:, None, :] + mu_b1[:, None, :] * ood[None]
                       + b2[:, None, :] * early[None])))
pf = 1 - (1 - leak[:, None]) * np.prod(1 - pz, axis=2)
resp = np.clip(pz / np.clip(pf, 1e-9, None)[:, :, None], 0, 1).mean(axis=0)

# ---- do they line up? -----------------------------------------------------
idx = [i for i in range(len(y)) if y[i] == 1 and label.get(keys[i]) in ("noise", "holds")]
lab = np.array([label[keys[i]] == "noise" for i in idx])
R = resp[idx]
base = lab.mean()
print(f"失败样本 {len(idx)} 个，其中换噪声能救回来的 {int(lab.sum())} 个（基准 {base:.0%}）\n")

print("按「嫌疑最大的环节」分组")
print(f"{'环节':<15}{'能救':>8}{'救不了':>8}{'合计':>8}{'能救占比':>12}")
print("-" * 52)
top = R.argmax(axis=1)
for j, c in enumerate(COMPONENTS):
    m = top == j
    if m.sum():
        print(f"{c:<15}{int(lab[m].sum()):>8}{int((~lab[m]).sum()):>8}"
              f"{int(m.sum()):>8}{lab[m].mean():>12.0%}")

print("\n按嫌疑高低直接比较（不做硬分组，样本更省）")
print(f"{'环节':<15}{'能救的均值':>12}{'救不了的均值':>14}{'差':>9}")
print("-" * 52)
for j, c in enumerate(COMPONENTS):
    a_, b_ = R[lab, j].mean(), R[~lab, j].mean()
    flag = "  <-- 方向符合预期" if (c == "plan_revision" and a_ > b_) else ""
    print(f"{c:<15}{a_:>12.3f}{b_:>14.3f}{a_ - b_:>+9.3f}{flag}")

print("\n预期：plan_revision 的嫌疑在「能救」组更高（它最贴近采样）；"
      "routing/denoise 在「救不了」组更高。")
print(f"注意：{len(idx)} 个样本很少，且数值误差本身会让 12~20% 的场景结果翻转，"
      "标签自带噪声，这个检验只能看方向。")
