"""The decisive test the top-N referee never took: within-state discrimination.

The replicate record contains real same-scene pairs -- the rep0 draw that
failed and an alternate draw that succeeded -- with the state features
recoverable from the labelled run of the same scene. If the referee cannot
rank the succeeding draw above the failing one HERE, candidate selection at
inference cannot work no matter how good its across-state AUC is: that AUC
measures scene-difficulty prediction, not action quality given the state.

Result on 2026-08-09 (clean MC referees): click_bell 10/19, click_alarmclock
6/14 -- 48% pooled, chance level. Explanation: the training buffer holds ONE
draw per state, so it contains zero within-state outcome contrast; the
capability was never learnable from that data structure.
"""

import glob
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/whn/AdpVLAFL/analysis")
from train_critic_bellman import TwinCritic

ROOT = "/data/whn/robotwin_eval"


def outcomes_by_rep(task):
    by_rep = {}
    for d in sorted(glob.glob(f"{ROOT}/rollouts_replicate/{task}_rep*")):
        o = {}
        for f in glob.glob(f"{d}/*/summary.jsonl"):
            for line in open(f):
                m = json.loads(line)
                o[int(m["seed"])] = bool(m["success"])
        by_rep[d.split("_rep")[-1]] = o
    return by_rep


def main():
    for task in ("click_bell", "click_alarmclock"):
        ck = torch.load(f"{ROOT}/rl_buffer/clean/criticmc_{task}.pt",
                        map_location="cpu", weights_only=False)
        critic = TwinCritic(*ck["dims"]).float().eval()
        critic.load_state_dict(ck["model"])
        st = {k: (torch.tensor(m), torch.tensor(s)) for k, (m, s) in ck["stats"].items()}

        def score(a25, j, sp):
            a = torch.tensor(a25.reshape(1, -1), dtype=torch.float32)
            a = (a - st["a"][0]) / st["a"][1]
            jt = (torch.tensor(j, dtype=torch.float32)[None] - st["j"][0]) / st["j"][1]
            spt = (torch.tensor(sp.reshape(1, -1), dtype=torch.float32)
                   - st["sp"][0]) / st["sp"][1]
            with torch.no_grad():
                q1, q2 = critic(a, jt, spt)
            return float((q1 + q2) / 2)

        by_rep = outcomes_by_rep(task)
        wins = tot = 0
        for seed, ok0 in sorted(by_rep.get("0", {}).items()):
            if ok0:
                continue
            lab_f = next(iter(glob.glob(
                f"{ROOT}/rollouts_labelled/{task}_*/episodes/*seed{seed}_*.npz")), None)
            if lab_f is None:
                continue
            zl = np.load(lab_f)
            if "intro_h_query_tokens" not in zl.files:
                continue
            sp = zl["intro_h_query_tokens"][0].astype(np.float32)
            j = zl["decision_states"][0].astype(np.float32)
            s_fail = score(zl["predicted_chunks"][0][:25], j, sp)
            for rep, o in by_rep.items():
                if rep == "0" or not o.get(seed):
                    continue
                rf = next(iter(glob.glob(
                    f"{ROOT}/rollouts_replicate/{task}_rep{rep}/*/episodes/*seed{seed}_*.npz")), None)
                if rf is None:
                    continue
                zr = np.load(rf)
                if "predicted_chunks" not in zr.files:
                    continue
                tot += 1
                wins += score(zr["predicted_chunks"][0][:25], j, sp) > s_fail
        pct = f" = {wins / tot:.0%}" if tot else ""
        print(f"{task}: 同场景 成功抽签 vs 失败抽签 {tot} 对, 裁判排对 {wins}/{tot}{pct}")


if __name__ == "__main__":
    main()
