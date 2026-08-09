"""Contrastive within-state ranker on the multi-draw corpus.

Training pairs are real same-scene contrasts: two draws of the SAME scene at
decision 0 (identical scene init; instruction phrasing may vary between draws,
which adds label-free noise but not systematic bias), one draw's episode
succeeded and the other's failed. RankNet objective: P(succ ranked first) =
sigmoid(s_succ - s_fail).

Feature ablations answer WHERE the predictable signal lives, if anywhere:
  action   executed chunk prefix (25x14) + joints + spatial query tokens
  noise    the raw draw eps (50x55) + joints + spatial query tokens
  both     union

Scene-grouped split; the reported number is held-out same-scene pair ranking
accuracy -- the exact capability the topn referee lacked (48% = chance).
Around 50% here despite proper contrastive training closes the q(eps|s) line:
the outcome difference is not predictable from these observables.
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_draws(root, task):
    """{scene: [per-draw dicts at decision 0]}"""
    scenes = defaultdict(list)
    for d in sorted(glob.glob(f"{root}/{task}_rep*")):
        for f in sorted(glob.glob(f"{d}/*/episodes/*.npz")):
            meta = json.load(open(f.replace(".npz", ".json")))
            with np.load(f) as z:
                if "intro_h_query_tokens" not in z.files:
                    continue
                scenes[int(meta["seed"])].append(dict(
                    a=z["predicted_chunks"][0][:25].reshape(-1).astype(np.float32),
                    eps=z["intro_noise"][0].reshape(-1).astype(np.float32),
                    j=z["decision_states"][0].astype(np.float32),
                    sp=z["intro_h_query_tokens"][0].reshape(-1).astype(np.float32),
                    y=bool(meta["success"]),
                ))
    return dict(scenes)


class Scorer(nn.Module):
    def __init__(self, d_main, d_j, d_sp, d_sp_proj=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(d_main, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.sp = nn.Sequential(nn.Linear(d_sp, d_sp_proj), nn.GELU(), nn.Dropout(0.3))
        self.head = nn.Sequential(
            nn.Linear(128 + d_j + d_sp_proj, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, m, j, sp):
        return self.head(torch.cat([self.main(m), j, self.sp(sp)], -1)).squeeze(-1)


def pairs_of(scenes, keys):
    out = []
    for s in keys:
        draws = scenes[s]
        succ = [d for d in draws if d["y"]]
        fail = [d for d in draws if not d["y"]]
        out += [(a, b) for a in succ for b in fail]
    return out


def run(task, scenes, feat, seed=0, epochs=300):
    keys = sorted(scenes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(keys)
    n_hold = max(5, len(keys) // 3)
    hold, train = set(order[:n_hold].tolist()), order[n_hold:].tolist()
    tr_pairs = pairs_of(scenes, train)
    te_pairs = pairs_of(scenes, hold)
    if len(tr_pairs) < 10 or len(te_pairs) < 5:
        print(f"  [{feat}] 配对不足: train {len(tr_pairs)} / test {len(te_pairs)}")
        return

    def fvec(d):
        m = {"action": d["a"], "noise": d["eps"],
             "both": np.concatenate([d["a"], d["eps"]])}[feat]
        return m, d["j"], d["sp"]

    all_tr = [fvec(d) for p in tr_pairs for d in p]
    mus = [np.stack([x[i] for x in all_tr]).mean(0) for i in range(3)]
    sds = [np.stack([x[i] for x in all_tr]).std(0) + 1e-6 for i in range(3)]

    def T(p):
        return [torch.tensor(np.stack([(fvec(d)[i] - mus[i]) / sds[i] for d, _ in
                [(p[0], 0), (p[1], 0)]]), dtype=torch.float32, device=DEV)
                for i in range(3)]

    torch.manual_seed(seed)
    net = Scorer(len(fvec(tr_pairs[0][0])[0]), 14, 20480).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    g = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = g.permutation(len(tr_pairs))
        for i in idx:
            m, j, sp = T(tr_pairs[i])
            s = net(m, j, sp)
            loss = nn.functional.softplus(-(s[0] - s[1]))   # -log sigmoid(margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        wins = 0
        for p in te_pairs:
            m, j, sp = T(p)
            s = net(m, j, sp)
            wins += bool(s[0] > s[1])
    n = len(te_pairs)
    print(f"  [{feat:<6}] train配对 {len(tr_pairs)}, 留出配对 {n}, "
          f"同场景排序正确 {wins}/{n} = {wins / n:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/whn/robotwin_eval/rollouts_multidraw")
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock"])
    ap.add_argument("--feats", nargs="*", default=["action", "noise", "both"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    cfg = ap.parse_args()
    for task in cfg.tasks:
        scenes = load_draws(cfg.root, task)
        n_draws = sum(len(v) for v in scenes.values())
        mixed = sum(1 for v in scenes.values()
                    if any(d["y"] for d in v) and any(not d["y"] for d in v))
        print(f"\n=== {task}: {len(scenes)} 场景 {n_draws} 抽签, 成败混合场景 {mixed} ===")
        for feat in cfg.feats:
            for seed in cfg.seeds:
                run(task, scenes, feat, seed=seed)


if __name__ == "__main__":
    main()
