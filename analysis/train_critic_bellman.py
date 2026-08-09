"""Bellman-trained twin critic on the decision-level semi-MDP (Phase 2, step 1).

Q(s, a) where s = joints + low-dim spatial-query projection, a = executed
25x14 chunk. Capacity allocation follows the Phase-1 gate: joints+action carry
the signal (0.81-0.90 OOS AUC); VLM-derived features get only a small,
heavily-dropped-out side branch the optimizer is free to ignore.

Targets are SARSA-style on logged transitions -- y = r + gamma * (1-term) *
minQ'(s', a'_logged) -- which evaluates the behavior policy. This is the stable
offline start; when actor fine-tuning begins, a' switches to GENERATE(s') via
the VINE sampler and this module's critics/targets carry over unchanged.

Terminal reward = episode success in {0,1}; Q is then a discounted success
probability, so targets stay in [0,1] and MSE is well-scaled.

Reported per epoch on held-out EPISODES (never seen in training):
  td      held-out TD error (fit quality)
  auc0    AUC of Q(s_0, a_0) against episode outcome (does the critic's value
          at the first decision rank episodes? compare with the gate's ~0.85/0.72)
  gap     mean Q(s_0,a_0) on eventual successes minus eventual failures
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = sums[inv] / cnt[inv]
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


class Critic(nn.Module):
    """Action-centric Q with a small state side-branch."""

    def __init__(self, d_action, d_joints, d_spatial, d_spatial_proj=64):
        super().__init__()
        self.a_tower = nn.Sequential(
            nn.Linear(d_action, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.s_proj = nn.Sequential(
            nn.Linear(d_spatial, d_spatial_proj), nn.GELU(), nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + d_joints + d_spatial_proj, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, a, j, sp):
        return self.head(torch.cat([self.a_tower(a), j, self.s_proj(sp)], -1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.q1 = Critic(*dims)
        self.q2 = Critic(*dims)

    def forward(self, a, j, sp):
        return self.q1(a, j, sp), self.q2(a, j, sp)


def load_task(buffer, task):
    z = np.load(f"{buffer}/{task}.npz")
    n = len(z["state"])
    d = {
        "a": z["chunk_exec"].reshape(n, -1).astype(np.float32),
        "j": z["state"].astype(np.float32),
        "sp": z["h_query_tokens"].reshape(n, -1).astype(np.float32),
        "y": z["success"].astype(np.float32),
        "ep": z["episode"],
        "dec": z["decision"],
        "term": z["terminal"],
    }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock"])
    ap.add_argument("--buffer", default="/data/whn/robotwin_eval/rl_buffer")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--mc", action="store_true",
                    help="Monte-Carlo targets (terminal outcome, no bootstrap): "
                         "the TD path collapsed on click_alarmclock once the "
                         "leaked episodes were removed, while the MC signal "
                         "survives; for top-N ranking MC IS the wanted quantity")
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="path prefix to save critic weights")
    cfg = ap.parse_args()

    for task in cfg.tasks:
        d = load_task(cfg.buffer, task)
        eps = np.unique(d["ep"])
        ep_y = np.array([d["y"][d["ep"] == e][0] for e in eps])
        rng = np.random.default_rng(cfg.seed)
        # stratified episode holdout
        te_eps = []
        for cls in (0, 1):
            c = eps[ep_y == cls]
            te_eps += list(rng.choice(c, max(1, int(len(c) * cfg.holdout)), replace=False))
        te_mask = np.isin(d["ep"], te_eps)
        tr_mask = ~te_mask

        # standardize on train
        stats = {}
        for k in ("a", "j", "sp"):
            mu, sd = d[k][tr_mask].mean(0), d[k][tr_mask].std(0) + 1e-6
            d[k] = (d[k] - mu) / sd
            stats[k] = (mu, sd)

        # next-decision index within the same episode (terminal rows unused)
        nxt = np.arange(len(d["y"])) + 1
        r = np.where(d["term"], d["y"], 0.0).astype(np.float32)

        t = {k: torch.tensor(v if v.dtype != np.bool_ else v.astype(np.float32),
                             dtype=torch.float32, device=DEV)
             for k, v in d.items() if k in ("a", "j", "sp", "y")}
        t["r"] = torch.tensor(r, device=DEV)
        t["term"] = torch.tensor(d["term"].astype(np.float32), device=DEV)

        torch.manual_seed(cfg.seed)
        dims = (d["a"].shape[1], d["j"].shape[1], d["sp"].shape[1])
        q = TwinCritic(*dims).to(DEV)
        qt = TwinCritic(*dims).to(DEV)
        qt.load_state_dict(q.state_dict())
        for p in qt.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(q.parameters(), lr=cfg.lr, weight_decay=1e-2)

        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        te0 = te_idx[d["dec"][te_idx] == 0]
        ep0_y = d["y"][te0]

        print(f"\n=== {task}: train {tr_mask.sum()} 决策 / test {te_mask.sum()} "
              f"(留出 {len(te_eps)} 集, 其中成功 {int(ep_y[np.isin(eps, te_eps)].sum())}) ===")
        print(f"{'epoch':>6} {'td(train)':>10} {'td(test)':>9} {'auc0':>7} {'gap':>7}")

        g = torch.Generator().manual_seed(cfg.seed)
        for epoch in range(1, cfg.epochs + 1):
            q.train()
            perm = tr_idx[torch.randperm(len(tr_idx), generator=g).numpy()]
            tds = []
            for i in range(0, len(perm), cfg.batch):
                b = perm[i : i + cfg.batch]
                bn = np.minimum(nxt[b], len(d["y"]) - 1)
                bt = torch.tensor(b, device=DEV)
                btn = torch.tensor(bn, device=DEV)
                with torch.no_grad():
                    if cfg.mc:
                        y = t["y"][bt]
                    else:
                        q1n, q2n = qt(t["a"][btn], t["j"][btn], t["sp"][btn])
                        y = t["r"][bt] + cfg.gamma * (1 - t["term"][bt]) * torch.minimum(q1n, q2n)
                q1, q2 = q(t["a"][bt], t["j"][bt], t["sp"][bt])
                loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tds.append(float(loss) / 2)
                with torch.no_grad():
                    for pt, p in zip(qt.parameters(), q.parameters()):
                        pt.mul_(1 - cfg.tau).add_(p, alpha=cfg.tau)

            if epoch % 25 == 0 or epoch == cfg.epochs:
                q.eval()
                with torch.no_grad():
                    bt = torch.tensor(te_idx, device=DEV)
                    btn = torch.tensor(np.minimum(nxt[te_idx], len(d["y"]) - 1), device=DEV)
                    if cfg.mc:
                        y = t["y"][bt]
                    else:
                        q1n, q2n = qt(t["a"][btn], t["j"][btn], t["sp"][btn])
                        y = t["r"][bt] + cfg.gamma * (1 - t["term"][bt]) * torch.minimum(q1n, q2n)
                    q1, q2 = q(t["a"][bt], t["j"][bt], t["sp"][bt])
                    td_te = float((((q1 - y) ** 2 + (q2 - y) ** 2) / 2).mean())
                    b0 = torch.tensor(te0, device=DEV)
                    q0 = ((q.q1(t["a"][b0], t["j"][b0], t["sp"][b0])
                           + q.q2(t["a"][b0], t["j"][b0], t["sp"][b0])) / 2).cpu().numpy()
                a0 = auc(ep0_y, q0)
                gap = q0[ep0_y == 1].mean() - q0[ep0_y == 0].mean()
                print(f"{epoch:>6} {np.mean(tds):>10.4f} {td_te:>9.4f} {a0:>7.3f} {gap:>7.3f}")

        if cfg.save:
            torch.save({"model": q.state_dict(), "stats": stats, "dims": dims,
                        "gamma": cfg.gamma, "task": task}, f"{cfg.save}_{task}.pt")
            print(f"saved -> {cfg.save}_{task}.pt")


if __name__ == "__main__":
    main()
