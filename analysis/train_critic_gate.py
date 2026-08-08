"""Phase-1 stop-loss gate for the VINE-RL pilot: is a chunk-action critic viable?

Three models per task, identical MLP capacity, 5-fold episode-grouped CV:
  V(s)    frozen-VLM prefix readouts + joint state          -> outcome
  Q(s,a)  the same, plus the executed action prefix (25x14) -> outcome
  A(a)    action prefix alone                               -> outcome (sanity)

The gate is NOT "is AUC high" but the decomposition:
  - Q(s,a) must clearly rank outcomes (else no critic, stop), AND
  - Q must beat V by a real margin: the value gradient used by VINE-RL exists
    only where Q's action-dependence is genuine. Q ~= V means the outcome is
    already decided by the state and grad_a Q is noise.

A learning curve (train on 25/50/75/100% of training episodes) makes the
data-quantity question an output instead of a guess: still climbing at 100%
-> collect more before concluding anything; flat -> the answer is intrinsic.

Targets are Monte-Carlo terminal outcomes (episodes are 4-12 decisions, no
bootstrapping needed for a ranking gate). Folds and curves are grouped by
episode; scenes are unique per episode in this buffer, so episode grouping
also separates scenes between train and test.
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
    # average ranks for ties so a constant predictor scores 0.5, not an artifact
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = sums[inv] / cnt[inv]
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_eval(Xtr, ytr, Xte, yte, seed, epochs=200):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    xe = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    # small data: full-batch with minibatch shuffling would overfit epochs apart;
    # 20% val-episode early stopping is replaced by fixed epochs + weight decay,
    # identical across V/Q/A so the comparison stays apples-to-apples
    n = len(xt)
    for ep in range(epochs):
        idx = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            b = idx[i : i + 256].to(DEV)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(xe).cpu().numpy()


def standardize(train, *rest):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(x - mu) / sd for x in (train, *rest)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock"])
    ap.add_argument("--buffer", default="/data/whn/robotwin_eval/rl_buffer")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--state-pca", type=int, default=256)
    ap.add_argument("--state-feats", default="pooled", choices=["pooled", "qtokens"],
                    help="pooled: h_image/h_lang/h_query means; qtokens: the 8 "
                         "unpooled spatial query tokens (Phase-1 found pooled "
                         "means poison state-action fusion)")
    cfg = ap.parse_args()

    for task in cfg.tasks:
        z = np.load(f"{cfg.buffer}/{task}.npz")
        if cfg.state_feats == "qtokens":
            S = z["h_query_tokens"].reshape(len(z["state"]), -1).astype(np.float32)
        else:
            S = np.concatenate([z["h_image"], z["h_lang"], z["h_query"]], 1).astype(np.float32)
        S = np.concatenate([S, z["state"]], 1)
        A = z["chunk_exec"].reshape(len(S), -1).astype(np.float32)
        y = z["success"].astype(np.float32)
        ep = z["episode"]
        eps = np.unique(ep)
        ep_y = np.array([y[ep == e][0] for e in eps])
        rng = np.random.default_rng(0)

        print(f"\n=== {task}: {len(eps)} 集 ({int(ep_y.sum())} 成功) "
              f"{len(y)} 决策, 特征 V:{S.shape[1]} A:{A.shape[1]} ===")
        print(f"{'训练集比例':>8} {'V(s) 状态':>12} {'Q(s,a) 状态+动作':>16} {'A(a) 仅动作':>12}")

        # stratified episode folds
        order = np.argsort(rng.random(len(eps)))
        eps_s, ep_y_s = eps[order], ep_y[order]
        folds = [[] for _ in range(cfg.folds)]
        for cls in (0, 1):
            for i, e in enumerate(eps_s[ep_y_s == cls]):
                folds[i % cfg.folds].append(e)

        for frac in cfg.fracs:
            scores = {k: [] for k in "VQA"}
            for k in range(cfg.folds):
                te_eps = set(folds[k])
                tr_eps = [e for e in eps if e not in te_eps]
                rng2 = np.random.default_rng(k)
                tr_eps = rng2.choice(tr_eps, max(4, int(len(tr_eps) * frac)),
                                     replace=False)
                tr = np.isin(ep, tr_eps)
                te = np.isin(ep, list(te_eps))
                Str, Ste = standardize(S[tr], S[te])
                Atr, Ate = standardize(A[tr], A[te])
                yy, yt_ = y[tr], y[te]
                for name, xtr, xte in (
                    ("V", Str, Ste),
                    ("Q", np.concatenate([Str, Atr], 1), np.concatenate([Ste, Ate], 1)),
                    ("A", Atr, Ate),
                ):
                    s = fit_eval(xtr, yy, xte, yt_, seed=1000 * k)
                    scores[name].append(auc(yt_, s))
            m = {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
            print(f"{frac:>8.0%} {m['V'][0]:>9.3f}±{m['V'][1]:.3f}"
                  f" {m['Q'][0]:>12.3f}±{m['Q'][1]:.3f}"
                  f" {m['A'][0]:>9.3f}±{m['A'][1]:.3f}")
        dq = np.array(scores["Q"]) - np.array(scores["V"])
        print(f"全量数据下 Q−V 逐折差: 均值 {dq.mean():+.3f}, "
              f"{int((dq > 0).sum())}/{cfg.folds} 折为正")

        # -- balanced-fusion diagnostic --------------------------------------
        # The naive concat lets 7694 state dims swamp 350 action dims. PCA the
        # state to a width comparable with the action before fusing, so a real
        # action contribution is visible if it exists. Also split test AUC by
        # decision index: late decisions in failing episodes may look abnormal
        # merely because the failure already happened (flailing), which inflates
        # action-only AUC without implying decision-time action sensitivity.
        dec = z["decision"]
        pooled = {k: (np.zeros(len(y)), np.zeros(len(y), dtype=bool))
                  for k in ("Vp", "Qp", "A", "Qs")}
        for k in range(cfg.folds):
            te_eps = set(folds[k])
            tr = ~np.isin(ep, list(te_eps))
            te = ~tr
            Str, Ste = standardize(S[tr], S[te])
            Atr, Ate = standardize(A[tr], A[te])
            # raw 14-d joint state, no VLM features: if fusing THIS with the
            # action behaves, the fusion mechanism is fine and the pooled VLM
            # features are the problem
            Jtr, Jte = standardize(z["state"][tr].astype(np.float32),
                                   z["state"][te].astype(np.float32))
            U, sv, _ = np.linalg.svd(Str, full_matrices=False)
            k_pc = min(cfg.state_pca, len(sv))
            P = (np.linalg.pinv(np.diag(sv[:k_pc])) @
                 (U[:, :k_pc].T @ Str)).T          # (D, k_pc) projection
            Sp_tr, Sp_te = Str @ P, Ste @ P
            Sp_tr, Sp_te = standardize(Sp_tr, Sp_te)
            for name, xtr, xte in (
                ("Vp", Sp_tr, Sp_te),
                ("Qp", np.concatenate([Sp_tr, Atr], 1), np.concatenate([Sp_te, Ate], 1)),
                ("A", Atr, Ate),
                ("Qs", np.concatenate([Jtr, Atr], 1), np.concatenate([Jte, Ate], 1)),
            ):
                s = fit_eval(xtr, y[tr], xte, y[te], seed=7000 + k)
                pooled[name][0][te] = s
                pooled[name][1][te] = True
        print(f"\n平衡融合诊断 (状态 PCA→{cfg.state_pca} 维), 汇总 5 折检验分数:")
        print(f"{'子集':>12} {'n决策':>6} {'V_pca':>8} {'Q_pca':>8} {'A':>8} {'Q(关节+a)':>10}")
        for lab, mask in (("全部", np.ones(len(y), bool)),
                          ("决策0", dec == 0), ("决策>0", dec > 0)):
            m = mask & pooled["A"][1]
            row = [auc(y[m], pooled[n][0][m]) for n in ("Vp", "Qp", "A", "Qs")]
            print(f"{lab:>12} {int(m.sum()):>6} "
                  f"{row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>10.3f}")


if __name__ == "__main__":
    main()
