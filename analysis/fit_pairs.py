"""Bayesian diagnosis over counterfactual pairs.

The pairing establishes the observation; this is the diagnosis. For each attempt,
every component reports whether it drifted abnormally far from the matched
counterfactual, and the noisy-OR arbitrates which component is answerable for the
outcome. Reading the marginal gap per component -- as an earlier pass did -- is
not diagnosis: it ignores that a weak signal can become decisive once competing
explanations are accounted for, and it yields no per-attempt attribution.

Decisions are aggregated to the attempt. Several decisions of one attempt share
an outcome, so treating them as independent observations would count the same
evidence repeatedly; the noisy-OR over decisions is the right aggregation anyway,
since one bad decision suffices to lose the attempt.

Only failures that actually have a counterfactual are analysed. A failure whose
nearest success is far away has no counterfactual at all -- the "match" is just
some other scene -- and forcing one in would feed every value node the difference
between two unrelated situations. The admission threshold comes from the
success-success distances: a pairing counts only if it is as close as two
successful decisions of that task typically are. Failures that fail this test are
reported separately and left alone, since there is nothing to borrow from.

Match quality also enters as a covariate among the admitted pairs, since even
within the threshold a slightly more distant match inflates every component's gap
at once.
"""

import argparse

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


def model(a, mdist, task_id, n_tasks, y=None):
    n_comp = a.shape[1]
    b0 = numpyro.sample("b0", dist.Normal(-3.0, 2.0).expand([n_comp]).to_event(1))
    mu_b1 = numpyro.sample("mu_b1", dist.Normal(0.0, 3.0).expand([n_comp]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.5).expand([n_comp]).to_event(1))
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([n_tasks, n_comp]).to_event(2))
    b1 = numpyro.deterministic("b1_task", mu_b1[None, :] + tau[None, :] * eps)
    # Absorbs "the counterfactual was not that close", which otherwise looks like
    # every component drifting at once.
    g = numpyro.sample("g_match", dist.Normal(0.0, 2.0))

    mu = numpyro.sample("mu_leak", dist.Normal(-2.0, 1.5))
    sd = numpyro.sample("sigma_leak", dist.HalfNormal(1.0))
    with numpyro.plate("tasks", n_tasks):
        raw = numpyro.sample("leak_raw", dist.Normal(0.0, 1.0))
    leak = numpyro.deterministic("leak", jax.nn.sigmoid(mu + sd * raw))

    pz = jax.nn.sigmoid(b0[None, :] + b1[task_id] * a + g * mdist[:, None])
    p = 1.0 - (1.0 - leak[task_id]) * jnp.prod(1.0 - pz, axis=1)
    numpyro.sample("y", dist.Bernoulli(probs=jnp.clip(p, 1e-6, 1 - 1e-6)), obs=y)


def fit(a, mdist, tid, n_tasks, y, warmup, samples, chains, seed=0):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), a=a, mdist=mdist, task_id=tid,
             n_tasks=n_tasks, y=y, extra_fields=("diverging",))
    return mcmc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/whn/robotwin_eval/pairs.npz")
    ap.add_argument("--quantile", type=float, default=0.97,
                    help="a component counts as drifted above this percentile of normal spread")
    ap.add_argument("--cf-quantile", type=float, default=0.50,
                    help="admit a pairing only if its distance is within this percentile "
                         "of the success-success distances, i.e. it is a real counterfactual")
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    cfg = ap.parse_args()

    d = np.load(cfg.data, allow_pickle=True)
    A, Y, TID, EPI, MD = d["a"], d["y"], d["task_id"], d["episode"], d["match_dist"]
    comps = [str(c) for c in d["components"]]
    tasks = [str(t) for t in d["tasks"]]

    # Admit only genuine counterfactuals, judged against how close two successful
    # decisions of the same task normally are. Applied to both arms, so failure
    # and success pairings are compared over the same range of state similarity.
    n_ep_all = len(np.unique(EPI))
    keep = np.zeros(len(Y), bool)
    for t in np.unique(TID):
        m = TID == t
        thr = np.quantile(MD[m & (Y == 0)], cfg.cf_quantile)
        keep |= m & (MD <= thr)
    A, Y, TID, EPI, MD = A[keep], Y[keep], TID[keep], EPI[keep], MD[keep]

    # Raw gaps -> percentile against the *admitted* success-success gaps only.
    # Using all success-success pairs as the reference would define "normal
    # spread" partly from pairings whose two decisions were never in the same
    # state, which is not a spread this comparison should be judged against.
    P = np.empty_like(A)
    for t in np.unique(TID):
        m = TID == t
        for j in range(A.shape[1]):
            base = np.sort(A[m & (Y == 0), j])
            P[m, j] = np.searchsorted(base, A[m, j]) / max(len(base), 1)
    A = P

    # decision -> attempt: a component counts as drifted if it drifted on any
    # decision of the attempt.
    hot = (A > cfg.quantile).astype(np.float32)
    eps_ = np.unique(EPI)
    a_ep = np.stack([hot[EPI == e].max(axis=0) for e in eps_])
    y_ep = np.array([Y[EPI == e][0] for e in eps_], dtype=np.int32)
    t_ep = np.array([TID[EPI == e][0] for e in eps_], dtype=np.int32)
    md_ep = np.array([MD[EPI == e].mean() for e in eps_], dtype=np.float32)
    md_ep = (md_ep - md_ep.mean()) / (md_ep.std() + 1e-6)
    n_tasks = int(t_ep.max()) + 1

    print(f"有反事实配对的尝试 {len(y_ep)} 次 / 全部 {n_ep_all} 次"
          f"（{n_ep_all - len(y_ep)} 次没有足够近的反事实，已排除）")
    print(f"{len(y_ep)} 次尝试（失败 {int(y_ep.sum())}）| {n_tasks} 个任务 | "
          f"{len(comps)} 个环节 | 异常判定：差异超过正常波动的 {cfg.quantile:.0%} 分位")
    print("每个环节被判为异常的比例（失败 / 成功）:")
    for j, c in enumerate(comps):
        print(f"  {c:<12} {a_ep[y_ep==1,j].mean():.2f} / {a_ep[y_ep==0,j].mean():.2f}")
    print(f"平均每次失败有 {a_ep[y_ep==1].sum(1).mean():.2f} 个环节同时异常\n")

    mc = fit(jnp.asarray(a_ep), jnp.asarray(md_ep), jnp.asarray(t_ep),
             n_tasks, jnp.asarray(y_ep), cfg.warmup, cfg.samples, cfg.chains)
    idata = az.from_numpyro(mc)
    post = mc.get_samples()
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    s = az.summary(idata, var_names=["b0", "mu_b1", "tau", "g_match"])
    print(f"采样质量: 失败采样 {div}/{cfg.chains*cfg.samples}, "
          f"收敛最差 {s['r_hat'].max():.3f}, 有效样本最少 {s['ess_bulk'].min():.0f}\n")

    mu_b1 = np.asarray(post["mu_b1"]); tau = np.asarray(post["tau"])
    b0 = np.asarray(post["b0"]); b1t = np.asarray(post["b1_task"])
    print("=== 各环节：与反事实样本偏离时，是否真的导致失败 ===")
    print(f"{'环节':<14}{'影响强度':>10}{'有影响的把握':>14}{'跨任务':>12}")
    print("-" * 52)
    for j, c in enumerate(comps):
        p = float((mu_b1[:, j] > 0).mean())
        v = "有影响" if p > 0.95 else ("说不准" if p > 0.8 else "没影响")
        cons = "一致" if np.percentile(tau[:, j], 3) < 0.3 else "随任务而变"
        print(f"{c:<14}{mu_b1[:, j].mean():>10.2f}{p:>14.3f}{cons:>12}  {v}")

    leak = np.asarray(post["leak"]); g = np.asarray(post["g_match"])
    pz = 1 / (1 + np.exp(-(b0[:, None, :] + b1t[:, t_ep] * a_ep[None]
                           + g[:, None, None] * md_ep[None, :, None])))
    pf = 1 - (1 - leak[:, t_ep]) * np.prod(1 - pz, axis=2)
    resp = np.clip(pz / np.clip(pf, 1e-9, None)[:, :, None], 0, 1).mean(axis=0)
    f = y_ep == 1
    top = resp[f].argmax(axis=1)
    print(f"\n=== {int(f.sum())} 次失败的责任归属 ===")
    print(f"{'环节':<14}{'平均嫌疑':>10}{'嫌疑最大的次数':>16}{'可矫正':>10}")
    print("-" * 52)
    for j, c in enumerate(comps):
        fix = "是" if c in ("noise", "routing") else "否"
        print(f"{c:<14}{resp[f][:, j].mean():>10.3f}{int((top == j).sum()):>16}{fix:>10}")
    n_fix = int(np.isin(top, [comps.index("noise"), comps.index("routing")]).sum())
    print(f"\n  其中 {n_fix} 次（{n_fix/max(f.sum(),1):.0%}）归到可矫正的环节")

    np.savez_compressed("/data/whn/robotwin_eval/pair_responsibility.npz",
                        resp=resp, y=y_ep, task_id=t_ep, episode=eps_,
                        components=np.array(comps), tasks=np.array(tasks))
    az.to_netcdf(idata, "/data/whn/robotwin_eval/post_pairs.nc")
    print("\n-> pair_responsibility.npz")


if __name__ == "__main__":
    main()
