"""Model B:责任同时落到「哪个环节」和「第几次决策」。

和之前的区别只有一处 —— 观测从「这次尝试里组件 c 异常吗」变成「第 k 次决策上组件 c
异常吗」。推理机器（分层 noisy-OR + NUTS）不变。

每个环节在每次决策上都有一次「搞砸」的机会，任意一次搞砸就足以让这次尝试失败：

    P(失败) = 1 − (1 − 任务基础失败率) × ∏(所有决策、所有环节) (1 − 搞砸概率)

所有尝试一律截到同样的决策次数，短的补「正常」，否则决策次数多的会自动显得更容易
失败 —— 而次数多本来就是失败造成的。

输出里最有用的是「责任」这一项：已知这次尝试失败了，回头看是哪个环节、在第几次决策
上最可能出的问题。
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


def model(a, task_id, n_tasks, y=None):
    """a: (episodes, decisions, components) 的 0/1 观测"""
    n_comp = a.shape[2]
    b0 = numpyro.sample("b0", dist.Normal(-4.0, 2.0).expand([n_comp]).to_event(1))
    mu_b1 = numpyro.sample("mu_b1", dist.Normal(0.0, 3.0).expand([n_comp]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.5).expand([n_comp]).to_event(1))
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([n_tasks, n_comp]).to_event(2))
    b1 = numpyro.deterministic("b1_task", mu_b1[None, :] + tau[None, :] * eps)

    mu = numpyro.sample("mu_leak", dist.Normal(-2.0, 1.5))
    sd = numpyro.sample("sigma_leak", dist.HalfNormal(1.0))
    with numpyro.plate("tasks", n_tasks):
        raw = numpyro.sample("leak_raw", dist.Normal(0.0, 1.0))
    leak = numpyro.deterministic("leak", jax.nn.sigmoid(mu + sd * raw))

    # (episodes, decisions, components)
    h = jax.nn.sigmoid(b0[None, None, :] + b1[task_id][:, None, :] * a)
    surv = jnp.prod(1.0 - h, axis=(1, 2))
    p = 1.0 - (1.0 - leak[task_id]) * surv
    numpyro.sample("y", dist.Bernoulli(probs=jnp.clip(p, 1e-6, 1 - 1e-6)), obs=y)


def fit(a, task_id, n_tasks, y, warmup, samples, chains, seed=0):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), a=a, task_id=task_id, n_tasks=n_tasks, y=y,
             extra_fields=("diverging",))
    return mcmc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/whn/robotwin_eval/node_values_perdecision.npz")
    ap.add_argument("--warmup", type=int, default=1200)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--chains", type=int, default=4)
    cfg = ap.parse_args()

    d = np.load(cfg.data, allow_pickle=True)
    a_np, y_np, tid_np = d["a"], d["y"], d["task_id"]
    comps = [str(c) for c in d["components"]]
    n_tasks = int(tid_np.max()) + 1
    n_ep, n_dec, n_comp = a_np.shape
    a, y, tid = jnp.asarray(a_np), jnp.asarray(y_np), jnp.asarray(tid_np)
    print(f"{n_ep} 次尝试 | {int(y_np.sum())} 次失败 | {n_tasks} 个任务 | "
          f"每次尝试 {n_dec} 次决策 × {n_comp} 个环节\n")

    mc = fit(a, tid, n_tasks, y, cfg.warmup, cfg.samples, cfg.chains)
    idata = az.from_numpyro(mc)
    post = mc.get_samples()
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    bad = az.summary(idata, var_names=["b0", "mu_b1", "tau"])
    print(f"采样是否可信: 失败的采样 {div} 次（越少越好，0 最理想）;"
          f" 收敛指标最差 {bad['r_hat'].max():.3f}（超过 1.01 就不可信）\n")

    mu_b1 = np.asarray(post["mu_b1"]); tau = np.asarray(post["tau"])
    print("=== 各环节：异常时会不会真的导致失败 ===")
    print(f"{'环节':<12}{'影响强度':>10}{'有影响的把握':>14}{'跨任务是否一致':>16}")
    print("-" * 56)
    for j, c in enumerate(comps):
        p = float((mu_b1[:, j] > 0).mean())
        consistent = "一致" if np.percentile(tau[:, j], 3) < 0.3 else "随任务而变"
        verdict = "有影响" if p > 0.95 else ("说不准" if p > 0.8 else "没影响")
        print(f"{c:<12}{mu_b1[:, j].mean():>10.2f}{p:>14.2f}   {consistent:<12}{verdict}")

    # 责任：已知失败，回头看每个「环节 × 决策」的嫌疑
    b0 = np.asarray(post["b0"]); b1t = np.asarray(post["b1_task"])
    leak = np.asarray(post["leak"])
    h = 1 / (1 + np.exp(-(b0[:, None, None, :] + b1t[:, tid_np][:, :, None, :] * a_np[None])))
    surv = np.prod(1 - h, axis=(2, 3))
    pf = 1 - (1 - leak[:, tid_np]) * surv
    resp = (h / np.clip(pf, 1e-9, None)[:, :, None, None]).mean(axis=0)   # (ep, dec, comp)
    resp = np.clip(resp, 0, 1)

    f = y_np == 1
    print("\n=== 失败尝试里，责任落在哪个环节、第几次决策 ===")
    print(f"{'环节':<12}" + "".join(f"{'第'+str(k)+'次':>9}" for k in range(n_dec)) + f"{'合计':>9}")
    print("-" * (12 + 9 * (n_dec + 1)))
    for j, c in enumerate(comps):
        row = resp[f][:, :, j].mean(axis=0)
        print(f"{c:<12}" + "".join(f"{v:>9.3f}" for v in row) + f"{row.sum():>9.3f}")

    flat = resp[f].reshape(f.sum(), -1).argmax(axis=1)
    dk, dc = np.unravel_index(flat, (n_dec, n_comp))
    print(f"\n每次失败「嫌疑最大」的落点统计（共 {int(f.sum())} 次失败）:")
    for j, c in enumerate(comps):
        if (dc == j).sum():
            print(f"  {c:<12} {int((dc == j).sum()):>4} 次，最常发生在第 "
                  f"{int(np.bincount(dk[dc == j]).argmax())} 次决策")

    # 换没见过的任务还管不管用
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n_tasks), 4)
    aucs = []
    for k, held in enumerate(folds):
        tr = ~np.isin(tid_np, held); te = ~tr
        if not (0 < y_np[te].sum() < te.sum()):
            continue
        tt = sorted(set(tid_np[tr].tolist())); remap = {t: i for i, t in enumerate(tt)}
        m2 = fit(jnp.asarray(a_np[tr]), jnp.asarray([remap[t] for t in tid_np[tr]]),
                 len(tt), jnp.asarray(y_np[tr]), 700, 700, 2, seed=k + 1)
        s = m2.get_samples()
        lk = jax.nn.sigmoid(np.asarray(s["mu_leak"]))[:, None]
        hh = 1 / (1 + np.exp(-(np.asarray(s["b0"])[:, None, None, :]
                               + np.asarray(s["mu_b1"])[:, None, None, :] * a_np[te][None])))
        pfx = (1 - (1 - lk) * np.prod(1 - hh, axis=(2, 3))).mean(axis=0)
        pos, neg = pfx[y_np[te] == 1], pfx[y_np[te] == 0]
        allv = np.concatenate([pos, neg]); r = np.argsort(np.argsort(allv)).astype(float) + 1
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    print(f"\n换成完全没见过的任务，判断失败的准确度: {np.mean(aucs):.3f}")
    print(f"  （0.5 = 瞎猜，1.0 = 完美。对照：只说「哪个环节先出问题」是 0.796）")

    az.to_netcdf(idata, "/data/whn/robotwin_eval/post_perdecision.nc")


if __name__ == "__main__":
    main()
