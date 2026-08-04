"""Fit the noisy-OR with a two-parent CPT per component, arbitration left to the model.

Each component's latent failure has a conditional probability table over two
observed binary parents -- did it drift, and did the drift start early:

    P(z_c = 1 | drifted, early) = sigmoid(b0_c + b1_{c,task}*drifted + b2_c*early)

b0 is the intermittency term: a component can be at fault on a run where it
reads normal. b1 varies by task (partially pooled); b2 is the extra weight
carried by drifting early rather than late.

Unlike the previous extraction, several components are abnormal on the same
episode (3.8 of 6 on average), so the noisy-OR has a genuine competition to
resolve and the per-episode responsibility is an inference rather than a
restatement of an argmin.
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


def model(ood, early, task_id, n_tasks, y=None):
    n_comp = ood.shape[1]
    b0 = numpyro.sample("b0", dist.Normal(-3.0, 2.0).expand([n_comp]).to_event(1))
    b2 = numpyro.sample("b2", dist.Normal(0.0, 2.0).expand([n_comp]).to_event(1))
    mu_b1 = numpyro.sample("mu_b1", dist.Normal(0.0, 3.0).expand([n_comp]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.5).expand([n_comp]).to_event(1))
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([n_tasks, n_comp]).to_event(2))
    b1 = numpyro.deterministic("b1_task", mu_b1[None, :] + tau[None, :] * eps)

    mu = numpyro.sample("mu_leak", dist.Normal(-2.0, 1.5))
    sd = numpyro.sample("sigma_leak", dist.HalfNormal(1.0))
    with numpyro.plate("tasks", n_tasks):
        raw = numpyro.sample("leak_raw", dist.Normal(0.0, 1.0))
    leak = numpyro.deterministic("leak", jax.nn.sigmoid(mu + sd * raw))

    pz = jax.nn.sigmoid(b0[None, :] + b1[task_id] * ood + b2[None, :] * early)
    p = 1.0 - (1.0 - leak[task_id]) * jnp.prod(1.0 - pz, axis=1)
    numpyro.sample("y", dist.Bernoulli(probs=jnp.clip(p, 1e-6, 1 - 1e-6)), obs=y)


def fit(ood, early, tid, n_tasks, y, warmup, samples, chains, seed=0):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), ood=ood, early=early, task_id=tid,
             n_tasks=n_tasks, y=y, extra_fields=("diverging",))
    return mcmc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/whn/robotwin_eval/node_values_cpt.npz")
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    cfg = ap.parse_args()

    d = np.load(cfg.data, allow_pickle=True)
    ood_np, early_np = d["ood"], d["early"]
    y_np, tid_np = d["y"], d["task_id"]
    comps = [str(c) for c in d["components"]]
    n_tasks = int(tid_np.max()) + 1
    ood, early = jnp.asarray(ood_np), jnp.asarray(early_np)
    y, tid = jnp.asarray(y_np), jnp.asarray(tid_np)
    print(f"{len(y_np)} 次尝试 | {int(y_np.sum())} 次失败 | {n_tasks} 个任务 | "
          f"{len(comps)} 个环节\n")

    mc = fit(ood, early, tid, n_tasks, y, cfg.warmup, cfg.samples, cfg.chains)
    idata = az.from_numpyro(mc)
    post = mc.get_samples()
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    s = az.summary(idata, var_names=["b0", "b1_task", "b2", "mu_b1", "tau"])
    print(f"采样质量: 失败采样 {div}/{cfg.chains*cfg.samples}，"
          f"收敛指标最差 {s['r_hat'].max():.3f}（>1.01 不可信），"
          f"有效样本最少 {s['ess_bulk'].min():.0f}\n")

    b0 = np.asarray(post["b0"]); b2 = np.asarray(post["b2"])
    mu_b1 = np.asarray(post["mu_b1"]); tau = np.asarray(post["tau"])

    print("=== 条件概率表：某环节在各观测状态下「真的出问题」的概率 ===")
    print(f"{'环节':<15}{'都正常':>10}{'偏离(晚)':>12}{'偏离(早)':>12}{'有影响的把握':>14}")
    print("-" * 66)
    for j, c in enumerate(comps):
        p_norm = 1 / (1 + np.exp(-b0[:, j]))
        p_late = 1 / (1 + np.exp(-(b0[:, j] + mu_b1[:, j])))
        p_early = 1 / (1 + np.exp(-(b0[:, j] + mu_b1[:, j] + b2[:, j])))
        conf = float((mu_b1[:, j] > 0).mean())
        print(f"{c:<15}{p_norm.mean():>10.3f}{p_late.mean():>12.3f}"
              f"{p_early.mean():>12.3f}{conf:>14.3f}")
    print("\n  「都正常」一列 = 该环节看起来没事时仍出问题的概率（间歇性）")
    print("  「偏离(早)」明显高于「偏离(晚)」= 早偏离更要紧")

    print(f"\n{'环节':<15}{'跨任务是否一致':>16}")
    print("-" * 32)
    for j, c in enumerate(comps):
        lo = np.percentile(tau[:, j], 3)
        print(f"{c:<15}{('一致' if lo < 0.3 else '随任务而变'):>16}")

    # 单样本责任：已知失败，各环节的嫌疑
    b1t = np.asarray(post["b1_task"]); leak = np.asarray(post["leak"])
    pz = 1 / (1 + np.exp(-(b0[:, None, :] + b1t[:, tid_np] * ood_np[None]
                           + b2[:, None, :] * early_np[None])))
    pf = 1 - (1 - leak[:, tid_np]) * np.prod(1 - pz, axis=2)
    resp = np.clip(pz / np.clip(pf, 1e-9, None)[:, :, None], 0, 1).mean(axis=0)
    f = y_np == 1
    top = resp[f].argmax(axis=1)
    print(f"\n=== {int(f.sum())} 次失败的责任归属 ===")
    print(f"{'环节':<15}{'平均嫌疑':>10}{'嫌疑最大的次数':>16}")
    print("-" * 42)
    for j, c in enumerate(comps):
        print(f"{c:<15}{resp[f][:, j].mean():>10.3f}{int((top == j).sum()):>16}")
    print(f"  六个环节嫌疑之和的中位数 {np.median(resp[f].sum(1)):.3f}"
          f"（其余归于任务基础难度）")

    # 换没见过的任务
    rng = np.random.default_rng(0)
    aucs = []
    for k, held in enumerate(np.array_split(rng.permutation(n_tasks), 4)):
        tr = ~np.isin(tid_np, held); te = ~tr
        if not (0 < y_np[te].sum() < te.sum()):
            continue
        tt = sorted(set(tid_np[tr].tolist())); rm = {t: i for i, t in enumerate(tt)}
        m2 = fit(jnp.asarray(ood_np[tr]), jnp.asarray(early_np[tr]),
                 jnp.asarray([rm[t] for t in tid_np[tr]]), len(tt),
                 jnp.asarray(y_np[tr]), 800, 800, 2, seed=k + 1)
        ss = m2.get_samples()
        lk = jax.nn.sigmoid(np.asarray(ss["mu_leak"]))[:, None]
        pzt = 1 / (1 + np.exp(-(np.asarray(ss["b0"])[:, None, :]
                                + np.asarray(ss["mu_b1"])[:, None, :] * ood_np[te][None]
                                + np.asarray(ss["b2"])[:, None, :] * early_np[te][None])))
        pfx = (1 - (1 - lk) * np.prod(1 - pzt, axis=2)).mean(axis=0)
        pos, neg = pfx[y_np[te] == 1], pfx[y_np[te] == 0]
        allv = np.concatenate([pos, neg]); r = np.argsort(np.argsort(allv)).astype(float) + 1
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    print(f"\n换成没见过的任务，判断失败的准确度 {np.mean(aucs):.3f}  各折 {np.round(aucs,3)}")
    print("  （对照：旧的 argmin 编码 0.796，但那是特征硬选的结果，不是推理）")

    np.savez_compressed("/data/whn/robotwin_eval/responsibility_cpt.npz",
                        resp=resp, y=y_np, task_id=tid_np,
                        components=np.array(comps), seeds=d["seeds"],
                        tasks=d["tasks"])
    az.to_netcdf(idata, "/data/whn/robotwin_eval/post_cpt.nc")


if __name__ == "__main__":
    main()
