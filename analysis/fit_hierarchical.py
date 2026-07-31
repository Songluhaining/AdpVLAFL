"""Failure localization with binary value nodes and per-task partial pooling.

Three changes from the previous fit, each aimed at a specific diagnosed fault:

1. Value nodes are Bernoulli, not continuous. The observation becomes
   "component c looked abnormal this episode", thresholded at a within-task
   quantile. That puts the task-specific scaling entirely in the threshold and
   leaves a genuinely task-agnostic CPT:
       P(z_c = 1 | a_c = 0) = sigmoid(b0_c)
       P(z_c = 1 | a_c = 1) = sigmoid(b0_c + b1_{c,t})
   Previously a continuous within-task percentile rank was fed to a sigmoid, so
   task-specific meaning leaked into b1 and stopped it transferring.

2. b1 is hierarchical over tasks. Previously `leak` was the only task-varying
   parameter, so every kind of task heterogeneity -- including the genuine
   "component c matters more in this task" -- had nowhere to go but leak, which
   is why leak absorbed ~87% of the responsibility.

3. tau_c, the across-task spread of b1_c, is reported as a direct measure of
   generality. Small tau means the component behaves the same everywhere and a
   general method can rest on it; large tau means it is task-specific. That
   replaces measuring generality only after the fact with held-out AUC.

Pooled (tau forced to 0) and hierarchical are both fit and compared by LOO.
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


def _leak(n_tasks):
    mu = numpyro.sample("mu_leak", dist.Normal(-2.0, 1.5))
    sd = numpyro.sample("sigma_leak", dist.HalfNormal(1.0))
    with numpyro.plate("tasks", n_tasks):
        raw = numpyro.sample("leak_raw", dist.Normal(0.0, 1.0))
    return numpyro.deterministic("leak", jax.nn.sigmoid(mu + sd * raw))


def _outcome(pz, leak, task_id, y):
    p = 1.0 - (1.0 - leak[task_id]) * jnp.prod(1.0 - pz, axis=1)
    numpyro.sample("y", dist.Bernoulli(probs=jnp.clip(p, 1e-6, 1 - 1e-6)), obs=y)


def model_pooled(a, task_id, n_tasks, y=None):
    n_comp = a.shape[1]
    b0 = numpyro.sample("b0", dist.Normal(-3.0, 2.0).expand([n_comp]).to_event(1))
    b1 = numpyro.sample("b1", dist.Normal(0.0, 3.0).expand([n_comp]).to_event(1))
    pz = jax.nn.sigmoid(b0[None, :] + b1[None, :] * a)
    _outcome(pz, _leak(n_tasks), task_id, y)


def model_hier(a, task_id, n_tasks, y=None):
    n_comp = a.shape[1]
    b0 = numpyro.sample("b0", dist.Normal(-3.0, 2.0).expand([n_comp]).to_event(1))
    mu_b1 = numpyro.sample("mu_b1", dist.Normal(0.0, 3.0).expand([n_comp]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.5).expand([n_comp]).to_event(1))
    # Non-centred: with only ~8 failures per task the centred form funnels badly.
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([n_tasks, n_comp]).to_event(2))
    b1 = numpyro.deterministic("b1_task", mu_b1[None, :] + tau[None, :] * eps)  # (T, C)
    pz = jax.nn.sigmoid(b0[None, :] + b1[task_id] * a)
    _outcome(pz, _leak(n_tasks), task_id, y)


def fit(m, a, task_id, n_tasks, y, cfg, seed=0):
    mcmc = MCMC(NUTS(m, target_accept_prob=0.95), num_warmup=cfg.warmup,
                num_samples=cfg.samples, num_chains=cfg.chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), a=a, task_id=task_id, n_tasks=n_tasks, y=y,
             extra_fields=("diverging",))
    return mcmc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/whn/robotwin_eval/node_values.npz")
    ap.add_argument("--thresh", type=float, default=0.75,
                    help="within-task percentile above which a component counts as abnormal")
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    cfg = ap.parse_args()

    d = np.load(cfg.data, allow_pickle=True)
    a_cont, y_np, tid_np = d["a"], d["y"], d["task_id"]
    comps = [str(c) for c in d["components"]]
    n_tasks = int(tid_np.max()) + 1

    # a is already a within-task percentile rank, so a fixed cut is exactly
    # "the worst (1 - thresh) fraction of this task's episodes".
    a_np = (a_cont > cfg.thresh).astype(np.float32)
    a, y, tid = jnp.asarray(a_np), jnp.asarray(y_np), jnp.asarray(tid_np)
    print(f"{len(y_np)} episodes | {int(y_np.sum())} failures | {n_tasks} tasks | "
          f"二值化阈值 = 任务内 {cfg.thresh:.0%} 分位")
    print(f"各组件被判为异常的比例: " +
          "  ".join(f"{c}={a_np[:, j].mean():.2f}" for j, c in enumerate(comps)) + "\n")

    mp = fit(model_pooled, a, tid, n_tasks, y, cfg)
    mh = fit(model_hier, a, tid, n_tasks, y, cfg)
    id_p, id_h = az.from_numpyro(mp), az.from_numpyro(mh)
    for nm, mc in (("pooled", mp), ("hier", mh)):
        print(f"[{nm}] divergences={int(np.asarray(mc.get_extra_fields()['diverging']).sum())}"
              f" / {cfg.chains * cfg.samples}")

    cmp = az.compare({"pooled": id_p, "hierarchical": id_h}, ic="loo")
    print("\n=== LOO: 池化 vs 分层 ===")
    print(cmp[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]].to_string())

    post = mh.get_samples()
    mu_b1 = np.asarray(post["mu_b1"]); tau = np.asarray(post["tau"]); b0 = np.asarray(post["b0"])
    sh = az.summary(id_h, var_names=["mu_b1", "tau"], hdi_prob=0.94)

    print("\n=== 分层模型: 组件效应与通用性 ===")
    print(f"{'组件':<12}{'mu_b1':>8}{'94% HDI':>18}{'P(>0)':>8}{'tau':>8}"
          f"{'tau 94% HDI':>18}  判定")
    print("-" * 84)
    for j, c in enumerate(comps):
        r1, rt = sh.iloc[j], sh.iloc[len(comps) + j]
        pgt = float((mu_b1[:, j] > 0).mean())
        if pgt > 0.95 and rt["mean"] < 1.0:
            verdict = "有效且跨任务一致"
        elif pgt > 0.95:
            verdict = "有效但任务特异"
        else:
            verdict = "无证据"
        print(f"{c:<12}{mu_b1[:, j].mean():>8.2f}  [{r1['hdi_3%']:>6.2f},{r1['hdi_97%']:>6.2f}]"
              f"{pgt:>8.2f}{tau[:, j].mean():>8.2f}  [{rt['hdi_3%']:>6.2f},{rt['hdi_97%']:>6.2f}]"
              f"  {verdict}")
    print("\n  mu_b1>0 = 观测异常确实抬高失败概率；tau 小 = 各任务表现一致（可作通用方法基础）")
    print(f"  CPT 基线 P(z=1 | 观测正常) = " +
          "  ".join(f"{c}={1/(1+np.exp(-b0[:, j])).mean():.3f}" for j, c in enumerate(comps)))

    # ---- held-out TASK validation, both models ----------------------------
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n_tasks), 4)
    print("\n=== 留出任务交叉验证 ===")
    for nm, m in (("pooled", model_pooled), ("hierarchical", model_hier)):
        aucs = []
        for k, held in enumerate(folds):
            tr = ~np.isin(tid_np, held); te = ~tr
            if not (0 < y_np[te].sum() < te.sum()):
                continue
            tr_tasks = sorted(set(tid_np[tr].tolist()))
            remap = {t: i for i, t in enumerate(tr_tasks)}
            mm = fit(m, jnp.asarray(a_np[tr]), jnp.asarray([remap[t] for t in tid_np[tr]]),
                     len(tr_tasks), jnp.asarray(y_np[tr]),
                     argparse.Namespace(warmup=800, samples=800, chains=2), seed=k + 1)
            s = mm.get_samples()
            # Unseen tasks: leak from the hyperprior, and for the hierarchical
            # model b1 from the population mean -- a new task has no eps of its own.
            lk = jax.nn.sigmoid(np.asarray(s["mu_leak"]))[:, None]
            bb = np.asarray(s["mu_b1"] if "mu_b1" in s else s["b1"])
            pz = 1 / (1 + np.exp(-(np.asarray(s["b0"])[:, None, :] + bb[:, None, :] * a_np[te][None])))
            pf = (1.0 - (1.0 - lk) * np.prod(1.0 - pz, axis=2)).mean(axis=0)
            pos, neg = pf[y_np[te] == 1], pf[y_np[te] == 0]
            allv = np.concatenate([pos, neg]); r = np.argsort(np.argsort(allv)) + 1.0
            aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        print(f"  {nm:<14} 各折 AUC {np.round(aucs, 3)}   平均 {np.mean(aucs):.3f}")

    az.to_netcdf(id_h, "/data/whn/robotwin_eval/post_hier.nc")
    print("\nposterior -> /data/whn/robotwin_eval/post_hier.nc")


if __name__ == "__main__":
    main()
