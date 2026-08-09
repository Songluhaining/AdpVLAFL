"""Advantage-weighted self-imitation fine-tune (online-RL loop, training step).

Clones the policy's own SUCCESSFUL rollouts with the model's native flow-
matching objective -- t ~ Beta(1.5,1), eps ~ N(0,I), x_t = t*eps + (1-t)*a,
L1 regression of v toward (eps - a) -- restricted to successes and weighted by
scene difficulty w = (1 - p_scene) + 0.1, where p_scene is the success rate
among this iteration's K rollouts of the scene. Episode-level outcomes and
scene-level baselines only: no critic, no per-draw credit, per the diagnosis
in RESEARCH_REPORT (within-state signals are ungrounded).

One suffix forward per sample (no BPTT): ~10x cheaper per step than the
value-gradient trainer. LoRA on the action expert; VLM frozen (prefix KV
under no_grad, identical to train_actor_vine).
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from train_actor_vine import CAM_KEYS, compute_prefix, make_obs  # noqa: E402


def build_success_index(roots, task):
    """Successful episodes with images, plus per-scene success rate over roots."""
    eps, by_scene = [], defaultdict(list)
    for root in roots:
        for f in sorted(glob.glob(f"{root}/{task}_*/episodes/*.npz")):
            meta = json.load(open(f.replace(".npz", ".json")))
            by_scene[int(meta["seed"])].append(bool(meta["success"]))
            if meta["success"]:
                with np.load(f) as z:
                    if "decision_images_head_camera" not in z.files:
                        continue
                eps.append((f, meta, int(meta["num_decisions"])))
    p_scene = {s: sum(v) / len(v) for s, v in by_scene.items()}
    return eps, p_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="rollout roots of THIS iteration (scene p comes from them)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--init_lora", default=None, help="LoRA to continue from")
    ap.add_argument("--max-decisions", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from peft import LoraConfig, inject_adapter_in_model

    srv = LingbotVLAv2Server(path_to_pi_model=cfg.model_path, use_length=50,
                             chunk_ret=True, use_bf16=True, use_compile=False)
    srv.reset(robo_name="robotwin")
    M = srv.vla.model
    for p in srv.vla.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(
        LoraConfig(r=cfg.rank, lora_alpha=2 * cfg.rank,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
        M.qwenvl_with_expert.qwen_expert)
    lora_params = []
    for n, p in srv.vla.named_parameters():
        if "lora" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
            lora_params.append(p)
    if cfg.init_lora:
        ck = torch.load(cfg.init_lora, map_location="cpu", weights_only=False)
        missing, unexpected = srv.vla.load_state_dict(
            {k: v.to(torch.float32) for k, v in ck["lora"].items()}, strict=False)
        assert not unexpected, unexpected[:3]
        print(f"continued from {cfg.init_lora}", flush=True)
    print(f"LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M", flush=True)

    eps, p_scene = build_success_index(cfg.roots, cfg.task)
    n_scenes = len(p_scene)
    print(f"{cfg.task}: {len(eps)} 成功集 / {n_scenes} 场景, "
          f"平均场景成功率 {np.mean(list(p_scene.values())):.2f}", flush=True)
    if not eps:
        sys.exit("no successful episodes with images")
    # normalize weights to mean 1 so the difficulty weighting shifts emphasis
    # without silently rescaling the learning rate (K=1 data would otherwise
    # put every weight at the 0.1 floor)
    w_mean = np.mean([(1.0 - p_scene[int(m["seed"])]) + 0.1 for _, m, _ in eps])

    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    K = M.config.n_action_steps
    D = M.config.max_action_dim

    step, t0 = 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(eps))
        done, loss_acc = 0, []
        for ei in order:
            f, meta, n = eps[ei]
            w = ((1.0 - p_scene[int(meta["seed"])]) + 0.1) / w_mean
            with np.load(f) as z:
                for k in range(n):
                    inp = srv._prepare_model_input(make_obs(z, meta, k))
                    p_pad, p_pos, past_kv = compute_prefix(M, inp, device, dtype)
                    state = inp["state"].unsqueeze(0).to(dtype=dtype, device=device)
                    a = torch.tensor(z["predicted_chunks_normalized"][k],
                                     device=device, dtype=torch.float32)[None]
                    # native pretraining recipe: t ~ Beta(1.5,1) -> [0.001, 1]
                    t_val = float(np.random.default_rng(step + cfg.seed).beta(1.5, 1.0)
                                  * 0.999 + 0.001)
                    t = torch.tensor(t_val, dtype=dtype, device=device)
                    eps_n = torch.randn(1, K, D, device=device, dtype=torch.float32)
                    x_t = (t_val * eps_n + (1 - t_val) * a).to(dtype)
                    v = M.predict_velocity(state, p_pad, past_kv, x_t, t.expand(1),
                                           prefix_position_ids=p_pos)
                    u = (eps_n - a)
                    loss = w * (v.to(torch.float32) - u).abs().mean() / cfg.accum
                    loss.backward()
                    loss_acc.append(float(loss) * cfg.accum)
                    step += 1
                    done += 1
                    if step % cfg.accum == 0:
                        gn = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if (step // cfg.accum) % 20 == 0:
                            print(f"ep{epoch} {done} loss={np.mean(loss_acc[-320:]):.4f} "
                                  f"gnorm={float(gn):.3f} "
                                  f"{(time.time() - t0)/max(step,1):.2f}s/dec", flush=True)
                    if cfg.max_decisions and done >= cfg.max_decisions:
                        break
                if cfg.max_decisions and done >= cfg.max_decisions:
                    break
        sd = {n_: p.detach().cpu() for n_, p in srv.vla.named_parameters() if "lora" in n_}
        path = f"{cfg.out}_{cfg.task}_ep{epoch}.pt"
        torch.save({"lora": sd, "cfg": vars(cfg), "epoch": epoch}, path)
        print(f"epoch {epoch}: mean_loss={np.mean(loss_acc):.4f} saved {path}", flush=True)


if __name__ == "__main__":
    main()
