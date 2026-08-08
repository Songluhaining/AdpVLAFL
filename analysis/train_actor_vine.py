"""VINE actor fine-tuning (Phase 2, step 2): LoRA + BPTT through the sampler.

The paper's actor update, adapted to LingBot-VLA:

  loss = -Q(s, GENERATE(s)) + alpha * || GENERATE(s) - a_logged ||^2

GENERATE is the VINE chain (10 endpoint-prediction steps with per-step fresh
noise, lingbot time convention t=1->0.1) run with gradients THROUGH the action
expert only: the VLM prefix is computed once per decision under no_grad and its
KV cache enters the chain as a constant, so BPTT never touches the 4B backbone.
Trainable parameters are LoRA adapters (fp32) on the 36 expert layers'
attention projections; everything else stays frozen bf16.

The critic is the Phase-2 Bellman twin Q (frozen here), which consumes the
EXECUTED UNNORMALIZED prefix (25x14). The bridge from the actor's normalized
50x55 chunk is recovered numerically from feature_transform.unapply at startup
(probe the affine map, verify linearity and cross-sample constancy, abort
loudly if the norm scheme is not a fixed affine) -- so the gradient path
actor -> unnormalize -> critic is exact, not re-implemented by hand.

Run from the lingbot repo root (relative config paths), e.g.:
  QWEN3VL_PATH=... python /path/to/train_actor_vine.py \
      --model_path ckpts/.../hf_ckpt --rollouts .../rollouts \
      --critic .../rl_buffer/critic_click_bell.pt --task click_bell
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the lingbot repo root: `deploy` / `lingbotvla` live here
from train_critic_bellman import TwinCritic  # noqa: E402

CAM_KEYS = {
    "observation.images.cam_high": "decision_images_head_camera",
    "observation.images.cam_left_wrist": "decision_images_left_camera",
    "observation.images.cam_right_wrist": "decision_images_right_camera",
}


def build_index(rollouts, task):
    """(npz_path, meta, decision_idx) for every decision that has images."""
    idx = []
    for f in sorted(glob.glob(f"{rollouts}/{task}_*/episodes/*.npz")):
        with np.load(f) as z:
            if "decision_images_head_camera" not in z.files:
                continue
            n = int(json.load(open(f.replace(".npz", ".json")))["num_decisions"])
        meta = json.load(open(f.replace(".npz", ".json")))
        idx.append((f, meta, n))
    return idx


def make_obs(z, meta, k):
    obs = {ok: np.ascontiguousarray(z[nk][k]) for ok, nk in CAM_KEYS.items()}
    obs["observation.state"] = z["decision_states"][k].astype(np.float32)
    obs["task"] = meta["instruction"]
    return obs


def build_unnorm_map(srv, transformed, chunk_len, act_dim):
    """Recover a = x @ W + b from feature_transform.unapply by probing."""
    a_keys = srv.action_key if isinstance(srv.action_key, (list, tuple)) else [srv.action_key]

    def U(x):
        s = dict(transformed)
        s["actions"] = torch.tensor(x, dtype=torch.float32)
        if "state" in s and torch.is_tensor(s["state"]):
            s["state"] = s["state"].to(torch.float32)
        d = srv.vla.feature_transform.unapply(s)
        out = d[a_keys[0]]
        return np.asarray(out, dtype=np.float64)

    off = U(np.zeros((chunk_len, act_dim), np.float32))
    n_out = off.shape[-1]
    W = np.zeros((act_dim, n_out))
    for j in range(act_dim):
        e = np.zeros((chunk_len, act_dim), np.float32)
        e[:, j] = 1.0
        resp = U(e) - off
        if np.abs(resp - resp[0]).max() > 1e-4:
            raise RuntimeError(f"unapply is timestep-dependent at dim {j}; not a fixed affine")
        W[j] = resp[0]
    x = np.random.default_rng(0).standard_normal((chunk_len, act_dim)).astype(np.float32)
    err = np.abs(U(x) - (x @ W + off)).max()
    if err > 1e-3:
        raise RuntimeError(f"unapply is not affine (probe error {err:.2e})")
    return W, off[0], err


@torch.no_grad()
def compute_prefix(M, inp, device, dtype):
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks

    images = inp["images"].to(dtype=dtype, device=device)
    if images.ndim == 4:
        images = images.unsqueeze(0)
    img_masks = inp["img_masks"].to(device=device)
    if img_masks.ndim == 1:
        img_masks = img_masks.unsqueeze(0)
    lang_tokens = inp["lang_tokens"].unsqueeze(0).to(device)
    lang_masks = inp["lang_masks"].unsqueeze(0).to(device)
    grid = inp.get("image_grid_thw")
    grid = grid.to(device=device, dtype=torch.long) if grid is not None else None

    (p_embs, p_pad, p_att, p_pos, vis_masks, deepstack) = M.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, image_grid_thw=grid)
    att2d = make_att_2d_masks(p_pad, p_att)
    _, past_kv, _ = M.qwenvl_with_expert.forward(
        attention_mask=att2d, position_ids=p_pos, vlm_position_ids=p_pos,
        past_key_values=None, inputs_embeds=[p_embs, None],
        use_cache=M.config.use_cache, fill_kv_cache=True,
        visual_pos_masks=vis_masks, deepstack_visual_embeds=deepstack)
    return p_pad, p_pos, past_kv


def vine_generate(M, state, p_pad, p_pos, past_kv, device, dtype, gen=None):
    """VINE chain with gradients; returns the normalized (1, 50, 55) chunk."""
    K = M.config.num_steps
    shape = (1, M.config.n_action_steps, M.config.max_action_dim)
    x_t = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    a_hat = None
    for k in range(K):
        t_val = 1.0 - k / K
        t = torch.tensor(t_val, dtype=dtype, device=device)
        v = M.predict_velocity(state, p_pad, past_kv, x_t, t.expand(1),
                               prefix_position_ids=p_pos)
        a_hat = x_t - t * v
        if k < K - 1:
            nt = 1.0 - (k + 1) / K
            z = torch.randn(shape, generator=gen, device=device, dtype=dtype)
            x_t = nt * z + (1 - nt) * a_hat
    return a_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--task", default="click_bell")
    ap.add_argument("--out", default="actor_lora")
    ap.add_argument("--alpha_q", type=float, default=0.2,
                    help="TD3+BC style weight of the normalized Q term relative "
                         "to the (spread-normalized) BC anchor")
    ap.add_argument("--bc_scale", type=float, default=2e-4,
                    help="typical BC value (2*sigma^2 of the policy's own real-dim "
                         "spread); bc/bc_scale ~ 1 inside the data spread")
    ap.add_argument("--q_cap", type=float, default=1.0,
                    help="semantic ceiling of the value (discounted success prob)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout-eps", type=int, default=15)
    ap.add_argument("--max-decisions", type=int, default=0, help="smoke test cap")
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

    expert = M.qwenvl_with_expert.qwen_expert
    inject_adapter_in_model(
        LoraConfig(r=cfg.rank, lora_alpha=2 * cfg.rank,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
        expert)
    lora_params = []
    for n, p in srv.vla.named_parameters():
        if "lora" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
            lora_params.append(p)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"LoRA params: {n_lora/1e6:.2f}M over {len(lora_params)} tensors", flush=True)

    ck = torch.load(cfg.critic, map_location="cpu", weights_only=False)
    critic = TwinCritic(*ck["dims"]).to(device).float().eval()
    critic.load_state_dict(ck["model"])
    for p in critic.parameters():
        p.requires_grad_(False)
    cstats = {k: (torch.tensor(m, dtype=torch.float32, device=device),
                  torch.tensor(s, dtype=torch.float32, device=device))
              for k, (m, s) in ck["stats"].items()}

    index = build_index(cfg.rollouts, cfg.task)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(index))
    hold = set(order[: cfg.holdout_eps].tolist())
    train_eps = [i for i in range(len(index)) if i not in hold]
    print(f"{cfg.task}: {len(index)} episodes with images -> "
          f"train {len(train_eps)} / probe {len(hold)}", flush=True)

    # affine unnormalization bridge, verified on 3 different decisions
    Wb = None
    for ei in list(hold)[:3]:
        f, meta, n = index[ei]
        with np.load(f) as z:
            inp = srv._prepare_model_input(make_obs(z, meta, 0))
        W, b, err = build_unnorm_map(
            srv, inp, M.config.n_action_steps, M.config.max_action_dim)
        if Wb is None:
            Wb = (W, b)
        elif np.abs(W - Wb[0]).max() > 1e-6 or np.abs(b - Wb[1]).max() > 1e-6:
            raise RuntimeError("unnormalization map varies across samples")
    W = torch.tensor(Wb[0], dtype=torch.float32, device=device)
    b = torch.tensor(Wb[1], dtype=torch.float32, device=device)
    # BC only on the dims that map to real joints: the other 41 of 55 are
    # padding, the model's output there is unconstrained noise, and averaging
    # it into the BC loss both inflates it (~0.87 vs ~0.1) and pollutes the
    # gradient with a signal about nothing.
    real_dims = torch.tensor((np.abs(Wb[0]).sum(1) > 1e-9), device=device)
    print(f"unnorm bridge verified (probe err {err:.2e}), output dim {W.shape[1]}, "
          f"BC over {int(real_dims.sum())}/{len(real_dims)} dims", flush=True)

    def critic_q(chunk_norm_f32, joints_np, sp_np):
        """Pessimistic min(Q1,Q2): with ~1k transitions the critic extrapolates
        loosely (raw Q was seen reaching 2.0 on a [0,1]-target scale when the
        actor pushed off-data), and the min is the standard damper."""
        a_exec = (chunk_norm_f32[0, :25].to(torch.float32) @ W + b).reshape(1, -1)
        a_std = (a_exec - cstats["a"][0]) / cstats["a"][1]
        j = (torch.tensor(joints_np, dtype=torch.float32, device=device)[None]
             - cstats["j"][0]) / cstats["j"][1]
        sp = (torch.tensor(sp_np, dtype=torch.float32, device=device)[None]
              - cstats["sp"][0]) / cstats["sp"][1]
        q1, q2 = critic(a_std, j, sp)
        return torch.minimum(q1, q2)

    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=0.0)
    gen = torch.Generator(device=device)

    def probe(tag):
        qs, q_logged = [], []
        with torch.no_grad():
            for ei in sorted(hold):
                f, meta, n = index[ei]
                with np.load(f) as z:
                    for k in range(min(n, 3)):     # first decisions matter most
                        inp = srv._prepare_model_input(make_obs(z, meta, k))
                        p_pad, p_pos, past_kv = compute_prefix(M, inp, device, dtype)
                        state = inp["state"].unsqueeze(0).to(dtype=dtype, device=device)
                        gen.manual_seed(10_000 + ei * 100 + k)  # fixed probe noise
                        a_hat = vine_generate(M, state, p_pad, p_pos, past_kv,
                                              device, dtype, gen)
                        sp = z["intro_h_query_tokens"][k].reshape(-1).astype(np.float32)
                        j = z["decision_states"][k].astype(np.float32)
                        qs.append(float(critic_q(a_hat, j, sp)))
                        logged = torch.tensor(z["predicted_chunks_normalized"][k],
                                              device=device, dtype=torch.float32)[None]
                        q_logged.append(float(critic_q(logged, j, sp)))
        print(f"[probe {tag}] Q(vine)={np.mean(qs):.4f}  Q(logged)={np.mean(q_logged):.4f}"
              f"  n={len(qs)}", flush=True)
        return np.mean(qs)

    probe("init")
    step = 0
    q_ema = 0.25
    bc_scale = cfg.bc_scale
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        ep_order = rng.permutation(train_eps)
        done = 0
        for ei in ep_order:
            f, meta, n = index[ei]
            with np.load(f) as z:
                for k in range(n):
                    inp = srv._prepare_model_input(make_obs(z, meta, k))
                    p_pad, p_pos, past_kv = compute_prefix(M, inp, device, dtype)
                    state = inp["state"].unsqueeze(0).to(dtype=dtype, device=device)
                    a_hat = vine_generate(M, state, p_pad, p_pos, past_kv, device, dtype)
                    sp = z["intro_h_query_tokens"][k].reshape(-1).astype(np.float32)
                    j = z["decision_states"][k].astype(np.float32)
                    q = critic_q(a_hat, j, sp)
                    # Q is a discounted success probability; values above 1 are
                    # semantically impossible and only reachable by exploiting
                    # the critic (observed: held-out Q hit 1.41). Capping stops
                    # the gradient from paying for impossible value.
                    q = torch.clamp(q, max=cfg.q_cap)
                    target = torch.tensor(z["predicted_chunks_normalized"][k],
                                          device=device, dtype=torch.float32)[None]
                    bc = ((a_hat.to(torch.float32) - target) ** 2)[..., real_dims].mean()
                    # TD3+BC balance: the policy's own spread on real dims is
                    # tiny (bc ~1e-4), so a fixed alpha either vanishes against
                    # Q or crushes it. Normalizing Q by its running magnitude
                    # keeps the two gradients at a fixed ratio regardless of
                    # where the critic's scale drifts.
                    q_ema = 0.95 * q_ema + 0.05 * abs(float(q))
                    lam = cfg.alpha_q / (q_ema + 1e-3)
                    loss = (-lam * q.squeeze() + bc / bc_scale) / cfg.accum
                    loss.backward()
                    step += 1
                    done += 1
                    if step % cfg.accum == 0:
                        gn = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if (step // cfg.accum) % 10 == 0:
                            dt_s = (time.time() - t0) / done
                            print(f"ep{epoch} {done}/? q={float(q):.4f} "
                                  f"bc={float(bc):.5f} lam={lam:.2f} "
                                  f"gnorm={float(gn):.3f} {dt_s:.2f}s/dec", flush=True)
                    if cfg.max_decisions and done >= cfg.max_decisions:
                        break
                if cfg.max_decisions and done >= cfg.max_decisions:
                    break
        probe(f"epoch{epoch}")
        sd = {n_: p.detach().cpu() for n_, p in srv.vla.named_parameters() if "lora" in n_}
        path = f"{cfg.out}_{cfg.task}_ep{epoch}.pt"
        torch.save({"lora": sd, "cfg": vars(cfg), "epoch": epoch}, path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
