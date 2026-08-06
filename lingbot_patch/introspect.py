"""Capture of inference-time internals for causal failure localization.

The Bayesian network that consumes this data needs one observed value per
component per decision. Those values already exist inside a normal forward pass
-- the prefix hidden states, the MoE router logits, the flow-matching trajectory
-- they are simply discarded on the way out. This module collects them.

Deliberately implemented as a side-channel registry rather than by widening
return signatures: `sample_actions` and `predict_velocity` are also reachable
from the training path, and threading extra outputs through every call site
would be both invasive and easy to get wrong. Capture is off by default and
costs nothing when disabled.

Everything stays on the GPU until `finalize`. A naive implementation that moves
each tensor to host as it is produced costs one device synchronization per MoE
layer per denoising step -- 360 syncs per decision -- which measured at +58%
inference latency. Batching the transfer removes essentially all of that.

Everything recorded here is computable from the model's own forward pass, with
no privileged simulator state, so the same instrumentation transfers unchanged
to a real robot.
"""

from contextlib import contextmanager

import torch

_STATE = {"enabled": False, "buf": None, "noise_seed": None, "sampler": "euler",
          "step_gen": None}

SAMPLERS = ("euler", "vine")


def enabled() -> bool:
    return _STATE["enabled"]


def set_noise_seed(seed):
    """Pin the flow-matching noise for the next decision, or None to free it.

    The sampling noise is the only stochastic input to a decision, so without
    controlling it two runs of the same scene diverge and no intervention can be
    attributed: "the outcome changed" would never distinguish the manipulated
    variable from a different random draw. Seeding rather than replaying a stored
    tensor keeps this usable when the intervention changes how many decisions an
    episode takes, which is exactly what varying exec_horizon does.
    """
    _STATE["noise_seed"] = seed
    # The per-step noise stream restarts with the decision, so replaying the
    # same (seed, sampler) is bit-reproducible even within one server process.
    _STATE["step_gen"] = None


def set_sampler(name):
    """Select the denoising sampler for the next decision: euler (default) or vine."""
    name = name or "euler"
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}, expected one of {SAMPLERS}")
    _STATE["sampler"] = name


def sampler() -> str:
    return _STATE["sampler"]


def make_noise(shape, device, dtype):
    if _STATE["noise_seed"] is None:
        return torch.randn(shape, device=device, dtype=dtype)
    g = torch.Generator(device=device)
    g.manual_seed(int(_STATE["noise_seed"]))
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


def make_step_noise(shape, device, dtype):
    """Fresh per-step noise for the VINE sampler (arXiv 2607.10369).

    VINE draws a new z_k at every denoising step. Under a pinned decision seed
    the stream must be (a) reproducible across replays and (b) distinct from the
    initial draw of make_noise. A single generator seeded by a splitmix-style
    scramble of the decision seed gives both: sequential draws differ per step,
    and set_noise_seed resets the stream at each decision boundary.
    """
    if _STATE["noise_seed"] is None:
        return torch.randn(shape, device=device, dtype=dtype)
    g = _STATE["step_gen"]
    if g is None:
        g = torch.Generator(device=device)
        g.manual_seed((int(_STATE["noise_seed"]) * 6364136223846793005
                       + 1442695040888963407) % (2 ** 63))
        _STATE["step_gen"] = g
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


@contextmanager
def capture():
    """Enable capture for one decision and yield the dict it fills."""
    prev_enabled, prev_buf = _STATE["enabled"], _STATE["buf"]
    buf = {"router_counts": [], "router_entropy": [], "denoise_x": [],
           "denoise_v_norm": [], "sel_pending": []}
    _STATE["enabled"], _STATE["buf"] = True, buf
    try:
        yield buf
    finally:
        _STATE["enabled"], _STATE["buf"] = prev_enabled, prev_buf


def _buf():
    return _STATE["buf"]


def record_noise(noise):
    if not _STATE["enabled"]:
        return
    _buf()["noise"] = (noise[0] if noise.ndim == 3 else noise).detach().to(torch.float16)


def record_prefix(hidden_states, visual_pos_masks, query_slice):
    """Pool the prefix hidden states into three group summaries.

    Image / language / distillation-query positions are pooled separately because
    they are the observable proxies for three different components: perception,
    language grounding, and the model's own depth+dynamics readout. Pooled
    vectors rather than scalars are stored so the reduction to a node value (a
    distance to the success-run distribution) stays a post-hoc choice.
    """
    if not _STATE["enabled"] or hidden_states is None:
        return
    h = hidden_states[0]  # (L, D), batch of 1 at eval
    b = _buf()

    vis = visual_pos_masks[0].bool() if visual_pos_masks is not None else None
    if vis is not None and vis.any():
        b["h_image"] = h[vis].mean(0).detach().to(torch.float16)
    if query_slice is not None:
        s, e = query_slice
        if e > s:
            q = h[s:e].detach()
            b["h_query"] = q.mean(0).to(torch.float16)
            b["h_query_tokens"] = q.to(torch.float16)  # per-token, 8 x D

    lang = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
    if vis is not None:
        lang &= ~vis
    if query_slice is not None:
        lang[query_slice[0] : query_slice[1]] = False
    if lang.any():
        b["h_lang"] = h[lang].mean(0).detach().to(torch.float16)


def record_layer_selection(selected_experts, num_experts):
    """True per-layer selection counts, called from inside the MoE block.

    The logits alone cannot reproduce the selection whenever a routing bias is
    active: the block picks top-k of scores *plus* e_score_correction_bias, so a
    counts summary recomputed from logits is blind to exactly the interventions
    this instrumentation exists to observe.
    """
    if not _STATE["enabled"]:
        return
    _buf()["sel_pending"].append(
        torch.bincount(selected_experts.reshape(-1), minlength=num_experts))


def record_router(router_logits_list, top_k=4):
    """Summarize one denoising step's MoE routing.

    Raw logits are 36 layers x ~51 tokens x 32 experts per denoising step, which
    at 10 steps x 60 decisions would dominate the episode on disk. The load
    histogram and the routing entropy retain what the routing node needs -- which
    experts fired and how peaked the choice was -- at a fraction of the size.
    """
    if not _STATE["enabled"] or not router_logits_list:
        return
    # All 36 layers at once. Looping in Python instead costs 36 iterations x ~6
    # kernel launches per denoising step, i.e. >2000 launches per decision, and
    # that Python/launch overhead — not the arithmetic — dominated the capture.
    lg = torch.stack([x.detach() for x in router_logits_list]).float()  # (L, T, E)
    n_layers, n_tokens, n_exp = lg.shape
    b = _buf()
    pend = b.pop("sel_pending", [])
    b["sel_pending"] = []
    if len(pend) == n_layers:
        # the block reported its actual (bias-inclusive) choices
        counts = torch.stack(pend).to(lg.dtype)
    else:
        k = min(top_k, n_exp)
        idx = lg.topk(k, dim=-1).indices.reshape(n_layers, -1)          # (L, T*k)
        counts = torch.zeros(n_layers, n_exp, device=lg.device, dtype=lg.dtype)
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=lg.dtype))

    logp = torch.log_softmax(lg, dim=-1)
    ent = -(logp.exp() * logp).sum(-1).mean(-1)                          # (L,)

    b["router_counts"].append(counts.to(torch.int16))
    b["router_entropy"].append(ent.to(torch.float16))


def record_denoise_step(x_t, v_t):
    if not _STATE["enabled"]:
        return
    b = _buf()
    b["denoise_x"].append((x_t[0] if x_t.ndim == 3 else x_t).detach().to(torch.float16))
    b["denoise_v_norm"].append(v_t.detach().float().norm())


def finalize(buf):
    """Stack on device, then make exactly one host transfer per field."""
    out = {}
    for k in ("noise", "h_image", "h_lang", "h_query", "h_query_tokens"):
        if k in buf:
            out[k] = buf[k]
    if buf["denoise_x"]:
        out["denoise_x"] = torch.stack(buf["denoise_x"])              # (steps, 50, 55)
        out["denoise_v_norm"] = torch.stack(buf["denoise_v_norm"])    # (steps,)
    if buf["router_counts"]:
        out["router_counts"] = torch.stack(buf["router_counts"])      # (steps, layers, experts)
        out["router_entropy"] = torch.stack(buf["router_entropy"])    # (steps, layers)
    return {k: v.cpu().numpy() for k, v in out.items()}
