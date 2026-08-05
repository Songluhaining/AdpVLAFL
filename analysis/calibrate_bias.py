"""Calibrate the routing-bias scale on one real recorded decision.

The correction target is a delta of expert-load shares (median ~0.005), but it
is added to the router's sigmoid *scores*, whose spacing is unknown. Guessing a
scale risks either a no-op (selections never flip) or a lobotomy (routing
scrambled everywhere). So: replay one failing decision's exact observation with
the noise pinned, sweep the scale, and measure two things --

  moved    how much the realized expert load changed at all
  aligned  how much of that movement is along the intended direction,
           i.e. the correction target delta

The scale to use is the smallest one that moves routing clearly *and* mostly in
the intended direction. Alignment collapsing while movement grows is the
signature of the bias overpowering the router rather than steering it.
"""

import glob
import json
import sys

import numpy as np
import websockets.sync.client

sys.path.insert(0, "/data/whn/robotwin_eval/RoboTwin/script")
from msgpack_numpy_min import Packer, unpackb

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9450
TASK = "click_bell"

scenes = json.load(open(f"/data/whn/robotwin_eval/correction/{TASK}_scenes.json"))
bias_bank = np.load(f"/data/whn/robotwin_eval/correction/{TASK}_bias.npz")
seed = scenes["A"][0]
delta = bias_bank[str(seed)].astype(np.float32)          # (36, 32)

ep_file = None
for f in glob.glob(f"/data/whn/robotwin_eval/rollouts/{TASK}_*/episodes/*seed{seed}_*.npz"):
    ep_file = f
meta = json.load(open(ep_file.replace(".npz", ".json")))
z = np.load(ep_file)
obs_base = {
    "observation.images.cam_high": z["decision_images_head_camera"][0],
    "observation.images.cam_left_wrist": z["decision_images_left_camera"][0],
    "observation.images.cam_right_wrist": z["decision_images_right_camera"][0],
    "observation.state": z["decision_states"][0].astype(np.float32),
    "task": meta["instruction"],
    "return_introspect": True,
    "noise_seed": 12345,
}
print(f"scene seed {seed} ({TASK}), decision 0, |delta| max {np.abs(delta).max():.4f}")

packer = Packer()
ws = websockets.sync.client.connect(f"ws://127.0.0.1:{PORT}", compression=None,
                                    max_size=None, ping_interval=None, proxy=None)
unpackb(ws.recv())


def call(extra):
    ws.send(packer.pack({**obs_base, **extra}))
    r = unpackb(ws.recv())
    rc = r["_introspect_router_counts"].astype(np.float32)   # (10, 36, 32)
    p = rc.mean(0)
    return p / (p.sum(-1, keepdims=True) + 1e-8), np.asarray(r["action"])


ws.send(packer.pack(dict(reset=True, robo_name="robotwin")))
unpackb(ws.recv())

p0, a0 = call({})
# repeat: the bf16 wobble floor for both metrics
p0b, _ = call({})
floor = np.abs(p0 - p0b).sum(-1).mean() / 2

dhat = delta / (np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-9)
print(f"\n{'scale':>8}{'moved(选择变动比例)':>22}{'aligned(沿目标方向)':>22}{'动作变化':>10}")
print(f"{'0(底噪)':>8}{floor:>22.4f}{'-':>22}{'-':>10}")
tv_target = float(np.abs(delta).sum(-1).mean() / 2)
print(f'目标移动量(|delta| TV): {tv_target:.4f}')
for scale in (0.01, 0.03, 0.1, 0.3, 1, 3):
    p1, a1 = call({"routing_bias": delta * scale})
    moved = np.abs(p1 - p0).sum(-1).mean() / 2               # per-layer TV distance
    shift = p1 - p0
    num = (shift * dhat).sum(-1)
    den = np.linalg.norm(shift, axis=-1) + 1e-9
    aligned = float((num / den).mean())                       # mean cosine per layer
    da = float(np.abs(a1 - a0).max())
    print(f"{scale:>8}{moved:>22.4f}{aligned:>22.3f}{da:>10.3f}")
ws.close()
print("\n选 moved 明显高于底噪、aligned 仍接近其峰值的最小 scale。")
