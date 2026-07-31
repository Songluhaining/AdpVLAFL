"""Measure end-to-end policy latency as the simulator will experience it:
observation over the websocket -> action chunk back. Uses a synthetic observation
shaped exactly like RoboTwin's D435 output (240x320 RGB from three cameras).
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/data/whn/robotwin_eval/RoboTwin/script")
from msgpack_numpy_min import Packer, unpackb  # noqa: E402

import websockets.sync.client  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9331
N_WARMUP, N_ITERS = 3, 12

packer = Packer()
# proxy=None: websockets>=14 honours ALL_PROXY/HTTPS_PROXY, and this box exports a
# SOCKS proxy, so a loopback connect would otherwise be tunnelled and fail.
ws = websockets.sync.client.connect(
    f"ws://127.0.0.1:{PORT}", compression=None, max_size=None, ping_interval=None, proxy=None
)
unpackb(ws.recv())


def call(obs):
    ws.send(packer.pack(obs))
    resp = ws.recv()
    if isinstance(resp, str):
        raise RuntimeError(resp)
    return unpackb(resp)


call(dict(reset=True, robo_name="robotwin"))

rng = np.random.default_rng(0)
obs = {
    "observation.images.cam_high": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
    "observation.images.cam_left_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
    "observation.images.cam_right_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
    "observation.state": rng.standard_normal(14).astype(np.float32),
    "task": "Pick up the cup and place it on the plate.",
}

for _ in range(N_WARMUP):
    call(obs)

lat = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    resp = call(obs)
    lat.append((time.perf_counter() - t0) * 1000)

chunk = np.asarray(resp["action"])
lat = np.array(lat)
print(f"action chunk shape : {chunk.shape}  dtype={chunk.dtype}")
print(f"chunk finite       : {np.isfinite(chunk).all()}  range=[{chunk.min():.3f}, {chunk.max():.3f}]")
print(f"latency mean       : {lat.mean():.1f} ms")
print(f"latency p50 / p90  : {np.percentile(lat, 50):.1f} / {np.percentile(lat, 90):.1f} ms")
print(f"latency min / max  : {lat.min():.1f} / {lat.max():.1f} ms")
print(f"server-side infer  : {resp.get('server_timing', {}).get('infer_ms', float('nan')):.1f} ms")
ws.close()
