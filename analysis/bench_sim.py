"""Measure the simulator-side cost of a RoboTwin episode on this box.

Care is taken to separate one-time costs from steady state: the first scene setup
pays warp/curobo kernel compilation and the first render pays Vulkan shader
compilation, both of which are irrelevant to per-episode throughput but large
enough to swamp a naive average.
"""

import statistics
import sys
import time

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

import numpy as np  # noqa: E402

sys.path.insert(0, "./script")
from eval_lingbot_vla import build_env_args, make_task_env  # noqa: E402

TASK = sys.argv[1] if len(sys.argv) > 1 else "place_empty_cup"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "demo_clean"

args, _ = build_env_args(TASK, CONFIG)
args["eval_mode"] = True
args["eval_video_log"] = False
args["render_freq"] = 0

env = make_task_env(TASK)
print(f"task={TASK}  config={CONFIG}", flush=True)

setup_times, expert_times = [], []
for i in range(3):
    t0 = time.perf_counter()
    env.setup_demo(now_ep_num=i, seed=100000 + i, is_test=True, **args)
    setup_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    env.play_once()
    expert_times.append(time.perf_counter() - t0)
    solved = bool(env.plan_success and env.check_success())

    if i == 0:
        # First iteration also warms the renderer, so the shader compile does not
        # land inside the get_obs timings below.
        env.get_obs()
    env.close_env()
    print(f"  iter {i}: setup {setup_times[-1]:.2f}s  expert {expert_times[-1]:.2f}s  solved={solved}",
          flush=True)

print(f"\nstep_lim               : {env.step_lim}")
print(f"setup_demo  first/rest : {setup_times[0]:.2f}s / {statistics.median(setup_times[1:]):.2f}s")
print(f"expert play first/rest : {expert_times[0]:.2f}s / {statistics.median(expert_times[1:]):.2f}s")

# Steady-state per-step costs inside the policy loop.
env.setup_demo(now_ep_num=99, seed=100000, is_test=True, **args)
env.set_instruction(instruction="benchmark")

obs_times = []
for _ in range(12):
    t0 = time.perf_counter()
    obs = env.get_obs()
    obs_times.append(time.perf_counter() - t0)
head = obs["observation"]["head_camera"]["rgb"]
obs_ms = statistics.median(obs_times) * 1000
print(f"head camera rgb        : {head.shape} {head.dtype}")
print(f"get_obs first          : {obs_times[0] * 1000:.0f} ms")
print(f"get_obs median (3 cams): {obs_ms:.0f} ms")

state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
act_times = []
for i in range(40):
    a = state.copy()
    a[:6] += 0.003 * (i + 1)
    a[7:13] += 0.003 * (i + 1)
    t0 = time.perf_counter()
    env.take_action(a)
    act_times.append(time.perf_counter() - t0)
act_ms = statistics.median(act_times) * 1000
print(f"take_action median     : {act_ms:.1f} ms  (p90 {np.percentile(act_times, 90) * 1000:.1f} ms)")
env.close_env()

exec_h = 25
per_decision_sim = obs_ms / 1000 + exec_h * act_ms / 1000
n_dec = env.step_lim / exec_h
print(f"\nper decision (sim only, exec_horizon={exec_h}) : {per_decision_sim:.2f} s")
print(f"worst-case full episode, sim only            : {n_dec * per_decision_sim:.0f} s "
      f"({n_dec:.0f} decisions x {env.step_lim} steps)")
print(f"+ expert check & 2x scene setup              : "
      f"~{statistics.median(expert_times[1:]) + 2 * statistics.median(setup_times[1:]):.0f} s per episode")
