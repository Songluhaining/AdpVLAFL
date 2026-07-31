"""Closed-loop RoboTwin 2.0 evaluation of a LingBot-VLA 2.0 websocket policy server,
with full per-episode trajectory logging.

Derived from RoboTwin's own ``script/eval_policy_client.py``. Two things differ:

1. Transport. RoboTwin's client speaks a bespoke length-prefixed JSON/base64
   socket protocol; LingBot's deploy server speaks msgpack over websockets and
   expects observations keyed by the *canonical* feature names declared in
   ``configs/robot_configs/<robo_name>.yaml``. So we talk to it directly.

2. Logging. Every episode is dumped to an .npz + .json pair holding the states,
   the full predicted action chunks, the actions actually executed, per-decision
   camera frames, inference latency and the success/failure outcome - the raw
   material for analysing where the policy's decisions go wrong.

Run from the RoboTwin repo root, in the simulation conda env:

    python script/eval_lingbot_vla.py --task_name place_empty_cup \
        --task_config demo_clean --port 9330 --test_num 50 \
        --output_dir /data/whn/robotwin_eval/rollouts
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Canonical observation keys the LingBot feature transform expects. They are the
# `origin_keys` side of configs/robot_configs/robotwin.yaml, so if you evaluate
# with a different --robo_name check that file first.
IMAGE_KEYS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}
STATE_KEY = "observation.state"
RAW_ACTION_KEY = "action"  # what robotwin.yaml's reverse mapping produces
ARM_ACTION_KEY = "action.arm.position"
GRIPPER_ACTION_KEY = "action.effector.position"


# --------------------------------------------------------------------------- #
# websocket policy client (mirrors deploy/websocket_client_policy.py)
# --------------------------------------------------------------------------- #
class PolicyClient:
    def __init__(self, host="127.0.0.1", port=9330, connect_timeout=1800):
        import websockets.sync.client

        from msgpack_numpy_min import Packer, unpackb

        self._packer = Packer()
        self._unpackb = unpackb
        uri = f"ws://{host}:{port}"
        deadline = time.time() + connect_timeout
        while True:
            try:
                # proxy=None matters here: websockets>=14 reads ALL_PROXY/HTTPS_PROXY
                # from the environment, and this machine exports a SOCKS proxy, so a
                # loopback connect would be tunnelled through it and fail the handshake.
                self._ws = websockets.sync.client.connect(
                    uri, compression=None, max_size=None,
                    ping_interval=None, ping_timeout=None, proxy=None,
                )
                self._metadata = self._unpackb(self._ws.recv())
                print(f"connected to policy server at {uri}", flush=True)
                return
            except (ConnectionRefusedError, OSError) as exc:
                if time.time() > deadline:
                    raise ConnectionError(f"policy server at {uri} never came up: {exc}")
                print("waiting for policy server ...", flush=True)
                time.sleep(5)

    def infer(self, obs):
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"policy server error:\n{response}")
        return self._unpackb(response)

    def reset(self, robo_name):
        return self.infer(dict(reset=True, robo_name=robo_name))

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# observation <-> action plumbing
# --------------------------------------------------------------------------- #
def encode_obs(observation, instruction):
    """RoboTwin observation dict -> LingBot canonical observation dict."""
    obs = {}
    for cam_name, key in IMAGE_KEYS.items():
        obs[key] = np.ascontiguousarray(observation["observation"][cam_name]["rgb"])
    obs[STATE_KEY] = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    obs["task"] = instruction
    return obs


def decode_action_chunk(response):
    """Server response -> (chunk_len, 14) RoboTwin qpos actions.

    FeatureTransform.unapply reverses the robot config all the way back to the
    dataset's *origin* keys, so for configs/robot_configs/robotwin.yaml the server
    already hands back a single ``action`` of 14 dims laid out exactly as
    RoboTwin's ``take_action`` wants it: [left arm 6 | left gripper | right arm 6 |
    right gripper]. The split arm/effector branch below is the fallback for robot
    configs whose canonical features do not recombine into one origin key.
    """
    if RAW_ACTION_KEY in response:
        chunk = np.asarray(response[RAW_ACTION_KEY], dtype=np.float32)
        return chunk[None] if chunk.ndim == 1 else chunk

    arm = np.asarray(response[ARM_ACTION_KEY], dtype=np.float32)
    gripper = np.asarray(response[GRIPPER_ACTION_KEY], dtype=np.float32)
    if arm.ndim == 1:
        arm, gripper = arm[None], gripper[None]
    return np.concatenate(
        [arm[:, 0:6], gripper[:, 0:1], arm[:, 6:12], gripper[:, 1:2]], axis=-1
    )


# --------------------------------------------------------------------------- #
# per-episode recorder
# --------------------------------------------------------------------------- #
class EpisodeRecorder:
    """Accumulates everything observed during one rollout."""

    def __init__(self, save_images=True):
        self.save_images = save_images
        self.decisions = []  # one entry per policy call
        self.states = []  # env state before every executed action
        self.executed = []  # action actually sent to take_action
        self.decision_of_step = []  # which decision produced each executed action
        self.endposes = []  # [left xyz+quat(7) | left grip | right xyz+quat(7) | right grip]

    def add_decision(self, step, obs, chunk, latency, normalized=None, introspect=None):
        entry = {
            "step": step,
            "state": np.asarray(obs[STATE_KEY], dtype=np.float32),
            "chunk": np.asarray(chunk, dtype=np.float32),
            "latency_s": float(latency),
        }
        if normalized is not None:
            entry["chunk_normalized"] = np.asarray(normalized, dtype=np.float32)
        if introspect:
            # Model internals for the failure-localization model: one observation
            # per component per decision. Kept at their native precision (fp16 /
            # int16) so a 60-decision episode stays a few MB.
            for k, v in introspect.items():
                entry[f"intro_{k}"] = v
        if self.save_images:
            for cam_name, key in IMAGE_KEYS.items():
                entry[f"image_{cam_name}"] = np.asarray(obs[key], dtype=np.uint8)
        self.decisions.append(entry)

    def add_step(self, state, action, decision_idx, endpose=None):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.executed.append(np.asarray(action, dtype=np.float32))
        self.decision_of_step.append(decision_idx)
        if endpose is not None:
            self.endposes.append(np.asarray(endpose, dtype=np.float32))

    def dump(self, out_dir, meta):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"episode_{meta['episode_idx']:04d}_seed{meta['seed']}_{'success' if meta['success'] else 'fail'}"

        arrays = {
            "states": np.stack(self.states) if self.states else np.zeros((0, 14), np.float32),
            "executed_actions": np.stack(self.executed) if self.executed else np.zeros((0, 14), np.float32),
            "decision_of_step": np.asarray(self.decision_of_step, dtype=np.int32),
            "decision_steps": np.asarray([d["step"] for d in self.decisions], dtype=np.int32),
            "decision_states": (
                np.stack([d["state"] for d in self.decisions]) if self.decisions else np.zeros((0, 14), np.float32)
            ),
            "predicted_chunks": (
                np.stack([d["chunk"] for d in self.decisions]) if self.decisions else np.zeros((0, 0, 14), np.float32)
            ),
            "latencies_s": np.asarray([d["latency_s"] for d in self.decisions], dtype=np.float32),
        }
        if self.endposes:
            arrays["endposes"] = np.stack(self.endposes)
        if self.decisions and "chunk_normalized" in self.decisions[0]:
            arrays["predicted_chunks_normalized"] = np.stack([d["chunk_normalized"] for d in self.decisions])
        if self.save_images and self.decisions:
            for cam_name in IMAGE_KEYS:
                arrays[f"decision_images_{cam_name}"] = np.stack(
                    [d[f"image_{cam_name}"] for d in self.decisions]
                )
        # Introspection fields, stacked over decisions. Collected by key union
        # rather than from decisions[0] so a decision that skipped capture (or a
        # field the server did not emit) cannot silently drop the whole array.
        if self.decisions:
            intro_keys = sorted({k for d in self.decisions for k in d if k.startswith("intro_")})
            for k in intro_keys:
                vals = [d.get(k) for d in self.decisions]
                if any(v is None for v in vals) or len({np.shape(v) for v in vals}) != 1:
                    print(f"  [warn] ragged/missing introspection field {k}, skipped", flush=True)
                    continue
                arrays[k] = np.stack(vals)

        np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return stem


# --------------------------------------------------------------------------- #
# environment setup (follows RoboTwin's eval_policy_client.main)
# --------------------------------------------------------------------------- #
def build_env_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = None

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def embodiment_file(name):
        path = embodiment_types[name]["file_path"]
        if path is None:
            raise ValueError(f"no embodiment file for {name}")
        return path

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    with open(os.path.join(args["left_robot_file"], "config.yml"), "r", encoding="utf-8") as f:
        args["left_embodiment_config"] = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(args["right_robot_file"], "config.yml"), "r", encoding="utf-8") as f:
        args["right_embodiment_config"] = yaml.load(f.read(), Loader=yaml.FullLoader)

    return args, camera_config[head_camera_type]


def make_task_env(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def find_ffmpeg():
    """Locate ffmpeg without relying on the conda env being activated.

    ffmpeg is installed into this env's bin/, but launching the interpreter by
    absolute path (as run_eval.sh does) leaves that directory off PATH, so a bare
    "ffmpeg" lookup fails. Prefer the one sitting next to our own interpreter.
    """
    candidate = Path(sys.executable).with_name("ffmpeg")
    if candidate.is_file():
        return str(candidate)
    from shutil import which

    return which("ffmpeg")


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def run_episode(TASK_ENV, client, args, cfg, recorder, episode_seed):
    """Drive one episode to termination. Returns (success, n_steps)."""
    instruction = TASK_ENV.get_instruction()
    client.reset(cfg.robo_name)

    decision_idx = -1
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation, instruction)
        if cfg.log_normalized:
            obs["return_normalized"] = True
        if cfg.log_introspect:
            obs["return_introspect"] = True
        if cfg.fixed_noise:
            # Derived from (scene seed, decision index) so two runs of the same
            # scene draw the same noise even when a changed setting alters how
            # many decisions the episode takes.
            obs["noise_seed"] = int(episode_seed) * 1000003 + decision_idx + 1

        t0 = time.time()
        response = client.infer(obs)
        latency = time.time() - t0

        chunk = decode_action_chunk(response)
        intro = {k[len("_introspect_"):]: v for k, v in response.items()
                 if k.startswith("_introspect_")}
        decision_idx += 1
        recorder.add_decision(
            TASK_ENV.take_action_cnt,
            obs,
            chunk,
            latency,
            normalized=response.get("_normalized_actions"),
            introspect=intro,
        )

        n_exec = min(cfg.exec_horizon, len(chunk)) if cfg.exec_horizon > 0 else len(chunk)
        for action in chunk[:n_exec]:
            if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
                break
            # Read joint state straight off the robot rather than via get_obs():
            # get_obs() re-renders all three cameras, which would roughly double
            # the cost of every simulation step just to log 14 numbers.
            state_before = np.asarray(
                TASK_ENV.robot.get_left_arm_jointState() + TASK_ENV.robot.get_right_arm_jointState(),
                dtype=np.float32,
            )
            endpose = None
            if cfg.log_endpose:
                endpose = np.concatenate(
                    [
                        np.asarray(TASK_ENV.get_arm_pose("left"), dtype=np.float32),
                        np.asarray(TASK_ENV.get_arm_pose("right"), dtype=np.float32),
                    ]
                )
            TASK_ENV.take_action(action)
            recorder.add_step(state_before, action, decision_idx, endpose=endpose)

    return bool(TASK_ENV.eval_success), TASK_ENV.take_action_cnt


def main(cfg):
    args, head_camera_cfg = build_env_args(cfg.task_name, cfg.task_config)
    args["policy_name"] = "LingBotVLA2"
    args["eval_mode"] = True
    args["eval_video_log"] = cfg.eval_video_log
    if cfg.render_freq is not None:
        # >0 opens a live SAPIEN viewer (needs DISPLAY). take_action() drives
        # viewer.render() itself, so nothing else has to change - but it does slow
        # the rollout down, so leave it off for bulk collection.
        args["render_freq"] = cfg.render_freq

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{cfg.task_name}_{cfg.task_config}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}", flush=True)

    video_size = f"{head_camera_cfg['w']}x{head_camera_cfg['h']}"
    ffmpeg_bin = None
    if cfg.eval_video_log:
        ffmpeg_bin = find_ffmpeg()
        if ffmpeg_bin is None:
            # Losing the whole run to a missing encoder would be absurd; the npz
            # trajectories are the actual deliverable.
            print("WARNING: ffmpeg not found, continuing without video", flush=True)
            cfg.eval_video_log = False
            args["eval_video_log"] = False
        else:
            video_dir = run_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            args["eval_video_save_dir"] = video_dir

    TASK_ENV = make_task_env(cfg.task_name)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    client = PolicyClient(host=cfg.host, port=cfg.port)

    clear_cache_freq = args["clear_cache_freq"]
    now_seed = 100000 * (1 + cfg.seed)
    succ_seed = 0
    now_id = 0
    summary = []

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2, ensure_ascii=False)

    while succ_seed < cfg.test_num:
        # RoboTwin only evaluates on seeds the scripted expert can solve, so that
        # a policy failure means the policy failed - not that the scene was
        # unsolvable. This replay is why each episode costs roughly 2x.
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
        except UnStableError as e:
            print(f"unstable seed {now_seed}: {e}", flush=True)
            TASK_ENV.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        except Exception:
            print(f"expert failed on seed {now_seed}:\n{traceback.format_exc()}", flush=True)
            TASK_ENV.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        succ_seed += 1
        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], cfg.test_num)
        instruction = np.random.choice(results[0][cfg.instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        ffmpeg = None
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    ffmpeg_bin, "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", video_size,
                    "-framerate", str(cfg.video_fps), "-i", "-",
                    "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        recorder = EpisodeRecorder(save_images=cfg.save_images)
        t_start = time.time()
        success, n_steps = run_episode(TASK_ENV, client, args, cfg, recorder, now_seed)
        wall_s = time.time() - t_start

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        meta = {
            "task_name": cfg.task_name,
            "task_config": cfg.task_config,
            "episode_idx": now_id,
            "seed": int(now_seed),
            "instruction": str(instruction),
            "instruction_type": cfg.instruction_type,
            "success": bool(success),
            "steps": int(n_steps),
            "step_limit": int(TASK_ENV.step_lim),
            "num_decisions": len(recorder.decisions),
            "exec_horizon": cfg.exec_horizon,
            "wall_seconds": round(wall_s, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        stem = recorder.dump(run_dir / "episodes", meta)

        TASK_ENV.suc += int(success)
        TASK_ENV.test_num += 1
        summary.append(meta)
        with open(run_dir / "summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        print(
            f"[{'SUCCESS' if success else 'FAIL'}] {stem} steps={n_steps} {wall_s:.1f}s\n"
            f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => "
            f"{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%, current seed: {now_seed}",
            flush=True,
        )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        now_seed += 1

    client.close()
    final = {
        "task_name": cfg.task_name,
        "task_config": cfg.task_config,
        "success": TASK_ENV.suc,
        "total": TASK_ENV.test_num,
        "success_rate": TASK_ENV.suc / max(TASK_ENV.test_num, 1),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"FINAL Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => "
          f"{round(TASK_ENV.suc / max(TASK_ENV.test_num,1) * 100, 1)}%", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", required=True)
    p.add_argument("--task_config", default="demo_clean")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9330)
    p.add_argument("--robo_name", default="robotwin",
                   help="picks configs/robot_configs/<robo_name>.yaml on the policy server")
    p.add_argument("--instruction_type", default="unseen", choices=["seen", "unseen"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_num", type=int, default=50)
    p.add_argument("--exec_horizon", type=int, default=25,
                   help="actions executed per policy call; <=0 means the whole chunk")
    p.add_argument("--output_dir", default="/data/whn/robotwin_eval/rollouts")
    p.add_argument("--eval_video_log", action="store_true")
    p.add_argument("--video_fps", type=int, default=10)
    p.add_argument("--render_freq", type=int, default=None,
                   help="override the task config's render_freq; >0 opens a live SAPIEN "
                        "viewer on $DISPLAY (slower - for watching, not for bulk runs)")
    p.add_argument("--save_images", action="store_true", default=True)
    p.add_argument("--no_save_images", dest="save_images", action="store_false")
    p.add_argument("--log_normalized", action="store_true", default=True,
                   help="also record the pre-unnormalization action chunk")
    p.add_argument("--fixed_noise", action="store_true", default=False,
                   help="pin the sampling noise from (scene seed, decision index); "
                        "required for any intervention comparison to be attributable")
    p.add_argument("--log_introspect", action="store_true", default=True,
                   help="record model internals (noise, prefix readouts, MoE routing, "
                        "denoising trajectory) for failure localization")
    p.add_argument("--no_log_introspect", dest="log_introspect", action="store_false")
    p.add_argument("--log_endpose", action="store_true", default=True,
                   help="also record both end-effector poses at every executed step")
    p.add_argument("--no_log_endpose", dest="log_endpose", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main(parse_args())
