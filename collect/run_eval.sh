#!/usr/bin/env bash
# One-command closed-loop RoboTwin eval on a single GPU.
#
# Starts the LingBot-VLA 2.0 websocket policy server (env: lingbotvla), waits for
# it to load, then runs the rollout collector (env: robotwin) against it. The two
# live in different conda envs on purpose - torch 2.8/py3.12 for the policy,
# torch 2.4/py3.10 + SAPIEN for the simulator - and only ever talk over the socket.
#
#   bash run_eval.sh --task_name place_empty_cup --test_num 50
set -euo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

task_name=""
task_config=demo_clean
test_num=50
port=9330
exec_horizon=25
use_length=50
use_compile=False
instruction_type=unseen
render_freq=""
video=--eval_video_log
seed=0
output_dir="$ROOT/rollouts"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name)        task_name="$2"; shift 2 ;;
        --task_config)      task_config="$2"; shift 2 ;;
        --test_num)         test_num="$2"; shift 2 ;;
        --port)             port="$2"; shift 2 ;;
        --exec_horizon)     exec_horizon="$2"; shift 2 ;;
        --use_length)       use_length="$2"; shift 2 ;;
        --use_compile)      use_compile="$2"; shift 2 ;;
        --instruction_type) instruction_type="$2"; shift 2 ;;
        --render_freq)      render_freq="$2"; shift 2 ;;
        --seed)             seed="$2"; shift 2 ;;
        --output_dir)       output_dir="$2"; shift 2 ;;
        --no_video)         video=""; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
[ -n "$task_name" ] || { echo "--task_name is required" >&2; exit 1; }

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/${task_name}_${task_config}_${stamp}"
mkdir -p "$log_dir"
server_log="$log_dir/policy_server.log"
client_log="$log_dir/rollout.log"

server_pid=""
cleanup() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        echo "stopping policy server ($server_pid)"
        kill -TERM "$server_pid" 2>/dev/null || true
        sleep 3
        kill -KILL "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# A 24GB 4090 fits exactly one policy server (~12.6GB) plus one simulator
# (~5.4GB). Starting a second one dies with a bare CUDA OOM traceback several
# minutes into weight loading, so refuse up front with an actionable message.
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mib" -lt 19000 ]; then
    echo "ERROR: only ${free_mib} MiB of GPU memory free; a run needs ~19000 MiB." >&2
    echo "Something else is already using the GPU:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    echo "Wait for it to finish, or stop it first." >&2
    exit 1
fi

echo "=== starting policy server (log: $server_log) ==="
cd "$REPO"
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" \
CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" \
        --use_length "$use_length" \
        --use_bf16 True \
        --use_compile "$use_compile" \
        --port "$port" > "$server_log" 2>&1 &
server_pid=$!
echo "policy server pid $server_pid"

# The client retries the connection for 30 min on its own, but fail fast here if
# the server dies during weight loading rather than waiting that out.
for _ in $(seq 1 240); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "policy server exited during startup; tail of $server_log:" >&2
        tail -30 "$server_log" >&2
        exit 1
    fi
    grep -q "Model initialized" "$server_log" && break
    sleep 5
done

echo "=== running rollouts (log: $client_log) ==="
cd "$ROOT/RoboTwin"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
        --task_name "$task_name" \
        --task_config "$task_config" \
        --port "$port" \
        --test_num "$test_num" \
        --exec_horizon "$exec_horizon" \
        --instruction_type "$instruction_type" \
        --seed "$seed" \
        --output_dir "$output_dir" \
        ${render_freq:+--render_freq "$render_freq"} \
        $video 2>&1 | tee "$client_log"

echo "=== done. rollouts under $output_dir ==="
