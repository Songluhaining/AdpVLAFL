#!/usr/bin/env bash
# Final arbiter of the VINE-RL pilot: paired eval of the fine-tuned actor.
#
#   C  base policy, euler sampler          -- the deployed baseline
#   T  base + actor LoRA, vine sampler     -- generation matched to training
#
# Same labelled seed-11 scenes, same pinned noise stream as every prior A/B,
# so per-scene flips line up with the vine_ab and replicate baselines. Each
# arm gets its own resident server (the LoRA changes weights, so arms cannot
# share one process).
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
LORA=${LORA:?set LORA=/path/to/actor_lora_task_epN.pt}
TASK=${TASK:-click_bell}
PORT=${PORT:-9470}
N=${N:-50}
SEED=${SEED:-11}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/rl_ab_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"
echo "rl_ab $stamp | task=$TASK lora=$LORA N=$N seed=$SEED" | tee "$summary"
printf "%s\t%s\t%s\t%s\n" task arm success_over_total minutes | tee -a "$summary"

server_pid=""
kill_server() {
    # The policy server forks a child that inherits the GPU allocation, so
    # killing only the launched pid leaves a 12GB zombie. Match every process
    # carrying this port's command line (the fork shares it), never this shell.
    for pid in $(pgrep -f "lingbot_vla_v2_policy.*--port $PORT"); do
        kill -TERM "$pid" 2>/dev/null
    done
    sleep 4
    for pid in $(pgrep -f "lingbot_vla_v2_policy.*--port $PORT"); do
        kill -KILL "$pid" 2>/dev/null
    done
    server_pid=""
}
trap kill_server EXIT INT TERM

run_arm() {  # arm server_extra client_extra
    local arm=$1 server_extra=$2 client_extra=$3
    cd "$REPO" || exit 1
    QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
        "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
            --model_path "$CKPT" --use_length 50 --use_bf16 True \
            --use_compile False --port "$PORT" $server_extra \
            > "$log_dir/server_${arm}.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 120); do
        kill -0 "$server_pid" 2>/dev/null || { echo "server died (arm $arm)" | tee -a "$summary"; return 1; }
        grep -q "Model initialized" "$log_dir/server_${arm}.log" && break
        sleep 5
    done
    cd "$ROOT/RoboTwin" || exit 1
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
            --task_name "$TASK" --task_config demo_randomized --port "$PORT" \
            --test_num "$N" --seed "$SEED" --exec_horizon 25 \
            --instruction_type unseen --fixed_noise --noise_replicate 0 \
            --no_log_introspect --no_save_images $client_extra \
            --output_dir "$ROOT/rollouts_rl_ab/${TASK}_${arm}" \
            > "$log_dir/${TASK}_${arm}.log" 2>&1
    local t1=$(date +%s)
    local r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${TASK}_${arm}.log" | tail -1)
    printf "%s\t%s\t%s\t%s\n" "$TASK" "$arm" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    kill_server
}

run_arm C "" ""
run_arm T "--lora_path $LORA" "${T_CLIENT_ARGS:---sampler vine}"
echo "=== done ===" | tee -a "$summary"
