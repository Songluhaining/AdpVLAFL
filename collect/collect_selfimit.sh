#!/usr/bin/env bash
# One self-imitation iteration's data: K free-noise rollouts per training-pool
# scene, images kept (the FM fine-tune needs them). Scene pool is the seed-21
# range -- zero overlap with every evaluation range (leak lesson).
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
TASKS=${TASKS:-"click_bell click_alarmclock"}
N=${N:-50}
SEED=${SEED:-21}
K=${K:-3}
ITER=${ITER:?iteration tag, e.g. 1}
LORA=${LORA:-}          # empty = base policy (iteration 1 collects under base+iter0 LoRA etc.)
PORT=${PORT:-9500}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/selfimit_it${ITER}_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"
echo "selfimit iter$ITER $stamp | tasks=[$TASKS] N=$N seed=$SEED K=$K lora=${LORA:-none}" | tee "$summary"
printf "%s\t%s\t%s\t%s\n" task arm success_over_total minutes | tee -a "$summary"

kill_server() {
    for pid in $(pgrep -f "lingbot_vla_v2_policy.*--port $PORT"); do
        kill -TERM "$pid" 2>/dev/null
    done
    sleep 4
    for pid in $(pgrep -f "lingbot_vla_v2_policy.*--port $PORT"); do
        kill -KILL "$pid" 2>/dev/null
    done
}
trap kill_server EXIT INT TERM

extra=""
[ -n "$LORA" ] && extra="--lora_path $LORA"
cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" $extra > "$log_dir/server.log" 2>&1 &
for _ in $(seq 1 140); do
    grep -q "Model initialized" "$log_dir/server.log" && break
    sleep 5
done

cd "$ROOT/RoboTwin" || exit 1
for task in $TASKS; do
    for k in $(seq 0 $((K - 1))); do
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config demo_randomized --port "$PORT" \
                --test_num "$N" --seed "$SEED" --exec_horizon 25 \
                --instruction_type unseen --no_log_introspect \
                --output_dir "$ROOT/rollouts_selfimit/${task}_it${ITER}_r${k}" \
                > "$log_dir/${task}_r${k}.log" 2>&1
        t1=$(date +%s)
        r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}_r${k}.log" | tail -1)
        printf "%s\t%s\t%s\t%s\n" "$task" "r$k" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    done
done
echo "=== done ===" | tee -a "$summary"
