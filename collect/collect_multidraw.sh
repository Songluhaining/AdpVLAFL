#!/usr/bin/env bash
# Multi-draw contrastive corpus: the training material the single-draw buffer
# provably lacks (within-state ranking measured at 48% = chance without it).
#
# Fresh scene range (seed base 15 -> 1600000), K noise draws per scene with
# full introspect features, no images. Same scenes across draws, so every
# failed/succeeded draw pair at the same scene is a real within-state
# contrast example. Also the meaningfulness arbiter for the q(eps|s) line:
# if a ranker trained on this still cannot separate same-state draws, the
# outcome difference is not predictable from these observables at all.
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
TASKS=${TASKS:-"click_bell click_alarmclock"}
N=${N:-50}
SEED=${SEED:-15}
REPS=${REPS:-"0 1 2 3 4 5"}
PORT=${PORT:-9490}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/multidraw_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"
echo "multidraw $stamp | tasks=[$TASKS] N=$N seed=$SEED reps=[$REPS]" | tee "$summary"
printf "%s\t%s\t%s\t%s\n" task rep success_over_total minutes | tee -a "$summary"

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

cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" > "$log_dir/server.log" 2>&1 &
for _ in $(seq 1 140); do
    grep -q "Model initialized" "$log_dir/server.log" && break
    sleep 5
done

cd "$ROOT/RoboTwin" || exit 1
for task in $TASKS; do
    for rep in $REPS; do
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config demo_randomized --port "$PORT" \
                --test_num "$N" --seed "$SEED" --exec_horizon 25 \
                --instruction_type unseen --fixed_noise --noise_replicate "$rep" \
                --no_save_images \
                --output_dir "$ROOT/rollouts_multidraw/${task}_rep${rep}" \
                > "$log_dir/${task}_rep${rep}.log" 2>&1
        t1=$(date +%s)
        r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}_rep${rep}.log" | tail -1)
        printf "%s\t%s\t%s\t%s\n" "$task" "$rep" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    done
done
echo "=== done ===" | tee -a "$summary"
