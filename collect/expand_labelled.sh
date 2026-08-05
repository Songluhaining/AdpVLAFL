#!/usr/bin/env bash
# Expand the labelled corpus: replicate each scene under several noise draws to
# learn whether its outcome swings, then re-run the same scenes once with
# instrumentation so the localization has internals to work from.
#
# 27 labelled failures was too few to settle whether the localization tracks the
# real cause. Six tasks x 50 scenes at roughly 45% failure should yield ~135 more.
#
# Both phases share one resident policy server -- reloading the 6B checkpoint
# costs about four minutes each time -- and both pin the noise stream from
# (scene seed, decision index) so a scene is reproducible across replicates.
#
# Scene seeds start at a base untouched by earlier runs, so these are new scenes.
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

TASKS=${TASKS:-"click_bell click_alarmclock stamp_seal place_can_basket hanging_mug open_microwave"}
N=${N:-50}
REPS=${REPS:-"0 0b 1 2 3 4"}
PORT=${PORT:-9440}
SEED=${SEED:-11}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/expand_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null && {
        kill -TERM "$server_pid" 2>/dev/null; sleep 3; kill -KILL "$server_pid" 2>/dev/null; }
    return 0
}
trap cleanup EXIT INT TERM

echo "expand $stamp | tasks=[$TASKS] N=$N reps=[$REPS] seed_base=$SEED" | tee "$summary"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free_mib" -lt 19000 ] && { echo "ERROR: only ${free_mib} MiB GPU free" | tee -a "$summary"; exit 1; }

cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" > "$log_dir/policy_server.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || { echo "server died" | tee -a "$summary"; exit 1; }
    grep -q "Model initialized" "$log_dir/policy_server.log" && break
    sleep 5
done
echo "server ready" | tee -a "$summary"

cd "$ROOT/RoboTwin" || exit 1
printf "%s\t%s\t%s\t%s\t%s\n" phase task replicate success_over_total minutes | tee -a "$summary"

run() {  # phase task rep extra_args...
    local phase=$1 task=$2 rep=$3; shift 3
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
            --task_name "$task" --task_config demo_randomized --port "$PORT" \
            --test_num "$N" --exec_horizon 25 --instruction_type unseen \
            --seed "$SEED" --fixed_noise --no_save_images "$@" \
            > "$log_dir/${phase}_${task}_${rep}.log" 2>&1
    local t1=$(date +%s)
    local r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${phase}_${task}_${rep}.log" | tail -1)
    printf "%s\t%s\t%s\t%s\t%s\n" "$phase" "$task" "$rep" "${r:--}" "$(( (t1-t0)/60 ))" | tee -a "$summary"
}

# Phase 1: replicates, no capture -- only the outcome is needed to label a scene.
for task in $TASKS; do
    for rep in $REPS; do
        run replicate "$task" "$rep" \
            --noise_replicate "${rep%b}" --no_log_introspect \
            --output_dir "$ROOT/rollouts_replicate/${task}_rep${rep}"
    done
done

# Phase 2: the same scenes once more, with capture, matching replicate 0.
for task in $TASKS; do
    run labelled "$task" "instr" \
        --noise_replicate 0 --output_dir "$ROOT/rollouts_labelled"
done

echo "=== done ===" | tee -a "$summary"
