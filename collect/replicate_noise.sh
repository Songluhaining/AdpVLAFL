#!/usr/bin/env bash
# Same scenes, same everything, different flow-matching noise draw.
#
# Asks one question: when an attempt fails, was it decided by that particular
# draw from the action distribution, or would the model have failed that scene
# no matter what it drew?
#
#   spread across replicates -> the draw decides, and choosing the draw is a
#                               real remedy with real headroom
#   every replicate fails    -> the scene is simply unwinnable for this model,
#                               no noise-selection strategy can help, and the
#                               responsibility the model puts on the opening
#                               decision is confounded rather than causal
#
# Replicate 0 is run twice. bf16 rounding alone perturbs a rollout even with the
# noise pinned, so the repeat measures how many outcomes flip from numerical
# noise; a spread across genuinely different draws only counts if it beats that.
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

TASKS=${TASKS:-"click_bell click_alarmclock open_microwave"}
N=${N:-25}
REPS=${REPS:-"0 0b 1 2 3 4 5 6 7"}
PORT=${PORT:-9420}
SEED=${SEED:-7}          # same scene seeds the intervention arms used
OUT="$ROOT/rollouts_replicate"

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/replicate_${stamp}"
mkdir -p "$log_dir" "$OUT"
summary="$log_dir/summary.tsv"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null && {
        kill -TERM "$server_pid" 2>/dev/null; sleep 3; kill -KILL "$server_pid" 2>/dev/null; }
    return 0
}
trap cleanup EXIT INT TERM

echo "replicate $stamp | tasks=[$TASKS] N=$N reps=[$REPS] seed_base=$SEED" | tee "$summary"

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
printf "%s\t%s\t%s\t%s\n" task replicate success_over_total minutes | tee -a "$summary"

for task in $TASKS; do
    for rep in $REPS; do
        r_num="${rep%b}"            # "0b" is the repeat of replicate 0
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config demo_randomized --port "$PORT" \
                --test_num "$N" --exec_horizon 25 --instruction_type unseen \
                --seed "$SEED" --fixed_noise --noise_replicate "$r_num" \
                --no_log_introspect --no_save_images \
                --output_dir "$OUT/${task}_rep${rep}" \
                > "$log_dir/${task}_rep${rep}.log" 2>&1
        t1=$(date +%s)
        r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}_rep${rep}.log" | tail -1)
        printf "%s\t%s\t%s\t%s\n" "$task" "$rep" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    done
done

echo "=== done ===" | tee -a "$summary"
