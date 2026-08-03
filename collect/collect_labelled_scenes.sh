#!/usr/bin/env bash
# Re-run the noise-replicate scenes *with instrumentation*, once per scene.
#
# The replicate experiment labelled each of those scenes as noise-decided (the
# outcome swung across draws) or scene-decided (it held). Those scenes were run
# without capture, so the labels currently have no internals attached.
#
# Recollecting them with capture gives something the corpus has never had: a
# ground truth for *why* an attempt failed, independent of the Bayesian model.
# The model can then be checked against it rather than only against itself --
# if the localization is real, the failures it attributes to a component should
# line up with how the scene behaves under a different draw.
#
# Same seed base and the same pinned noise stream as replicate 0, so the outcome
# is the one already labelled.
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

TASKS=${TASKS:-"click_bell click_alarmclock open_microwave"}
N=${N:-25}
PORT=${PORT:-9430}
SEED=${SEED:-7}
OUT="$ROOT/rollouts_labelled"

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/labelled_${stamp}"
mkdir -p "$log_dir" "$OUT"
summary="$log_dir/summary.tsv"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null && {
        kill -TERM "$server_pid" 2>/dev/null; sleep 3; kill -KILL "$server_pid" 2>/dev/null; }
    return 0
}
trap cleanup EXIT INT TERM

echo "labelled collection $stamp | tasks=[$TASKS] N=$N seed_base=$SEED" | tee "$summary"

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
printf "%s\t%s\t%s\n" task success_over_total minutes | tee -a "$summary"

for task in $TASKS; do
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
            --task_name "$task" --task_config demo_randomized --port "$PORT" \
            --test_num "$N" --exec_horizon 25 --instruction_type unseen \
            --seed "$SEED" --fixed_noise --noise_replicate 0 \
            --no_save_images \
            --output_dir "$OUT" \
            > "$log_dir/${task}.log" 2>&1
    t1=$(date +%s)
    r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}.log" | tail -1)
    printf "%s\t%s\t%s\n" "$task" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
done

echo "=== done ===" | tee -a "$summary"
