#!/usr/bin/env bash
# Paired intervention on exec_horizon.
#
# The observational model ranks `execution` as the leading suspect, but that
# ranking cannot separate causation from a within-task confound: a hard scene can
# produce both an odd chunk seam and a failure. This does the actual do():
# identical scenes, identical noise stream, exec_horizon 25 vs 5, nothing else
# changed. Any difference in outcome is caused by exec_horizon.
#
# --fixed_noise is what makes the pairing valid. Without it the two arms draw
# different flow-matching noise and a flipped outcome proves nothing.
#
# Introspection and image capture are off: only the outcome is needed here, and
# leaving them on would cost ~20% latency and gigabytes for nothing.
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

TASKS=${TASKS:-"click_bell hanging_mug open_microwave click_alarmclock"}
N=${N:-25}
PORT=${PORT:-9400}
# Three arms, not two. bf16 rounding leaves ~1.3e-2 of run-to-run drift that
# no seeding removes (fp32 would, but the weights do not fit on a 4090), so a
# repeat of the baseline is needed to measure how many outcomes flip from pure
# numerical noise. The intervention only counts if it beats that floor.
ARMS=${ARMS:-"A:25 A2:25 B:5"}
SEED=${SEED:-7}          # a seed base untouched by the sweep (0) or collection (1,2)
OUT="$ROOT/rollouts_intervene"

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/intervene_${stamp}"
mkdir -p "$log_dir" "$OUT"
server_log="$log_dir/policy_server.log"
summary="$log_dir/summary.tsv"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null && {
        kill -TERM "$server_pid" 2>/dev/null; sleep 3; kill -KILL "$server_pid" 2>/dev/null; }
    return 0
}
trap cleanup EXIT INT TERM

echo "intervention $stamp | tasks=[$TASKS] N=$N arms=[$ARMS] seed_base=$SEED" | tee "$summary"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free_mib" -lt 19000 ] && { echo "ERROR: only ${free_mib} MiB GPU free" | tee -a "$summary"; exit 1; }

cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" > "$server_log" 2>&1 &
server_pid=$!
for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || { echo "server died" | tee -a "$summary"; exit 1; }
    grep -q "Model initialized" "$server_log" && break
    sleep 5
done
echo "server ready" | tee -a "$summary"

cd "$ROOT/RoboTwin" || exit 1
printf "%s\t%s\t%s\t%s\t%s\n" task arm horizon success_over_total minutes | tee -a "$summary"

for task in $TASKS; do
    for arm in $ARMS; do
        label="${arm%%:*}"; h="${arm##*:}"
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config demo_randomized --port "$PORT" \
                --test_num "$N" --exec_horizon "$h" --instruction_type unseen \
                --seed "$SEED" --fixed_noise \
                --no_log_introspect --no_save_images \
                --output_dir "$OUT/${label}_h${h}" \
                > "$log_dir/${task}_${label}.log" 2>&1
        t1=$(date +%s)
        r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}_${label}.log" | tail -1)
        printf "%s\t%s\t%s\t%s\t%s\n" "$task" "$label" "$h" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    done
done

echo "=== done ===" | tee -a "$summary"
