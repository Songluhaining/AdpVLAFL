#!/usr/bin/env bash
# Screen every remaining RoboTwin task for failure rate, with full instrumentation.
#
# Purpose is selection, not precision: the goal is to find which tasks this
# checkpoint actually fails at, so the Bayesian model gets evidence from several
# tasks rather than one. A task at 100% or 0% carries no signal either way.
#
# The episodes are recorded exactly like a real collection run, so nothing here
# is throwaway -- whatever the sweep produces goes straight into the corpus.
#
# Launch detached so a multi-hour run outlives the shell that started it:
#   setsid nohup bash sweep_all_tasks.sh > logs/sweep_launch.log 2>&1 &
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
N=${N:-12}
PORT=${PORT:-9380}
OUT="$ROOT/rollouts"

# The 6 already probed are excluded; their rates are known and re-running the
# three that scored 100% would buy nothing.
#   hanging_mug 41.7% | open_microwave 58.3% | stack_bowls_three 83.3%
#   put_bottles_dustbin / blocks_ranking_rgb / stack_blocks_three  100%
TASKS=(
    lift_pot scan_object handover_block click_bell put_object_cabinet place_shoe
    adjust_bottle beat_block_hammer blocks_ranking_size click_alarmclock
    dump_bin_bigbin grab_roller handover_mic move_can_pot move_pillbottle_pad
    move_playingcard_away place_cans_plasticbox place_container_plate
    place_dual_shoes place_empty_cup place_fan place_mouse_pad
    place_object_basket place_object_scale place_object_stand place_phone_stand
    move_stapler_pad open_laptop pick_diverse_bottles pick_dual_bottles
    place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
    place_burger_fries place_can_basket press_stapler rotate_qrcode
    shake_bottle_horizontally shake_bottle stack_blocks_two stack_bowls_two
    stamp_seal turn_switch
)

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/sweep_${TASK_CONFIG}_${stamp}"
mkdir -p "$log_dir"
server_log="$log_dir/policy_server.log"
summary="$log_dir/sweep_summary.tsv"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null && {
        kill -TERM "$server_pid" 2>/dev/null; sleep 3; kill -KILL "$server_pid" 2>/dev/null; }
    return 0
}
trap cleanup EXIT INT TERM

echo "sweep $stamp | ${#TASKS[@]} tasks x $N episodes | config=$TASK_CONFIG" | tee "$summary"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mib" -lt 19000 ]; then
    echo "ERROR: only ${free_mib} MiB GPU free, need ~19000" | tee -a "$summary"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$summary"
    exit 1
fi

echo "starting policy server ..." | tee -a "$summary"
cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" > "$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || { echo "server died, see $server_log" | tee -a "$summary"; exit 1; }
    grep -q "Model initialized" "$server_log" && break
    sleep 5
done
grep -q "Model initialized" "$server_log" || { echo "server never ready" | tee -a "$summary"; exit 1; }
echo "server ready (pid $server_pid)" | tee -a "$summary"

cd "$ROOT/RoboTwin" || exit 1
printf "%s\t%s\t%s\t%s\n" task success_over_total rate minutes | tee -a "$summary"

done_n=0
for task in "${TASKS[@]}"; do
    done_n=$((done_n + 1))
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] ($done_n/${#TASKS[@]}) $task" >&2
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
            --task_name "$task" --task_config "$TASK_CONFIG" --port "$PORT" \
            --test_num "$N" --exec_horizon 25 --instruction_type unseen \
            --output_dir "$OUT" \
            > "$log_dir/${task}.log" 2>&1
    t1=$(date +%s)
    line=$(grep -oP 'FINAL Success rate: \K.*' "$log_dir/${task}.log" | tail -1)
    printf "%s\t%s\t%s\t%s\n" "$task" \
        "$(echo "$line" | grep -oP '^\d+/\d+' || echo '-')" \
        "$(echo "$line" | grep -oP '=> \K[\d.]+%' || echo '-')" \
        "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
done

echo "[$(date +%H:%M:%S)] === sweep complete ===" | tee -a "$summary"
