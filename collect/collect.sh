#!/usr/bin/env bash
# Bulk trajectory collection. Designed to be left alone for hours:
#
#  * one resident policy server for every task and chunk (reloading the 6B
#    checkpoint costs ~4 min each time)
#  * work is split into chunks with disjoint seed bases, so a crash loses at most
#    one chunk instead of the whole run
#  * meant to be launched under setsid (see launch_collect.sh) so it survives the
#    terminal, the ssh session, and the agent that started it
#
# TASKS="open_microwave stack_bowls_three" TOTAL=300 CHUNK=50 bash collect.sh
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"

TASKS=${TASKS:?set TASKS to a space-separated task list}
TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
TOTAL=${TOTAL:-300}
CHUNK=${CHUNK:-50}
# Chunk c draws scenes from seed base 100000*(1+c). The screening sweep already
# used c=0, so start past it or the new episodes replay scenes we already have.
SEED_START=${SEED_START:-0}
PORT=${PORT:-9360}
EXEC_HORIZON=${EXEC_HORIZON:-25}
OUT=${OUT:-$ROOT/rollouts}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/collect_${TASK_CONFIG}_${stamp}"
mkdir -p "$log_dir"
server_log="$log_dir/policy_server.log"
progress="$log_dir/progress.txt"

server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null && {
        kill -TERM "$server_pid" 2>/dev/null
        sleep 3
        kill -KILL "$server_pid" 2>/dev/null
    }
    return 0
}
trap cleanup EXIT INT TERM

{
    echo "collect run $stamp"
    echo "  tasks       : $TASKS"
    echo "  task_config : $TASK_CONFIG"
    echo "  total/chunk : $TOTAL / $CHUNK"
    echo "  out         : $OUT"
} | tee "$progress"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mib" -lt 19000 ]; then
    echo "ERROR: only ${free_mib} MiB GPU memory free, need ~19000" | tee -a "$progress"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$progress"
    exit 1
fi

echo "=== starting policy server ===" | tee -a "$progress"
cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" > "$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || {
        echo "server died during load; see $server_log" | tee -a "$progress"; exit 1; }
    grep -q "Model initialized" "$server_log" && break
    sleep 5
done
grep -q "Model initialized" "$server_log" || {
    echo "server never became ready" | tee -a "$progress"; exit 1; }
echo "server ready (pid $server_pid)" | tee -a "$progress"

cd "$ROOT/RoboTwin" || exit 1
n_chunks=$(( (TOTAL + CHUNK - 1) / CHUNK ))

for task in $TASKS; do
    for c in $(seq "$SEED_START" $((SEED_START + n_chunks - 1))); do
        # --seed shifts the scene seed base (now_seed = 100000 * (1 + seed)), so
        # each chunk explores a disjoint set of scenes rather than repeating them.
        echo "[$(date +%H:%M:%S)] $task chunk $((c+1))/$n_chunks (seed base $c)" | tee -a "$progress"
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config "$TASK_CONFIG" --port "$PORT" \
                --test_num "$CHUNK" --exec_horizon "$EXEC_HORIZON" \
                --instruction_type unseen --seed "$c" \
                --output_dir "$OUT" \
                > "$log_dir/${task}_chunk${c}.log" 2>&1
        rc=$?
        rate=$(grep -oP 'FINAL Success rate: \K.*' "$log_dir/${task}_chunk${c}.log" | tail -1)
        echo "    -> rc=$rc  ${rate:-<no result>}" | tee -a "$progress"
    done
done

echo "[$(date +%H:%M:%S)] === collection complete ===" | tee -a "$progress"
