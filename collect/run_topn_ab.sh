#!/usr/bin/env bash
# Attribution-conditioned arbitration of top-N noise-draw selection.
#
# Scenes are the replicate-labelled sets (attribution labels built BEFORE the
# run from existing multi-draw records): noise-rescuable failures, scene-decided
# failures, stable successes. The method passes only if rescues concentrate in
# the noise-rescuable class -- an aggregate gain without that concentration
# does not count (user's criterion: only act where re-drawing noise is the
# established fix).
#
#   C  single pinned draw (rep0 stream)   -- baseline
#   T  sample_topn=N, critic picks        -- candidate 0 IS the C draw, so any
#                                            outcome change traces to selection
#
# EXTRA_REPS runs additional plain replicate arms first (fills in attribution
# labels for scenes whose rescuability is still unknown).
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
REPO=/data/whn/codes/lingbot-vla-v2
CKPT="$ROOT/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
TASK=${TASK:?}
CRITIC=${CRITIC:?}
SCENES=${SCENES:?json list of scene seeds}
TOPN=${TOPN:-8}
EXTRA_REPS=${EXTRA_REPS:-}
PORT=${PORT:-9480}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/topn_ab_${TASK}_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"
echo "topn_ab $stamp | task=$TASK topn=$TOPN critic=$CRITIC scenes=$SCENES" | tee "$summary"
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

cd "$REPO" || exit 1
QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct" \
ADPVLAFL_ANALYSIS=/data/whn/AdpVLAFL/analysis CUDA_VISIBLE_DEVICES=0 \
    "$CONDA/envs/lingbotvla/bin/python" -u -m deploy.lingbot_vla_v2_policy \
        --model_path "$CKPT" --use_length 50 --use_bf16 True \
        --use_compile False --port "$PORT" --critic_path "$CRITIC" \
        > "$log_dir/server.log" 2>&1 &
for _ in $(seq 1 120); do
    grep -q "Model initialized" "$log_dir/server.log" && break
    sleep 5
done
grep -q "top-N critic loaded" "$log_dir/server.log" || {
    echo "ERROR: critic did not load" | tee -a "$summary"; exit 1; }

run_client() {  # tag extra...
    local tag=$1; shift
    cd "$ROOT/RoboTwin" || exit 1
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
            --task_name "$TASK" --task_config demo_randomized --port "$PORT" \
            --scene_seeds_file "$SCENES" --exec_horizon 25 \
            --instruction_type unseen --fixed_noise \
            --no_log_introspect --no_save_images "$@" \
            > "$log_dir/${TASK}_${tag}.log" 2>&1
    local t1=$(date +%s)
    local r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${TASK}_${tag}.log" | tail -1)
    printf "%s\t%s\t%s\t%s\n" "$TASK" "$tag" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
}

for rep in $EXTRA_REPS; do
    run_client "rep${rep}" --noise_replicate "$rep" \
        --output_dir "$ROOT/rollouts_replicate/${TASK}_rep${rep}"
done
run_client C --noise_replicate 0 --output_dir "$ROOT/rollouts_topn/${TASK}_C"
run_client T --noise_replicate 0 --sample_topn "$TOPN" \
    --output_dir "$ROOT/rollouts_topn/${TASK}_T"
echo "=== done ===" | tee -a "$summary"
