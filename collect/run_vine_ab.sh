#!/usr/bin/env bash
# Paired sampler comparison: euler (original) vs VINE (arXiv 2607.10369),
# pure inference-time swap on the pretrained checkpoint -- no RL, no weights.
#
# Two arms over the same labelled scene sets (seed base 11, the expand_labelled
# scenes) with the same pinned noise stream:
#   C  euler  -- re-establishes each scene's outcome in this session
#   T  vine   -- per-step noise re-injection + endpoint prediction
#
# With a pinned seed both samplers make an identical first velocity query and
# diverge from step 2, so per-scene outcome flips are attributable to the
# sampler (up to the known bf16 wobble). The historical euler replicates
# rollouts_replicate/{task}_rep0,rep0b on the same scenes calibrate that
# wobble: a real sampler effect must show flip *asymmetry* (rescued >> broken)
# beyond what euler-vs-euler reruns produce.
#
# Expects the policy server already resident on $PORT.
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
TASKS=${TASKS:-"click_bell click_alarmclock stamp_seal place_can_basket"}
PORT=${PORT:-9460}
N=${N:-50}
SEED=${SEED:-11}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/vine_ab_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"
echo "vine_ab $stamp | tasks=[$TASKS] N=$N seed_base=$SEED port=$PORT" | tee "$summary"

curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || {
    echo "ERROR: no policy server on $PORT" | tee -a "$summary"; exit 1; }

cd "$ROOT/RoboTwin" || exit 1
printf "%s\t%s\t%s\t%s\n" task arm success_over_total minutes | tee -a "$summary"

for task in $TASKS; do
    for arm in C T; do
        extra=""
        [ "$arm" = T ] && extra="--sampler vine"
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config demo_randomized --port "$PORT" \
                --test_num "$N" --seed "$SEED" --exec_horizon 25 \
                --instruction_type unseen --fixed_noise --noise_replicate 0 \
                --no_log_introspect --no_save_images $extra \
                --output_dir "$ROOT/rollouts_vine_ab/${task}_${arm}" \
                > "$log_dir/${task}_${arm}.log" 2>&1
        t1=$(date +%s)
        r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}_${arm}.log" | tail -1)
        printf "%s\t%s\t%s\t%s\n" "$task" "$arm" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    done
done
echo "=== done ===" | tee -a "$summary"
