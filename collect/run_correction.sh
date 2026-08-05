#!/usr/bin/env bash
# The routing-correction contrast the diagnosis is judged by.
#
# Two arms over the same scenes with the same pinned noise stream:
#   C  no correction              -- re-establishes each scene's outcome
#   T  routing bias, scale 0.01   -- pushes each scene's routing toward its
#                                    matched counterfactual (calibrated so the
#                                    load moves by about the intended delta,
#                                    ~70% along the intended direction)
#
# Group A scenes are the ones the diagnosis blamed on routing, group B the ones
# it blamed elsewhere; both receive the identical treatment. The diagnosis is
# supported only if A's rescue rate beats B's -- an absolute rescue rate proves
# nothing, since bf16 wobble alone flips outcomes at a known floor and hits both
# groups alike.
#
# Expects the policy server already resident on $PORT (it takes ~4 min to load;
# the calibration step used the same one).
set -uo pipefail

CONDA=/data/whn/miniconda3
ROOT=/data/whn/robotwin_eval
TASKS=${TASKS:-"click_bell click_alarmclock stamp_seal place_can_basket open_microwave hanging_mug"}
PORT=${PORT:-9450}
SCALE=${SCALE:-0.01}

stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$ROOT/logs/correction_${stamp}"
mkdir -p "$log_dir"
summary="$log_dir/summary.tsv"
echo "correction $stamp | tasks=[$TASKS] scale=$SCALE port=$PORT" | tee "$summary"

curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || {
    echo "ERROR: no policy server on $PORT" | tee -a "$summary"; exit 1; }

cd "$ROOT/RoboTwin" || exit 1
printf "%s\t%s\t%s\t%s\n" task arm success_over_total minutes | tee -a "$summary"

for task in $TASKS; do
    scenes="$ROOT/correction/${task}_scenes.json"
    [ -f "$scenes" ] || continue
    list="$log_dir/${task}_list.json"
    "$CONDA/envs/robotwin/bin/python" -c \
        "import json;d=json.load(open('$scenes'));json.dump(d['scenes'],open('$list','w'))"
    for arm in C T; do
        extra=""
        [ "$arm" = T ] && extra="--routing_bias_file $ROOT/correction/${task}_bias.npz --routing_bias_scale $SCALE"
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
            "$CONDA/envs/robotwin/bin/python" -u script/eval_lingbot_vla.py \
                --task_name "$task" --task_config demo_randomized --port "$PORT" \
                --scene_seeds_file "$list" --exec_horizon 25 --instruction_type unseen \
                --fixed_noise --noise_replicate 0 \
                --no_log_introspect --no_save_images $extra \
                --output_dir "$ROOT/rollouts_correction/${task}_${arm}" \
                > "$log_dir/${task}_${arm}.log" 2>&1
        t1=$(date +%s)
        r=$(grep -oP 'FINAL Success rate: \K\d+/\d+' "$log_dir/${task}_${arm}.log" | tail -1)
        printf "%s\t%s\t%s\t%s\n" "$task" "$arm" "${r:--}" "$(( (t1 - t0) / 60 ))" | tee -a "$summary"
    done
done
echo "=== done ===" | tee -a "$summary"
