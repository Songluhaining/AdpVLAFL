#!/usr/bin/env bash
# Detach collect.sh from this terminal so a multi-hour run survives the shell,
# the ssh session, and whatever agent kicked it off.
#
#   TASKS="open_microwave" TOTAL=300 bash launch_collect.sh
#
# Then follow along with:
#   tail -f $(ls -td /data/whn/robotwin_eval/logs/collect_* | head -1)/progress.txt
set -euo pipefail

ROOT=/data/whn/robotwin_eval
: "${TASKS:?set TASKS to a space-separated task list}"

cd "$ROOT"
setsid nohup env \
    TASKS="$TASKS" \
    TASK_CONFIG="${TASK_CONFIG:-demo_randomized}" \
    TOTAL="${TOTAL:-300}" \
    CHUNK="${CHUNK:-50}" \
    PORT="${PORT:-9360}" \
    bash collect.sh > "$ROOT/logs/collect_launch.log" 2>&1 < /dev/null &

echo "launched detached (pid $!)"
sleep 5
echo "progress file will appear at:"
ls -td "$ROOT"/logs/collect_* 2>/dev/null | head -1
