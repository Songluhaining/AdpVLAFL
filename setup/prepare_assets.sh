#!/usr/bin/env bash
# Unpack the RoboTwin 2.0 simulator assets into the repo and fix the absolute
# paths baked into the embodiment configs. Run once, after download_all.py.
set -euo pipefail

ROOT=/data/whn/robotwin_eval
SRC="$ROOT/robotwin_assets"
DST="$ROOT/RoboTwin/assets"
PY="/data/whn/miniconda3/envs/robotwin/bin/python"

cd "$DST"
for z in embodiments objects background_texture; do
    if [ -f "$SRC/$z.zip" ]; then
        echo "=== unzip $z ==="
        # -n: never clobber, so re-running is cheap and safe
        unzip -q -n "$SRC/$z.zip" -d "$DST"
    else
        echo "missing $SRC/$z.zip" >&2
        exit 1
    fi
done

cd "$ROOT/RoboTwin"
echo "=== rewriting embodiment config paths ==="
# The script prompts if it cannot find assets/embodiments; it is right here, so
# it never reaches the prompt - but keep stdin closed so a surprise cannot hang.
"$PY" ./script/update_embodiment_config_path.py < /dev/null

echo "=== done ==="
ls -d assets/embodiments/aloha-agilex assets/objects | head
