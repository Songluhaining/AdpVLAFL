#!/usr/bin/env bash
# curobo: RoboTwin's motion planner for the scripted expert. envs/robot/robot.py
# imports CuroboPlanner unconditionally, so this is not optional.
#
# Split out of setup_env_robotwin.sh because the github clone runs over the
# metered proxy and kept dying mid-transfer; here it retries on its own.
set -euo pipefail

CONDA=/data/whn/miniconda3
ENV_NAME=robotwin
ROOT=/data/whn/robotwin_eval
PIP="$CONDA/envs/$ENV_NAME/bin/pip"
PY="$CONDA/envs/$ENV_NAME/bin/python"

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

PIP_IDX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

cd "$ROOT/RoboTwin/envs"

if [ ! -f curobo/setup.py ]; then
    rm -rf curobo
    for attempt in 1 2 3 4 5 6; do
        echo "=== clone curobo (attempt $attempt) ==="
        # A shallow single-branch clone keeps this under ~50MB of proxy traffic.
        # postBuffer/lowSpeedLimit make git give up and retry instead of hanging.
        if env -u all_proxy -u ALL_PROXY git \
                -c http.proxy=http://127.0.0.1:7897 \
                -c https.proxy=http://127.0.0.1:7897 \
                -c http.postBuffer=524288000 \
                -c http.lowSpeedLimit=1000 \
                -c http.lowSpeedTime=60 \
                clone --branch v0.7.8 --depth 1 --single-branch \
                https://github.com/NVlabs/curobo.git; then
            break
        fi
        rm -rf curobo
        sleep 15
    done
fi
[ -f curobo/setup.py ] || { echo "curobo clone failed" >&2; exit 1; }

cd curobo
echo "=== building curobo CUDA extensions (sm_89, this takes a while) ==="
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=8 \
    $PIP install $PIP_IDX -e . --no-build-isolation

env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    $PIP install $PIP_IDX warp-lang==1.12.0 setuptools==69.5.1

echo "=== verify ==="
$PY -c "from curobo.wrap.reacher.motion_gen import MotionGen; print('curobo OK')"
