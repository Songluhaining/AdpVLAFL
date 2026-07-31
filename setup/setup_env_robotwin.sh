#!/usr/bin/env bash
# Simulation-side environment: SAPIEN 3 + mplib + curobo (RoboTwin 2.0).
# Kept separate from the policy env on purpose - the two only talk over a
# websocket, so their torch versions never have to agree.
set -euo pipefail

CONDA=/data/whn/miniconda3
ENV_NAME=robotwin
ROOT=/data/whn/robotwin_eval

# Everything domestic and proxy-free: the user's proxy quota is metered.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
PIP_IDX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
TORCH_IDX="https://mirrors.aliyun.com/pytorch-wheels/cu124/"

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

# The mirror drops connections while the 50GB weight download is saturating the
# link, so every network step gets retried rather than aborting the whole build.
retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        if [ "$n" -ge 8 ]; then echo "FAILED after $n attempts: $*" >&2; return 1; fi
        echo "--- retry $n: $* ---" >&2
        sleep 20
    done
}

export CONDA_REMOTE_READ_TIMEOUT_SECS=120
PY="$CONDA/envs/$ENV_NAME/bin/python"
PIP="$CONDA/envs/$ENV_NAME/bin/pip"
if [ ! -x "$PY" ]; then
    retry "$CONDA/bin/conda" create -n "$ENV_NAME" python=3.10 -y
fi

echo "=== torch 2.4.1+cu124 ==="
# Direct wheel URLs, not --find-links: the mirror's flat index lists thousands of
# wheels and pip spends minutes parsing it before it downloads anything.
retry $PIP install $PIP_IDX \
    "${TORCH_IDX}torch-2.4.1%2Bcu124-cp310-cp310-linux_x86_64.whl" \
    "${TORCH_IDX}torchvision-0.19.1%2Bcu124-cp310-cp310-linux_x86_64.whl"

echo "=== RoboTwin sim deps ==="
retry $PIP install $PIP_IDX \
    transforms3d==0.4.2 sapien==3.0.0b1 scipy==1.10.1 mplib==0.2.1 \
    gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 \
    pydantic zarr huggingface_hub==0.25.0 h5py pyglet'<2' moviepy \
    termcolor av matplotlib toppra numpy'<2' opencv-python

echo "=== websocket client deps (talk to the policy server) ==="
retry $PIP install $PIP_IDX websockets==15.0.1 msgpack==1.1.1 pyyaml tqdm

echo "=== ffmpeg (video recording of rollouts) ==="
retry "$CONDA/bin/conda" install -n "$ENV_NAME" -y -c conda-forge ffmpeg

echo "=== patch sapien urdf_loader + mplib planner (per RoboTwin script/_install.sh) ==="
SAPIEN_LOC=$($PIP show sapien | awk '/^Location/{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN_LOC/wrapper/urdf_loader.py"
MPLIB_LOC=$($PIP show mplib | awk '/^Location/{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "$MPLIB_LOC/planner.py"

echo "=== curobo (needs nvcc; builds CUDA kernels) ==="
cd "$ROOT/RoboTwin/envs"
if [ ! -d curobo ]; then
    # github is unreachable directly here, so this one clone goes over the proxy
    http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 \
        git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 \
        clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
fi
cd curobo
retry env TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=12 $PIP install $PIP_IDX -e . --no-build-isolation
retry $PIP install $PIP_IDX warp-lang==1.12.0 setuptools==69.5.1

echo "=== done: $ENV_NAME ==="
$PY -c "import sapien, mplib, torch; print('sapien', sapien.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
