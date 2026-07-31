#!/usr/bin/env bash
# Policy-side environment: Qwen3-VL + LingBot-VLA 2.0 inference server.
set -euo pipefail

CONDA=/data/whn/miniconda3
ENV_NAME=lingbotvla
REPO=/data/whn/codes/lingbot-vla-v2
ROOT=/data/whn/robotwin_eval

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
PIP_IDX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
TORCH_IDX="https://mirrors.aliyun.com/pytorch-wheels/cu128/"

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

retry() { local n=0; until "$@"; do n=$((n+1)); if [ "$n" -ge 8 ]; then echo "FAILED after $n: $*" >&2; return 1; fi; echo "--- retry $n ---" >&2; sleep 20; done; }
export CONDA_REMOTE_READ_TIMEOUT_SECS=120
if [ ! -x "$CONDA/envs/$ENV_NAME/bin/python" ]; then retry "$CONDA/bin/conda" create -n "$ENV_NAME" python=3.12 -y; fi
PY="$CONDA/envs/$ENV_NAME/bin/python"
PIP="$CONDA/envs/$ENV_NAME/bin/pip"

echo "=== torch 2.8.0+cu128 ==="
retry $PIP install $PIP_IDX \
    "${TORCH_IDX}torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "${TORCH_IDX}torchvision-0.23.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "${TORCH_IDX}torchaudio-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"

echo "=== repo requirements (minus the torch lines already installed) ==="
grep -vE '^(torch|torchvision|torchaudio)==' "$REPO/requirements.txt" > "$ROOT/req_no_torch.txt"
retry $PIP install $PIP_IDX -r "$ROOT/req_no_torch.txt"

echo "=== flash-attn 2.8.3 prebuilt wheel ==="
# Only file we knowingly pull over the proxy (~240MB): building it from source
# would take hours. abiTRUE matches torch>=2.7 wheels.
WHL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHL}"
if [ ! -f "$ROOT/$WHL" ]; then
    http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 \
        curl -fL --retry 5 --retry-delay 5 -o "$ROOT/$WHL" "$URL"
fi
retry $PIP install $PIP_IDX "$ROOT/$WHL"

echo "=== done: $ENV_NAME ==="
$PY -c "import torch, transformers, flash_attn; print('torch', torch.__version__, '| tf', transformers.__version__, '| fa', flash_attn.__version__, '| cuda', torch.cuda.is_available())"
