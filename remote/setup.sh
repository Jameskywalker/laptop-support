#!/usr/bin/env bash
# 在租的 GPU 机器上（Vast.ai，镜像 vastai/base-image）装环境：解包代码 → uv + Python 3.12 → requirements-gpu（不装 build123d）→ 试 CuPy。
# 本地先打包：git archive --format=tar.gz -o remote_bundle.tar.gz HEAD -- $(git ls-files '*.py') requirements.txt requirements-gpu.txt out/desk40/model/profile.json
# 再 scp remote_bundle.tar.gz 和本脚本到 /workspace/，ssh 上去 bash /workspace/setup.sh（5090 上约 1 分钟）
set -euo pipefail
mkdir -p /workspace/laptop_support && cd /workspace/laptop_support
tar xzf /workspace/remote_bundle.tar.gz
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.12 -q
grep -v build123d requirements.txt > /tmp/req.txt
sed 's#-r requirements.txt#-r /tmp/req.txt#' requirements-gpu.txt > /tmp/req-gpu.txt
uv pip install --python .venv/bin/python -q -r /tmp/req-gpu.txt
.venv/bin/python -c "import cupy as cp; p=cp.cuda.runtime.getDeviceProperties(0); print('GPU', p['name'].decode(), 'CC', p['major'], p['minor'], 'mem', round(p['totalGlobalMem']/2**30,1), 'GB'); a=cp.arange(10)**2; print('cupy ok', int(a.sum()))"
echo SETUP_DONE
