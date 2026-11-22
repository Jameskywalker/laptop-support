#!/usr/bin/env bash
# 支撑首选：无填充三维优化「贴床连到主框」版（45° 悬空规则 + --island_connect，屈曲 ≥ 20，最薄 1.9 mm）
# nz16 粗网格 → nz24 细网格（从粗网格结果插值起步）→ 导出 STL + 复核 → out/<方案>/model_noinfill_<TAG>/
# 需要 CUDA 显卡（CuPy）；RTX 5090 上约 12 分钟。租机装环境见 remote/setup.sh
# 用法: ./run_noinfill.sh            环境变量 STAND_VARIANT（默认 desk40）、TAG（默认 isc）
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export STAND_VARIANT=${STAND_VARIANT:-desk40}
TAG=${TAG:-isc}
O=out/$STAND_VARIANT
FLAGS=(--tmin 1.9 --buck 20 --island_connect)
$PY -u topopt3d_noinfill.py --nz 16 --maxit 240 "${FLAGS[@]}" --save_every 20 --out "$O/noinfill_${TAG}_nz16"
$PY -u topopt3d_noinfill.py --nz 24 --maxit 110 "${FLAGS[@]}" --init "$O/noinfill_${TAG}_nz16/result.npz" --beta0 4 --beta_every 30 \
    --save_every 10 --out "$O/noinfill_${TAG}"
$PY -u export3d_noinfill.py --dir "$O/noinfill_${TAG}" --no-gable --smooth 0
# 预览网页（和不限悬空版对照，需先有 $O/model_noinfill_noam）：
#   $PY make_web_noinfill.py --dir "$O/model_noinfill_${TAG}" --cmp-dir "$O/model_noinfill_noam" --names 连到主框,连到主框,不限悬空,不限悬空
