#!/usr/bin/env bash
# 支撑首选：无填充三维优化「中段断开」版（2026-11-22）：45° 悬空规则 + --island_connect，屈曲 ≥ 20，最薄 1.9 mm；
# 斜面中段（前、顶两个面板接口区之间，s ≈ 69–153 mm）不贴笔记本：载荷只加在头尾，中段斜面下 1.5 mm 固定为空
# nz16 粗网格 → nz24 细网格（从粗网格结果插值起步）→ 导出 STL + 复核 → out/<方案>/model_noinfill_<TAG>/
# 需要 CUDA 显卡（CuPy）；RTX 5090 上约 15 分钟。租机装环境见 remote/setup.sh
# 用法: ./run_noinfill.sh            环境变量 STAND_VARIANT（默认 desk40）、TAG（默认 ends）、
#                                   MID（默认 joints；none = 整条顶面都贴，即之前的「贴床连到主框」版）、GAP（默认 1.5 mm）
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export STAND_VARIANT=${STAND_VARIANT:-desk40}
TAG=${TAG:-ends}
MID=${MID:-joints}
GAP=${GAP:-1.5}
O=out/$STAND_VARIANT
FLAGS=(--tmin 1.9 --buck 20 --island_connect)
if [ "$MID" != none ]; then
  FLAGS+=(--mid "$MID" --gap "$GAP")
fi
$PY -u topopt3d_noinfill.py --nz 16 --maxit 240 "${FLAGS[@]}" --save_every 20 --out "$O/noinfill_${TAG}_nz16"
$PY -u topopt3d_noinfill.py --nz 24 --maxit 110 "${FLAGS[@]}" --init "$O/noinfill_${TAG}_nz16/result.npz" --beta0 4 --beta_every 30 \
    --save_every 10 --out "$O/noinfill_${TAG}"
$PY -u export3d_noinfill.py --dir "$O/noinfill_${TAG}" --no-gable --smooth 0
# TPU 卡套（顶面垫默认分头、尾两段），检查和支撑的贴合：
#   $PY pads.py --support "$O/model/support_left.stl" "$O/model_noinfill_${TAG}/support_noinfill_left.stl"
# 预览网页（和整条顶面版对照：先 TAG=iscfix MID=none 跑一遍，再按头尾载荷复核
#   $PY export3d_noinfill.py --dir $O/noinfill_iscfix --no-gable --smooth 0 --mid joints --md $O/model_noinfill_iscfix_atends）：
#   $PY make_web_noinfill.py --dir "$O/model_noinfill_${TAG}" --cmp-dir "$O/model_noinfill_iscfix_atends" --names 中段断开三维,中段断开,整条顶面三维,整条顶面
