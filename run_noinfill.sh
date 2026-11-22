#!/usr/bin/env bash
# 支撑首选：无填充三维「夹头 + 中段」版（2026-11-22）：横梁接口不再整块固定，只在方槽两端、卡扣臂贴着的地方留 2 mm 厚的实心夹头，
# 两条卡扣臂之间的槽中段也交给优化（--joint clampfill）；横梁工况正反两个方向都算（--beam_rev）。
# 其余同上一个首选「中段断开」：45° 悬空规则 + --island_connect，屈曲 ≥ 20，最薄 1.9 mm；斜面中段（前、顶两个面板接口区之间，s ≈ 69–153 mm）
# 不贴笔记本：载荷只加在头尾，中段斜面下 1.5 mm 固定为空
# nz16 粗网格 → nz24 细网格（从粗网格结果插值起步）→ 导出 STL + 复核 → out/<方案>/model_noinfill_<TAG>/
# 需要 CUDA 显卡（CuPy）；RTX 5090 上约 15 分钟。租机装环境见 remote/setup.sh
# 用法: ./run_noinfill.sh            环境变量 STAND_VARIANT（默认 desk40）、TAG（默认 clampfillrev）、
#                                   JOINT（默认 clampfill = 只固定方槽两端的夹头，两臂之间的槽中段也交给优化；clamp = 槽中段留空；
#                                          block = 接口区 26 × 50 整块固定，墙 + 填充）、
#                                   REV（默认 1 = 加反向横梁工况；0 = 只有正向）、
#                                   MID（默认 joints；none = 整条顶面都贴，即之前的「贴床连到主框」版）、GAP（默认 1.5 mm）、
#                                   EXTRA（其余参数原样加给两段优化）
#   上一个首选「中段断开」：JOINT=block REV=0 EXTRA=--legacy_slot TAG=ends ./run_noinfill.sh（--legacy_slot：当时横梁力摊在整圈槽壁上）
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export STAND_VARIANT=${STAND_VARIANT:-desk40}
TAG=${TAG:-clampfillrev}
MID=${MID:-joints}
GAP=${GAP:-1.5}
JOINT=${JOINT:-clampfill}
REV=${REV:-1}
O=out/$STAND_VARIANT
FLAGS=(--tmin 1.9 --buck 20 --island_connect --joint "$JOINT")
if [ "$REV" = 1 ]; then
  FLAGS+=(--beam_rev)
fi
if [ -n "${EXTRA:-}" ]; then
  read -r -a _ex <<< "$EXTRA"; FLAGS+=("${_ex[@]}")
fi
if [ "$MID" != none ]; then
  FLAGS+=(--mid "$MID" --gap "$GAP")
fi
$PY -u topopt3d_noinfill.py --nz 16 --maxit 240 "${FLAGS[@]}" --save_every 20 --out "$O/noinfill_${TAG}_nz16"
$PY -u topopt3d_noinfill.py --nz 24 --maxit 110 "${FLAGS[@]}" --init "$O/noinfill_${TAG}_nz16/result.npz" --beta0 4 --beta_every 30 \
    --save_every 10 --out "$O/noinfill_${TAG}"
$PY -u export3d_noinfill.py --dir "$O/noinfill_${TAG}" --no-gable --smooth 0
# TPU 卡套（顶面垫默认分头、尾两段），检查和支撑的贴合：
#   $PY pads.py --support "$O/model/support_left.stl" "$O/model_noinfill_${TAG}/support_noinfill_left.stl"
# 预览网页（和「只留夹头」「中段断开」对照：先 JOINT=clamp TAG=clamprev 跑一遍；中段断开按正反横梁工况复核
#   $PY export3d_noinfill.py --dir $O/noinfill_ends --no-gable --smooth 0 --beam_rev --md $O/model_noinfill_ends_rev）：
#   $PY make_web_noinfill.py --dir "$O/model_noinfill_${TAG}" --cmp-dir "$O/model_noinfill_clamprev" "$O/model_noinfill_ends_rev" \
#       --names "夹头 + 中段优化,夹头+中段,只留夹头,只留夹头,现首选（整块接口区）,现首选" --title "无填充支撑 · 横梁接口只留夹头" --lead "…" --cmp-note "…"
