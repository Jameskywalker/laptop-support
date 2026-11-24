#!/usr/bin/env bash
# 面板 / 后横梁的无填充三维优化（端头是插指 + U 形卡子，link3d.py）
# 粗网格 1 mm（本地 6 GB 显卡也放得下，每个约 10–15 分钟）→ 0.5 mm 正式网格（约 17 GB 显存，A100 / 5090 上每个约 20 分钟；租机装环境见 remote/setup.sh）
# 新默认权重（--weights relaxed）：面内主承力 ~85%，面外保底 ~15%；面板默认 60 g（投影放宽到 1.4×），后横梁默认 30 g（不再强迫闭口尖顶管）
# 需要 out/<方案>/clip_trial/{beam,panel}_clip_outline.json（clip_trial.py 生成）：它是对照的"现在的零件"
# 用法: ./run_link3d.sh [beam|panel|all]   环境变量 STAND_VARIANT（默认 desk40）、PANEL_G（默认 60）、BEAM_G（默认 30）、OPEN_SCALE（默认 1.4）、WEIGHTS（默认 relaxed）
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export STAND_VARIANT=${STAND_VARIANT:-desk40}
O=out/$STAND_VARIANT
WHAT=${1:-all}
WEIGHTS=${WEIGHTS:-relaxed}
OPEN_SCALE=${OPEN_SCALE:-1.4}
FINE=(--h 0.5 --beta0 4 --beta_every 30 --maxit 150 --save_every 10 --weights "$WEIGHTS")
if [ "$WHAT" = beam ] || [ "$WHAT" = all ]; then
  $PY -u link3d.py --part beam --h 1 --weights "$WEIGHTS" --mass_g "${BEAM_G:-30}" --init_ref 0.1 --out "$O/link3d_beam_h1"
  $PY -u link3d.py --part beam "${FINE[@]}" --mass_g "${BEAM_G:-30}" --init "$O/link3d_beam_h1/result.npz" --out "$O/link3d_beam"
fi
if [ "$WHAT" = panel ] || [ "$WHAT" = all ]; then
  $PY -u link3d.py --part panel --h 1 --weights "$WEIGHTS" --mass_g "${PANEL_G:-60}" --open_scale "$OPEN_SCALE" --init_ref 0.1 --out "$O/link3d_panel_h1"
  $PY -u link3d.py --part panel "${FINE[@]}" --mass_g "${PANEL_G:-60}" --open_scale "$OPEN_SCALE" --init "$O/link3d_panel_h1/result.npz" --out "$O/link3d_panel"
fi
# 预览网页（对照用的 STL 先跑 clip_trial.py --skip_fe）：
#   $PY make_web_link3d.py --beam "轻量化三维=$O/link3d_beam" --panel "轻量化三维=$O/link3d_panel"
