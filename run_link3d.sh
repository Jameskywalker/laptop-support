#!/usr/bin/env bash
# 面板 / 后横梁的无填充三维优化（端头是插指 + U 形卡子，link3d.py；2026-11-23 的选定版）
# 粗网格 1 mm（6 GB 显卡也放得下，每个约 10–15 分钟）→ 0.5 mm 正式网格（约 17 GB 显存，RTX 5090 上每个约 20 分钟；租机装环境见 remote/setup.sh）
# 后横梁从尖顶闭口管起步（--init_tube 17,10：均匀或实心起步会长成开口 U 形槽）；面板从现在的插指版面板起步（--init_ref 0.1）
# 需要 out/<方案>/clip_trial/{beam,panel}_clip_outline.json（clip_trial.py 生成）：它是对照的"现在的零件"
# 用法: ./run_link3d.sh [beam|panel|all]   环境变量 STAND_VARIANT（默认 desk40）、PANEL_G（面板用料 g，默认 83）
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export STAND_VARIANT=${STAND_VARIANT:-desk40}
O=out/$STAND_VARIANT
WHAT=${1:-all}
FINE=(--h 0.5 --beta0 4 --beta_every 30 --maxit 150 --save_every 10)
if [ "$WHAT" = beam ] || [ "$WHAT" = all ]; then
  $PY -u link3d.py --part beam --h 1 --init_tube 17,10 --init_ref 0.05 --beta0 4 --out "$O/link3d_beam_h1"
  $PY -u link3d.py --part beam "${FINE[@]}" --init "$O/link3d_beam_h1/result.npz" --out "$O/link3d_beam"
fi
if [ "$WHAT" = panel ] || [ "$WHAT" = all ]; then
  $PY -u link3d.py --part panel --h 1 --mass_g "${PANEL_G:-83}" --init_ref 0.1 --out "$O/link3d_panel_h1"
  $PY -u link3d.py --part panel "${FINE[@]}" --mass_g "${PANEL_G:-83}" --init "$O/link3d_panel_h1/result.npz" --out "$O/link3d_panel"
fi
# 预览网页（对照用的 STL 先跑 clip_trial.py --skip_fe）：
#   $PY make_web_link3d.py --beam "三维=$O/link3d_beam" --panel "三维 83 g=$O/link3d_panel"
