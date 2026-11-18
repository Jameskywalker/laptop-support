#!/usr/bin/env bash
# 一键重算：拓扑优化 → 后处理/FE 复核 → 三维建模(STEP/STL) → 校验与渲染 → 网页预览
# 用法: ./run_all.sh [方案...]   默认 desk35 desk40 desk45；上一版为 raised25
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
VARIANTS=("${@:-desk35 desk40 desk45}")
for v in ${VARIANTS[@]}; do
  echo "=== $v ==="
  export STAND_VARIANT=$v
  $PY topopt2d.py --robust --h 1 --rmin 4 --maxit 450
  $PY postprocess.py
  $PY build3d.py
  $PY render.py
done
$PY make_web.py
