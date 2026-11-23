#!/usr/bin/env bash
# 一键重算：支撑拓扑优化 → 后处理/FE 复核 → 面板拓扑优化 → 三维建模(STEP/STL) → 校验与渲染 → 网页预览
# 用法: ./run_all.sh [方案...]   默认 desk40；上一版为 raised25，任意角度用 deskNN
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
VARIANTS=("${@:-desk40}")
# 装了 requirements-gpu.txt 且有可用显卡时，拓扑优化自动用 GPU；TOPO_GPU=0 强制用 CPU
GPU_FLAG=()
if [ "${TOPO_GPU:-auto}" != 0 ] && $PY -c "import cupy, nvmath; cupy.cuda.runtime.getDeviceCount()" 2>/dev/null; then
  GPU_FLAG=(--gpu)
fi
for v in ${VARIANTS[@]}; do
  echo "=== $v ==="
  export STAND_VARIANT=$v
  $PY topopt2d.py --robust --h 1 --rmin 4 --maxit 450 "${GPU_FLAG[@]}"
  $PY postprocess.py
  $PY panel.py "${GPU_FLAG[@]}"      # 有面板的方案：面板拓扑优化（卡扣当刚性夹住，X 斜撑）→ 作为下一步的参考设计；没有则跳过
  $PY panel_contact.py "${GPU_FLAG[@]}"   # 卡扣按真实接触、加支撑扭转工况、四分之一对称重新优化 → out/<方案>/panel_contact/（build3d 用这个）
  $PY build3d.py
  $PY render.py
  if [ "${PADS:-0}" = 1 ]; then      # 可选：TPU 卡套（顶面垫、前挡垫、前后脚套），检查和支撑的贴合
    $PY pads.py --support "out/$v/model/support_left.stl"
  fi
done
$PY make_web.py
