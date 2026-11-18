"""
笔记本支架 —— 全部几何/载荷参数（单位 mm, N, MPa）

通过环境变量 STAND_VARIANT 选择方案：
  desk35 / desk40 / desk45 ：前挡直接落在桌面，靠 35–45° 倾角抬高屏幕（默认 desk40）
  raised25                  ：上一版，25° 倾角、整体抬高（笔记本前沿离桌约 100 mm）
输出目录：out/<方案>/topo（拓扑优化）、out/<方案>/model（模型与校核）

坐标系（支撑侧面轮廓平面）：x 由前向后，y 向上，原点在支撑最前端的桌面上。
支撑件沿 z 方向拉伸（厚度 T），打印时侧面贴平台，因此拉伸体天然无悬空。
"""
import os
import re
import numpy as np
import shapely
from shapely.geometry import Polygon, box, Point

VARIANT = os.environ.get("STAND_VARIANT", "desk40")
_m = re.fullmatch(r"(desk|raised)(\d+(?:\.\d+)?)", VARIANT)
assert _m, f"未知方案 {VARIANT}，应为 desk35/desk40/desk45/raised25 这类名字"
MODE, THETA_DEG = _m.group(1), float(_m.group(2))
OUT_DIR = f"out/{VARIANT}"
TOPO_DIR = f"{OUT_DIR}/topo"
MODEL_DIR = f"{OUT_DIR}/model"

# ---------------- 笔记本 ----------------
LAPTOP_W = 310.0          # 宽
LAPTOP_D = 220.0          # 深
LAPTOP_MASS = 1.6         # kg（13-14 寸上限）
DESIGN_MASS = 2.0         # kg，留余量

# ---------------- 姿态 ----------------
TH = np.deg2rad(THETA_DEG)
U = np.array([np.cos(TH), np.sin(TH)])     # 沿斜面向上（向后）
N = np.array([-np.sin(TH), np.cos(TH)])    # 斜面外法向（向上）
S_END = 205.0             # 支撑顶面沿斜面的长度（笔记本后部悬出 15 mm）
LIP_T = 8.0               # 前挡厚（沿斜面）
TB = 5.0                  # 顶面实体带厚度（非设计区）
FOOT_H = 5.0              # 脚垫实体厚度（非设计区）
BASE_H = 5.0              # 底边拉杆最小厚度（非设计区）
FILLET_R = 2.0            # 前挡内角圆角（降低前挡根部应力集中；笔记本边缘本身有圆角，不影响贴合）

if MODE == "raised":
    LIP_H = 12.0
    P0 = np.array([22.0, 100.0])           # 笔记本前下沿位置（s=0 处）
    X_MAX = 255.0
    Y_CAP_REAR = 165.0
    FOOT_LEN = 28.0
    FRONT_FIX, REAR_FIX = "xy", "y"        # 前脚铰支，后脚滑动
    _JOINT = (212.0, 75.0)
else:
    LIP_H = 15.0                           # 倾角大，前挡加高
    # 让前挡最前点落在 x=0、顶面实体带最低点落在桌面 y=0 —— 前挡直接接触桌面
    P0 = np.array([LIP_T * np.cos(TH) + LIP_H * np.sin(TH), LIP_T * np.sin(TH) + TB * np.cos(TH)])
    REAR_EXT = 95.0                        # 后脚伸出顶面后端的距离（防后翻；H3C 热床允许，2026-11-18 由 45 加长到 95）
    _E = P0 + S_END * U
    X_MAX = round(float(_E[0] + REAR_EXT), 1)
    Y_CAP_REAR = float(_E[1] - 25.0)       # 顶面后端之后的区域限高（给转轴/屏幕下沿留空间）
    FOOT_LEN = 25.0
    FRONT_FIX, REAR_FIX = "y", "xy"        # 前挡下方只给竖向支撑：前挡推力必须由结构自身传到后脚，不靠桌面摩擦
    # 横梁接口：放在"顶面后端 → 后脚"连线上、离地约 45% 的位置（后腿的天然传力路径上）
    _Fr = np.array([X_MAX - FOOT_LEN / 2, 0.0])
    _J = _E + 0.55 * (_Fr - _E)
    _JOINT = (round(float(_J[0]), 1), round(float(_J[1]), 1))

VOLFRAC = 0.25 if MODE == "raised" else 0.30   # 拓扑优化体积分数（相对设计域）

# ---------------- 打印机 ----------------
PRINTER = "H3C"           # 拓竹 H3C，打印尺寸 330 × 320 × 325 mm
BED_W, BED_D = 330.0, 320.0

# ---------------- 支撑件 ----------------
T = 15.0                  # 支撑板厚（拉伸方向 = 打印 z 方向）
SPACING = 220.0           # 两支撑中心距

# ---------------- 横梁接口：卡扣（无螺钉） ----------------
# 横梁两端各一对弹性卡扣臂，穿过支撑上的贯通方槽，卡钩扣住支撑外侧面；梁身台肩顶住内侧面。
# 卡扣臂在打印平面内弯曲（应力沿层内），卡钩/斜面/让位槽都是平面图形沿 z 拉伸 → 无悬空。
USE_JOINT = True
JX, JY = _JOINT           # 横梁截面中心（在支撑轮廓平面内）
BEAM_W, BEAM_H = 16.0, 40.0     # 横梁截面（x 向宽 × y 向高）
TENON_W, TENON_H = 16.0, 30.0   # 卡扣臂外包络：与梁身同宽，上下各留 5 mm 台肩
CLEAR = 0.2               # 方槽单边间隙
JOINT_MARGIN = 5.0        # 接口周围实体壁厚
SLOT_CORNER_R = 0.6       # 方槽四角让位圆（打印内角会被喷嘴倒圆）
ARM_T = 2.5               # 卡扣臂厚
ARM_ROOT = 15.0           # 卡扣臂伸进梁身的长度（两侧开让位槽，加长弹性段、降低应变）
RELIEF_W = 1.5            # 卡扣臂外侧让位槽宽
HOOK_H = 1.4              # 卡钩高度：扣住外侧面的搭接量 = HOOK_H - CLEAR = 1.2 mm
HOOK_FLAT = 1.0           # 卡钩顶部平台长
HOOK_RAMP = 3.0           # 导入斜面长
SNAP_PLAY = 0.15          # 卡钩与支撑外侧面之间的轴向间隙

# ---------------- 材料（等效 PLA/PETG 打印件） ----------------
E_MAT = 2500.0            # MPa，考虑填充/层间后的等效模量
NU = 0.35

# ---------------- 载荷（单个支撑） ----------------
G = 9.81
W1 = DESIGN_MASS * G / 2 * 1.5     # 笔记本重量分到单侧 × 1.5 动载系数 ≈ 14.7 N
F_PRESS = 30.0                     # 按压
F_PUSH = 10.0                      # 推屏 / 推机身
F_LIP_IMPACT = 40.0                # 把笔记本滑进支架时撞击前挡（沿斜面向下）
F_JOINT = (8.0, -15.0)             # 横梁传来的力（搬动、两侧受力不均）


def sn2xy(s, n):
    s = np.asarray(s, float); n = np.asarray(n, float)
    return P0[0] + s * U[0] + n * N[0], P0[1] + s * U[1] + n * N[1]


def xy2sn(x, y):
    dx = np.asarray(x) - P0[0]; dy = np.asarray(y) - P0[1]
    return dx * U[0] + dy * U[1], dx * N[0] + dy * N[1]


def sn_rect(s0, s1, n0, n1):
    xs, ys = sn2xy([s0, s1, s1, s0], [n0, n0, n1, n1])
    return Polygon(list(zip(xs, ys)))


def joint_rect(w, h):
    return box(JX - w / 2, JY - h / 2, JX + w / 2, JY + h / 2)


def joint_slot():
    """支撑上供卡扣穿过的贯通方槽（含四角让位圆）"""
    w, h = TENON_W + 2 * CLEAR, TENON_H + 2 * CLEAR
    corners = [Point(JX + sx * w / 2, JY + sy * h / 2).buffer(SLOT_CORNER_R, quad_segs=8) for sx in (-1, 1) for sy in (-1, 1)]
    return shapely.unary_union([joint_rect(w, h)] + corners)


def snap_strain():
    """卡扣臂扣入时的最大弯曲应变（悬臂梁近似 ε = 1.5·t·δ/L²）"""
    delta = HOOK_H - CLEAR
    L = ARM_ROOT + T + SNAP_PLAY
    return 1.5 * ARM_T * delta / L ** 2


def geometry(use_joint=USE_JOINT):
    """返回 (设计域, 非设计实体区 dict)"""
    big = 2000.0
    band = sn_rect(-LIP_T, S_END, -TB, 0.0)
    lip = sn_rect(-LIP_T, 0.0, 0.0, LIP_H)
    if MODE == "raised":
        below = sn_rect(-big, big, -big, 0.0)
        s_le_end = sn_rect(-big, S_END, -big, big)
        main = box(0, 0, X_MAX, big).intersection(below).intersection(s_le_end)
        rear = box(0, 0, X_MAX, Y_CAP_REAR).intersection(below)
        domain = shapely.unary_union([main, rear, lip, band]).intersection(box(0, 0, X_MAX, big))
        extra = {}
    else:
        E = sn2xy(S_END, 0); F = sn2xy(0, 0); Gp = sn2xy(0, LIP_H); H = sn2xy(-LIP_T, LIP_H)
        D = sn2xy(S_END, (Y_CAP_REAR - float(E[1])) / np.cos(TH))
        pts = [(0, 0), (X_MAX, 0), (X_MAX, Y_CAP_REAR), D, E, F, Gp, H, (0.0, float(H[1]))]
        domain = Polygon([(float(x), float(y)) for x, y in pts]).buffer(0)
        r = FILLET_R
        # 方块向 s<0、n<0 方向多伸 0.05 mm 与主体重叠，避免浮点误差导致圆角区与主体分离
        fillet = sn_rect(-0.05, r, -0.05, r).difference(Point(*map(float, sn2xy(r, r))).buffer(r, quad_segs=32))
        domain = shapely.unary_union([domain, fillet, band.intersection(box(0, 0, X_MAX, big))])
        extra = {"fillet": fillet}
    solids = {
        "band": band.intersection(domain),
        "lip": lip.intersection(domain),
        "foot_f": box(0, 0, FOOT_LEN, FOOT_H),
        "foot_r": box(X_MAX - FOOT_LEN, 0, X_MAX, FOOT_H),
        "base": box(0, 0, X_MAX, BASE_H),
        **extra,
    }
    if use_joint:
        solids["joint"] = joint_rect(BEAM_W + 2 * JOINT_MARGIN, BEAM_H + 2 * JOINT_MARGIN)
    return domain, solids
