"""
笔记本支架 —— 全部几何/载荷参数（单位 mm, N, MPa）

通过环境变量 STAND_VARIANT 选择方案：
  deskNN（如 desk40）：前挡直接落在桌面，靠 NN° 倾角抬高屏幕（默认 desk40）
  raised25                  ：上一版，25° 倾角、整体抬高（笔记本前沿离桌约 100 mm）
输出目录：out/<方案>/topo（拓扑优化）、out/<方案>/model（模型与校核）

坐标系（支撑侧面轮廓平面）：x 由前向后，y 向上，原点在支撑最前端的桌面上。
支撑件沿 z 方向拉伸（厚度 T），打印时侧面贴平台，因此拉伸体天然无悬空。
"""
import os
import re
import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import Polygon, box, Point, LineString

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

# 拓扑优化体积约束：raised 方案按整个设计域计（上一版做法）；前挡落桌方案只按可优化区域计
# （非设计区越多——比如加横梁接口——留给结构的材料也不会被挤占）
VF_MODE = "domain" if MODE == "raised" else "active"
VOLFRAC = 0.25 if MODE == "raised" else 0.20

# ---------------- 打印机 ----------------
PRINTER = "H3C"           # 拓竹 H3C，打印尺寸 330 × 320 × 325 mm
BED_W, BED_D = 330.0, 320.0

# ---------------- 支撑件 ----------------
T = 15.0                  # 支撑板厚（拉伸方向 = 打印 z 方向）
SPACING = 220.0           # 两支撑中心距

# ---------------- 横梁接口：卡扣（无螺钉） ----------------
# 横梁两端各一对弹性卡扣臂，穿过支撑上的贯通方槽，卡钩扣住支撑外侧面；梁身台肩顶住内侧面。
# 卡扣臂在打印平面内弯曲（应力沿层内），卡钩/斜面/让位槽都是平面图形沿 z 拉伸 → 无悬空。
# 所有横梁是同一个零件；每根横梁在支撑轮廓里的位置 (x, y) 和截面方向 phi 由 JOINTS 给出：
#   phi = 横梁截面长边（40 mm，卡扣臂所在方向）与水平方向的夹角（度）。
USE_JOINT = True
BEAM_W, BEAM_H = 16.0, 40.0     # 横梁截面（短边 × 长边）
TENON_W, TENON_H = 16.0, 30.0   # 卡扣臂外包络：与梁身同宽，上下各留 5 mm 台肩
CLEAR = 0.2               # 方槽单边间隙
JOINT_MARGIN = 5.0        # 接口周围实体壁厚
SLOT_CORNER_R = 0.6       # 方槽四角让位圆（打印内角会被喷嘴倒圆）
ARM_T = 2.5               # 卡扣臂厚
ARM_ROOT = 15.0           # 卡扣臂伸进梁身的长度（两侧开让位槽，加长弹性段、降低应变）
RELIEF_W = 1.5            # 卡扣臂外侧让位槽宽
SLIT_W = 1.5              # 卡扣臂内侧让位缝宽（扣入时臂往里弯 HOOK_H − CLEAR = 1.2 mm）；两臂之间其余部分实心（2026-11-22 前是整块挖空 25 × 15 mm）
HOOK_H = 1.4              # 卡钩高度：扣住外侧面的搭接量 = HOOK_H - CLEAR = 1.2 mm
HOOK_FLAT = 1.0           # 卡钩顶部平台长
HOOK_RAMP = 3.0           # 导入斜面长
SNAP_PLAY = 0.15          # 卡钩与支撑外侧面之间的轴向间隙

_HB = BEAM_W / 2 + JOINT_MARGIN          # 接口实体区半宽（短边方向）= 13
_HA = BEAM_H / 2 + JOINT_MARGIN          # 接口实体区半长（长边方向）= 25
if MODE == "raised":
    JOINTS = [dict(name="rear", cn="后", x=212.0, y=75.0, phi=90.0, part="beam")]
else:
    # 三个接口放在支撑三角形的三个顶点附近（2026-11-20 起）。
    # 前、顶两个接口的截面都与斜面平行、在同一平面上（离笔记本底面 5 mm），由一块拓扑优化的面板连起来（part="panel"，见 panel.py）；
    # 后接口仍是单独一根横梁（part="beam"）。
    # 接口实体区顶边贴齐斜面（2026-11-21 起；原来在顶面实体带下方，孔中心离斜面 18 mm）：
    # 接口区和斜面直接连在一起，无填充三维优化不用再长材料去连；面板离笔记本底面从 10 mm 变成 5 mm
    # 前（前下顶点）：尽量靠前，直到接口区最低角碰到桌面
    _nc = -_HB
    _sc = _HA + ((_HB - _nc) * np.cos(TH) - P0[1]) / np.sin(TH) + 2.0
    _Jf = P0 + _sc * U + _nc * N
    # 顶（上顶点）：顶面后端
    _Jt = P0 + (S_END - _HA - 2.0) * U + _nc * N
    # 后（后下顶点）：后脚正上方，截面平放，离桌面 BASE_H
    JOINTS = [
        dict(name="front", cn="前", x=round(float(_Jf[0]), 1), y=round(float(_Jf[1]), 1), phi=THETA_DEG, part="panel"),
        dict(name="top", cn="顶", x=round(float(_Jt[0]), 1), y=round(float(_Jt[1]), 1), phi=THETA_DEG, part="panel"),
        dict(name="rear", cn="后", x=round(X_MAX - _HA - 2.0, 1), y=_HB, phi=0.0, part="beam"),
    ]

# ---------------- 材料（等效 PLA/PETG 打印件） ----------------
E_MAT = 2500.0            # MPa，考虑填充/层间后的等效模量
NU = 0.35

# ---------------- 三维优化（topopt3d.py） ----------------
# 打印时支撑侧面贴床，z = 打印方向。三维优化用"高度场"：每根 1×1 mm 柱子只有一个高度，
# 材料只能从顶面往下去掉 → 每层都完整落在下一层之上，无悬空；底面平整贴床。
LAYER_H = 0.2             # 层高 = z 向单元高度
KZ = 0.5                  # z 向（层间）模量、层间剪切模量的折减系数，模拟层间结合弱于层内
T3_MAX = 19.2             # 初始模型厚度（96 层）；多重网格要求层数能被 8 整除
T3_DILATE = 3.0           # 二维结果向外膨胀量，作为三维初始模型的轮廓
T3_MIN = 3.0              # 最小厚度：比它薄的柱子刚度按 (t/T3_MIN)^3 惩罚，避免出现薄皮
F_SIDE = 10.0             # 侧推：笔记本被横向推动，经顶面摩擦传给单侧支撑（z 向）
F_JOINT_Z = 10.0          # 横梁传来的轴向力（z 向）
# 打印结构（三维优化 --model print 与复核用）：墙、上下表皮为实心，其余为稀疏填充
LINE_W = 0.42             # 挤出线宽
NWALL = 3                 # 墙圈数 → 墙厚 1.26 mm
NSKIN = 4                 # 顶/底实心层数（各 0.8 mm）
INFILL = 0.15             # gyroid 填充率；刚度按 INFILL² 折算，用料按 INFILL 计

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


def _place(g, j):
    """局部坐标（u 沿横梁截面长边，v 沿短边）→ 支撑轮廓坐标"""
    return shapely.affinity.translate(shapely.affinity.rotate(g, j["phi"], origin=(0, 0)), j["x"], j["y"])


def joint_rect(b, a, j):
    """以横梁 j 为中心、短边 b × 长边 a 的矩形"""
    return _place(box(-a / 2, -b / 2, a / 2, b / 2), j)


def joint_slot(j=None):
    """支撑上供卡扣穿过的贯通方槽（含四角让位圆）；j=None 时返回所有横梁方槽的并集"""
    if j is None:
        return shapely.unary_union([joint_slot(jj) for jj in JOINTS])
    a, b = TENON_H + 2 * CLEAR, TENON_W + 2 * CLEAR
    corners = [Point(su * a / 2, sv * b / 2).buffer(SLOT_CORNER_R, quad_segs=8) for su in (-1, 1) for sv in (-1, 1)]
    return _place(shapely.unary_union([box(-a / 2, -b / 2, a / 2, b / 2)] + corners), j)


SLOT_A, SLOT_B = TENON_H + 2 * CLEAR, TENON_W + 2 * CLEAR    # 方槽长边（卡扣臂所在方向）× 短边


def slot_uv(j, x, y):
    """全局坐标 → 横梁 j 的方槽局部坐标（u 沿长边，v 沿短边，原点在槽中心）"""
    ph = np.deg2rad(j["phi"]); c, s = np.cos(ph), np.sin(ph)
    dx, dy = np.asarray(x, float) - j["x"], np.asarray(y, float) - j["y"]
    return dx * c + dy * s, -dx * s + dy * c


def slot_bearing(j, x, y, f, tol):
    """横梁面内力 f 的承压权重：卡扣臂顶在法向朝向 f 的槽壁上，权重 = max(0, 槽壁外法向·f 方向)。
    只取离槽壁 tol 以内的点；其余为 0。"""
    u, v = slot_uv(j, x, y)
    a2, b2 = SLOT_A / 2, SLOT_B / 2
    near = (np.abs(u) <= a2 + tol) & (np.abs(v) <= b2 + tol) & ((np.abs(u) >= a2 - tol) | (np.abs(v) >= b2 - tol))
    on_u = np.abs(u) / a2 >= np.abs(v) / b2                     # 落在短边壁（u = ±a/2）上
    nu = np.where(on_u, np.sign(u), 0.0); nv = np.where(on_u, 0.0, np.sign(v))
    ph = np.deg2rad(j["phi"]); c, s = np.cos(ph), np.sin(ph)
    fx, fy = np.asarray(f, float) / np.hypot(*f)
    w = np.maximum(0.0, (nu * c - nv * s) * fx + (nu * s + nv * c) * fy)
    return np.where(near, w, 0.0)


def slot_bearing_arms(j, x, y, f, tol):
    """同 slot_bearing，但只算卡扣臂真正贴着的槽壁：短边壁全长；长边壁只在两端 |u| ≥ TENON_H/2 − ARM_T（臂内侧面）以外。
    支撑里只有两条卡扣臂（2.5 × 16 mm，方槽长边两端各一条），两臂之间（|u| &lt; 12.5）没有任何东西碰到长边壁"""
    u, v = slot_uv(j, x, y)
    on_u = np.abs(u) / (SLOT_A / 2) >= np.abs(v) / (SLOT_B / 2)          # 短边壁（同 slot_bearing 的分法）
    return np.where(on_u | (np.abs(u) >= TENON_H / 2 - ARM_T - tol), slot_bearing(j, x, y, f, tol), 0.0)


# ---------------- 接口「夹头」（无填充三维 --joint clamp / clampfill） ----------------
# 横梁在支撑里只有两条卡扣臂，限位全在方槽两端：短边壁挡 ±u；长边壁两端（臂贴着的 2.7 mm）挡 ±v；
# 卡钩搭接带（外侧面 z = 0）挡往里拉；梁身台肩（内侧面 z = T，|u| 到 BEAM_H/2）挡往外推。
# 夹头 = 方槽每一端 |u| ∈ [CLAMP_U0, BEAM_H/2]、|v| ≤ SLOT_B/2 + CLAMP_WALL 的全厚度实心块（去掉方槽）；
# 原来 26 × 50 mm 接口区的其余部分（长边壁中段、外圈墙）交给优化。
CLAMP_WALL = 2.0                                  # 夹头上下槽壁厚（≥ 最小厚度 1.9 mm）
CLAMP_U0 = TENON_H / 2 - ARM_T - 0.5              # 夹头内端（臂内侧面 12.5 再让 0.5）= 12.0
ARM_SWEEP = HOOK_H - CLEAR                        # 扣入时卡扣臂向内弯 1.2 mm
KEEP_U0 = TENON_H / 2 - ARM_T - ARM_SWEEP - 0.5   # 槽中段可填时，两端留空口袋的内端 = 10.8（臂弯进来再让 0.5）


def _uv_box(j, u0, u1, v0, v1):
    return _place(box(u0, v0, u1, v1), j)


def joint_clamps(j):
    """横梁 j 方槽两端的夹头（全厚度实心，已去掉方槽）"""
    b = SLOT_B / 2 + CLAMP_WALL
    g = shapely.union(_uv_box(j, CLAMP_U0, BEAM_H / 2, -b, b), _uv_box(j, -BEAM_H / 2, -CLAMP_U0, -b, b))
    return g.difference(joint_slot(j))


def joint_keepout(j, fill_mid=False):
    """必须留空的部分：fill_mid=False → 整个方槽；True → 只留方槽两端卡扣臂（含扣入时向内弯的 1.2 mm）的口袋，
    两臂之间的槽中段也交给优化（梁身在两臂之间开了让位槽，伸进支撑的只有两条臂）"""
    s = joint_slot(j)
    if not fill_mid:
        return s
    big = SLOT_A
    return s.intersection(shapely.union(_uv_box(j, KEEP_U0, big, -big, big), _uv_box(j, -big, -KEEP_U0, -big, big)))


def hook_strip(j, x, y, tol):
    """卡钩扣住支撑外侧面的搭接带：方槽两端（长边方向）各一条，宽 HOOK_H − CLEAR"""
    u, v = slot_uv(j, x, y)
    return (np.abs(u) >= SLOT_A / 2 - tol) & (np.abs(u) <= TENON_H / 2 + HOOK_H + tol) & (np.abs(v) <= TENON_W / 2 + tol)


# ---------------- 面板（前、顶两个接口合成一块，见 panel.py） ----------------
PANEL_T = BEAM_W          # 面板厚 = 横梁截面短边，四角卡扣与横梁端头完全相同；平放打印
PANEL_VF = 0.20           # 面板拓扑优化的体积分数（四角实体区不计入）
PANEL_BODY = 25.0         # 四角实体区沿横向的长度（卡扣臂根部和让位槽都在里面）
F_RACK = 10.0             # 面板工况的力（两片支撑上部相对下部横向错动、互相撑开、前后错动）


def panel_joints():
    """属于面板的接口，按沿斜面位置从前到后排"""
    js = [j for j in JOINTS if j.get("part") == "panel"]
    return sorted(js, key=lambda j: float(xy2sn(j["x"], j["y"])[0]))


def mid_span(mid):
    """斜面中段不贴笔记本的范围（沿斜面 s，mm）：None / "none" → None（整条贴）；
    "joints" → 前、顶两个面板接口区之间（接口区顶边贴齐斜面，笔记本只压在前挡到前接口区、顶接口区到顶面后端这两段上）；
    "s0,s1" 或 (s0, s1) → (s0, s1)"""
    if mid is None or (isinstance(mid, str) and mid.strip().lower() in ("", "none")):
        return None
    if isinstance(mid, str) and mid.strip() == "joints":
        s = [float(xy2sn(j["x"], j["y"])[0]) for j in panel_joints()]
        return (s[0] + _HA, s[-1] - _HA)
    v = [float(t) for t in mid.split(",")] if isinstance(mid, str) else [float(t) for t in mid]
    assert len(v) == 2 and v[0] < v[1], f"中段范围应为 s0,s1：{mid}"
    return tuple(v)


def snap_end(sx, half, yc=0.0):
    """横梁/面板一端的卡扣，打印平面坐标：X 沿横向，支撑内侧面在 X = sx·half；Y 沿截面长边，卡扣中心在 Y = yc。
    返回 (要并上的多边形, 要挖掉的多边形)：两条卡扣臂 + 卡钩 + 导入斜面；臂外侧让位槽、臂内侧让位缝（SLIT_W），
    两臂之间离支撑内侧面 0.5 mm 以内留空（不伸进方槽）。两臂之间原来整块挖空，卡扣和本体之间只剩两条台肩条连着。"""
    yA = TENON_H / 2; yI = yA - ARM_T
    uc = T + SNAP_PLAY; uL = uc + HOOK_FLAT + HOOK_RAMP

    def T_(u, y):
        return (sx * (half + u), yc + y)
    parts, cuts = [], []
    for sy in (1, -1):
        parts.append(Polygon([T_(-1, sy * yI), T_(uL, sy * yI), T_(uL, sy * yA), T_(-1, sy * yA)]))
        parts.append(Polygon([T_(uc, sy * (yA - 0.01)), T_(uc, sy * (yA + HOOK_H)), T_(uc + HOOK_FLAT, sy * (yA + HOOK_H)), T_(uL, sy * (yA - 0.01))]))
        w = RELIEF_W
        cuts.append(LineString([T_(-ARM_ROOT + w / 2, sy * (yA + w / 2)), T_(2, sy * (yA + w / 2))]).buffer(w / 2))
        s = SLIT_W
        cuts.append(LineString([T_(-ARM_ROOT + s / 2, sy * (yI - s / 2)), T_(3, sy * (yI - s / 2))]).buffer(s / 2))
    cuts.append(Polygon([T_(-0.5, -yI), T_(3, -yI), T_(3, yI), T_(-0.5, yI)]))
    return parts, cuts


def snap_strain():
    """卡扣臂扣入时的最大弯曲应变（悬臂梁近似 ε = 1.5·t·δ/L²）"""
    delta = HOOK_H - CLEAR
    L = ARM_ROOT + T + SNAP_PLAY
    return 1.5 * ARM_T * delta / L ** 2


# ---------------- 刚性插指 + U 形卡子（2026-11-22 试验：SNAP_END=clip） ----------------
# 端头不再用弹性卡扣臂：两根实心插指正好占住原来卡扣臂和它往里弯的口袋（|y| FING_IN–15），
# 后接口槽中间那根穿槽斜撑（clampfill 的设计区）可以保留。插指伸出支撑外侧面，外侧各开一个缺口；
# U 形卡子从截面短边方向压进去：两条实心腿夹在支撑外侧面和缺口外壁之间，承受往外拔的力；
# 中间两条细卡扣往里弯一下、卡钩扣住插指底面，只管卡子自己不掉，弯的方向和受力方向垂直。
SNAP_END = os.environ.get("SNAP_END", "arm")      # "arm" = 弹性卡扣臂（现在），"clip" = 插指 + 卡子
FING_IN = KEEP_U0 + 0.3        # 插指内侧面 |y| = 11.1（离 clampfill 让位口袋内端 0.3 mm）
FING_LEN = 24.0                # 插指从支撑内侧面量起的长度
NECK_Y = 13.5                  # 缺口处插指只剩 |y| ∈ [FING_IN, NECK_Y]
CLIP_T = 5.0                   # 卡子厚（沿横梁轴向）
CLIP_PLAY = 0.2                # 卡子腿和缺口外壁之间的间隙
NOTCH_U0 = T - 0.2             # 缺口从支撑外侧面里面 0.2 mm 开始：卡子腿只压支撑，不碰缺口内壁
NOTCH_U1 = T + CLIP_T + CLIP_PLAY                 # 缺口外壁 = 20.2
LEG_Y = (NECK_Y + 0.2, 19.0)   # 卡子腿 |y| 范围：压在支撑外侧面 |y| 15.2–19（夹头实心区 |u| ≤ 20）
CLIP_ARM_T = 2.0               # 中间细卡扣宽（弯的方向）
CLIP_HOOK = 0.8                # 卡钩扣住插指底面的搭接
CLIP_GAP = 0.2                 # 卡子横梁和插指顶面、卡钩和插指底面之间的间隙
NOTCH_R = 0.0                  # 缺口内角倒圆半径（卡子腿对应倒角）


def clip_end(sx, half, yc=0.0):
    """插指端头，打印平面坐标同 snap_end。返回 (要并上的, 要挖掉的)：两根插指；插指外侧缺口；两插指之间离支撑内侧面 0.5 mm 以内留空"""
    def T_(u, y):
        return (sx * (half + u), yc + y)

    def B(u0, u1, y0, y1):
        return Polygon([T_(u0, y0), T_(u1, y0), T_(u1, y1), T_(u0, y1)])
    parts, cuts = [], []
    for sy in (1, -1):
        y0, y1 = sorted([sy * FING_IN, sy * TENON_H / 2])
        parts.append(B(-1.0, FING_LEN, y0, y1))
        n0, n1 = sorted([sy * NECK_Y, sy * (TENON_H / 2 + 1.0)])
        c = B(NOTCH_U0, NOTCH_U1, n0, n1)
        cuts.append(c.buffer(-NOTCH_R, quad_segs=8).buffer(NOTCH_R, quad_segs=8) if NOTCH_R > 0 else c)
    cuts.append(B(-0.5, FING_LEN + 1.0, -FING_IN, FING_IN))
    return parts, cuts


def end_geometry(sx, half, yc=0.0, style=None):
    return (clip_end if (style or SNAP_END) == "clip" else snap_end)(sx, half, yc)


def clip_outline():
    """U 形卡子的打印轮廓（平面朝下打印，厚 CLIP_T）：x 沿截面长边（插指 y），y 沿截面短边（v，0 = 横梁中面，卡子从 +v 一侧压进去）"""
    h = BEAM_W / 2
    top0 = h + CLIP_GAP; top1 = top0 + 4.0
    a1 = FING_IN - 0.3; a0 = a1 - CLIP_ARM_T; hk = FING_IN + CLIP_HOOK
    hb = -h - CLIP_GAP                     # 卡钩扣合面
    g = [box(-LEG_Y[1], top0, LEG_Y[1], top1)]
    for s in (1, -1):
        g.append(box(*sorted([s * LEG_Y[0], s * LEG_Y[1]])[:1], -h - 1.0, *sorted([s * LEG_Y[0], s * LEG_Y[1]])[1:], top0 + 0.01))
        # 卡钩下端只伸到插指底面下 1.5 mm：后接口离支撑底边只有 5 mm，后脚套侧壁包到底边上方 3 mm
        pts = [(a0, top0 + 0.01), (a0, hb - 1.3), (a1, hb - 1.3), (hk, hb - 0.4), (hk, hb), (a1, hb), (a1, top0 + 0.01)]
        g.append(Polygon([(s * x, y) for x, y in pts]))
    return shapely.unary_union(g)


def clip_strain():
    """卡子中间细卡扣扣入时的最大弯曲应变（悬臂梁 ε = 1.5·t·δ/L²，L 从卡子横梁到卡钩中部）"""
    L = (BEAM_W / 2 + CLIP_GAP) + (BEAM_W / 2 + CLIP_GAP + 0.4)
    return 1.5 * CLIP_ARM_T * CLIP_HOOK / L ** 2


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
        # 接口区实体，方槽是贯通孔：从设计域和所有实体区里挖掉，优化时就按有孔计算
        for j in JOINTS:
            solids["joint_" + j["name"]] = joint_rect(2 * _HB, 2 * _HA, j)
        slots = joint_slot()
        domain = domain.difference(slots)
        solids = {k: g.difference(slots) for k, g in solids.items()}
    return domain, solids
