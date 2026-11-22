"""
TPU 垫（可选，所有步骤之后再做，不参与优化）：U 形卡套，靠弹性卡在支撑边上，不用胶、不改支撑。
  顶面垫：笔记本底面压在上面；前端顶住前挡垫。默认（--mid joints）分头、尾两段，只盖前挡到前接口区后沿、顶接口区前沿到顶面后端，
         中段不盖 —— 支撑首选「中段断开」版斜面中段比斜面低 1.5 mm，笔记本只压在头尾；二维版装上也一样只靠头尾。--mid none 为整条
  前挡垫：前挡内侧，笔记本前缘靠在上面；下端坐在顶面垫上
  前脚套、后脚套：脚底 + 端面包住，防滑；受桌面摩擦时不会往外滑脱
每件都是直的 U 形槽：底面朝下打印，侧壁竖直，没有悬空。U 形对厚度方向对称，左右支撑通用，各一套。
侧壁内侧比支撑厚度每边窄 --squeeze（过盈），靠 TPU 弹性夹紧。

坐标（与支撑 STL 相同）：x、y 是支撑轮廓平面，z 是厚度方向 0..T。
给了 --support 时检查：卡套和支撑的过盈体积、侧壁实际贴到支撑的面积比例、离卡扣卡钩的距离。
用法: STAND_VARIANT=desk40 python pads.py [--support out/desk40/model/support_left.stl ...]
"""
import argparse, json, os
import numpy as np
import shapely
from shapely.geometry import Polygon, box
import manifold3d as m3d
import trimesh
import params as P

RHO_TPU = 1.21      # g/cm³


def channel(L, tb, wall, depth, sq, caps=(False, False), cap_t=None):
    """局部坐标 (a 沿边, q 沿外法向, z 厚度) 的 U 形槽：
    底 a∈[0,L], q∈[0,tb]（在支撑外面）；两侧壁 q∈[−depth, tb]，内侧面在 z = sq 和 T − sq（每边过盈 sq）；
    caps：在 a=0 / a=L 端再包住支撑端面（厚 cap_t）"""
    T = P.T; ct = cap_t or tb
    parts = [m3d.Manifold.cube((L, tb, T + 2 * wall)).translate((0, 0, -wall)),
             m3d.Manifold.cube((L, depth + tb, wall + sq)).translate((0, -depth, -wall)),
             m3d.Manifold.cube((L, depth + tb, wall + sq)).translate((0, -depth, T - sq))]
    if caps[0]:
        parts.append(m3d.Manifold.cube((ct, depth + tb, T + 2 * wall)).translate((-ct, -depth, -wall)))
    if caps[1]:
        parts.append(m3d.Manifold.cube((ct, depth + tb, T + 2 * wall)).translate((L, -depth, -wall)))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def placed(man, A, t):
    """局部 (a, q, z) → 支撑坐标：点 = A + a·t + q·n，n = t 顺时针转 90°（外法向在 t 的右手边）"""
    t = np.asarray(t, float) / np.linalg.norm(t)
    n = np.array([t[1], -t[0]])
    M = np.array([[t[0], n[0], 0, A[0]], [t[1], n[1], 0, A[1]], [0, 0, 1, 0]], float)
    return man.transform(M)


def printed(man, tb, wall):
    """局部 → 打印姿态：底面外侧（q = tb）贴床，x' = a，y' = z + wall，z' = tb − q"""
    M = np.array([[1, 0, 0, 0], [0, 0, 1, wall], [0, -1, 0, tb]], float)
    return man.transform(M)


def to_tm(man):
    mm = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def pad_specs(a):
    """各垫：名称 → (局部 U 形槽, 起点 A, 沿边方向 t, 底厚, 长度)。外法向 n = t 顺时针转 90°，指向支撑外面"""
    F = P.sn2xy(0.0, 0.0)
    U, N = P.U, P.N
    specs = {}
    # 顶面：从前挡内侧（s=0）沿斜面到 S_END − gap；t = U 时右手边是 −N，所以反过来从后往前走，外法向 = +N
    Ltop = P.S_END - a.gap_end
    mid = P.mid_span(a.mid)
    if mid is None:
        specs["顶面垫"] = (channel(Ltop, a.base, a.wall, a.depth, a.squeeze), F + Ltop * U, -U, a.base, Ltop)
    else:      # 中段断开：头 s = 0..s0，尾 s = s1..S_END − gap
        s0, s1 = mid
        specs["顶面垫·头"] = (channel(s0, a.base, a.wall, a.depth, a.squeeze), F + s0 * U, -U, a.base, s0)
        specs["顶面垫·尾"] = (channel(Ltop - s1, a.base, a.wall, a.depth, a.squeeze), F + Ltop * U, -U, a.base, Ltop - s1)
    # 前挡内侧：s=0 的面朝 +U（朝笔记本）；从顶面垫上面（n = base）到前挡顶（n = LIP_H）；t = +N 时右手边 = +U
    Llip = P.LIP_H - a.base
    specs["前挡垫"] = (channel(Llip, a.base, a.wall, a.depth, a.squeeze), F + a.base * N, N, a.base, Llip)
    # 脚：底边 y = 0，外法向朝下；t = +x 时右手边是 −y。前脚在 x = 0 端包住前端面，后脚在 x = X_MAX 端包住后端面
    Lf = P.FOOT_LEN
    specs["前脚套"] = (channel(Lf, a.foot, a.wall, a.depth, a.squeeze, caps=(True, False)), np.array([0.0, 0.0]), np.array([1.0, 0.0]), a.foot, Lf)
    specs["后脚套"] = (channel(Lf, a.foot, a.wall, a.depth, a.squeeze, caps=(False, True)), np.array([P.X_MAX - Lf, 0.0]), np.array([1.0, 0.0]), a.foot, Lf)
    return specs


def check_support(fn, specs, a):
    """卡套和支撑：过盈体积（应接近两侧壁 × 过盈的理论值）、底与支撑的重叠（应为 0）、侧壁贴合比例"""
    sm = trimesh.load(fn)
    sup = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(sm.vertices, np.float32), tri_verts=np.asarray(sm.faces, np.uint32)))
    out = {}
    T = P.T
    sl = {}
    for zz in (0.1, T - 0.1):
        rings = [Polygon(r).buffer(0) for r in sup.slice(zz).to_polygons() if len(r) >= 3]
        g = Polygon()
        for r in rings:
            g = g.symmetric_difference(r)
        sl[zz] = g
    for nm, (ch, A, t, tb, Lseg) in specs.items():
        pm = placed(ch, A, t)
        inter = (pm ^ sup).volume()
        t_ = np.asarray(t, float) / np.linalg.norm(t); n_ = np.array([t_[1], -t_[0]])
        # 侧壁在支撑面上的投影：a∈[0,L], q∈[−depth, 0]
        q0 = np.array(A); foot = Polygon([q0, q0 + Lseg * t_, q0 + Lseg * t_ - a.depth * n_, q0 - a.depth * n_])
        grip = [round(sl[zz].intersection(foot).area / foot.area, 3) for zz in sl]
        # 离卡钩：卡扣卡钩在外侧面（z<0）伸出，取方槽两端的搭接带，按 1 mm 网格采样
        xs, ys = np.meshgrid(np.arange(foot.bounds[0], foot.bounds[2], 0.25), np.arange(foot.bounds[1], foot.bounds[3], 0.25))
        xs, ys = xs.ravel(), ys.ravel()
        inside = shapely.contains_xy(foot.buffer(1.0), xs, ys)
        hook = np.zeros(xs.size, bool)
        for j in P.JOINTS:
            hook |= P.hook_strip(j, xs, ys, 0.0)
        out[nm] = dict(过盈体积_mm3=round(inter, 1), 理论过盈_mm3=round(2 * Lseg * a.depth * a.squeeze, 1),
                       侧壁贴合比例_z0_zT=grip, 离卡钩不到1mm=bool((inside & hook).any()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=1.5, help="顶面垫、前挡垫的底厚 mm（笔记本压在上面）")
    ap.add_argument("--foot", type=float, default=2.0, help="脚套底厚 mm")
    ap.add_argument("--wall", type=float, default=1.2, help="侧壁厚 mm")
    ap.add_argument("--depth", type=float, default=3.0, help="侧壁往支撑里伸的深度 mm（夹住支撑的高度）")
    ap.add_argument("--squeeze", type=float, default=0.15, help="每边过盈 mm（侧壁内侧比支撑厚度每边窄这么多）")
    ap.add_argument("--gap_end", type=float, default=2.0, help="顶面垫比顶面短多少 mm（后端留空）")
    ap.add_argument("--mid", default="joints", help="顶面垫在斜面中段断开：joints = 前、顶两个面板接口区之间（默认，配支撑首选「中段断开」版）；s0,s1；none = 整条")
    ap.add_argument("--support", nargs="*", default=[], help="支撑 STL，检查贴合")
    ap.add_argument("--out", default=f"{P.OUT_DIR}/pads")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    specs = pad_specs(a)
    info = dict(params=vars(a), parts={})
    asm = []
    for nm, (ch, A, t, tb, L) in specs.items():
        pr = printed(ch, tb, a.wall)
        ext = to_tm(pr).extents
        fn = {"顶面垫": "pad_top", "顶面垫·头": "pad_top_front", "顶面垫·尾": "pad_top_rear", "前挡垫": "pad_lip", "前脚套": "boot_front", "后脚套": "boot_rear"}[nm]
        to_tm(pr).export(f"{a.out}/{fn}.stl")
        info["parts"][nm] = dict(file=f"{fn}.stl", 数量=2, 尺寸_mm=[round(float(v), 1) for v in ext],
                                 体积_cm3=round(ch.volume() / 1000, 2), 重量_g=round(ch.volume() / 1000 * RHO_TPU, 1))
        asm.append(placed(ch, A, t))
    to_tm(m3d.Manifold.batch_boolean(asm, m3d.OpType.Add)).export(f"{a.out}/pads_assembled_one_support.stl")
    info["合计_g（两片支撑）"] = round(2 * sum(p["重量_g"] for p in info["parts"].values()), 1)
    info["checks"] = {os.path.relpath(fn): check_support(fn, specs, a) for fn in a.support}
    json.dump(info, open(f"{a.out}/pads.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
