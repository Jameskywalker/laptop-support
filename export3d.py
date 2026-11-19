"""
三维优化结果 → 可打印模型 + 复核
  1. 高度场取整到层后，逐层取平滑轮廓（列高 ≥ 该层的区域），并上精确的非设计几何、扣掉横梁方槽；
     上层轮廓强制包含在下层轮廓内 → 每层都落在下一层之上，无悬空。
  2. 各层轮廓逐段拉伸叠起，导出 STL（底面 z=0 平整贴床）。右支撑为镜像件。
  3. 把导出的几何体素化回三维网格，与二维方案（profile.json 拉伸 T，同样带方槽）在同一模型下对比柔度、应力和用料。
     主要按打印结构比较：每层轮廓向内 NWALL 圈墙（距离变换）、上下 NSKIN 层表皮为实心，其余为填充；另附按实心的比较。
用法: STAND_VARIANT=desk40 python export3d.py
"""
import argparse, json, os, time
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt
from skimage import measure
import manifold3d as m3d
import trimesh
import cupy as cp
import params as P
import topopt3d as T3

D3 = f"{P.OUT_DIR}/topo3d"
MD3 = f"{P.OUT_DIR}/model3d"


def polys(g):
    if g.is_empty: return []
    return list(g.geoms) if isinstance(g, MultiPolygon) else [g] if isinstance(g, Polygon) else [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def level_regions(tq, NX, NY, h, dz):
    """每个层高 L → 列高 ≥ L 的平滑区域"""
    domain, solids = P.geometry(True)
    for k in [k for k in solids if k.startswith("joint_")]:
        solids[k] = Polygon(solids[k].exterior).buffer(-3, join_style="mitre").buffer(3).difference(P.joint_slot())
    solid_all = shapely.unary_union(list(solids.values()))
    slots = P.joint_slot()
    d = 0.2
    xs = np.arange(-2, NX * h + 2, d); ys = np.arange(-2, NY * h + 2, d)
    XX, YY = np.meshgrid(xs, ys)
    G = tq.reshape(NX, NY).T
    levels = np.unique(np.round(tq[tq > 0] / dz).astype(int))
    regs, prev = {}, None
    for n in levels:
        L = n * dz
        ind = gaussian_filter((G >= L - 1e-6).astype(float), 0.6)
        Fd = map_coordinates(ind, [YY.ravel() / h - 0.5, XX.ravel() / h - 0.5], order=1, mode="constant", cval=0).reshape(XX.shape)
        rings = [Polygon(np.c_[xs[0] + c[:, 1] * d, ys[0] + c[:, 0] * d]).buffer(0) for c in measure.find_contours(Fd, 0.5) if len(c) >= 8]
        reg = shapely.unary_union([]) if not rings else rings[0]
        for pg in rings[1:]:
            reg = reg.symmetric_difference(pg)
        if L <= P.T + 1e-6:
            reg = reg.union(solid_all)
        reg = reg.intersection(domain)
        reg = reg.buffer(1.5).buffer(-1.5).buffer(-1.0).buffer(1.0).intersection(domain)   # 内凹 R1.5、外凸 R1
        if L <= P.T + 1e-6:
            reg = reg.difference(slots)
        if prev is not None:
            reg = reg.intersection(prev)                    # 上层包含于下层
        reg = shapely.unary_union([p for p in polys(reg) if p.area > 4.0]).simplify(0.03)
        if reg.is_empty:
            break
        regs[n] = reg; prev = reg
    return regs


def build_mesh(regs, dz):
    parts, z0 = [], 0.0
    for n, reg in regs.items():
        z1 = n * dz
        rings = []
        for p in polys(reg):
            rings.append(np.asarray(shapely.geometry.polygon.orient(p, 1.0).exterior.coords)[:-1])
            rings += [np.asarray(i.coords)[:-1] for i in shapely.geometry.polygon.orient(p, 1.0).interiors]
        cs = m3d.CrossSection(rings, m3d.FillRule.EvenOdd)
        parts.append(m3d.Manifold.extrude(cs, z1 - z0).translate((0, 0, z0)))
        z0 = z1
    man = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)   # 合并近邻顶点会弄破网格


def voxel_heights(regs, M):
    """导出几何 → 每根柱子的高度（柱中心落在哪些层的轮廓内）"""
    I, J = np.meshgrid(np.arange(M.NX), np.arange(M.NY), indexing="ij")
    xc, yc = ((I + 0.5) * M.h).ravel(), ((J + 0.5) * M.h).ravel()
    t = np.zeros(M.NX * M.NY)
    for n, reg in regs.items():
        t[shapely.contains_xy(reg, xc, yc)] = n * M.dz
    return t


def printed_fields(M, t, nwall, nskin, infill):
    """打印结构的体素刚度系数、实心材料掩码、用料（实心等效 mm³）。墙按每层轮廓的距离变换，与优化里的邻域腐蚀独立实现"""
    wall = nwall * P.LINE_W
    n = np.round(t / M.dz).astype(int)
    g = np.full((M.NX * M.NY, M.NZ), M.gmin, np.float32); solid = np.zeros_like(g, bool); m = 0.0
    for k in range(M.NZ):
        mk = (n > k).reshape(M.NX, M.NY)
        if not mk.any():
            continue
        db = distance_transform_edt(mk) * M.h - 0.5 * M.h            # 柱中心到轮廓的距离
        fw = np.clip((wall - (db - 0.5 * M.h)) / M.h, 0, 1).ravel()   # 柱里墙所占比例
        skin = (k < nskin) | (k >= n - nskin)
        on = mk.ravel()
        g[on, k] = (fw + (1 - fw) * np.where(skin, 1.0, infill ** 2))[on]
        m += float((fw + (1 - fw) * np.where(skin, 1.0, infill))[on].sum())
        solid[on, k] = ((fw > 0.5) | skin)[on]
    return cp.asarray(g.ravel()), cp.asarray(solid.ravel()), m * M.h ** 2 * M.dz


def main():
    t0 = time.time()
    os.makedirs(MD3, exist_ok=True)
    r = np.load(f"{D3}/result3d.npz")
    hist = json.load(open(f"{D3}/history3d.json"))
    ha = dict(model="solid", nwall=P.NWALL, nskin=P.NSKIN, infill=P.INFILL) | hist["args"]
    args = argparse.Namespace(**{k: ha[k] for k in ("tmax", "dilate", "tmin", "kz", "rmin", "nu_s", "omega", "model", "nwall", "nskin", "infill")})
    NX, NY, h, dz = int(r["NX"]), int(r["NY"]), float(r["h"]), float(r["dz"])
    regs = level_regions(r["tq"], NX, NY, h, dz)
    print(f"层数 {len(regs)}（最高 {max(regs) * dz:.1f} mm），轮廓 {time.time()-t0:.0f}s", flush=True)
    left = build_mesh(regs, dz)
    assert left.is_watertight
    left.export(f"{MD3}/support3d_left.stl")
    right = left.copy(); right.apply_transform(np.diag([-1, 1, 1, 1])); right.apply_translation((float(left.bounds[1][0]), 0, 0))
    assert right.is_watertight and right.volume > 0
    right.export(f"{MD3}/support3d_right.stl")                 # 镜像件：左右镜像，平面仍在 z=0 贴床
    prof = json.load(open(f"{P.MODEL_DIR}/profile.json"))
    pg2 = Polygon(prof["exterior"], prof["holes"])
    print(f"STL：{len(left.faces)} 面，体积 {left.volume/1000:.1f} cm³（二维方案 {pg2.area*P.T/1000:.1f} cm³）", flush=True)
    # 横梁梁身从接口区内侧面（z = T）伸出，截面范围内 T 以上不能有材料
    above = shapely.unary_union([g for n, g in regs.items() if n * dz > P.T + 1e-6])
    clash = sum(above.intersection(P.joint_rect(P.BEAM_W, P.BEAM_H, j)).area for j in P.JOINTS) if not above.is_empty else 0.0
    print(f"横梁截面与 T 以上材料的重叠面积 {clash:.3f} mm²", flush=True)
    assert clash < 1e-6

    # 复核：两种几何都体素化，带方槽
    M = T3.Model3D(args)
    I, J = np.meshgrid(np.arange(M.NX), np.arange(M.NY), indexing="ij")
    in2 = shapely.contains_xy(pg2, ((I + 0.5) * h).ravel(), ((J + 0.5) * h).ravel())
    t2 = np.where(in2, P.T, 0.0)
    t3 = voxel_heights(regs, M)
    res, res_s, mass = {}, {}, {}
    for nm, t in (("二维方案拉伸", t2), ("三维优化", t3)):
        M.reload(t >= P.T - 1e-6)                 # 载荷只加在实际几何的节点上（方槽里没有材料）
        gp, sm, mp = printed_fields(M, t, args.nwall, args.nskin, args.infill)
        M.U = [None] * len(M.F)
        comps, its = M.solve_g(gp)
        res[nm] = dict(comp=comps.tolist(), stress=M.stresses(cp.ones_like(gp), sm))   # 墙和表皮里实心材料的应力
        mass[nm] = dict(solid_equiv_cm3=mp / 1000, petg_g=mp / 1000 * 1.27)
        o = cp.clip((cp.asarray(t, cp.float32)[:, None] - M.zb[None, :]) / np.float32(M.dz), 0, 1).ravel()
        gs = (np.float32(M.gmin) + np.float32(1 - M.gmin) * o).astype(cp.float32)
        M.U = [None] * len(M.F)
        comps_s, _ = M.solve_g(gs)
        res_s[nm] = dict(comp=comps_s.tolist(), stress=M.stresses(gs, gs > 0.99))
        print(f"{nm}: 打印用料 {mp/1000:.1f} cm³（PETG 约 {mp/1000*1.27:.0f} g），CG {its}", flush=True)
    rel = np.array(res["三维优化"]["comp"]) / np.array(res["二维方案拉伸"]["comp"])
    rel_s = np.array(res_s["三维优化"]["comp"]) / np.array(res_s["二维方案拉伸"]["comp"])
    table = []
    for k, nm in enumerate(M.names):
        a2, a3 = res["二维方案拉伸"]["stress"][k], res["三维优化"]["stress"][k]
        table.append(dict(工况=nm, 柔度比=round(float(rel[k]), 3), 柔度比_实心=round(float(rel_s[k]), 3),
                          最大位移_mm=[round(a2["最大位移_mm"], 4), round(a3["最大位移_mm"], 4)],
                          VM_99_9=[round(a2["VM_99_9pct_MPa"], 2), round(a3["VM_99_9pct_MPa"], 2)],
                          VM_max=[round(a2["VM_max_MPa"], 2), round(a3["VM_max_MPa"], 2)],
                          层间拉_99_9=[round(a2["层间拉应力_99_9pct_MPa"], 3), round(a3["层间拉应力_99_9pct_MPa"], 3)],
                          层间剪_99_9=[round(a2["层间剪应力_99_9pct_MPa"], 3), round(a3["层间剪应力_99_9pct_MPa"], 3)]))
        print(table[-1])
    summary = dict(weighted_ratio=float((M.imp * rel).sum() / M.imp.sum()), weighted_ratio_solid=float((M.imp * rel_s).sum() / M.imp.sum()),
                   names=M.names, beam_clash_mm2=clash, mass=mass, print_setting=dict(nwall=args.nwall, nskin=args.nskin, infill=args.infill),
                   opt_model=args.model, stl_volume_cm3=left.volume / 1000, profile2d_volume_cm3=pg2.area * P.T / 1000,
                   layers=len(regs), max_height_mm=max(regs) * dz, table=table, raw=res, raw_solid=res_s)
    json.dump(summary, open(f"{MD3}/verify3d.json", "w"), ensure_ascii=False, indent=1)
    print(f"加权柔度 三维/二维：按打印结构 {summary['weighted_ratio']:.3f}，按实心 {summary['weighted_ratio_solid']:.3f}", flush=True)
    render_preview(regs, dz, f"{MD3}/preview3d.png")
    print(f"用时 {time.time()-t0:.0f}s")


def render_preview(regs, dz, fn):
    """预览图：每 5 层取一段、轮廓放粗，网格小一些好画"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    keys = list(regs)
    sub = {n: regs[n].simplify(0.4) for i, n in enumerate(keys) if i % 5 == 4 or n == keys[-1] or i == 0}
    m = build_mesh(sub, dz)
    fig = plt.figure(figsize=(20, 8))
    tri = m.vertices[m.faces]
    light = np.array([-0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    shade = 0.3 + 0.7 * np.clip(m.face_normals @ light, 0, 1)
    col = np.c_[0.32 * shade + 0.1, 0.5 * shade + 0.1, 0.72 * shade + 0.1, np.ones_like(shade)]
    for k, (elev, azim, title) in enumerate(((38, -58, "台阶面朝上（打印姿态；装配时朝内）"), (70, -90, "俯视"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors="none"))
        ax.set_xlim(0, 270); ax.set_ylim(0, 145); ax.set_zlim(0, 20)
        ax.set_box_aspect((270, 145, 20), zoom=1.15)
        ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=13)
    fig.savefig(fn, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
