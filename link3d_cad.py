"""
将面板 / 后横梁正式版三维拓扑优化结果参数化重构为纯直线杆件 + 圆角 CAD 模型（manifold3d），
并用 link3d.py 的非线性单向接触 GPU 有限元复核刚度与悬空检查。

用法:
  STAND_VARIANT=desk40 .venv/bin/python link3d_cad.py --part panel --out out/desk40/link3d_panel --mass-g 59.5
  STAND_VARIANT=desk40 .venv/bin/python link3d_cad.py --part beam  --out out/desk40/link3d_beam  --mass-g 29.5
"""
import argparse
import json
import os
import time
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely import affinity
import manifold3d as m3d
import trimesh

import params as P
import link3d as L3


def fillet_2d(geom, r_in=3.0, r_out=0.0):
    """用形态学闭/开运算在直杆交汇内角生成精确圆弧倒圆，保持直线段平直。"""
    g = geom.buffer(0)
    if r_in > 0:
        g = g.buffer(r_in, join_style=1, quad_segs=16).buffer(-r_in, join_style=1, quad_segs=16)
    if r_out > 0:
        g = g.buffer(-r_out, join_style=1, quad_segs=12).buffer(r_out, join_style=1, quad_segs=12)
    return g.simplify(0.012, preserve_topology=True)


def quad_mirror(geom, xc=0.0, yc=0.0):
    """将第一象限几何沿 x=xc 和 y=yc 四象限镜像合并。"""
    g_lr = geom.union(affinity.scale(geom, -1.0, 1.0, origin=(xc, yc)))
    return g_lr.union(affinity.scale(g_lr, 1.0, -1.0, origin=(xc, yc))).buffer(0)


def bar(p0, p1, w):
    """从 p0 到 p1 构造全宽 w 的平直杆（方头端盖，避免节点凸出圆包）。"""
    return LineString([p0, p1]).buffer(0.5 * w, cap_style=2, join_style=2)


def extrude_2d(geom, z0, z1):
    """将 2D Shapely Polygon/MultiPolygon 沿 Z 从 z0 拉伸到 z1 为 manifold。"""
    parts = []
    for p in L3._polys(geom):
        if p.area < 1e-6:
            continue
        p = shapely.geometry.polygon.orient(p, 1.0)
        rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(i.coords)[:-1] for i in p.interiors]
        parts.append(m3d.Manifold.extrude(m3d.CrossSection(rings, m3d.FillRule.EvenOdd), z1 - z0).translate((0.0, 0.0, z0)))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add) if parts else m3d.Manifold()


def extrude_xz_prism(xz_pts, y0, y1):
    """由 (x, z) 多边形沿 Y ∈ [y0, y1] 拉伸生成正交修剪体（保证零阶梯斜坡面）。"""
    p = shapely.geometry.polygon.orient(Polygon(xz_pts), 1.0)
    rings = [np.asarray(p.exterior.coords)[:-1]]
    man = m3d.Manifold.extrude(m3d.CrossSection(rings, m3d.FillRule.EvenOdd), y1 - y0)
    return man.rotate([90.0, 0.0, 0.0]).translate((0.0, y1, 0.0))


def _clip_and_finalize(man_print, G):
    """裁剪到允许设计域、合并精确插指与台肩固定区（全在打印坐标中），清理并返回单连通流形。"""
    dx = -(L3.HALF + L3.UE)
    dy = -G.V0
    pg = G.polys
    low = affinity.translate(pg["body"].difference(pg["cut"]).difference(pg["rec"]), dx, dy)
    high = affinity.translate(pg["body"].difference(pg["cut"]).difference(pg["rec_all"]), dx, dy)
    allowed = extrude_2d(low, 0.0, min(L3.TS, G.H))
    fixed = affinity.translate(pg["fing"].union(pg["band"]).difference(pg["cut"]), dx, dy)
    man = (man_print ^ allowed) + extrude_2d(fixed, 0.0, L3.TS)
    # Remove any detached tiny islands if created at boolean seam; keep largest body
    parts = man.decompose()
    if len(parts) > 1:
        pos_parts = [p for p in parts if p.volume() > 0]
        main_body = max(pos_parts, key=lambda p: p.volume())
        man = main_body
    return man


def build_panel_cad(G, scale=1.0):
    """构造正式版轻量化面板的纯直线杆件四象限对称 CAD 模型（打印坐标）。"""
    yc = 0.5 * (G.VC[0] + G.VC[1]) - G.V0  # 66.97 mm
    u_junc, dy_junc = 71.5, 39.5
    u_end = 102.5

    # 主 X 斜撑（底框内缘至三叉节点，再折向台肩内臂）
    w_x1 = 4.3 * scale
    w_x2 = 4.0 * scale
    w_diag = 3.5 * scale
    w_tie = 2.0 * scale

    q_x = bar((0.0, 0.0), (u_junc, dy_junc), w_x1).union(bar((u_junc, dy_junc), (u_end, 50.5), w_x2))
    q_x_pad = q_x.union(box(u_end - 3.5, 46.5, u_end + 0.5, 56.0))
    x_2d = fillet_2d(quad_mirror(affinity.translate(q_x_pad, 0.0, yc), 0.0, yc), r_in=4.5, r_out=0.3)

    # 低层（0..4.8 mm）加强桁架：四角斜撑 + 竖直细拉条 + 端头台肩之间桥接壁 + 主 X 撑底座
    q_base = (
        q_x
        .union(bar((u_junc, dy_junc), (u_end, 82.0), w_diag))
        .union(bar((u_junc, 0.0), (u_junc, dy_junc), w_tie))
        .union(box(u_end - 3.5, 46.5, u_end + 0.5, 56.0))
        .union(box(u_end - 3.5, 77.5, u_end + 0.5, 87.0))
        .union(box(98.5, 55.0, u_end + 0.5, 79.0))
    )
    base_2d = fillet_2d(quad_mirror(affinity.translate(q_base, 0.0, yc), 0.0, yc), r_in=3.0, r_out=0.25)

    # 端头指间桥接壁（|X| >= 98.5，z=0..6.8 mm）：直接传递前后错动剪力与扭转
    q_bridge = box(98.5, 54.0, u_end + 0.5, 80.0)
    bridge_2d = quad_mirror(affinity.translate(q_bridge, 0.0, yc), 0.0, yc)

    # 主 X 撑顶面高度轮廓（平坦 13 mm 中段 + 平面斜坡升至 16 mm）
    xz_panel = [(-104.0, 0.0), (104.0, 0.0), (104.0, 16.0), (76.0, 16.0), (54.0, 13.0),
                (-54.0, 13.0), (-76.0, 16.0), (-104.0, 16.0)]
    man_x = extrude_2d(x_2d, 0.0, 16.0) ^ extrude_xz_prism(xz_panel, -25.0, 160.0)
    man_base = extrude_2d(base_2d, 0.0, 4.8)
    man_bridge = extrude_2d(bridge_2d, 0.0, 6.8)
    return _clip_and_finalize(man_x + man_base + man_bridge, G)


def build_beam_cad(G, scale=1.0):
    """构造正式版轻量化后横梁的纯直线杆件四象限对称 CAD 模型（打印坐标）。"""
    u_end = 102.5
    # 两侧贯通立肋中心线：中段 y=15.0，外段外移至 y=17.0（最大化抗压拢/扭转惯性矩），末端顺接入台肩 [11.1, 20.0]
    w_rib = 3.1 * scale
    w_truss = 2.9 * scale

    def y_rib(u):
        if u <= 26.0:
            return 15.0
        if u <= 36.0:
            return 15.0 + 2.0 * (u - 26.0) / 10.0
        return 17.0

    rib_pts = [(0.0, 15.0), (26.0, 15.0), (36.0, 17.0), (u_end, 17.0)]
    q_rib = LineString(rib_pts).buffer(0.5 * w_rib, cap_style=2, join_style=2).union(box(95.0, 11.1, u_end + 0.5, 20.0))
    rib_2d = fillet_2d(quad_mirror(q_rib, 0.0, 0.0), r_in=2.5, r_out=0.25)

    # 两侧立肋对称拱顶 XZ 剖面：中段 6.2 mm 缓升至 12.2 mm 再平面斜坡接 16.0 mm 插指台肩（斜率 ≤ 38°）
    xz_rib = [(-104.0, 0.0), (104.0, 0.0), (104.0, 16.0), (99.0, 16.0), (93.0, 12.0),
              (75.0, 10.2), (50.0, 8.2), (24.0, 6.2), (-24.0, 6.2), (-50.0, 8.2),
              (-75.0, 10.2), (-93.0, 12.0), (-99.0, 16.0), (-104.0, 16.0)]
    man_ribs = extrude_2d(rib_2d, 0.0, 16.0) ^ extrude_xz_prism(xz_rib, -25.0, 25.0)

    # 底面 6 格对称 X 桁架（0..2.0 mm）：对齐拓扑优化节距的纯直线斜杆 + 圆形倒圆
    nodes_c = (0.0, 25.0, 45.0, 79.0)
    nodes_r = (12.0, 35.0, 61.0, 93.5)
    q_truss = q_rib
    segs = [
        ((nodes_c[0], 0.0), (nodes_r[0], y_rib(nodes_r[0]))),
        ((nodes_c[1], 0.0), (nodes_r[0], y_rib(nodes_r[0]))),
        ((nodes_c[1], 0.0), (nodes_r[1], y_rib(nodes_r[1]))),
        ((nodes_c[2], 0.0), (nodes_r[1], y_rib(nodes_r[1]))),
        ((nodes_c[2], 0.0), (nodes_r[2], y_rib(nodes_r[2]))),
        ((nodes_c[3], 0.0), (nodes_r[2], y_rib(nodes_r[2]))),
        ((nodes_c[3], 0.0), (nodes_r[3], y_rib(nodes_r[3]))),
    ]
    for p0, p1 in segs:
        q_truss = q_truss.union(bar(p0, p1, w_truss))
    truss_2d = fillet_2d(quad_mirror(q_truss, 0.0, 0.0), r_in=2.0, r_out=0.2)
    man_truss = extrude_2d(truss_2d, 0.0, 2.0)

    return _clip_and_finalize(man_ribs + man_truss, G)


def voxelize_for_fe(man_print, G):
    """按 h=1.0 mm 体素中心平面切片重建 B_cad (NX, NY, NZ) 供 link3d 接触有限元复核。"""
    ox = L3.HALF + L3.UE
    oy = G.V0
    A, V = np.meshgrid(G.ac - ox, G.vc - oy, indexing="ij")
    ax = A.ravel()
    vy = V.ravel()
    B_slice = np.zeros((G.NX, G.NY, G.NZ), dtype=bool)
    for k in range(G.NZ):
        z = float(G.zc[k])
        if z > L3.TS:
            continue
        cs = man_print.slice(z)
        rings = [Polygon(r).buffer(0) for r in cs.to_polygons() if len(r) >= 3]
        if not rings:
            continue
        g = Polygon()
        for r in rings:
            g = g.symmetric_difference(r)
        if not g.is_empty:
            B_slice[:, :, k] = shapely.contains_xy(g, ax, vy).reshape(G.NX, G.NY)
    return np.where(G.design3, B_slice & G.dom3, G.solid3)


def fit_mass_cad(G, part, target_mass_g):
    """二分调整杆件名义宽度系数，使精准 CAD 实体重量匹配目标质量。"""
    builder = build_panel_cad if part == "panel" else build_beam_cad
    lo, hi = 0.72, 1.38
    best = None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        man = builder(G, scale=mid)
        mass_g = man.volume() / 1000.0 * L3.RHO
        best = (mid, man, mass_g)
        if abs(mass_g - target_mass_g) < 0.08:
            break
        if mass_g > target_mass_g:
            hi = mid
        else:
            lo = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("panel", "beam"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mass-g", type=float, default=None, help="目标实体重量 g（默认面板 59.5、横梁 29.5）")
    ap.add_argument("--no-verify", action="store_true", help="跳过 GPU 非线性接触有限元复核")
    ap.add_argument("--simp", type=float, default=0.0, help="导出前可选 manifold 简化容差（默认 0 保持绝对平直）")
    a = ap.parse_args()

    t0 = time.time()
    rep_path = f"{a.out}/report.json"
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    h_val = float(rep.get("h", 1.0))
    G = L3.Geo(a.part, h_val, rep.get("H"), None)
    target_g = a.mass_g if a.mass_g is not None else (59.5 if a.part == "panel" else 29.5)

    scale, man, mass_g = fit_mass_cad(G, a.part, target_g)
    if a.simp > 0:
        man = man.simplify(a.simp)
    parts = man.decompose()
    cav = sorted([-p.volume() for p in parts if p.volume() < 0])
    bodies = int(sum(p.volume() > 0 for p in parts))
    mm = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
    stl_path = f"{a.out}/{a.part}_3d.stl"
    tm.export(stl_path)
    oh = L3.overhang_check(man, float(tm.bounds[1][2]))
    stl_meta = dict(
        volume_cm3=round(man.volume() / 1000.0, 2),
        mass_g=round(man.volume() / 1000.0 * L3.RHO, 1),
        watertight=bool(tm.is_watertight),
        bbox_mm=[round(float(x), 1) for x in (tm.bounds[1] - tm.bounds[0])],
        genus=int(man.genus()),
        faces=int(len(tm.faces)),
        bodies=bodies,
        cavities_mm3=[round(float(v), 1) for v in cav],
        cad_scale=round(float(scale), 4),
    )
    print(f"[{a.part}] CAD 直线拟合 (scale={scale:.4f}): STL={stl_meta}", flush=True)
    print(f"悬空检查：{oh}", flush=True)

    if not a.no_verify:
        G.contacts()
        cases = L3.make_cases(G, rep.get("weights_preset", "relaxed"))
        fe = L3.GPUFE(G, tol=1e-5)
        E_ref, m_ref, A_ref, _ = G.reference_fields()
        if "ref" in rep and len(rep["ref"]) == len(cases):
            ref = rep["ref"]
            ref_C = np.array([r["柔度"] for r in ref])
        else:
            ref = L3.evaluate(fe, G, cases, E_ref, E_ref >= 0.5, "现在")
            for r in ref:
                r.pop("act")
            ref_C = np.array([r["柔度"] for r in ref])
            rep["ref"] = ref
            rep["ref_mass_g"] = round(float(m_ref.sum()) * G.vox / 1000.0 * L3.RHO, 1)
            rep["ref_proj_mm2"] = A_ref

        B_cad = voxelize_for_fe(man, G)
        vox_mass_g = float(B_cad.sum()) * G.vox / 1000.0 * L3.RHO
        proj_mm2 = float((B_cad.any(2) & G.body2).sum()) * G.h ** 2
        new_cad = L3.evaluate(fe, G, cases, B_cad.astype(float), B_cad, "CAD直线")
        fe.drop()
        for r in new_cad:
            r.pop("act")
        w = np.array([q.w for q in cases])
        ratio = np.array([n["柔度"] for n in new_cad]) / ref_C
        in_mask = np.array([q.group == "面内" for q in cases])
        out_mask = np.array([q.group == "面外" for q in cases])
        w_all = float(w @ ratio / w.sum())
        w_in = float(w[in_mask] @ ratio[in_mask] / w[in_mask].sum())
        w_out = float(w[out_mask] @ ratio[out_mask] / w[out_mask].sum())

        old_w = rep.get("weighted")
        print(f"\n=== 有限元复核对比 ({a.part}) ===", flush=True)
        print(f"  实体质量 {stl_meta['mass_g']} g（体素化 {vox_mass_g:.1f} g），投影 {proj_mm2:.0f} mm²", flush=True)
        print(f"  加权柔度比：{w_all:.4f}" + (f"（原体素版 {old_w:.4f}）" if old_w else "") + f" | 面内 {w_in:.4f} | 面外 {w_out:.4f}", flush=True)
        for q, r, ro in zip(cases, ratio, rep.get("ratio", [None] * len(cases))):
            s_old = f" (原 {ro:.3f})" if ro is not None else ""
            print(f"    {q.name}: {r:.3f}{s_old}", flush=True)

        rep.update(
            new=new_cad,
            ratio=ratio.tolist(),
            weighted=w_all,
            mass_g=stl_meta["mass_g"],
            proj_mm2=proj_mm2,
            weights=[float(x) for x in w],
        )

    rep.update(stl=stl_meta, overhang=oh, iso="cad_straight", smooth=0.0, cad_fit=True)
    json.dump(rep, open(rep_path, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"完成 {time.time() - t0:.1f}s -> {stl_path}", flush=True)


if __name__ == "__main__":
    main()
