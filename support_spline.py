"""
对无填充三维支撑按打印方向（Z）逐层水平截面（0.2 mm 层高）做保角三次 B-样条拟合，
消除面内 X/Y 方向 0.2 mm 体素锯齿（不改变层高 Z 向结构），再叠加为水密三维流形导出 STL 与预览对比。

用法:
  STAND_VARIANT=desk40 .venv/bin/python support_spline.py \
      --src out/desk40/model_noinfill_clampfillrev_lv020_oct \
      --out out/desk40/model_noinfill_clampfillrev_lv020_oct_spline \
      --rms 0.16
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import time
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from scipy.interpolate import splprep, splev
import manifold3d as m3d
import trimesh

import params as P
import topopt3d_noinfill as N3
import export3d_noinfill as E3


def _polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return [p for p in g.geoms if p.area > 0]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and p.area > 0]


def _resample_ring(coords, ds=0.25):
    ls = LineString(coords)
    L = ls.length
    n = max(8, int(round(L / ds)))
    us = np.linspace(0.0, L, n, endpoint=False)
    return np.array([[ls.interpolate(u).x, ls.interpolate(u).y] for u in us], dtype=np.float64)


def fit_ring_spline(coords, rms_tol=0.16, ds=0.25, corner_deg=45.0, win_mm=1.2):
    """对单条闭合轮廓做保角三次 B-样条平滑；尖角处打断保留端点，光顺段按自适应 RMS 拟合。"""
    ls0 = LineString(coords)
    if ls0.length < 2.0:
        return coords
    pts = _resample_ring(coords, ds=ds)
    n = len(pts)
    if n < 8:
        return coords

    w = max(2, int(round(win_mm / ds)))
    v_prev = pts - np.roll(pts, w, axis=0)
    v_next = np.roll(pts, -w, axis=0) - pts
    l1 = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-9
    l2 = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-9
    cos_a = np.clip((v_prev[:, 0] * v_next[:, 0] + v_prev[:, 1] * v_next[:, 1]) / (l1 * l2), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_a))

    is_c = np.zeros(n, dtype=bool)
    for i in range(n):
        if ang[i] >= corner_deg:
            nbr = [ang[(i + k) % n] for k in range(-w, w + 1)]
            if ang[i] >= max(nbr) - 1e-9:
                is_c[i] = True
    c_idx = np.nonzero(is_c)[0].tolist()

    def smooth_open(seg_pts, tol):
        m_s = len(seg_pts)
        if m_s < 5:
            return seg_pts
        seg_len = LineString(seg_pts).length
        for scale in (1.0, 0.5, 0.25):
            s_val = m_s * ((tol * scale) ** 2)
            try:
                tck, u = splprep([seg_pts[:, 0], seg_pts[:, 1]], s=s_val, k=3, per=0)
                xe, ye = splev(u, tck)
                max_d = float(np.max(np.hypot(xe - seg_pts[:, 0], ye - seg_pts[:, 1])))
                if max_d <= max(0.32, 2.2 * tol) or scale == 0.25:
                    out_n = max(6, int(round(seg_len / ds)))
                    xn, yn = splev(np.linspace(0.0, 1.0, out_n), tck)
                    res = np.column_stack([xn, yn])
                    res[0] = seg_pts[0]
                    res[-1] = seg_pts[-1]
                    return res
            except Exception:
                return seg_pts
        return seg_pts

    if not c_idx:
        p_cl = np.vstack([pts, pts[0:1]])
        for scale in (1.0, 0.5, 0.25):
            s_val = n * ((rms_tol * scale) ** 2)
            try:
                tck, u = splprep([p_cl[:, 0], p_cl[:, 1]], s=s_val, k=3, per=1)
                xe, ye = splev(u, tck)
                max_d = float(np.max(np.hypot(xe - p_cl[:, 0], ye - p_cl[:, 1])))
                if max_d <= max(0.32, 2.2 * rms_tol) or scale == 0.25:
                    out_n = max(12, int(round(ls0.length / ds)))
                    xn, yn = splev(np.linspace(0.0, 1.0, out_n, endpoint=False), tck)
                    res = np.column_stack([xn, yn])
                    return np.vstack([res, res[0:1]])
            except Exception:
                return coords
        return coords

    out_segs = []
    nc = len(c_idx)
    for k in range(nc):
        i0 = c_idx[k]
        i1 = c_idx[(k + 1) % nc]
        seg = pts[i0 : i1 + 1] if i1 > i0 else np.vstack([pts[i0:], pts[: i1 + 1]])
        sm = smooth_open(seg, rms_tol)
        out_segs.append(sm[:-1])
    res = np.vstack(out_segs)
    return np.vstack([res, res[0:1]])


def smooth_layer_geom(g, dom_outer, fixed, keepout, rms_tol=0.16, corner_deg=45.0):
    """拟合单层全部外环与内孔并严格按 CAD 边界/固定区/方槽保留槽裁剪。"""
    out = []
    for p in _polys(g):
        if p.area < 0.05:
            continue
        ext = fit_ring_spline(list(p.exterior.coords), rms_tol=rms_tol, corner_deg=corner_deg)
        ext_ls = LineString(ext).simplify(0.015, preserve_topology=False)
        ints = []
        for hole in p.interiors:
            if Polygon(hole).area < 0.05:
                continue
            hr = fit_ring_spline(list(hole.coords), rms_tol=rms_tol, corner_deg=corner_deg)
            hl = LineString(hr).simplify(0.015, preserve_topology=False)
            if len(hl.coords) >= 4:
                ints.append(list(hl.coords))
        if len(ext_ls.coords) >= 4:
            pg = Polygon(list(ext_ls.coords), ints).buffer(0)
            if not pg.is_empty:
                out.append(pg)
    res = shapely.unary_union(out) if out else Polygon()
    res = res.intersection(dom_outer).union(fixed).difference(keepout).buffer(0)
    # 清理极小孤立噪点 (< 0.05 mm²)
    kept = [p for p in _polys(res) if p.area >= 0.05]
    return shapely.unary_union(kept) if kept else Polygon()


def _worker_smooth(task):
    k, wkb_g, wkb_dom, wkb_fix, wkb_ko, rms_tol, corner_deg = task
    g = shapely.from_wkb(wkb_g)
    dom_outer = shapely.from_wkb(wkb_dom)
    fixed = shapely.from_wkb(wkb_fix)
    keepout = shapely.from_wkb(wkb_ko)
    n0 = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) for p in _polys(g))
    gs = smooth_layer_geom(g, dom_outer, fixed, keepout, rms_tol=rms_tol, corner_deg=corner_deg)
    n1 = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) for p in _polys(gs))
    sd = g.symmetric_difference(gs).area
    return k, shapely.to_wkb(gs), n0, n1, g.area, gs.area, sd


def extrude_2d(geom, z0, z1):
    parts = []
    for p in _polys(geom):
        p = shapely.geometry.polygon.orient(p, 1.0)
        rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(i.coords)[:-1] for i in p.interiors]
        parts.append(m3d.Manifold.extrude(m3d.CrossSection(rings, m3d.FillRule.EvenOdd), z1 - z0).translate((0.0, 0.0, z0)))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add) if parts else m3d.Manifold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{P.OUT_DIR}/model_noinfill_clampfillrev_lv020_oct")
    ap.add_argument("--out", default=f"{P.OUT_DIR}/model_noinfill_clampfillrev_lv020_oct_spline")
    ap.add_argument("--rms", type=float, default=0.16, help="样条平滑 RMS 容差 mm（默认 0.16）")
    ap.add_argument("--corner-deg", type=float, default=45.0, help="打断尖角的最小转角阈值度（默认 45）")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    ver = json.load(open(f"{a.src}/verify.json"))
    ha = ver["args"]
    M = N3.Free3D(
        ha["nz"], kz=ha["kz"], skin=ha["skin"], misuse=True, lateral=True, warp=True,
        asm_z=ha.get("asm_z", True), hxy=ha.get("hxy"), dz=ha.get("dz"), mid=ha.get("mid"),
        gap=ha.get("gap", 0.0), joint=ha.get("joint", "block"), beam_rev=ha.get("beam_rev", False),
        geo_only=True,
    )
    dom_outer, _ = P.geometry(False)  # 外包络（不含方槽孔，因 clampfill 允许槽中段有优化材料）
    dom_slots, _ = P.geometry(True)
    fixed = shapely.unary_union([g.intersection(dom_slots) for g in M.fixed_regions.values()])
    keepout = M.keepout_geom

    src_stl = f"{a.src}/support_noinfill_left.stl"
    m_src = trimesh.load(src_stl)
    man_src = m3d.Manifold(
        m3d.Mesh(vert_properties=np.asarray(m_src.vertices, np.float32), tri_verts=np.asarray(m_src.faces, np.uint32))
    )

    lh = P.LAYER_H
    nl = int(round(P.T / lh))
    zs = [(k + 0.5) * lh for k in range(nl)]
    orig_layers = []
    for z in zs:
        cs = man_src.slice(float(z))
        rings = [Polygon(r).buffer(0) for r in cs.to_polygons() if len(r) >= 3]
        g = Polygon()
        for r in rings:
            g = g.symmetric_difference(r)
        orig_layers.append(g)
    print(f"Loaded & sliced {nl} layers from {src_stl} in {time.time() - t0:.2f}s", flush=True)

    wkb_dom = shapely.to_wkb(dom_outer)
    wkb_fix = shapely.to_wkb(fixed)
    wkb_ko = shapely.to_wkb(keepout)
    tasks = [
        (k, shapely.to_wkb(orig_layers[k]), wkb_dom, wkb_fix, wkb_ko, a.rms, a.corner_deg)
        for k in range(nl)
    ]

    ts = time.time()
    gs = [None] * nl
    tot_v0 = tot_v1 = 0
    tot_a0 = tot_a1 = tot_sd = 0.0
    if a.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, wkb_gs, n0, n1, a0, a1, sd in ex.map(_worker_smooth, tasks):
                gs[k] = shapely.from_wkb(wkb_gs)
                tot_v0 += n0
                tot_v1 += n1
                tot_a0 += a0
                tot_a1 += a1
                tot_sd += sd
    else:
        for tsk in tasks:
            k, wkb_gs, n0, n1, a0, a1, sd = _worker_smooth(tsk)
            gs[k] = shapely.from_wkb(wkb_gs)
            tot_v0 += n0
            tot_v1 += n1
            tot_a0 += a0
            tot_a1 += a1
            tot_sd += sd
    print(
        f"Spline-fitted {nl} horizontal layers in {time.time() - ts:.2f}s: "
        f"verts {tot_v0} -> {tot_v1} ({tot_v0 / max(tot_v1, 1):.1f}x fewer), "
        f"area {tot_a0:.1f} -> {tot_a1:.1f} ({100 * (tot_a1 - tot_a0) / tot_a0:+.2f}%), "
        f"mean symdiff {100 * tot_sd / tot_a0:.2f}%",
        flush=True,
    )

    # 针对样条轻微过冲引起的宽悬空，按层对越界块定向裁剪（不改无悬空区域）
    off = lh * np.tan(np.deg2rad(45.0)) + 0.04
    n_clamped = 0
    trimmed_area = 0.0
    for k in range(1, nl):
        un = [p for p in _polys(gs[k].difference(gs[k - 1].buffer(off, join_style=1, quad_segs=6))) if p.area > 0.05]
        wide = [p for p in un if not p.buffer(-1.0).is_empty]
        if wide:
            n_clamped += 1
            for wp in wide:
                trimmed_area += wp.area
            gs[k] = gs[k].difference(shapely.unary_union(wide)).union(fixed).difference(keepout).buffer(0)
    print(f"Overhang clamp: adjusted {n_clamped} layers, trimmed wide area {trimmed_area:.2f} mm²", flush=True)

    # 逐层 0.2 mm 薄片叠加合成水密流形
    tb = time.time()
    slabs = [
        extrude_2d(gs[k], k * lh, (k + 1) * lh + (0.002 if k < nl - 1 else 0.0))
        for k in range(nl)
    ]
    man = m3d.Manifold.batch_boolean(slabs, m3d.OpType.Add)
    parts = man.decompose()
    pos_parts = [p for p in parts if p.volume() > 0]
    if len(pos_parts) > 1:
        biggest = max(pos_parts, key=lambda p: p.volume())
        dust = [p for p in pos_parts if p is not biggest]
        if dust:
            man = man - m3d.Manifold.batch_boolean(dust, m3d.OpType.Add)
            parts = man.decompose()
    cav = sorted([-p.volume() for p in parts if p.volume() < 0])
    left = E3.to_trimesh(man)
    print(
        f"Manifold assembled in {time.time() - tb:.2f}s: faces={len(left.faces)}, watertight={left.is_watertight}, "
        f"vol={left.volume / 1000:.2f} cm³, bodies={sum(p.volume() > 0 for p in parts)}, cavities={len(cav)}",
        flush=True,
    )

    left_stl = f"{a.out}/support_noinfill_left.stl"
    right_stl = f"{a.out}/support_noinfill_right.stl"
    left.export(left_stl)
    right = left.copy()
    right.apply_transform(np.diag([-1, 1, 1, 1]))
    right.apply_translation((float(left.bounds[1][0]), 0.0, 0.0))
    if right.volume < 0:
        right.invert()
    right.export(right_stl)

    oh = E3.overhang_check(man, tol=0.05)
    print("悬空检查：", oh, flush=True)

    jv = E3.extrude(M.joint_geom, 0, P.T).volume()
    jm = float(M.m_fix[M.cols["jnt"]].sum()) * M.vox
    mass_mesh = (left.volume - jv + jm) / 1000.0
    ver["mesh"] = dict(
        faces=len(left.faces),
        watertight=bool(left.is_watertight),
        volume_cm3=left.volume / 1000.0,
        printed_cm3=mass_mesh,
        petg_g=mass_mesh * 1.27,
        bodies=int(sum(p.volume() > 0 for p in parts)),
        cavities_mm3=cav,
    )
    ver["overhang"] = oh
    ver["iso"] = "spline_xy"
    ver["smooth"] = dict(
        method="horizontal_bspline",
        layer_mm=lh,
        rms_tol=a.rms,
        corner_deg=a.corner_deg,
        clamped_layers=n_clamped,
    )
    json.dump(ver, open(f"{a.out}/verify.json", "w"), ensure_ascii=False, indent=1)
    if os.path.exists(f"{a.src}/binary.npz"):
        shutil.copy2(f"{a.src}/binary.npz", f"{a.out}/binary.npz")
    print(f"Done in {time.time() - t0:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
