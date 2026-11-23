"""
刚性插指 + U 形卡子（params.clip_end / clip_outline）试验，后横梁和面板都换上，和现在的弹性卡扣臂对比：
  1. 接触复核：同 beam_contact.py / panel_contact.py 的非线性单向接触，网格、工况相同。
     现在：台肩 + 槽短边壁（卡扣臂外侧面）+ 卡钩；插指：台肩 + 槽短边壁（插指外侧面）+ 卡子腿顶住缺口外壁（卡子当刚体）。
     杆身 / 面板本体不变：后横梁仍是 40 × 16 实心杆；面板用 panel_contact 的当前面板，角上原来的让位槽、让位缝补实。
  2. 和现在首选三维支撑（noinfill_clampfillrev 体素）对干涉：插指占的地方有没有料，卡子腿压的外侧面有没有料。
  3. 导出横梁、面板、卡子的 STL / STEP。
结果放 out/<方案>/clip_trial/。
用法: STAND_VARIANT=desk40 python clip_trial.py [--h_beam 0.25] [--h_panel 0.5] [--skip_fe] [--skip_export]
注意：接触点在直角边上（卡钩、卡子腿），单点峰值随网格加密一直涨，比较时看本体最大应力和 99.9% 分位
"""
import argparse, json, os, time
import numpy as np
import shapely
from shapely.geometry import Polygon, box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2
import beam_contact as BC
import panel_contact as PC

OUT = f"{P.OUT_DIR}/clip_trial"
UE_TRIAL = 26.0                    # 优化 / 复核网格伸进支撑一侧的长度：盖住插指（24 mm）


def geometry_clip(M):
    """插指端头的设计域和固定实心（优化坐标），结构同 M.geometry()"""
    UE, L, VLEN, HALF, V0, BH2 = M.UE, M.L, M.VLEN, M.HALF, M.V0, M.BH2
    full = box(0, 0, L + 2 * UE, VLEN)
    parts, cuts = [], []
    for vc in M.VC:
        for sx in (1, -1):
            p_, c_ = P.clip_end(sx, HALF, vc - V0); parts += p_; cuts += c_
    parts = M.to_opt(shapely.unary_union(parts)); cuts = M.to_opt(shapely.unary_union(cuts))
    body = box(UE, 0, UE + L, VLEN)
    keep = shapely.unary_union([box(0, vc - BH2, L + 2 * UE, vc + BH2) for vc in M.VC])
    recess = shapely.unary_union([box(UE, 0, UE + 0.5, VLEN), box(UE + L - 0.5, 0, UE + L, VLEN)]).difference(keep)
    outside = full.difference(body).difference(parts)
    domain = full.difference(shapely.unary_union([outside, cuts, recess]))
    corner = shapely.unary_union([box(UE, vc - BH2, UE + 18, vc + BH2) for vc in M.VC] +
                                 [box(UE + L - 18, vc - BH2, UE + L, vc + BH2) for vc in M.VC])
    solid = shapely.unary_union([parts, corner]).intersection(domain)
    polys = [g for g in getattr(solid, "geoms", [solid]) if g.geom_type == "Polygon" and g.area > 1e-6]
    return domain, {f"snap{i}": g for i, g in enumerate(polys)}


def contact_sets_clip(M, m):
    """同 M.contact_sets，卡钩换成卡子腿：缺口外壁（离支撑内侧面 NOTCH_U1）只能往支撑方向顶，|dv| 在 NECK_Y + NOTCH_R..15"""
    h = m.h; tol = 1e-6; YA = P.TENON_H / 2
    FX = np.zeros((m.nelx + 2, m.nely + 2), bool); FX[1:-1, 1:-1] = m.solid.reshape(m.nelx, m.nely)
    ni, nj = np.round(m.xn / h).astype(int), np.round(m.yn / h).astype(int)
    ll, lr, ul, ur = FX[ni, nj], FX[ni + 1, nj], FX[ni, nj + 1], FX[ni + 1, nj + 1]
    out = []
    for side, k in M.JOINTS:
        sx = -1 if side == "L" else 1
        af = M.UE if side == "L" else M.UE + M.L
        dv = m.yn - M.VC[k]
        sh = np.isclose(m.xn, af) & (np.abs(dv) <= M.BH2 + tol) & (np.abs(dv) >= P.SLOT_A / 2 - tol)
        sh &= (lr | ur) if side == "L" else (ll | ul)
        for n_ in np.nonzero(sh)[0]: out.append((2 * n_, sx, (side, k), "台肩"))
        inx = (sx * (m.xn - af) >= -tol) & (sx * (m.xn - af) <= P.T + tol)
        for sy in (1, -1):
            face = inx & (np.abs(dv - sy * YA) < 0.75 * h)
            face &= ((ll | lr) & ~(ul | ur)) if sy > 0 else ((ul | ur) & ~(ll | lr))
            for n_ in np.nonzero(face)[0]: out.append((2 * n_ + 1, sy, (side, k), "槽壁"))
        hx = af + sx * P.NOTCH_U1
        hk = (np.abs(m.xn - hx) < 0.75 * h) & (np.abs(dv) >= P.NECK_Y + P.NOTCH_R - tol) & (np.abs(dv) <= YA + tol)   # 卡子腿倒角，避开缺口内角的倒圆
        hk &= ((ll | ul) & ~(lr | ur)) if side == "L" else ((lr | ur) & ~(ll | ul))
        for n_ in np.nonzero(hk)[0]: out.append((2 * n_, -sx, (side, k), "卡子"))
    return out


def clip_outline_of(M, arm_outline):
    """现在的轮廓（打印坐标）→ 本体不变、端头换成插指：本体取两支撑内侧面之间，角上 18 mm 补实（原来的让位槽、让位缝），再拼插指"""
    body = arm_outline.intersection(box(-M.HALF, -1000, M.HALF, 1000))
    fill = [box(sx * M.HALF - (18 if sx > 0 else 0), vc - M.V0 - M.BH2, sx * M.HALF + (18 if sx < 0 else 0), vc - M.V0 + M.BH2)
            for vc in M.VC for sx in (1, -1)]
    shape = shapely.unary_union([body] + fill)
    old = M.snap_v2
    M.snap_v2 = lambda sx, yc: P.clip_end(sx, M.HALF, yc)
    try:
        return M.with_snaps_v2(shape)
    finally:
        M.snap_v2 = old


def kinds_of(cons):
    k = {}
    for c in cons: k[f"{c[2][0]}{c[2][1]}{c[3]}"] = k.get(f"{c[2][0]}{c[2][1]}{c[3]}", 0) + 1
    return k


def run_fe(M, name, arm_outline, h):
    M.UE = UE_TRIAL
    scs = M.scenarios()
    out = {}
    m_a = T2.Mesh(h, True, geom=M.geometry())
    cons_a = M.contact_sets(m_a)
    x_a = M.raster(m_a, arm_outline)
    t = time.time(); res_a, f_a = M.evaluate(m_a, x_a, scs, cons_a, "卡扣臂")
    print(f"[{name}] 卡扣臂 接触点 {kinds_of(cons_a)}，{time.time() - t:.0f}s", flush=True)
    clip_poly = clip_outline_of(M, arm_outline)
    m_c = T2.Mesh(h, True, geom=geometry_clip(M))
    cons_c = contact_sets_clip(M, m_c)
    x_c = M.raster(m_c, clip_poly)
    t = time.time(); res_c, f_c = M.evaluate(m_c, x_c, scs, cons_c, "插指")
    print(f"[{name}] 插指 接触点 {kinds_of(cons_c)}，{time.time() - t:.0f}s", flush=True)
    for ra, rc in zip(res_a, res_c):
        print(f"[{name}] {ra['工况']}: 柔度 {ra['柔度']:.4g} → {rc['柔度']:.4g}（{rc['柔度'] / ra['柔度']:.2f}）；"
              f"本体最大 {ra['最大应力_MPa']:.2f} → {rc['最大应力_MPa']:.2f} MPa；伸进支撑那段最大 {ra['卡扣段最大应力_MPa']:.2f} → {rc['卡扣段最大应力_MPa']:.2f}；"
              f"99.9% {ra['VM_99_9pct_MPa']:.2f} → {rc['VM_99_9pct_MPa']:.2f}", flush=True)
    out = dict(arm=res_a, clip=res_c, contacts=dict(arm=kinds_of(cons_a), clip=kinds_of(cons_c)),
               area_mm2=dict(arm=round(arm_outline.area, 1), clip=round(clip_poly.area, 1)))
    fields = dict(m_a=m_a, m_c=m_c, x_a=x_a, x_c=x_c, f_a=f_a, f_c=f_c)
    return out, clip_poly, fields


def plot_fields(M, name, fields, res, path, cases):
    """左端（左支撑一侧）附近的 von Mises 场：上排卡扣臂，下排插指"""
    fig, axs = plt.subplots(2, len(cases), figsize=(4.2 * len(cases), 6.4), squeeze=False)
    vmax = max(max(r["VM_99_9pct_MPa"] for r in res["arm"] if r["工况"] in cases) * 1.3, 1e-6)   # 按 99.9% 分位定色标：接触点的单点峰值随网格变
    for col, cn in enumerate(cases):
        for row, (mk, fk, tag) in enumerate((("m_a", "f_a", "arm"), ("m_c", "f_c", "clip"))):
            m = fields[mk]; f = fields[fk][cn].reshape(m.nelx, m.nely)
            ax = axs[row, col]
            ax.imshow(f.T, origin="lower", extent=[0, m.nelx * m.h, 0, m.nely * m.h], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            vc = M.VC[0]
            ax.set_xlim(0, M.UE + 40); ax.set_ylim(vc - 28, vc + 28); ax.set_aspect("equal")
            ax.axvline(M.UE, color="#4f77a5", lw=1); ax.axvline(M.UE - P.T, color="#4f77a5", lw=1)
            r = [x for x in res[tag] if x["工况"] == cn][0]
            ax.set_title(f"{'arm' if tag == 'arm' else 'clip'}  max {r['最大应力_MPa']:.2f} / in-support {r['卡扣段最大应力_MPa']:.2f} MPa", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{name}: von Mises (MPa), 0..{vmax:.1f}; blue lines = support faces", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def support_check():
    """插指、卡子和现在首选三维支撑（体素）的关系，按每个接口的方槽坐标 (u 沿长边, v 沿短边) 统计"""
    fn = f"{P.OUT_DIR}/noinfill_clampfillrev/result.npz"
    if not os.path.exists(fn):
        return dict(note=f"没有 {fn}")
    d = np.load(fn, allow_pickle=True); xi = d["xi"] > 0.5; h = float(d["h"]); NX, NY, NZ = xi.shape
    I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij"); xc, yc = (I + 0.5) * h, (J + 0.5) * h
    out = {}
    for j in P.JOINTS:
        u, v = P.slot_uv(j, xc, yc)
        fing = (np.abs(u) >= P.FING_IN - 0.15) & (np.abs(u) <= P.TENON_H / 2 + 0.15) & (np.abs(v) <= P.BEAM_W / 2 + 0.15)
        leg = (np.abs(u) >= P.SLOT_A / 2) & (np.abs(u) <= P.LEG_Y[1]) & (np.abs(v) <= P.BEAM_W / 2 + 2.0)
        hook = (np.abs(u) >= P.SLOT_A / 2) & (np.abs(u) <= P.TENON_H / 2 + P.HOOK_H) & (np.abs(v) <= P.TENON_W / 2)
        mid = (np.abs(u) < P.FING_IN) & (np.abs(v) <= P.SLOT_B / 2)
        out[j["name"]] = dict(插指占的地方实心体素数=int(xi[fing].sum()), 插指占的格子数=int(fing.sum()) * NZ,
                              卡子腿压的外侧面实心比例=round(float(xi[:, :, 0][leg].mean()), 3),
                              卡钩现在压的外侧面实心比例=round(float(xi[:, :, 0][hook].mean()), 3),
                              槽中间实心比例=round(float(xi[mid].mean()), 3))
    return out


def export(beam_poly, panel_poly):
    from build123d import Wire, Face, Vector, extrude, export_step, export_stl
    from shapely.geometry.polygon import orient

    def solid(poly, hgt):
        poly = orient(poly, 1.0)
        wire = lambda cs: Wire.make_polygon([Vector(x, y, 0) for x, y in list(cs)[:-1]], close=True)
        return extrude(Face(wire(poly.exterior.coords), [wire(i.coords) for i in poly.interiors]), amount=hgt)
    rep = {}
    for nm, poly, hgt in (("beam_clip", beam_poly, P.BEAM_W), ("panel_clip", panel_poly, P.PANEL_T), ("clip", P.clip_outline(), P.CLIP_T)):
        s = solid(poly, hgt)
        export_step(s, f"{OUT}/{nm}.step"); export_stl(s, f"{OUT}/{nm}.stl", tolerance=0.01, angular_tolerance=0.1)
        bb = s.bounding_box()
        rep[nm] = dict(volume_cm3=round(s.volume / 1000, 2), bbox_mm=[round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)], valid=s.is_valid)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_beam", type=float, default=0.25)
    ap.add_argument("--h_panel", type=float, default=0.5)
    ap.add_argument("--skip_fe", action="store_true")
    ap.add_argument("--skip_export", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rep = dict(params=dict(FING_IN=P.FING_IN, FING_LEN=P.FING_LEN, NECK_Y=P.NECK_Y, NOTCH_U=[P.NOTCH_U0, P.NOTCH_U1], LEG_Y=list(P.LEG_Y),
                           CLIP_T=P.CLIP_T, CLIP_ARM_T=P.CLIP_ARM_T, CLIP_HOOK=P.CLIP_HOOK, clip_strain_pct=round(100 * P.clip_strain(), 2),
                           arm_strain_pct=round(100 * P.snap_strain(), 2), clip_area_mm2=round(P.clip_outline().area, 1)))
    rep["support_check"] = support_check()
    print(json.dumps(rep["support_check"], ensure_ascii=False), flush=True)
    beam_arm = Polygon(*(lambda d: (d["exterior"], d["holes"]))(json.load(open(f"{P.MODEL_DIR}/beam_outline.json"))))
    panel_arm = Polygon(*(lambda d: (d["exterior"], d["holes"]))(json.load(open(f"{P.OUT_DIR}/panel_contact/panel_outline.json"))))
    if not a.skip_fe:
        rb, beam_clip, fb = run_fe(BC, "后横梁", beam_arm, a.h_beam)
        rep["beam"] = rb
        plot_fields(BC, "rear beam", fb, rb, f"{OUT}/beam_vm.png", ["前后错动", "左支撑绕竖直轴正转", "撑开"])
        rp, panel_clip, fp_ = run_fe(PC, "面板", panel_arm, a.h_panel)
        rep["panel"] = rp
        plot_fields(PC, "panel", fp_, rp, f"{OUT}/panel_vm.png", ["前后错动", "左支撑绕法线正转", "侧向错动"])
    else:
        BC.UE = PC.UE = UE_TRIAL
        beam_clip, panel_clip = clip_outline_of(BC, beam_arm), clip_outline_of(PC, panel_arm)
    for nm, poly in (("beam", beam_clip), ("panel", panel_clip)):
        json.dump(dict(exterior=list(poly.exterior.coords), holes=[list(i.coords) for i in poly.interiors]), open(f"{OUT}/{nm}_clip_outline.json", "w"))
    if not a.skip_export:
        rep["export"] = export(beam_clip, panel_clip)
        print(json.dumps(rep["export"], ensure_ascii=False), flush=True)
    json.dump(rep, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
