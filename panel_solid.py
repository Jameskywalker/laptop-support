"""
面板（实心杆件版）：杆件按实心打印（墙圈数加够或 100% 填充），四个卡扣角仍是墙 + 表皮 + 15% 填充。
现在的面板杆件是"薄壳盒子"（两侧墙 + 上下表皮 + 稀疏填充），每平方毫米很轻、刚度又主要靠墙和表皮；
同样的打印用料做成 16 mm 全高的实心杆，投影面积（挡在笔记本下面的面积）小得多，散热好。

做法：总打印用料与现在的面板相同（四个角按打印结构算），杆件按实心做二维拓扑优化（工况同 panel.py）；
两种面板都按打印结构复核（墙、表皮实心，填充刚度按 0.15²），报投影面积和三个工况的刚度。
结果放 out/<方案>/panel_solid/，不覆盖现有面板。
用法: STAND_VARIANT=desk40 python panel_solid.py [--gpu]
"""
import argparse, json, os, time, copy
import numpy as np
import shapely
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2
import panel as PN

OUT = f"{P.OUT_DIR}/panel_solid"
WALL = P.NWALL * P.LINE_W; SKIN = 2 * P.NSKIN * P.LAYER_H; T = P.PANEL_T


def thickness(shape, m, corners_print=True, members_solid=False, sub=4):
    """面板坐标的轮廓 → 每个单元的 (刚度等效厚度, 用料等效厚度)。墙（离轮廓 WALL 以内）和表皮实心，填充刚度按 0.15²"""
    h = m.h
    o = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(o * h, o * h)
    xs = (m.xc[:, None] + ox.ravel()[None, :]).ravel(); ys = (m.yc[:, None] + oy.ravel()[None, :]).ravel()
    ins = shapely.contains_xy(shape, xs, ys)
    d = np.full(xs.size, 99.0); d[ins] = shapely.distance(shape.boundary, shapely.points(xs[ins], ys[ins]))
    tE = np.where(d < WALL, T, SKIN + (T - SKIN) * P.INFILL ** 2)
    tm = np.where(d < WALL, T, SKIN + (T - SKIN) * P.INFILL)
    if members_solid:
        corner = shapely.contains_xy(shapely.unary_union(list(PN.geometry()[1].values())), xs, ys)
        tE = np.where(corner, tE, T); tm = np.where(corner, tm, T)
    return (tE * ins).reshape(m.nel, -1).mean(1), (tm * ins).reshape(m.nel, -1).mean(1)


def fe_comp(m, t, gpu=False):
    F, names, imp, groups = PN.cases(m)
    inside = t > 1e-6
    used = np.zeros(m.nn, bool); used[(m.edof[:, 0::2] // 2)[inside].ravel()] = True; m.node_in = used
    F, names, imp, groups = PN.cases(m)
    fe = T2.MultiFE(m, T2.element_stiffness(P.NU), groups, gpu)
    Ek = P.E_MAT * np.maximum(t, T * 1e-9)
    U = fe.solve(Ek if not gpu else __import__("cupy").asarray(Ek)); U = np.asarray(U.get() if hasattr(U, "get") else U); fe.close()
    return np.array([float(F[:, k] @ U[:, k]) for k in range(F.shape[1])]), names


def main():
    ap = argparse.ArgumentParser()
    T2.add_args(ap, 0.1)
    ap.set_defaults(robust=True, rmin=3.0, maxit=450, h=0.5)
    ap.add_argument("--hv", type=float, default=0.5)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    # 现在的面板（按打印结构）
    r = np.load(f"{PN.OUT}/topo/result.npz")
    shape0 = PN.extract(r["xphys"], T2.Mesh(float(r["h"]), True, geom=PN.geometry()))
    mv = T2.Mesh(a.hv, True, geom=PN.geometry())
    tE0, tm0 = thickness(shape0, mv)
    c0, names = fe_comp(mv, tE0, a.gpu)
    vol0 = float(tm0.sum()) * a.hv ** 2; area0 = float((tE0 > 0).sum()) * a.hv ** 2
    # 四个角按打印结构的用料（两种面板相同）→ 杆件可用的实心体积 → 体积分数
    corners = shapely.unary_union(list(PN.geometry()[1].values()))
    tEc, tmc = thickness(corners, mv)
    vol_c = float(tmc.sum()) * a.hv ** 2
    mo = T2.Mesh(a.h, True, geom=PN.geometry())
    vf = (vol0 - vol_c) / T / (float(mo.active.sum()) * a.h ** 2)
    print(f"现在的面板：投影面积 {area0:.0f} mm²，打印用料 {vol0/1000:.1f} cm³（四角 {vol_c/1000:.1f}）；"
          f"柔度（按打印结构）{dict(zip(names, np.round(c0, 5).tolist()))}；杆件实心可用 {(vol0-vol_c)/1000:.1f} cm³ → 体积分数 {vf:.4f}", flush=True)
    b = copy.copy(a); b.vf = vf; b.out = f"{OUT}/topo"
    F, nm, imp, groups = PN.cases(mo)
    free0 = groups[0][1]; fixed0 = np.setdiff1d(np.arange(mo.ndof), free0)
    T2.optimize(mo, F, nm, imp, free0, fixed0, b, thick=P.PANEL_T, vf_mode="active", fe_groups=groups, t0=t0)
    r1 = np.load(f"{b.out}/result.npz")
    shape1 = PN.extract(r1["xphys"], mo)
    tE1, tm1 = thickness(shape1, mv, members_solid=True)
    c1, _ = fe_comp(mv, tE1, a.gpu)
    vol1 = float(tm1.sum()) * a.hv ** 2; area1 = float((tE1 > 0).sum()) * a.hv ** 2
    outline = PN.with_snaps(shape1)
    json.dump(dict(exterior=list(outline.exterior.coords), holes=[list(h.coords) for h in outline.interiors],
                   half=PN.HALF, yc=[vc - PN.V0 for vc in PN.VC], thick=P.PANEL_T, joints=[j["name"] for j in PN.PJ], members_solid=True),
              open(f"{OUT}/panel_outline.json", "w"))
    res = dict(names=names, now=dict(area_mm2=round(area0), vol_cm3=round(vol0 / 1000, 2), mass_g=round(vol0 / 1000 * 1.24, 1), comp=c0.tolist()),
               solid=dict(area_mm2=round(area1), vol_cm3=round(vol1 / 1000, 2), mass_g=round(vol1 / 1000 * 1.24, 1), comp=c1.tolist(), vf=vf),
               ratio_comp=(c1 / c0).tolist(), ratio_area=area1 / area0)
    json.dump(res, open(f"{OUT}/verify.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (shp, ttl) in zip(axs, ((PN.with_snaps(shape0), f"现在：投影 {area0:.0f} mm²，{vol0/1000*1.24:.0f} g（杆件是墙 + 表皮 + 15% 填充的盒子）"),
                                    (outline, f"实心杆件：投影 {area1:.0f} mm²（{100*area1/area0:.0f}%），{vol1/1000*1.24:.0f} g；柔度比 {np.round(c1/c0, 2).tolist()}"))):
        ex = np.array(shp.exterior.coords); ax.fill(ex[:, 0], ex[:, 1], color="#e0913a", lw=0)
        for it in shp.interiors:
            c = np.array(it.coords); ax.fill(c[:, 0], c[:, 1], color="white", lw=0)
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=10)
    fig.savefig(f"{OUT}/panel_compare.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
