"""
面板（实心杆件版）：杆件按实心打印（100% 填充，或墙圈数加够），四个卡扣角仍是墙 + 表皮 + 15% 填充。
现在的面板杆件是"薄壳盒子"（两侧墙 + 上下表皮 + 稀疏填充），每平方毫米很轻、刚度又主要靠墙和表皮；
同样的打印用料做成 16 mm 全高的实心杆，投影面积（挡在笔记本下面的面积）小得多，散热好。

做法：总打印用料与现在的面板相同，杆件按实心做二维拓扑优化。
  工况：panel.py 的三个（侧向错动 / 撑开 / 前后错动）+ 每个角单独受力的 8 个（另外三个角固定）：
        单角插拔 30 N（横向）、单角剪切 10 N（沿斜面），让每个卡扣角和杆件的连接都单独扛得住。
  四个卡扣角在优化里也按打印结构折算刚度（墙实心、里面 15% 填充），不当成刚体：
        否则优化以为角块是刚的，杆件只在角块一个顶点处接一下就够了，实际打印出来连接处就是软的。
  过滤时网格外按空算（否则贴边的细拉条在鲁棒公式下太便宜，优化沿前缘 / 顶缘长出 0.5 mm 的拉条，导出时被开运算去掉）。
  目标：各工况"相对现在面板的柔度比"加权和（主工况 1 / 0.5 / 0.5，单角工况各 w_corner）。按第一轮灰度设计归一化时，
        侧向错动的分量太小，优化会丢掉 X 撑改成口字框（单角工况好了，侧向错动软 15 倍）。
  导出轮廓时调等值线阈值，让轮廓的用料与优化结果相同（panel.extract 的开运算会把细杆两边各削掉零点几毫米）。
两种面板都按打印结构复核（墙、表皮实心，填充刚度按 0.15²），报投影面积、各工况柔度和受力点平均位移。
结果放 out/<方案>/panel_solid[_tag]/，不覆盖现有面板。
用法: STAND_VARIANT=desk40 python panel_solid.py [--gpu] [--w_corner 0.125] [--tag x]
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
CORNERS = [("L", 0, "左前"), ("R", 0, "右前"), ("L", 1, "左上"), ("R", 1, "右上")]
F_SNAP, F_SHEAR = 30.0, 10.0


def corner_union():
    return shapely.unary_union(list(PN.geometry()[1].values()))


def thickness(shape, m, members_solid=False, sub=4):
    """面板坐标的轮廓 → 每个单元的 (刚度等效厚度, 用料等效厚度)。墙（离轮廓 WALL 以内）和表皮实心，填充刚度按 0.15²。
    members_solid：四个角以外按实心"""
    h = m.h
    o = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(o * h, o * h)
    xs = (m.xc[:, None] + ox.ravel()[None, :]).ravel(); ys = (m.yc[:, None] + oy.ravel()[None, :]).ravel()
    ins = shapely.contains_xy(shape, xs, ys)
    d = np.full(xs.size, 99.0); d[ins] = shapely.distance(shape.boundary, shapely.points(xs[ins], ys[ins]))
    tE = np.where(d < WALL, T, SKIN + (T - SKIN) * P.INFILL ** 2)
    tm = np.where(d < WALL, T, SKIN + (T - SKIN) * P.INFILL)
    if members_solid:
        corner = shapely.contains_xy(corner_union(), xs, ys)
        tE = np.where(corner, tE, T); tm = np.where(corner, tm, T)
    return (tE * ins).reshape(m.nel, -1).mean(1), (tm * ins).reshape(m.nel, -1).mean(1)


def geometry():
    """panel.py 的设计域，两侧（支撑内侧面处）四角以外退让 0.75 mm 的带子去掉：成品面板两侧边条比支撑内侧面缩进 0.5 mm
    （with_snaps 里切掉），优化时不让材料贴着这条边长，否则导出时被切断"""
    domain, solids = PN.geometry()
    relief = shapely.unary_union([box(-1, -1, 0.75, PN.VLEN + 1), box(PN.L - 0.75, -1, PN.L + 1, PN.VLEN + 1)])
    return domain.difference(relief.difference(corner_union())), solids


def cases_ext(m, w_corner):
    """panel.py 的三个工况 + 每个角单独受力的 8 个工况（另外三个角的卡扣固定）。
    返回 F, names, imp, groups, ftot（每个工况的总外力，用来把柔度换成受力点平均位移）"""
    F0, names, imp, groups = PN.cases(m)
    tol = 1e-9

    def tenon(side, k):
        u0 = 0.0 if side == "L" else PN.L
        return np.nonzero(m.node_in & (np.abs(m.xn - u0) < m.h * 0.5 + tol) & (np.abs(m.yn - PN.VC[k]) <= P.TENON_H / 2 + tol))[0]
    T_ = {(s, k): tenon(s, k) for s, k, _ in CORNERS}
    cols, names, imp, gs, ftot = [F0[:, i] for i in range(F0.shape[1])], list(names), list(imp), [], [P.F_RACK] * F0.shape[1]
    for s, k, cn in CORNERS:
        others = [c for c in T_ if c != (s, k)]
        fixed = np.concatenate([np.concatenate([2 * T_[c], 2 * T_[c] + 1]) for c in others])
        free = np.setdiff1d(np.arange(m.ndof), fixed)
        ids = []
        for lab, d, Fm in (("单角插拔", 0, F_SNAP), ("单角剪切", 1, F_SHEAR)):
            f = np.zeros(m.ndof); idx = T_[(s, k)]; f[2 * idx + d] += Fm / idx.size
            ids.append(len(cols)); cols.append(f); names.append(f"{lab}·{cn}"); imp.append(w_corner); ftot.append(Fm)
        gs.append((free, ids))                                   # 同一个角的两个工况约束相同，共用一次分解
    F = np.stack(cols, 1)
    groups = [(F[:, g[2]], g[1], g[2]) for g in groups] + [(F[:, ids], fr, ids) for fr, ids in gs]
    return F, names, np.array(imp), groups, np.array(ftot)


def fe_comp(m, t):
    """按每个单元的等效厚度算各工况柔度（CPU：细网格、多组约束，显卡显存不够）"""
    inside = t > 1e-6
    used = np.zeros(m.nn, bool); used[(m.edof[:, 0::2] // 2)[inside].ravel()] = True; m.node_in = used
    F, names, imp, groups, ftot = cases_ext(m, 1.0)
    fe = T2.MultiFE(m, T2.element_stiffness(P.NU), groups, False)
    U = fe.solve(P.E_MAT * np.maximum(t, T * 1e-9)); fe.close()
    c = np.array([float(F[:, k] @ U[:, k]) for k in range(F.shape[1])])
    return c, names, c / ftot * 1000                               # 柔度 N·mm，受力点平均位移 µm


def extract_matched(xp, m):
    """导出轮廓，调等值线阈值让轮廓在设计区的面积等于优化结果的实心面积"""
    design = m.active
    target = float(xp[design].sum()) * m.h ** 2
    dpoly = geometry()[0].difference(corner_union())
    area = lambda s: s.intersection(dpoly).area
    s5 = PN.extract(xp, m); a5 = area(s5)
    if a5 >= 0.99 * target:
        return s5, 0.5, a5 / target
    lo, hi, best = 0.15, 0.5, (s5, 0.5, a5)
    for _ in range(10):
        lv = 0.5 * (lo + hi); s = PN.extract(xp, m, level=lv); a_ = area(s)
        if a_ < target: hi = lv
        else: lo = lv; best = (s, lv, a_)
    return best[0], best[1], best[2] / target


def main():
    ap = argparse.ArgumentParser()
    T2.add_args(ap, 0.1)
    ap.set_defaults(robust=True, rmin=3.0, maxit=450, h=0.5)
    ap.add_argument("--hv", type=float, default=0.5)
    ap.add_argument("--w_corner", type=float, default=0.125, help="每个单角工况的权重（8 个；原三个工况 1 / 0.5 / 0.5）。0.25 时优化跳到口字框，侧向错动软 15 倍")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = OUT + (f"_{a.tag}" if a.tag else "")
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    # 现在的面板（按打印结构）
    r = np.load(f"{PN.OUT}/topo/result.npz")
    shape0 = PN.extract(r["xphys"], T2.Mesh(float(r["h"]), True, geom=PN.geometry()))
    mv = T2.Mesh(a.hv, True, geom=geometry())
    tE0, tm0 = thickness(shape0, mv)
    c0, names, d0 = fe_comp(mv, tE0)
    vol0 = float(tm0.sum()) * a.hv ** 2; area0 = float((tE0 > 0).sum()) * a.hv ** 2
    # 四个角按打印结构的用料（两种面板相同）→ 杆件可用的实心体积 → 体积分数
    tEc, tmc = thickness(corner_union(), mv)
    vol_c = float(tmc.sum()) * a.hv ** 2
    mo = T2.Mesh(a.h, True, geom=geometry())
    vf = (vol0 - vol_c) / T / (float(mo.active.sum()) * a.h ** 2)
    print(f"现在的面板：投影面积 {area0:.0f} mm²，打印用料 {vol0/1000:.1f} cm³（四角 {vol_c/1000:.1f}）；"
          f"杆件实心可用 {(vol0-vol_c)/1000:.1f} cm³ → 体积分数 {vf:.4f}；单角工况权重 {a.w_corner}", flush=True)
    # 优化：四个角按打印结构折算刚度
    tEco, _ = thickness(corner_union(), mo)
    escale = np.where(mo.solid, np.maximum(tEco / T, 0.05), 1.0)
    b = copy.copy(a); b.vf = vf; b.out = f"{out}/topo"; b.pad_void = True   # 面板外是空气：过滤时网格外按空算
    F, nm, imp, groups, _ = cases_ext(mo, a.w_corner)
    free0 = groups[0][1]; fixed0 = np.setdiff1d(np.arange(mo.ndof), free0)
    T2.optimize(mo, F, nm, imp, free0, fixed0, b, thick=P.PANEL_T, vf_mode="active", fe_groups=groups, t0=t0, escale=escale, cnorm=c0)
    r1 = np.load(f"{b.out}/result.npz")
    shape1, level, afrac = extract_matched(r1["xphys"], mo)
    print(f"导出等值线阈值 {level:.3f}，轮廓面积 / 优化实心面积 = {afrac:.3f}", flush=True)
    tE1, tm1 = thickness(shape1, mv, members_solid=True)
    c1, _, d1 = fe_comp(mv, tE1)
    vol1 = float(tm1.sum()) * a.hv ** 2; area1 = float((tE1 > 0).sum()) * a.hv ** 2
    outline = PN.with_snaps(shape1)
    json.dump(dict(exterior=list(outline.exterior.coords), holes=[list(h.coords) for h in outline.interiors],
                   half=PN.HALF, yc=[vc - PN.V0 for vc in PN.VC], thick=P.PANEL_T, joints=[j["name"] for j in PN.PJ], members_solid=True),
              open(f"{out}/panel_outline.json", "w"))
    res = dict(names=names, w_corner=a.w_corner, level=level,
               now=dict(area_mm2=round(area0), vol_cm3=round(vol0 / 1000, 2), mass_g=round(vol0 / 1000 * 1.24, 1), comp=c0.tolist(), disp_um=d0.tolist()),
               solid=dict(area_mm2=round(area1), vol_cm3=round(vol1 / 1000, 2), mass_g=round(vol1 / 1000 * 1.24, 1), comp=c1.tolist(), disp_um=d1.tolist(), vf=vf),
               ratio_comp=(c1 / c0).tolist(), ratio_area=area1 / area0)
    json.dump(res, open(f"{out}/verify.json", "w"), ensure_ascii=False, indent=1)
    rc = c1 / c0
    print("各工况 柔度比（实心/现在）、平均位移 µm（现在 → 实心）：")
    for n_, r_, x0, x1 in zip(names, rc, d0, d1):
        print(f"  {n_:8s} {r_:5.2f}   {x0:7.1f} → {x1:7.1f}")
    print(f"投影面积 {area0:.0f} → {area1:.0f} mm²（{100*area1/area0:.0f}%），用料 {vol0/1000*1.24:.1f} → {vol1/1000*1.24:.1f} g", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (shp, ttl) in zip(axs, ((PN.with_snaps(shape0), f"现在：投影 {area0:.0f} mm²，{vol0/1000*1.24:.0f} g（杆件是墙 + 表皮 + 15% 填充的盒子）"),
                                    (outline, f"实心杆件：投影 {area1:.0f} mm²（{100*area1/area0:.0f}%），{vol1/1000*1.24:.0f} g；"
                                              f"柔度比 主工况 {rc[:3].min():.2f}–{rc[:3].max():.2f}，单角 {rc[3:].min():.2f}–{rc[3:].max():.2f}"))):
        ex = np.array(shp.exterior.coords); ax.fill(ex[:, 0], ex[:, 1], color="#e0913a", lw=0)
        for it in shp.interiors:
            c = np.array(it.coords); ax.fill(c[:, 0], c[:, 1], color="white", lw=0)
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=10)
    fig.savefig(f"{out}/panel_compare.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
