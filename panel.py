"""
面板：把前、顶两根横梁合成一块平板（与笔记本底面平行，离底面 10 mm，厚 PANEL_T），对它做二维拓扑优化。
两片支撑和面板组成一个带斜撑的框，左右晃动时上部相对下部的错动由面板的面内刚度承担。

面板坐标（优化用）：u 沿横向，0 = 左支撑内侧面，L = 右支撑内侧面；v 沿斜面向上，0 = 面板前缘。
打印坐标（建模用）：X = u − L/2，Y = v − v_前卡扣中心，Z 沿斜面法向 0..PANEL_T（平放打印，无悬空）。
四个角各有一个与横梁端头相同的卡扣，插进支撑上原来的方槽。

工况（四个卡扣在支撑内侧面处加载/约束）：
  侧向错动：前端两个卡扣固定；顶端两个卡扣只约束沿斜面方向，各受横向力 F_RACK/2   → 斜撑
  撑开：    左边两个卡扣固定；右边两个卡扣只约束沿斜面方向，各受向外 F_RACK/2       → 横向拉杆
  前后错动：前端两个卡扣固定；顶端两个卡扣沿斜面一前一后各 F_RACK/2              → 抗扭
用法: STAND_VARIANT=desk40 python panel.py [--gpu]
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
from shapely import affinity
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2

PJ = P.panel_joints()
if len(PJ) != 2:
    print("这个方案没有面板，跳过"); sys.exit(0)
L = P.SPACING - P.T                       # 两支撑内侧面间距
HALF = L / 2
S = [float(P.xy2sn(j["x"], j["y"])[0]) for j in PJ]
NS = [float(P.xy2sn(j["x"], j["y"])[1]) for j in PJ]
assert all(abs(j["phi"] - P.THETA_DEG) < 1e-9 for j in PJ) and abs(NS[0] - NS[1]) < 0.15, \
    "面板的两个接口必须与斜面平行且在同一平面（坐标取整到 0.1 mm，容差 0.15 mm，远小于方槽单边间隙）"
BH2 = P.BEAM_H / 2
V0 = BH2                                  # 前卡扣中心的 v 坐标
VC = [V0, V0 + S[1] - S[0]]               # 两个卡扣中心
VLEN = VC[1] + BH2
OUT = f"{P.OUT_DIR}/panel"


def geometry():
    domain = box(0, 0, L, VLEN)
    solids = {}
    for k, vc in enumerate(VC):
        solids[f"corner{k}L"] = box(0, vc - BH2, P.PANEL_BODY, vc + BH2)
        solids[f"corner{k}R"] = box(L - P.PANEL_BODY, vc - BH2, L, vc + BH2)
    return domain, solids


def cases(m):
    """返回 (F, names, imp, groups)。groups 用于 MultiFE：[(F 子矩阵, free, 工况下标)]"""
    tol = 1e-9

    def tenon(side, k):
        u0 = 0.0 if side == "L" else L
        return np.nonzero(m.node_in & (np.abs(m.xn - u0) < m.h * 0.5 + tol) & (np.abs(m.yn - VC[k]) <= P.TENON_H / 2 + tol))[0]
    T_ = {(s, k): tenon(s, k) for s in "LR" for k in (0, 1)}
    f2 = P.F_RACK / 2

    def load(pairs):
        f = np.zeros(m.ndof)
        for (s, k), (fu, fv) in pairs:
            idx = T_[(s, k)]
            f[2 * idx] += fu / idx.size; f[2 * idx + 1] += fv / idx.size
        return f

    def free_of(fix):                      # fix: {(side,k): "uv" 或 "v"}
        fixed = []
        for key, how in fix.items():
            idx = T_[key]
            if "u" in how: fixed.append(2 * idx)
            if "v" in how: fixed.append(2 * idx + 1)
        return np.setdiff1d(np.arange(m.ndof), np.concatenate(fixed))
    defs = [
        ("侧向错动", 1.0, {("L", 0): "uv", ("R", 0): "uv", ("L", 1): "v", ("R", 1): "v"}, [(("L", 1), (f2, 0)), (("R", 1), (f2, 0))]),
        ("撑开", 0.5, {("L", 0): "uv", ("L", 1): "uv", ("R", 0): "v", ("R", 1): "v"}, [(("R", 0), (f2, 0)), (("R", 1), (f2, 0))]),
        ("前后错动", 0.5, {("L", 0): "uv", ("R", 0): "uv"}, [(("L", 1), (0, f2)), (("R", 1), (0, -f2))]),
    ]
    F = np.stack([load(d[3]) for d in defs], 1)
    groups = [(F[:, [i]], free_of(d[2]), [i]) for i, d in enumerate(defs)]
    return F, [d[0] for d in defs], np.array([d[1] for d in defs]), groups


def extract(xphys, m):
    """密度场 → 面板轮廓（面板坐标，不含卡扣）"""
    G = xphys.reshape(m.nelx, m.nely).T
    d = 0.2
    xs = np.arange(-2, m.nelx * m.h + 2, d); ys = np.arange(-2, m.nely * m.h + 2, d)
    XX, YY = np.meshgrid(xs, ys)
    Fd = map_coordinates(gaussian_filter(G, 0.6), [YY.ravel() / m.h - 0.5, XX.ravel() / m.h - 0.5], order=1, mode="constant", cval=0).reshape(XX.shape)
    rings = [Polygon(np.c_[xs[0] + c[:, 1] * d, ys[0] + c[:, 0] * d]).buffer(0) for c in measure.find_contours(Fd, 0.5) if len(c) >= 8]
    region = rings[0]
    for pg in rings[1:]:
        region = region.symmetric_difference(pg)
    domain, solids = geometry()
    shape = shapely.unary_union([region] + list(solids.values())).intersection(domain)
    shape = shape.buffer(1.5).buffer(-1.5).buffer(-1.0).buffer(1.0).intersection(domain)
    if isinstance(shape, MultiPolygon):
        parts = sorted(shape.geoms, key=lambda g: -g.area)
        print("警告：面板有分离块，面积", [round(g.area, 1) for g in parts])
        shape = parts[0]
    holes = [h for h in shape.interiors if Polygon(h).area >= 20.0]
    return Polygon(shape.exterior, holes).simplify(0.02)


def with_snaps(shape):
    """面板坐标 → 打印坐标，加上四个卡扣"""
    g = affinity.translate(shape, -HALF, -V0)
    parts, cuts = [g], []
    for vc in VC:
        for sx in (1, -1):
            p_, c_ = P.snap_end(sx, HALF, vc - V0)
            parts += p_; cuts += c_
    # 两侧边条比支撑内侧面缩进 0.5 mm，只让四角台肩贴住支撑，打印误差不会顶住支撑导致卡不到位
    relief = shapely.unary_union([box(-HALF - 30, -500, -HALF + 0.5, 500), box(HALF - 0.5, -500, HALF + 30, 500)])
    keep = shapely.unary_union([box(-HALF - 30, vc - V0 - BH2, HALF + 30, vc - V0 + BH2) for vc in VC])
    cuts.append(relief.difference(keep))
    out = shapely.unary_union(parts).difference(shapely.unary_union(cuts))
    assert out.geom_type == "Polygon" and out.is_valid, out.geom_type
    return shapely.geometry.polygon.orient(out, 1.0)


def verify(shape, gpu=False):
    """最终轮廓（0.5 mm 网格）复核：三个工况的位移和应力"""
    hv = 0.5
    m = T2.Mesh(hv, True, geom=geometry())
    inside = shapely.contains_xy(shape, m.xc, m.yc)
    m.in_dom = inside
    used = np.zeros(m.nn, bool); used[(m.edof[:, 0::2] // 2)[inside].ravel()] = True; m.node_in = used
    F, names, imp, groups = cases(m)
    KE = T2.element_stiffness(P.NU)
    Ee = np.where(inside, P.E_MAT * P.PANEL_T, P.E_MAT * P.PANEL_T * 1e-9)
    fe = T2.MultiFE(m, KE * 1.0, groups, False)
    U = fe.solve(Ee)
    nu = P.NU; D = P.E_MAT / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    xin = np.array([-1, 1, 1, -1]); etn = np.array([-1, -1, 1, 1])
    B = np.zeros((3, 8)); B[0, 0::2] = 0.25 * xin * 2 / hv; B[1, 1::2] = 0.25 * etn * 2 / hv; B[2, 0::2] = 0.25 * etn * 2 / hv; B[2, 1::2] = 0.25 * xin * 2 / hv
    res = []; vms = []
    for k, nm in enumerate(names):
        sig = U[m.edof, k] @ (D @ B).T
        vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2 + 3 * sig[:, 2] ** 2); vm[~inside] = 0
        vms.append(vm)
        f = F[:, k]; loaded = np.nonzero(np.abs(f) > 0)[0]
        dl = float(np.abs(U[loaded, k]).max())
        res.append(dict(工况=nm, 力_N=P.F_RACK, 加载点最大位移_mm=dl, 刚度_N_per_mm=P.F_RACK / 2 / dl if dl > 0 else None,
                        最大VonMises_MPa=float(vm.max()), VM_99_9pct_MPa=float(np.percentile(vm[inside], 99.9))))
    return m, inside, res, vms


def main():
    ap = argparse.ArgumentParser()
    T2.add_args(ap, P.PANEL_VF)
    ap.set_defaults(robust=True, rmin=4.0, maxit=450)
    ap.add_argument("--out", default=f"{OUT}/topo")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print(f"面板 {L:.1f} × {VLEN:.1f} mm（横向 × 沿斜面），厚 {P.PANEL_T:g} mm；卡扣中心沿斜面相距 {VC[1]-VC[0]:.1f} mm")
    m = T2.Mesh(a.h, True, geom=geometry())
    F, names, imp, groups = cases(m)
    free0 = groups[0][1]; fixed0 = np.setdiff1d(np.arange(m.ndof), free0)
    T2.optimize(m, F, names, imp, free0, fixed0, a, thick=P.PANEL_T, vf_mode="active", fe_groups=groups, t0=t0)

    r = np.load(f"{a.out}/result.npz")
    shape = extract(r["xphys"], m)
    outline = with_snaps(shape)
    json.dump(dict(exterior=list(outline.exterior.coords), holes=[list(h.coords) for h in outline.interiors],
                   half=HALF, yc=[vc - V0 for vc in VC], thick=P.PANEL_T, joints=[j["name"] for j in PJ]),
              open(f"{OUT}/panel_outline.json", "w"))
    mv, inside, res, vms = verify(shape)
    info = dict(size_mm=[round(outline.bounds[2] - outline.bounds[0], 1), round(outline.bounds[3] - outline.bounds[1], 1), P.PANEL_T],
                area_mm2=round(outline.area, 1), holes=len(outline.interiors), volume_cm3=round(outline.area * P.PANEL_T / 1000, 1))
    json.dump(dict(info=info, fe=res), open(f"{OUT}/verify.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))
    for rr in res:
        print(rr)
    # 图：轮廓 + 侧向错动工况应力
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax = axs[0]
    ex = np.array(outline.exterior.coords); ax.fill(ex[:, 0], ex[:, 1], color="#e0913a", lw=0)
    for it in outline.interiors:
        c = np.array(it.coords); ax.fill(c[:, 0], c[:, 1], color="white", lw=0)
    ax.plot(ex[:, 0], ex[:, 1], "k-", lw=0.6)
    for sx in (-1, 1):
        ax.axvline(sx * HALF, color="#4a78a8", ls="--", lw=1)
    ax.set_aspect("equal"); ax.set_title(f"面板轮廓（打印坐标）：{info['size_mm'][0]} × {info['size_mm'][1]} × {P.PANEL_T:g} mm；蓝虚线 = 支撑内侧面")
    ax = axs[1]
    vg = mv.grid(np.where(inside, vms[0], np.nan))
    im = ax.imshow(vg, origin="lower", extent=[0, mv.nelx * mv.h, 0, mv.nely * mv.h], cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Von Mises (MPa)")
    ax.set_aspect("equal"); ax.set_title(f"侧向错动 {P.F_RACK:g} N：最大 {res[0]['最大VonMises_MPa']:.2f} MPa，顶端卡扣位移 {res[0]['加载点最大位移_mm']*1000:.1f} µm")
    fig.savefig(f"{OUT}/panel_and_stress.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"面板完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
