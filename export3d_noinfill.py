"""
无填充三维优化结果 → 可打印模型 + 复核
1. 取二值（ξ > 0.5），从床面往上逐层检查悬空（体素要落在下一层十字邻域上），支撑不住的去掉；去掉不连通的碎块。
2. 复核（同一个三维有限元模型，含 150 N 误用工况）：
   - 二维方案：profile.json 拉伸 T，按 3 圈墙 + 上下 4 层表皮 + 15% 填充打印；
   - 二维方案 + 尖拱空腔：同一轮廓不用填充，杆件内部挖成 45° 尖顶空腔（屋脊沿空腔中轴线，顶上留 4 层），
     墙厚调到与二维方案同样用料 —— 手工"尖拱封顶"的对照；
   - 无填充三维优化。
   三者的横梁接口区都按墙 + 表皮 + 15% 填充。
3. 网格：二值体素先平滑（--smooth σ 个体素），再插值到打印层高（0.2 mm）的立方网格，逐层做 45° 悬空处理
   （默认先在伸出的地方下面补斜坡、再削掉补不到的细边，见 smooth_fine），用料调回与体素模型相同；
   取等值面（z 两面镜像延拓后在 z=0、z=T 处切平），并上精确的固定区几何、挖掉精确的方槽 → STL。
   直接用 0.625 mm 体素做 marching cubes 的话，缓坡上是 0.625 mm 高的台阶（--smooth 0 仍可这样导出）。
   复核、薄壁统计用平滑后的形状（细网格按体素取多数）。
   悬空检查：按 0.2 mm 层高切片，每层超出下一层外扩 0.2 mm（45°）+ 0.05 mm 容差范围的面积。
用法: STAND_VARIANT=desk40 python export3d_noinfill.py [--dir out/desk40/noinfill]
"""
import argparse, json, os, time
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage import label, distance_transform_edt, binary_opening, binary_dilation, gaussian_filter, spline_filter, map_coordinates, grey_dilation
from skimage import measure, morphology
import manifold3d as m3d
import trimesh
import cupy as cp
import params as P
import topopt3d_noinfill as N


# ---------------- 二值化与清理 ----------------
def binarize(M, xi, no_am=False, am_angle=45.0, ground=False, ground_r=0.0, island_area=0.0, island_connect=False):
    d3 = M.design3
    B = np.where(d3, xi > 0.5, M.E_fix > 0)                # 固定区（含接口区）算有材料
    B00 = B; n_small = 0
    if island_area > 0:                                     # 底面孤岛：贴床那层面积小于 island_area 的连通块（8 连通）去掉，上面靠它托着的由 AM 修整去掉
        lab0, n0 = label(B[:, :, 0], structure=np.ones((3, 3), bool))
        small = np.bincount(lab0.ravel(), minlength=n0 + 1) * M.h ** 2 < island_area; small[0] = False
        B = B.copy(); B[:, :, 0] &= ~(small[lab0] & d3[:, :, 0]); n_small = int(small.sum())
    if island_connect:                                      # 连到主框：贴床那层（4 连通）不含固定实心格的块去掉
        lab0, n0 = label(B[:, :, 0])
        fix0 = (M.E_fix[:, :, 0] > 0) & ~d3[:, :, 0]
        keep = np.zeros(n0 + 1, bool); keep[np.unique(lab0[fix0 & B[:, :, 0]])] = True; keep[0] = True
        B = B.copy(); B[:, :, 0] &= ~(~keep[lab0] & d3[:, :, 0]); n_small += int((~keep).sum())
    B0 = B
    if ground or ground_r > 0:                              # 接地：床面那层（膨胀 ground_r 后）没料的柱子整柱去掉
        g0 = B[:, :, 0]
        if ground_r > 0:
            Rg = int(np.floor(ground_r / M.h + 1e-9)); og = np.arange(-Rg, Rg + 1)
            gi, gj = np.meshgrid(og, og, indexing="ij")
            g0 = binary_dilation(g0, structure=(gi * M.h) ** 2 + (gj * M.h) ** 2 <= ground_r ** 2 + 1e-9)
        B = np.where(d3, B & g0[:, :, None], B)
    B1 = B if no_am else N.am_binary(B, *N.am_stencil(M.h, M.dz, am_angle))
    lab, n = label(B1)                                      # 6 连通
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    B2 = lab == int(sizes.argmax())
    info = dict(底面小孤岛=n_small, 孤岛去掉床面体素=int((B00 & ~B0).sum()), 接地去掉体素=int((B0 & ~B).sum()), 悬空去掉体素=int((B & ~B1).sum()), 碎块数=int(n - 1), 碎块体素=int((B1 & ~B2).sum()))
    return B2 & d3, info                                    # 只返回设计区部分


def thin_stats(M, B, tmin):
    """B：设计区二值。用直径 tmin 的球做形态学开运算（网格外按空），被去掉的设计区体素就是比 tmin 薄的板 / 杆"""
    S = np.where(M.design3, B, M.E_fix > 0)
    r = tmin / 2 / M.h; R = int(np.ceil(r)); o = np.arange(-R, R + 1)
    rz = tmin / 2 / M.dz; Rz = int(np.ceil(rz)); oz = np.arange(-Rz, Rz + 1)
    di, dj, dk = np.meshgrid(o, o, oz, indexing="ij")
    ball = (di / r) ** 2 + (dj / r) ** 2 + (dk / rz) ** 2 <= 1 + 1e-9         # 扁体素时是椭球（物理上仍是直径 tmin 的球）
    thin = S & ~binary_opening(S, structure=ball) & M.design3
    sol = S & M.design3
    t = sol.sum(2)
    return dict(tmin_mm=tmin, 薄于tmin体积比例=round(float(thin.sum() / max(1, sol.sum())), 4),
                含薄处柱子比例=round(float(thin.any(2)[t > 0].mean()), 4), 单层柱子比例=round(float((t[t > 0] <= 1).mean()), 4))


# ---------------- 尖拱空腔对照 ----------------
def gable_fields(M, wall, bot=0.8, top=0.8, ss=4):
    """二维轮廓（接口区外）不用填充：墙厚 wall，底面 bot 实心，内部空腔顶为 45° 尖顶，屋脊沿空腔中轴线、离顶面 top。
    返回体素实心比例（接口区、固定条按 Free3D 的固定区）"""
    NX, NY, NZ, h = M.NX, M.NY, M.NZ, M.h
    pg = N.profile2d().difference(M.joint_geom)
    hs = h / ss
    X, Y = np.meshgrid((np.arange(NX * ss) + 0.5) * hs, (np.arange(NY * ss) + 0.5) * hs, indexing="ij")
    ins = shapely.contains_xy(pg, X, Y)
    d = distance_transform_edt(np.pad(ins, 1))[1:-1, 1:-1] * hs - 0.5 * hs
    I = ins & (d > wall)
    sk = morphology.skeletonize(I)
    ds = distance_transform_edt(~sk) * hs
    zc = (np.arange(NZ) + 0.5) * h
    frac = np.zeros((NX, NY, NZ))
    for k in range(NZ):
        cav = I & (zc[k] >= bot) & (zc[k] <= P.T - top - ds)
        frac[:, :, k] = (ins & ~cav).reshape(NX, ss, NY, ss).mean((1, 3))
    d3 = M.design3
    return np.where(d3, frac, M.E_fix), np.where(d3, frac, M.m_fix), np.where(d3, frac > 0.5, M.E_fix > 0.5)


def gable_matched(M, target):
    lo, hi = 0.42, 6.0
    for _ in range(30):
        w = 0.5 * (lo + hi)
        E, m, s = gable_fields(M, w)
        if float(m.sum()) * M.h ** 3 > target: hi = w
        else: lo = w
        if hi - lo < 0.01: break
    return w, gable_fields(M, w)


# ---------------- 网格 ----------------
def polys(g):
    if g.is_empty: return []
    return list(g.geoms) if isinstance(g, MultiPolygon) else [g] if isinstance(g, Polygon) else [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def extrude(geom, z0, z1):
    parts = []
    for p in polys(geom):
        p = shapely.geometry.polygon.orient(p, 1.0)
        rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(i.coords)[:-1] for i in p.interiors]
        parts.append(m3d.Manifold.extrude(m3d.CrossSection(rings, m3d.FillRule.EvenOdd), z1 - z0).translate((0, 0, z0)))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


# ---------------- 按打印层高平滑重建 ----------------
def _sdf_layer(f, c, lh, band=3):
    """一层的平面有向距离（mm，里面为正）：f 在 c 处的等值线用 marching squares 取（与最终网格在这一层的切片一致），
    离等值线 band 格以内的点算到等值线段的精确距离，更远的用像素距离变换"""
    b = f > c
    phi = np.where(b, distance_transform_edt(b) - 0.5, 0.5 - distance_transform_edt(~b))
    segs = [np.stack([cc[:-1], cc[1:]], 1) for cc in measure.find_contours(f, c) if len(cc) >= 2]
    if segs:
        segs = np.concatenate(segs)
        tree = shapely.STRtree(shapely.linestrings(segs))
        near = np.nonzero(np.abs(phi) <= band)
        _, d = tree.query_nearest(shapely.points(near[0], near[1]), return_distance=True, all_matches=False)
        phi[near] = np.where(b[near], d, -d)
    return (phi * lh).astype(np.float32)


def smooth_fine(M, F, sigma, lh=P.LAYER_H, pad=4, am="fill+trim", angle=45.0, match_volume=True):
    """体素场 F（0..1，固定区为 1）→ 打印层高 lh 的立方细网格场：
    1. 高斯平滑（σ 个体素；网格外按空，z 两面镜像），三次样条插值到细网格（采样点在每层正中，就是切片平面）。
       σ 可以是一个数，或 (σ_xy, σ_z) 平面内和厚度方向分开取；
    2. 阈值取成与体素模型同体积；
    3. 逐层悬空限制，在每层的平面距离场上做（_sdf_layer，轮廓附近是到等值线的精确距离；每层改完重新算距离再传下去，
       误差不会一层层累积）。δ = lh·tan(angle)，各方向一样（圆形外扩）：
       am="fill+trim"（默认）：先从顶层往下补——每层要包住上一层往里缩 δ 的部分，伸出超过 angle 的地方在下面补斜坡，
         只加料不删料；再从床面往上削——每层只能落在下一层外扩 δ 的范围内，保证每层都有支撑。补过之后削掉的只剩
         比 2δ 还窄、补不到的细边，不会连锁削断杆件；
       am="levelset"：只削；平滑后贴着 45° 的斜面局部稍陡就会被削，往上连锁，细杆可能断；
       am="cross"：只削，而且每层只能沿 x 或 y 外扩一格（斜着走的杆实际只放行到约 35°），连锁削断最严重，只留作对照；
    4. 只留最大的连通块。
    match_volume：补的料会让体积变大；先算一次补了多少，再把阈值调到少这么多重算一次，最后体积与体素模型相同。
    返回 dict(G, t, lh, info)，G 的 t 等值面就是零件表面"""
    NX, NY, NZ = F.shape; h = M.h
    sxy, sz = (sigma, sigma) if np.isscalar(sigma) else sigma
    G = np.pad(np.asarray(F, np.float32), ((pad, pad), (pad, pad), (0, 0)))
    G = np.concatenate([G[:, :, pad - 1::-1], G, G[:, :, :-pad - 1:-1]], 2)
    if sxy > 0 or sz > 0:
        G = gaussian_filter(G, (sxy, sxy, sz), mode="nearest")
    C = spline_filter(G, order=3, mode="nearest")
    nxf, nyf, nzf = (int(round(n * h / lh)) for n in (NX, NY, NZ))
    X, Y = np.meshgrid(((np.arange(nxf) + 0.5) * lh) / h - 0.5 + pad, ((np.arange(nyf) + 0.5) * lh) / h - 0.5 + pad, indexing="ij")
    G0 = np.empty((nxf, nyf, nzf), np.float32)
    for k in range(nzf):
        G0[:, :, k] = map_coordinates(C, [X, Y, np.full_like(X, ((k + 0.5) * lh) / h - 0.5 + pad)], order=3, prefilter=False, mode="nearest")
    del C, G, X, Y
    vc = float(np.sum(F)) * h ** 3
    dl = lh * np.tan(np.deg2rad(angle))

    def threshold(target):
        lo, hi = 0.2, 0.8
        for _ in range(30):
            t = 0.5 * (lo + hi)
            if (G0 > t).sum() * lh ** 3 > target: lo = t
            else: hi = t
        return t

    def run(target):
        t = threshold(target)
        Gf = G0.copy()
        n0 = int((Gf > t).sum())
        if am in ("fill", "levelset", "fill+trim"):
            if am != "levelset":   # 先去掉平滑后分离出来的小碎块（不然下面会给它们补出一根柱子）
                lab, n = label(Gf > t); cnt = np.bincount(lab.ravel()); cnt[0] = 0
                drop = (lab > 0) & (lab != int(cnt.argmax())); del lab
                Gf[drop] = np.minimum(Gf[drop], t - 0.05)
            lev = t
            for ps in {"fill": ["fill"], "levelset": ["trim"], "fill+trim": ["fill", "trim"]}[am]:
                prev = None
                for k in (range(nzf - 1, -1, -1) if ps == "fill" else range(nzf)):
                    phi = _sdf_layer(Gf[:, :, k], lev, lh)
                    if prev is not None:
                        phi = np.maximum(phi, prev - dl) if ps == "fill" else np.minimum(phi, prev + dl)
                        phi = _sdf_layer(phi, 0.0, lh)          # 改过的这一层重新算精确距离，再传给下一层
                    Gf[:, :, k] = phi; prev = phi
                lev = 0.0
            t1 = 0.0
        elif am == "cross":
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
            for k in range(1, nzf):
                Gf[:, :, k] = np.minimum(Gf[:, :, k], grey_dilation(Gf[:, :, k - 1], footprint=cross))
            t1 = t
        else:
            t1 = t
        L = Gf > t1
        n1 = int(L.sum())
        lab, n = label(L); cnt = np.bincount(lab.ravel()); cnt[0] = 0
        drop = L & (lab != int(cnt.argmax())); del lab
        Gf[drop] = np.minimum(Gf[drop], t1 - 0.05)
        n2 = n1 - int(drop.sum())
        return Gf, t1, dict(threshold=round(float(t), 4), 悬空处理增减_mm3=round((n1 - n0) * lh ** 3, 1),
                            碎块去掉_mm3=round((n1 - n2) * lh ** 3, 1), 碎块数=int(n - 1), 体积_cm3=round(n2 * lh ** 3 / 1000, 3))

    Gf, t1, r1 = run(vc)
    info = dict(sigma_vox=[sxy, sz], am=am, angle=angle, layer_mm=lh, 体素模型体积_cm3=round(vc / 1000, 3), 第一次=r1)
    if match_volume and am in ("fill", "fill+trim") and abs(r1["体积_cm3"] * 1000 - vc) > 1.0:
        del Gf
        Gf, t1, r2 = run(vc - (r1["体积_cm3"] * 1000 - vc))
        info["第二次（同体积）"] = r2
    info["平滑后体积_cm3"] = list(info.values())[-1]["体积_cm3"]
    return dict(G=Gf, t=float(t1), lh=lh, info=info)


def coarse_from_fine(M, fine):
    """细网格实体 → 体素二值（体素里的细网格采样点过半算有材料），给复核和薄壁统计用"""
    L = fine["G"] > fine["t"]
    r = M.h / fine["lh"]
    idx = [((np.arange(n) + 0.5) / r).astype(int) for n in L.shape]
    st = [np.r_[0, np.nonzero(np.diff(c))[0] + 1] for c in idx]
    assert tuple(len(s) for s in st) == (M.NX, M.NY, M.NZ)
    S = L
    for ax in range(3):
        S = np.add.reduceat(S, st[ax], axis=ax, dtype=np.uint16)
    cnt = [np.bincount(c) for c in idx]
    return S > 0.5 * (cnt[0][:, None, None] * cnt[1][None, :, None] * cnt[2][None, None, :])


def build_mesh(M, F, step=1, fine=None):
    """F：体素场（0..1，固定区为 1）→ manifold；给了 fine（smooth_fine 的结果）就用细网格场的等值面。
    z 两面镜像延拓，切平到 [0, T]；并上精确固定区，挖掉精确方槽"""
    h, F, lev = (M.h, F, 0.5) if fine is None else (fine["lh"], fine["G"], fine["t"])
    hz = M.dz if fine is None else h                        # 扁体素：层向尺寸不同
    pz = 3
    G = np.concatenate([F[:, :, pz - 1::-1], F, F[:, :, :-pz - 1:-1]], 2)
    G = np.pad(G, 2)
    v, f, _, _ = measure.marching_cubes(G.astype(np.float32, copy=False), lev, spacing=(h, h, hz), step_size=step, allow_degenerate=False)
    del G
    v = v + np.array([-1.5 * h, -1.5 * h, -(pz + 1.5) * hz])
    mesh = m3d.Manifold(m3d.Mesh(vert_properties=v.astype(np.float32), tri_verts=f[:, ::-1].astype(np.uint32)))
    if mesh.volume() < 0:
        mesh = m3d.Manifold(m3d.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32)))
    box = m3d.Manifold.cube((M.NX * M.h + 20, M.NY * M.h + 20, P.T)).translate((-10, -10, 0))
    mesh = mesh ^ box
    dom, _ = P.geometry(True)
    fixed = shapely.unary_union([g.intersection(dom) for g in M.fixed_regions.values()])
    mesh = mesh + extrude(fixed, 0, P.T)
    mesh = mesh - extrude(P.joint_slot(), -1, P.T + 1)
    if fine is not None:
        mesh = mesh.simplify(0.02)          # 0.02 mm 以内合并近似共面的三角形，面数降到约 1/4
    elif M.dz != M.h:
        mesh = mesh.simplify(0.01)          # 扁体素：表面全是平台阶，合并共面三角形（不然 STL 有几百 MB）
    return mesh


def to_trimesh(man):
    mm = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def overhang_check(man, lh=P.LAYER_H, ang=45.0, bridge=2.0, tol=0.05):
    """按层高切片：每层超出下一层外扩 lh·tan(ang) + tol 的部分算悬空（tol 是几何误差的容差，远小于 0.42 mm 线宽）。
    窄于 bridge（mm）的条带两边都有支撑，是空腔收顶时的短桥，单独统计；宽的才算真正的悬空。返回面积、最大单块面积、所在高度"""
    off = lh * np.tan(np.deg2rad(ang)) + tol
    zs = np.arange(lh / 2, P.T, lh)
    prev = None; tot = 0.0; tot_b = 0.0; worst = (0.0, 0.0); per = []
    for z in zs:
        cs = man.slice(float(z))
        # to_polygons 给出所有环（外环、孔），用奇偶规则重建
        rings = [Polygon(r).buffer(0) for r in cs.to_polygons() if len(r) >= 3]
        g = Polygon()
        for r in rings:
            g = g.symmetric_difference(r)
        if prev is not None:
            un = [p for p in polys(g.difference(prev.buffer(off, join_style="round", quad_segs=4))) if p.area > 0.05]
            wide = [p for p in un if not p.buffer(-bridge / 2).is_empty]
            a = [p.area for p in wide]
            tot += sum(a); per.append(sum(a)); tot_b += sum(p.area for p in un) - sum(a)
            if a and max(a) > worst[0]:
                worst = (max(a), float(z))
        prev = g
    return dict(悬空总面积_mm2=round(tot, 2), 最大单块_mm2=round(worst[0], 3), 最大单块高度_mm=round(worst[1], 2),
                有悬空的层数=int(sum(1 for a in per if a > 0)), 短桥面积_mm2=round(tot_b, 2), 层数=len(zs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=N.OUT)
    ap.add_argument("--no-gable", action="store_true")
    ap.add_argument("--no-buckling", action="store_true")
    ap.add_argument("--ref", default=None, help="另一次三维结果（同 nz 的 result.npz 所在目录）一起复核，例如无约束的那次")
    ap.add_argument("--tmin", type=float, default=None, help="薄壁统计用的厚度 mm（默认取优化时的 --tmin，没有则 1.9）")
    ap.add_argument("--smooth", default="1.0", help="STL 按打印层高平滑重建的高斯半径（体素）；可写 σ_xy,σ_z 分开取；0 = 直接用体素做 marching cubes")
    ap.add_argument("--am", default="fill+trim", choices=["fill+trim", "fill", "levelset", "cross"], help="平滑后的逐层悬空处理（见 smooth_fine）")
    ap.add_argument("--mid", default=None, help="改用另一套顶面载荷复核：斜面中段 s0,s1 / joints 不加载（默认取优化时的 --mid）")
    ap.add_argument("--legacy_top", action="store_true", help="按旧的顶面加载节点复核（复现 2026-11-22 以前的数）")
    ap.add_argument("--md", default=None, help="输出目录（默认由 --dir 推出；换载荷复核旧设计时另给一个，免得覆盖）")
    a = ap.parse_args()
    sm = [float(v) for v in str(a.smooth).split(",")]
    a.smooth = sm[0] if len(sm) == 1 else tuple(sm)
    smooth_on = max(sm) > 0
    t0 = time.time()
    r = np.load(f"{a.dir}/result.npz")
    hist = json.load(open(f"{a.dir}/history.json"))
    ha = hist["args"]
    md = a.md or (a.dir.replace("noinfill", "model_noinfill") if "noinfill" in a.dir else a.dir + "/model")
    os.makedirs(md, exist_ok=True)
    M = N.Free3D(ha["nz"], kz=ha["kz"], skin=ha["skin"], misuse=True, lateral=True, warp=True, asm_z=ha.get("asm_z", True),
                 hxy=ha.get("hxy"), dz=ha.get("dz"), mid=a.mid or ha.get("mid"), gap=ha.get("gap", 0.0),
                 legacy_top=a.legacy_top or ha.get("legacy_top", False))    # 复核总带 z 向体力工况
    if M.mid or M.legacy_top:
        print(f"顶面载荷：" + (f"中段 s = {M.mid[0]:.1f}–{M.mid[1]:.1f} 不加载（中段空隙 {M.gap:g} mm）" if M.mid else "") + ("旧加载节点" if M.legacy_top else ""), flush=True)
    if M.dz != M.h and smooth_on:
        print("扁体素：层向已经是打印层高，不做平滑（--smooth 只用于立方体素）", flush=True)
        smooth_on = False
    xi = r["xi"].astype(float)
    target = float(r["target"])
    B, info = binarize(M, xi, no_am=bool(ha.get("no_am")), am_angle=ha.get("am_angle", 45.0), ground=bool(ha.get("ground")), ground_r=float(ha.get("ground_r") or 0.0), island_area=float(ha.get("island_area") or 0.0), island_connect=bool(ha.get("island_connect")))
    _S = np.where(M.design3, B, M.E_fix > 0).astype(np.float64)
    sup_est = N.SupportEst(M.h, M.dz, M.NT, angle=ha.get("sup_angle", 30.0), c=ha.get("sup_c", 2.0),
                           e=ha.get("sup_e", 0.47), r_mm=ha.get("sup_r", 0.0))(cp.asarray(_S))[0]
    print(f"估算树状支撑（阈值 {ha.get('sup_angle', 30.0):g}°）：{sup_est/1000:.2f} cm³", flush=True)
    print("二值化：", info, flush=True)
    fine = None
    if smooth_on:
        tf = time.time()
        fine = smooth_fine(M, np.where(M.design3, B, M.E_fix > 0), a.smooth, am=a.am)
        Bs = coarse_from_fine(M, fine) & M.design3
        fine["info"]["体素变化"] = dict(增加=int((Bs & ~B).sum()), 去掉=int((B & ~Bs).sum()), 原有=int(B.sum()))
        print(f"按 {P.LAYER_H} mm 层平滑重建（复核也用平滑后的形状）：", fine["info"], f"{time.time()-tf:.0f}s", flush=True)
        B = Bs

    designs = {}; keys = {}
    E2, m2, s2 = M.baseline_fields()
    nm = "二维方案（墙+表皮+15%填充）"; designs[nm] = (E2, m2, s2); keys[nm] = "flat"
    gable_wall = None
    if not a.no_gable:
        gable_wall, (Eg, mg, sg) = gable_matched(M, float(m2.sum()) * M.vox)
        print(f"尖拱空腔对照：墙厚 {gable_wall:.2f} mm 时用料相同", flush=True)
        nm = f"二维方案 + 尖拱空腔（无填充，墙 {gable_wall:.2f} mm）"; designs[nm] = (Eg, mg, sg); keys[nm] = "gable"
    tmin = a.tmin or ha.get("tmin") or 1.9
    thin = {}
    if a.ref:
        rr = np.load(f"{a.ref}/result.npz")
        if rr["xi"].shape == xi.shape:
            Bo, info_o = binarize(M, rr["xi"].astype(float))
            if smooth_on:               # 对照也按同样方法平滑，网格另存一份给预览
                fo = smooth_fine(M, np.where(M.design3, Bo, M.E_fix > 0), a.smooth, am=a.am)
                Bo = coarse_from_fine(M, fo) & M.design3
                to_trimesh(build_mesh(M, None, fine=fo)).export(f"{md}/support_ref_left.stl")
                del fo
            nm = "无填充三维（无约束）"; designs[nm] = M.design_fields(Bo.astype(float)); keys[nm] = "free_old"
            thin[nm] = thin_stats(M, Bo, tmin)
        else:
            print(f"--ref 网格 {rr['xi'].shape} 与本次 {xi.shape} 不同，跳过", flush=True)
    cons = [f"屈曲 ≥ {ha['buck']:g}" if ha.get("buck") else "", f"最薄 {ha['tmin']:g} mm" if ha.get("tmin") else ""]
    nm = "无填充三维优化" + (f"（{'，'.join(c for c in cons if c)}）" if any(cons) else "")
    designs[nm] = M.design_fields(B.astype(float)); keys[nm] = "free"
    thin[nm] = thin_stats(M, B, tmin)
    print("薄壁统计：", thin, flush=True)
    res = {}
    buck_cases = [k for k, nm in enumerate(M.names) if nm in ("前部误用150N", "后部误用150N")]
    for nm, (E, m, s) in designs.items():
        M.U = [None] * len(M.F)
        M.set_mass(m)                      # 第一次调用（二维方案）定下 z 向体力的单位体积力
        comps, its = M.solve(E)
        st = M.stresses(s)
        mass = float(m.sum()) * M.vox
        res[nm] = dict(key=keys[nm], comp=comps.tolist(), stress=st, mass_cm3=mass / 1000, petg_g=mass / 1000 * 1.27, cg=its, thin=thin.get(nm),
                       warp_um={k: [round(v[0] * 1000, 2), round(v[1] * 1000, 2)] for k, v in M.warp_measures().items()})   # 顶面平均 z 位移：全长、后半段
        print(f"{nm}: 用料 {mass/1000:.2f} cm³（PETG {mass/1000*1.27:.0f} g），CG {sum(its)}", flush=True)
        if not a.no_buckling:
            tb = time.time(); lam = {}
            for k in buck_cases:
                lam[M.names[k]], _, nit = M.buckling(E, k, nvec=4, iters=20)
            res[nm]["buckling"] = lam
            print(f"  线性屈曲载荷系数 {dict((k, round(v, 2)) for k, v in lam.items())}（{time.time()-tb:.0f}s）", flush=True)
    names = M.names
    k0 = list(res)[0]
    opt = [k for k in range(len(names)) if M.imp[k] > 0]
    orig = [k for k in opt if names[k] != "z向体力"]
    base = np.array(res[k0]["comp"])
    table = []
    for nm in res:
        rel = np.array(res[nm]["comp"]) / base
        res[nm]["rel"] = rel.tolist()
        res[nm]["weighted"] = float((M.imp * rel)[opt].sum() / M.imp[opt].sum())
        res[nm]["weighted_orig"] = float((M.imp * rel)[orig].sum() / M.imp[orig].sum())
    for k, nm in enumerate(names):
        row = dict(工况=nm)
        for d in res:
            st = res[d]["stress"][k]
            row[d] = dict(柔度比=round(res[d]["rel"][k], 3), VM_99_9=round(st["VM_99_9pct_MPa"], 2), VM_max=round(st["VM_max_MPa"], 2),
                          面内位移_um=round(st["面内位移_mm"] * 1000, 1), z向位移_um=round(st["z向位移_mm"] * 1000, 1))
        table.append(row)
    for nm in res:
        print(f"  {nm}: 加权柔度 {res[nm]['weighted']:.3f}（不计 z 向体力 {res[nm]['weighted_orig']:.3f}），用料 {res[nm]['petg_g']:.1f} g")
    for row in table:
        print("  " + row["工况"] + "：" + "；".join(f"{d[:6]} {v['柔度比']:.3f}/{v['VM_99_9']:.1f}MPa/z{v['z向位移_um']:.0f}µm" for d, v in row.items() if d != "工况"))

    # 网格
    tm = time.time()
    F = np.where(M.design3, B.astype(float), (M.E_fix > 0).astype(float))   # 用检查过悬空的二值体素（立方体素 + 十字支撑 → 斜面不超过 45°）
    man = build_mesh(M, F, fine=fine)
    if fine is not None:
        del fine["G"]
    left = to_trimesh(man)
    parts = man.decompose()
    cav = sorted([-p.volume() for p in parts if p.volume() < 0])
    print(f"网格 {len(left.faces)} 面，水密 {left.is_watertight}，体积 {left.volume/1000:.2f} cm³，实体 {sum(p.volume() > 0 for p in parts)} 块，"
          f"封闭空腔 {len(cav)} 个（{', '.join(f'{v:.0f}' for v in cav[::-1][:6])} mm³），{time.time()-tm:.0f}s", flush=True)
    left.export(f"{md}/support_noinfill_left.stl")
    right = left.copy(); right.apply_transform(np.diag([-1, 1, 1, 1])); right.apply_translation((float(left.bounds[1][0]), 0, 0))
    right.invert() if right.volume < 0 else None
    right.export(f"{md}/support_noinfill_right.stl")
    # 扁体素：45° 规则是"每 m 层外扩一格"，单层最多伸出一格（hxy），检查时容差相应放宽到一格
    oh = overhang_check(man, tol=0.05 + max(0.0, M.h - M.dz * np.tan(np.deg2rad(45.0))))
    print("悬空检查：", oh, flush=True)
    if gable_wall is not None:      # 尖拱对照也导出网格（预览用）
        Fg = np.where(M.design3, designs[[k for k in keys if keys[k] == "gable"][0]][2], M.E_fix > 0).astype(float)
        mg_ = build_mesh(M, Fg)
        to_trimesh(mg_).export(f"{md}/support_gable_left.stl")
        oh_g = overhang_check(mg_)
        print("尖拱对照悬空检查：", oh_g, flush=True)
    # 实际打印用料：网格体积里接口区按墙+表皮+填充折算
    jv = extrude(M.joint_geom, 0, P.T).volume()
    jm = float(M.m_fix[M.cols["jnt"]].sum()) * M.vox
    mass_mesh = (left.volume - jv + jm) / 1000
    summary = dict(names=names, binarize=info, mesh=dict(faces=len(left.faces), watertight=bool(left.is_watertight), volume_cm3=left.volume / 1000,
                   printed_cm3=mass_mesh, petg_g=mass_mesh * 1.27, bodies=int(sum(p.volume() > 0 for p in parts)), cavities_mm3=cav), overhang=oh,
                   target_cm3=target / 1000, results=res, table=table, args=ha, gable_wall_mm=gable_wall,
                   overhang_gable=oh_g if gable_wall is not None else None, smooth=fine["info"] if fine is not None else None,
                   ref_model_dir=(a.ref.replace("noinfill", "model_noinfill") if a.ref and "noinfill" in a.ref else None))
    json.dump(summary, open(f"{md}/verify.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(f"{md}/binary.npz", B=B, h=M.h, dz=M.dz, NT=M.NT)
    N.fig_sections(M, np.where(M.design3, B, M.E_fix > 0).astype(float), f"{md}/sections_binary.png", "二值化后的剖面（x = 常数，纵轴为打印方向）")
    print(f"导出用料（按网格）{mass_mesh:.2f} cm³ ≈ {mass_mesh*1.27:.0f} g PETG；用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
