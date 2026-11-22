"""
面板（卡扣按真实接触建模）：panel.py 把四个角当 25 × 40 mm 的整块、卡扣处整面刚性夹住，实物却是
  台肩条（3.5 mm 宽）顶住支撑内侧面、卡扣臂外侧面贴槽短边壁、卡钩钩住支撑外侧面——都是单向接触；两臂之间还挖了 25 × 15 mm 的缺口。
  这样单个角在面板平面内受扭（支撑绕面板法线相对面板转）时，力挤过台肩条和开孔的内凹角，那里应力集中。
做法：
  - 优化区域两侧各延伸 UE，把卡扣臂、卡钩伸进支撑的那段也放进网格（固定实心）；卡扣用 params.snap_end（每条臂内侧 SLIT_W 宽的让位缝，
    两臂之间离支撑内侧面 0.5 mm 以内留空），两臂之间交给优化；角上只固定卡扣臂、台肩条和臂根 3 mm 的一条，其余交给优化。
  - 强制左右、前后四分之一对称（每轮在 4 个对称像上平均灵敏度和设计变量；导出时取四分之一镜像）。
  - 每个接口当一块刚性支撑，和面板之间只有单向接触：台肩（横向）、卡钩（横向，反方向）、槽短边壁（沿斜面）。
    各工况先用现在的面板解一遍非线性接触（主动集迭代），定下哪些接触点在受力；优化时固定支撑的受力接触点当约束，
    受载支撑的接触力当节点力（线性化）。最后用非线性接触复核新旧面板。
  - 工况：panel.py 的三个（侧向错动 / 撑开 / 前后错动，各 F_RACK）+ 左、右支撑各绕面板法线正反转（±M_TWIST）。
  - 目标：各工况柔度相对现在面板的比值加权和；用料与现在面板相同。导出轮廓时内凹圆角半径 R_FILLET。
结果放 out/<方案>/panel_contact/，不覆盖现有面板。
用法: STAND_VARIANT=desk40 python panel_contact.py [--h 0.5] [--maxit 300] [--gpu] [--verify_only]
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely import affinity
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import measure
import pypardiso
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2
import panel as PN

OUT = f"{P.OUT_DIR}/panel_contact"
L, HALF, VC, V0, VLEN, BH2 = PN.L, PN.HALF, PN.VC, PN.V0, PN.VLEN, PN.BH2
YA = P.TENON_H / 2; YI = YA - P.ARM_T
SLIT = P.SLIT_W               # 卡扣臂内侧让位缝宽（见 params.snap_end）
ROOT_BAND = 3.0               # 臂根后面固定实心的一条
R_FILLET = 4.0                # 导出轮廓时内凹圆角半径
M_TWIST = 2000.0              # N·mm：支撑绕面板法线相对面板转的扭矩
UE = 20.0                     # 优化区域两侧延伸（盖住卡扣臂伸进支撑的一段和卡钩）
KP_REL = 50.0                 # 接触罚刚度（× E·t）


def snap_v2(sx, yc):
    """一个卡扣（打印坐标）：(要并上的, 要挖掉的)，即 P.snap_end（臂内侧让位缝，两臂之间实心）"""
    return P.snap_end(sx, HALF, yc)


def to_opt(g):
    """打印坐标 → 优化坐标（a = X + HALF + UE，v = Y + V0）"""
    return affinity.translate(g, HALF + UE, V0)


def geometry():
    """优化坐标的设计域和固定实心"""
    full = box(0, 0, L + 2 * UE, VLEN)
    parts, cuts = [], []
    for vc in VC:
        for sx in (1, -1):
            p_, c_ = snap_v2(sx, vc - V0); parts += p_; cuts += c_
    parts = to_opt(shapely.unary_union(parts)); cuts = to_opt(shapely.unary_union(cuts))
    panel_reg = box(UE, 0, UE + L, VLEN)
    keep = shapely.unary_union([box(0, vc - BH2, L + 2 * UE, vc + BH2) for vc in VC])
    recess = shapely.unary_union([box(UE, 0, UE + 0.5, VLEN), box(UE + L - 0.5, 0, UE + L, VLEN)]).difference(keep)
    outside = full.difference(panel_reg).difference(parts)            # 支撑里除了卡扣以外都空
    void = shapely.unary_union([outside, cuts, recess])
    domain = full.difference(void)
    corner = shapely.unary_union([box(UE, vc - BH2, UE + P.ARM_ROOT + ROOT_BAND, vc + BH2) for vc in VC] +
                                 [box(UE + L - P.ARM_ROOT - ROOT_BAND, vc - BH2, UE + L, vc + BH2) for vc in VC])
    middle = shapely.unary_union([box(UE + 0.5, vc - YI + SLIT, UE + P.ARM_ROOT, vc + YI - SLIT) for vc in VC] +
                                 [box(UE + L - P.ARM_ROOT, vc - YI + SLIT, UE + L - 0.5, vc + YI - SLIT) for vc in VC])
    solid = shapely.unary_union([parts, corner.difference(middle)]).intersection(domain)
    polys = [g for g in getattr(solid, "geoms", [solid]) if g.geom_type == "Polygon" and g.area > 1e-6]
    return domain, {f"snap{i}": g for i, g in enumerate(polys)}


JOINTS = [("L", 0), ("L", 1), ("R", 0), ("R", 1)]


def contact_sets(m):
    """接触约束：[(自由度, n, 接口, 种类)]。n：面板沿该自由度往支撑里压的方向（穿入 = n·(u_面板 − u_支撑) > 0）"""
    h = m.h; tol = 1e-6
    S = np.zeros((m.nelx + 2, m.nely + 2), bool)
    S[1:-1, 1:-1] = (m.solid | m.active).reshape(m.nelx, m.nely)        # 有可能有料的单元
    FX = np.zeros_like(S); FX[1:-1, 1:-1] = m.solid.reshape(m.nelx, m.nely)
    ni, nj = np.round(m.xn / h).astype(int), np.round(m.yn / h).astype(int)
    ll, lr, ul, ur = FX[ni, nj], FX[ni + 1, nj], FX[ni, nj + 1], FX[ni + 1, nj + 1]
    anyS = S[ni, nj] | S[ni + 1, nj] | S[ni, nj + 1] | S[ni + 1, nj + 1]
    out = []
    for side, k in JOINTS:
        sx = -1 if side == "L" else 1
        af = UE if side == "L" else UE + L
        dv = m.yn - VC[k]
        # 台肩：x = af 的端面，槽口以外，面板一侧有固定实心
        sh = np.isclose(m.xn, af) & (np.abs(dv) <= BH2 + tol) & (np.abs(dv) >= P.SLOT_A / 2 - tol)
        sh &= (lr | ur) if side == "L" else (ll | ul)
        for n_ in np.nonzero(sh)[0]: out.append((2 * n_, sx, (side, k), "台肩"))
        # 槽短边壁：卡扣臂外侧面（|dv| ≈ YA），在支撑厚度内
        inx = (sx * (m.xn - af) >= -tol) & (sx * (m.xn - af) <= P.T + tol)
        for sy in (1, -1):
            face = inx & (np.abs(dv - sy * YA) < 0.75 * h)
            face &= ((ll | lr) & ~(ul | ur)) if sy > 0 else ((ul | ur) & ~(ll | lr))
            for n_ in np.nonzero(face)[0]: out.append((2 * n_ + 1, sy, (side, k), "槽壁"))
        # 卡钩：钩内侧面，面向支撑外侧面（离 af 为 T + SNAP_PLAY），|dv| 在 YA..YA+HOOK_H
        hx = af + sx * (P.T + P.SNAP_PLAY)
        hk = (np.abs(m.xn - hx) < 0.75 * h) & (np.abs(dv) >= YA - tol) & (np.abs(dv) <= YA + P.HOOK_H + tol)
        hk &= ((ll | ul) & ~(lr | ur)) if side == "L" else ((lr | ur) & ~(ll | ul))
        for n_ in np.nonzero(hk)[0]: out.append((2 * n_, -sx, (side, k), "卡钩"))
    return out


def scenarios():
    """(名称, 权重, 刚体列表)。刚体：(接口列表, {u/v/t: ("fix", 0) 或 ("force", 值)})。t 的力是扭矩 N·mm"""
    f2 = P.F_RACK / 2
    fx = {"u": ("fix", 0), "v": ("fix", 0), "t": ("fix", 0)}
    one = lambda j, c: ([j], c)
    sc = [
        ("侧向错动", 1.0, [one(("L", 0), fx), one(("R", 0), fx), one(("L", 1), {"u": ("force", f2), "v": ("fix", 0), "t": ("fix", 0)}),
                           one(("R", 1), {"u": ("force", f2), "v": ("fix", 0), "t": ("fix", 0)})]),
        ("撑开", 0.5, [one(("L", 0), fx), one(("L", 1), fx), one(("R", 0), {"u": ("force", f2), "v": ("fix", 0), "t": ("fix", 0)}),
                       one(("R", 1), {"u": ("force", f2), "v": ("fix", 0), "t": ("fix", 0)})]),
        ("前后错动", 0.5, [one(("L", 0), fx), one(("R", 0), fx), one(("L", 1), {"u": ("force", 0), "v": ("force", f2), "t": ("fix", 0)}),
                           one(("R", 1), {"u": ("force", 0), "v": ("force", -f2), "t": ("fix", 0)})]),
    ]
    for side, nm in (("L", "左"), ("R", "右")):
        other = "R" if side == "L" else "L"
        for sg, s_nm in ((1, "正"), (-1, "反")):
            sc.append((f"{nm}支撑绕法线{s_nm}转", 0.25, [([(side, 0), (side, 1)], {"u": ("force", 0), "v": ("force", 0), "t": ("force", sg * M_TWIST)}),
                                                        ([(other, 0), (other, 1)], fx)]))
    return sc


def body_center(joints):
    a = np.mean([UE - P.T / 2 if s == "L" else UE + L + P.T / 2 for s, k in joints]); v = np.mean([VC[k] for s, k in joints])
    return a, v


def assemble(m, Ek):
    KE = T2.element_stiffness(P.NU)
    return sp.csr_matrix(((KE.ravel()[None, :] * Ek[:, None]).ravel(), (m.iK, m.jK)), shape=(m.ndof, m.ndof))


def contact_solve(m, K, cons, bodies, maxit=60):
    """非线性单向接触（主动集 + 罚）。返回 U、主动集、接触力（作用在面板自由度上）、各刚体位移"""
    nb = len(bodies); N = m.ndof + 3 * nb
    dof = np.array([c[0] for c in cons]); n = np.array([c[1] for c in cons], float); jt = [c[2] for c in cons]
    bidx = np.full(len(cons), -1)
    for b, (joints, ctrl) in enumerate(bodies):
        for i, j in enumerate(jt):
            if j in joints: bidx[i] = b
    node = dof // 2; isu = dof % 2 == 0
    R = np.zeros((len(cons), 3))                                       # 接触点处支撑位移 = R · q_b
    for b, (joints, ctrl) in enumerate(bodies):
        ac, vc = body_center(joints); sel = bidx == b
        R[sel & isu, 0] = 1; R[sel & isu, 2] = -(m.yn[node[sel & isu]] - vc)
        R[sel & ~isu, 1] = 1; R[sel & ~isu, 2] = (m.xn[node[sel & ~isu]] - ac)
    kp = KP_REL * P.E_MAT * P.PANEL_T
    fixq = []; Fq = np.zeros(3 * nb)
    for b, (joints, ctrl) in enumerate(bodies):
        for c, key in enumerate("uvt"):
            kind, val = ctrl[key]
            if kind == "fix": fixq.append(m.ndof + 3 * b + c)
            else: Fq[3 * b + c] = val
    fixq = np.array(fixq, int); freeq = np.setdiff1d(np.arange(N), fixq)
    act = np.ones(len(cons), bool); solver = pypardiso.PyPardisoSolver(mtype=2)
    for it in range(maxit):
        rows, cols, vals = [], [], []
        a_ = np.nonzero(act)[0]
        for i in a_:                                                   # 罚项 k·(e_d − R_i)ᵀ(e_d − R_i)
            idx = [dof[i]] + [m.ndof + 3 * bidx[i] + c for c in range(3)]
            w = np.concatenate([[1.0], -R[i]])
            rows.append(np.repeat(idx, 4)); cols.append(np.tile(idx, 4)); vals.append(kp * np.outer(w, w).ravel())
        Kc = sp.csr_matrix((np.concatenate(vals) if vals else [], (np.concatenate(rows) if rows else [], np.concatenate(cols) if cols else [])), shape=(N, N))
        Kt = sp.bmat([[K, None], [None, sp.csr_matrix((3 * nb, 3 * nb))]], format="csr") + Kc + sp.diags(np.r_[np.zeros(m.ndof), np.full(3 * nb, 1e-6 * kp)])
        Fv = np.zeros(N); Fv[m.ndof:] = Fq
        Kf = sp.triu(Kt[freeq][:, freeq], format="csr")
        X = np.zeros(N); X[freeq] = solver.solve(Kf, Fv[freeq]); solver.free_memory()
        U = X[:m.ndof]; q = X[m.ndof:].reshape(nb, 3)
        us = np.einsum("ij,ij->i", R, q[bidx])
        g = n * (U[dof] - us)                                          # > 0：穿进支撑
        new = g > 0
        if (new == act).all(): break
        act = new
    fc = np.where(act, -kp * g * n, 0.0)                               # 支撑对面板的力
    return dict(U=U, act=act, fc=fc, q=q, it=it, dof=dof, bidx=bidx, n=n)


def linearize(m, cons, sc, sol):
    """把非线性接触的结果变成线性工况：受力的接触点，若它所在刚体在该方向（横向 = u，沿斜面 = v）是固定的 → 约束，
    否则 → 节点力（取非线性解的接触力）。全固定的刚体就是全部约束"""
    name, w, bodies = sc
    a = sol["act"]; dof, bidx = sol["dof"], sol["bidx"]
    isu = dof % 2 == 0
    fixdir = np.array([bodies[b][1]["u" if u_ else "v"][0] == "fix" for b, u_ in zip(bidx, isu)])
    fixdof = np.unique(dof[a & fixdir])
    F = np.zeros(m.ndof); ld = a & ~fixdir
    np.add.at(F, dof[ld], sol["fc"][ld])
    return F, fixdof


def vm_field(m, U, Ek_mask):
    nu, h = P.NU, m.h
    D = P.E_MAT / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    xin = np.array([-1, 1, 1, -1]); etn = np.array([-1, -1, 1, 1])
    B = np.zeros((3, 8)); B[0, 0::2] = 0.25 * xin * 2 / h; B[1, 1::2] = 0.25 * etn * 2 / h; B[2, 0::2] = 0.25 * etn * 2 / h; B[2, 1::2] = 0.25 * xin * 2 / h
    sig = U[m.edof] @ (D @ B).T
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2 + 3 * sig[:, 2] ** 2)
    return np.where(Ek_mask, vm, 0.0)


def raster(m, outline_print):
    """打印坐标的面板轮廓（含卡扣）→ 每个单元是否有料"""
    return shapely.contains_xy(to_opt(outline_print), m.xc, m.yc)


def evaluate(m, x_bin, scs, cons, tag):
    """非线性接触复核：每个工况的柔度（外力功）、最大应力、角上几处的应力"""
    E0 = P.E_MAT * P.PANEL_T
    K = assemble(m, np.where(x_bin, E0, E0 * 1e-9))
    res = []; fields = {}
    zones = {}
    for side, k in JOINTS:                                             # 离支撑内侧面 0–18 mm：臂根和台肩条；18–45 mm：角块和 X 撑 / 侧边条交汇处
        sx = -1 if side == "L" else 1; af = UE if side == "L" else UE + L
        b0, b1 = sorted([af, af - sx * (P.ARM_ROOT + ROOT_BAND)])
        zones[f"{side}{k}臂根/台肩条"] = (b0, b1, VC[k] - BH2, VC[k] + BH2)
        a0, a1 = sorted([af - sx * (P.ARM_ROOT + ROOT_BAND), af - sx * 45])
        zones[f"{side}{k}交汇处"] = (a0, a1, VC[k] - BH2 - 20, VC[k] + BH2 + 20)
    for nm, w, bodies in scs:
        sol = contact_solve(m, K, cons, bodies)
        F_ext = np.zeros(m.ndof); np.add.at(F_ext, sol["dof"], sol["fc"])
        vm = vm_field(m, sol["U"], x_bin)
        # 柔度：受载刚体的广义力 · 位移
        comp = 0.0
        for b, (joints, ctrl) in enumerate(bodies):
            for c, key in enumerate("uvt"):
                if ctrl[key][0] == "force": comp += ctrl[key][1] * sol["q"][b, c]
        zr = {}
        for zn, (a0, a1, v0, v1) in zones.items():
            mm = x_bin & (m.xc >= a0) & (m.xc <= a1) & (m.yc >= v0) & (m.yc <= v1)
            zr[zn] = float(vm[mm].max()) if mm.any() else 0.0
        inp = x_bin & (m.xc >= UE) & (m.xc <= UE + L)                  # 面板本体（不含伸进支撑的卡扣段：卡钩点接触处有离散奇异）
        vmp = np.where(inp, vm, 0.0); i = int(np.argmax(vmp))
        res.append(dict(工况=nm, 柔度=float(comp), 迭代=sol["it"], 最大应力_MPa=float(vmp.max()), 最大应力位置_打印坐标=[float(m.xc[i] - HALF - UE), float(m.yc[i] - V0)],
                        VM_99_9pct_MPa=float(np.percentile(vm[inp], 99.9)), 卡扣段最大应力_MPa=float(vm[x_bin & ~inp].max()), 分区最大应力_MPa=zr))
        fields[nm] = vm
    return res, fields


def sym_operator(m):
    """左右（a → L + 2UE − a）、前后（v → VLEN − v）两个镜像：每个单元在 4 个对称像上取平均。网格和严格对称差不到一格，取最近的单元"""
    W = L + 2 * UE
    ii = np.clip(np.round((W - m.xc) / m.h - 0.5).astype(int), 0, m.nelx - 1)
    jj = np.clip(np.round((VLEN - m.yc) / m.h - 0.5).astype(int), 0, m.nely - 1)
    e = np.arange(m.nel); I, J = m.I, m.J
    imgs = [e, ii * m.nely + J, I * m.nely + jj, ii * m.nely + jj]
    S = sp.csr_matrix((np.full(4 * m.nel, 0.25), (np.tile(e, 4), np.concatenate(imgs))), shape=(m.nel, m.nel))
    return S


def symmetrize(shape):
    """打印坐标的面板本体轮廓 → 取左下四分之一，左右、前后镜像拼成严格对称"""
    ym = VLEN / 2 - V0
    q = shape.intersection(box(-HALF - 50, -V0 - 50, 0, ym))
    parts = [q, affinity.scale(q, -1, 1, origin=(0, 0)), affinity.scale(q, 1, -1, origin=(0, ym)), affinity.scale(q, -1, -1, origin=(0, ym))]
    g = shapely.unary_union(parts).buffer(0.01).buffer(-0.01)
    if isinstance(g, MultiPolygon):
        g = sorted(g.geoms, key=lambda t: -t.area)[0]
    return Polygon(g.exterior, [hh for hh in g.interiors if Polygon(hh).area >= 20.0])


def extract(xphys, m, level=0.5):
    """密度场 → 面板本体轮廓（打印坐标，不含伸进支撑的卡扣段；with_snaps_v2 再精确补上卡扣）"""
    G = xphys.reshape(m.nelx, m.nely).T
    d = 0.2
    xs = np.arange(-2, m.nelx * m.h + 2, d); ys = np.arange(-2, m.nely * m.h + 2, d)
    XX, YY = np.meshgrid(xs, ys)
    Fd = map_coordinates(gaussian_filter(G, 0.6), [YY.ravel() / m.h - 0.5, XX.ravel() / m.h - 0.5], order=1, mode="constant", cval=0).reshape(XX.shape)
    rings = [Polygon(np.c_[xs[0] + c[:, 1] * d, ys[0] + c[:, 0] * d]).buffer(0) for c in measure.find_contours(Fd, level) if len(c) >= 8]
    region = Polygon()
    for pg in rings: region = region.symmetric_difference(pg)
    region = region.intersection(box(UE, 0, UE + L, VLEN))
    region = shapely.unary_union([region, shapely.unary_union(list(m.solids.values())).intersection(box(UE, 0, UE + L, VLEN))])
    region = region.buffer(R_FILLET).buffer(-R_FILLET).buffer(-1.0).buffer(1.0).intersection(box(UE, 0, UE + L, VLEN))
    if isinstance(region, MultiPolygon):
        parts = sorted(region.geoms, key=lambda g: -g.area); print("警告：面板有分离块，面积", [round(g.area, 1) for g in parts]); region = parts[0]
    holes = [hh for hh in region.interiors if Polygon(hh).area >= 20.0]
    return affinity.translate(Polygon(region.exterior, holes).simplify(0.02), -(HALF + UE), -V0)   # 打印坐标


def with_snaps_v2(shape):
    parts, cuts = [shape], []
    for vc in VC:
        for sx in (1, -1):
            p_, c_ = snap_v2(sx, vc - V0); parts += p_; cuts += c_
    relief = shapely.unary_union([box(-HALF - 30, -500, -HALF + 0.5, 500), box(HALF - 0.5, -500, HALF + 30, 500)])
    keep = shapely.unary_union([box(-HALF - 30, vc - V0 - BH2, HALF + 30, vc - V0 + BH2) for vc in VC])
    cuts.append(relief.difference(keep))
    out = shapely.unary_union(parts).difference(shapely.unary_union(cuts))
    if out.geom_type != "Polygon":
        parts_ = sorted(getattr(out, "geoms", [out]), key=lambda g: -g.area); print("警告：卡扣拼上后有分离块", [round(g.area, 1) for g in parts_]); out = parts_[0]
    return shapely.geometry.polygon.orient(out, 1.0)


def main():
    ap = argparse.ArgumentParser()
    T2.add_args(ap, P.PANEL_VF)
    ap.set_defaults(robust=True, rmin=4.0, maxit=300, h=0.5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--verify_only", action="store_true", help="只用非线性接触复核现在的面板")
    ap.add_argument("--check_only", action="store_true", help="只检查线性化工况（不优化）")
    ap.add_argument("--post_only", action="store_true", help="跳过优化，读 topo/result.npz 做导出和复核")
    a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    m = T2.Mesh(a.h, True, geom=geometry())
    cons = contact_sets(m)
    kinds = {}
    for c in cons: kinds[(c[2], c[3])] = kinds.get((c[2], c[3]), 0) + 1
    print("接触点:", {f"{k[0][0]}{k[0][1]}{k[1]}": v for k, v in kinds.items()})
    cur = json.load(open(f"{P.OUT_DIR}/panel/panel_outline.json"))
    cur_poly = Polygon(cur["exterior"], cur["holes"])
    x_ref = raster(m, cur_poly)
    scs = scenarios()
    ref, ref_f = evaluate(m, x_ref, scs, cons, "现在")
    for r in ref: print("现在", json.dumps(r, ensure_ascii=False), flush=True)
    if a.verify_only:
        json.dump(dict(ref=ref), open(f"{a.out}/verify_ref.json", "w"), ensure_ascii=False, indent=1); return
    E0 = P.E_MAT * P.PANEL_T
    if a.post_only:
        return post(a, m, cons, scs, cur_poly, ref, ref_f, x_ref, t0)
    # 线性化工况（用现在面板的接触状态）
    Kref = assemble(m, np.where(x_ref, E0, E0 * 1e-9))
    Fs, groups_d, names, imp = [], {}, [], []
    for i, sc in enumerate(scs):
        sol = contact_solve(m, Kref, cons, sc[2])
        F, fixdof = linearize(m, cons, sc, sol)
        Fs.append(F); names.append(sc[0]); imp.append(sc[1])
        groups_d.setdefault(tuple(fixdof), []).append(i)
        print(f"{sc[0]}: 固定 {fixdof.size} 个自由度，受载 {np.count_nonzero(F)} 个", flush=True)
    F = np.stack(Fs, 1)
    groups = [(F[:, idx], np.setdiff1d(np.arange(m.ndof), np.array(k, int)), idx) for k, idx in groups_d.items()]
    # 现在面板在线性化工况下的柔度（归一化基准）
    cref = np.zeros(len(scs))
    for Fg, free, idx in groups:
        Kf = sp.triu(Kref[free][:, free], format="csr")
        Uf = pypardiso.spsolve(Kf.tocsr(), Fg[free]) if False else pypardiso.PyPardisoSolver(mtype=2).solve(Kf, Fg[free])
        Uf = Uf.reshape(free.size, -1)
        for c, i in enumerate(idx): cref[i] = float(Fg[free, c] @ Uf[:, c])
    print("线性化工况下现在面板的柔度", dict(zip(names, np.round(cref, 4))), flush=True)
    assert np.all(np.isfinite(cref)) and np.all(cref > 0) and np.all(cref < 1e3), "线性化工况可能奇异"
    if getattr(a, "check_only", False): return
    # 用料：与现在面板相同（面板本体部分）
    A_ref = cur_poly.intersection(box(-HALF, -500, HALF, 500)).area
    A_fix = shapely.unary_union(list(m.solids.values())).intersection(box(UE, 0, UE + L, VLEN)).area
    A_act = float(m.active.sum()) * a.h ** 2
    a.vf = (A_ref - A_fix) / A_act
    a.pad_void = True
    print(f"用料：现在面板本体 {A_ref:.0f} mm²，固定 {A_fix:.0f}，可设计 {A_act:.0f} → 体积分数 {a.vf:.4f}", flush=True)
    free0 = groups[0][1]; fixed0 = np.setdiff1d(np.arange(m.ndof), free0)
    a.out_topo = a.out + "/topo"; out_save = a.out; a.out = a.out_topo
    if not a.post_only:
        T2.optimize(m, F, names, np.array(imp), free0, fixed0, a, thick=P.PANEL_T, vf_mode="active", fe_groups=groups, t0=t0, cnorm=cref, sym=sym_operator(m))
    a.out = out_save
    return post(a, m, cons, scs, cur_poly, ref, ref_f, x_ref, t0)


def post(a, m, cons, scs, cur_poly, ref, ref_f, x_ref, t0):
    a.out_topo = a.out + "/topo"
    r = np.load(f"{a.out_topo}/result.npz")
    shape = symmetrize(extract(r["xphys"], m))
    outline = with_snaps_v2(shape)
    json.dump(dict(exterior=list(outline.exterior.coords), holes=[list(hh.coords) for hh in outline.interiors],
                   half=HALF, yc=[vc - V0 for vc in VC], thick=P.PANEL_T, joints=[j["name"] for j in PN.PJ]),
              open(f"{a.out}/panel_outline.json", "w"))
    x_new = raster(m, outline)
    new, new_f = evaluate(m, x_new, scs, cons, "新")
    info = dict(现在_面积_mm2=round(cur_poly.area, 1), 新_面积_mm2=round(outline.area, 1), 现在_用料_cm3=round(cur_poly.area * P.PANEL_T / 1000, 1),
                新_用料_cm3=round(outline.area * P.PANEL_T / 1000, 1), 开孔数=len(outline.interiors), size_mm=[round(outline.bounds[2] - outline.bounds[0], 1), round(outline.bounds[3] - outline.bounds[1], 1)])
    json.dump(dict(info=info, ref=ref, new=new, args={k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, bool))}), open(f"{a.out}/verify.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))
    for r0, r1 in zip(ref, new):
        print(f"{r0['工况']}: 柔度 {r0['柔度']:.4g} → {r1['柔度']:.4g}（{r1['柔度'] / r0['柔度']:.2f}），最大应力 {r0['最大应力_MPa']:.2f} → {r1['最大应力_MPa']:.2f} MPa")
    np.savez_compressed(f"{a.out}/fields.npz", x_ref=x_ref, x_new=x_new, **{f"ref_{i}": f for i, f in enumerate(ref_f.values())}, **{f"new_{i}": f for i, f in enumerate(new_f.values())})
    print(f"完成，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
