"""
后横梁（卡扣按真实接触建模，方法同 panel_contact.py）：现在的后横梁是一根 40 × 16 mm 的实心方杆，每边只有一个卡扣；
支撑之间的力、扭矩全靠两个卡扣和杆身传，连接处和面板一样只靠台肩条、卡扣臂和卡钩的单向接触。
做法：
  - 设计范围前后加宽到 W_BEAM（以卡扣中心为准，默认 80 mm），厚度不变（BEAM_W = 16，平放打印）；用料与现在的实心杆相同。
  - 卡扣用 params.snap_end（臂内侧让位缝）；卡扣臂、台肩条和臂根 3 mm 一条固定实心，两臂之间和其余部分交给优化；
    两端在支撑内侧面处、卡扣角（±BEAM_H/2）以外退 0.5 mm，只让台肩贴住支撑。
  - 每个接口当一块刚性支撑，台肩 / 槽壁 / 卡钩单向接触；先用现在的横梁解非线性接触定接触状态，再线性化做优化，最后非线性复核。
  - 工况：往外撑开、往里压拢（各 F_BEAM）；两片支撑一前一后错动（F_BEAM，两端都不许转）；左、右支撑各绕竖直轴正反转（±M_TWIST）。
  - 强制左右、前后四分之一对称。目标：各工况柔度相对现在横梁的比值加权和。
结果放 out/<方案>/beam_contact/，不覆盖现有横梁。
用法: STAND_VARIANT=desk40 python beam_contact.py [--h 0.5] [--maxit 300] [--gpu] [--post_only]
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
from shapely import affinity
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import measure
import pypardiso
import params as P
import topopt2d as T2

OUT = f"{P.OUT_DIR}/beam_contact"
HALF = P.SPACING / 2 - P.T / 2         # 支撑内侧面 X = ±HALF（与 build3d 的 half 相同）
L = 2 * HALF
W_BEAM = 80.0                          # 设计范围前后总宽（以卡扣中心为准）
VLEN = W_BEAM; V0 = W_BEAM / 2; VC = [V0]
BH2 = P.BEAM_H / 2
YA = P.TENON_H / 2; YI = YA - P.ARM_T
SLIT = P.SLIT_W
ROOT_BAND = 3.0
R_FILLET = 4.0
F_BEAM = 10.0
M_TWIST = 2000.0
UE = 20.0
KP_REL = 50.0
JOINTS = [("L", 0), ("R", 0)]


def snap_v2(sx, yc):
    return P.snap_end(sx, HALF, yc)


def to_opt(g):
    return affinity.translate(g, HALF + UE, V0)


def geometry():
    full = box(0, 0, L + 2 * UE, VLEN)
    parts, cuts = [], []
    for sx in (1, -1):
        p_, c_ = snap_v2(sx, 0.0); parts += p_; cuts += c_
    parts = to_opt(shapely.unary_union(parts)); cuts = to_opt(shapely.unary_union(cuts))
    body = box(UE, 0, UE + L, VLEN)
    keep = box(0, V0 - BH2, L + 2 * UE, V0 + BH2)
    recess = shapely.unary_union([box(UE, 0, UE + 0.5, VLEN), box(UE + L - 0.5, 0, UE + L, VLEN)]).difference(keep)
    outside = full.difference(body).difference(parts)
    domain = full.difference(shapely.unary_union([outside, cuts, recess]))
    corner = shapely.unary_union([box(UE, V0 - BH2, UE + P.ARM_ROOT + ROOT_BAND, V0 + BH2),
                                  box(UE + L - P.ARM_ROOT - ROOT_BAND, V0 - BH2, UE + L, V0 + BH2)])
    middle = shapely.unary_union([box(UE + 0.5, V0 - YI + SLIT, UE + P.ARM_ROOT, V0 + YI - SLIT),
                                  box(UE + L - P.ARM_ROOT, V0 - YI + SLIT, UE + L - 0.5, V0 + YI - SLIT)])
    solid = shapely.unary_union([parts, corner.difference(middle)]).intersection(domain)
    polys = [g for g in getattr(solid, "geoms", [solid]) if g.geom_type == "Polygon" and g.area > 1e-6]
    return domain, {f"snap{i}": g for i, g in enumerate(polys)}


def scenarios():
    F = F_BEAM
    fx = {"u": ("fix", 0), "v": ("fix", 0), "t": ("fix", 0)}
    sc = [
        ("撑开", 0.5, [([("L", 0)], fx), ([("R", 0)], {"u": ("force", F), "v": ("fix", 0), "t": ("fix", 0)})]),
        ("压拢", 0.5, [([("L", 0)], fx), ([("R", 0)], {"u": ("force", -F), "v": ("fix", 0), "t": ("fix", 0)})]),
        ("前后错动", 1.0, [([("L", 0)], fx), ([("R", 0)], {"u": ("force", 0), "v": ("force", F), "t": ("fix", 0)})]),
    ]
    for side, nm in (("L", "左"), ("R", "右")):
        other = "R" if side == "L" else "L"
        for sg, s_nm in ((1, "正"), (-1, "反")):
            sc.append((f"{nm}支撑绕竖直轴{s_nm}转", 0.25, [([(side, 0)], {"u": ("force", 0), "v": ("force", 0), "t": ("force", sg * M_TWIST)}),
                                                          ([(other, 0)], fx)]))
    return sc


def body_center(joints):
    a = np.mean([UE - P.T / 2 if s == "L" else UE + L + P.T / 2 for s, k in joints]); v = np.mean([VC[k] for s, k in joints])
    return a, v


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
    kp = KP_REL * P.E_MAT * P.BEAM_W
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
    fixdof = dof[a & fixdir]
    # 全固定的刚体在某个方向上一个受力接触点都没有时（如撑开时左端只有卡钩受力），把它这个方向的接触点都约束上：
    # 非线性解里这些点不受力，只是锁住刚体运动，否则线性化后的刚度矩阵奇异（显卡上的 cuDSS 会出 NaN）
    for b, (j, c) in enumerate(bodies):
        if not all(c[k][0] == "fix" for k in "uvt"): continue
        for comp in (True, False):
            sel = (bidx == b) & (isu == comp)
            if not (a & sel).any(): fixdof = np.concatenate([fixdof, dof[sel]])
    fixdof = np.unique(fixdof)
    F = np.zeros(m.ndof); ld = a & ~fixdir
    np.add.at(F, dof[ld], sol["fc"][ld])
    return F, fixdof


def assemble(m, Ek):
    KE = T2.element_stiffness(P.NU)
    return sp.csr_matrix(((KE.ravel()[None, :] * Ek[:, None]).ravel(), (m.iK, m.jK)), shape=(m.ndof, m.ndof))


def vm_field(m, U, Ek_mask):
    nu, h = P.NU, m.h
    D = P.E_MAT / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    xin = np.array([-1, 1, 1, -1]); etn = np.array([-1, -1, 1, 1])
    B = np.zeros((3, 8)); B[0, 0::2] = 0.25 * xin * 2 / h; B[1, 1::2] = 0.25 * etn * 2 / h; B[2, 0::2] = 0.25 * etn * 2 / h; B[2, 1::2] = 0.25 * xin * 2 / h
    sig = U[m.edof] @ (D @ B).T
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2 + 3 * sig[:, 2] ** 2)
    return np.where(Ek_mask, vm, 0.0)


def evaluate(m, x_bin, scs, cons, tag):
    """非线性接触复核：每个工况的柔度（外力功）、最大应力、角上几处的应力"""
    E0 = P.E_MAT * P.BEAM_W
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


def sym_operator(m):
    """左右（a → L + 2UE − a）、前后（v → VLEN − v）两个镜像：每个单元在 4 个对称像上取平均。网格和严格对称差不到一格，取最近的单元"""
    W = L + 2 * UE
    ii = np.clip(np.round((W - m.xc) / m.h - 0.5).astype(int), 0, m.nelx - 1)
    jj = np.clip(np.round((VLEN - m.yc) / m.h - 0.5).astype(int), 0, m.nely - 1)
    e = np.arange(m.nel); I, J = m.I, m.J
    imgs = [e, ii * m.nely + J, I * m.nely + jj, ii * m.nely + jj]
    S = sp.csr_matrix((np.full(4 * m.nel, 0.25), (np.tile(e, 4), np.concatenate(imgs))), shape=(m.nel, m.nel))
    return S


def raster(m, outline_print):
    return shapely.contains_xy(to_opt(outline_print), m.xc, m.yc)


def symmetrize(shape):
    """打印坐标的杆身轮廓 → 取 X ≤ 0、Y ≤ 0 的四分之一，左右、前后镜像拼成严格对称"""
    q = shape.intersection(box(-HALF - 50, -V0 - 50, 0, 0))
    parts = [q, affinity.scale(q, -1, 1, origin=(0, 0)), affinity.scale(q, 1, -1, origin=(0, 0)), affinity.scale(q, -1, -1, origin=(0, 0))]
    g = shapely.unary_union(parts).buffer(0.01).buffer(-0.01)
    if isinstance(g, MultiPolygon):
        g = sorted(g.geoms, key=lambda t: -t.area)[0]
    return Polygon(g.exterior, [hh for hh in g.interiors if Polygon(hh).area >= 20.0])


def with_snaps_v2(shape):
    parts, cuts = [shape], []
    for sx in (1, -1):
        p_, c_ = snap_v2(sx, 0.0); parts += p_; cuts += c_
    relief = shapely.unary_union([box(-HALF - 30, -500, -HALF + 0.5, 500), box(HALF - 0.5, -500, HALF + 30, 500)])
    cuts.append(relief.difference(box(-HALF - 30, -BH2, HALF + 30, BH2)))
    out = shapely.unary_union(parts).difference(shapely.unary_union(cuts))
    if out.geom_type != "Polygon":
        parts_ = sorted(getattr(out, "geoms", [out]), key=lambda g: -g.area); print("警告：卡扣拼上后有分离块", [round(g.area, 1) for g in parts_]); out = parts_[0]
    return shapely.geometry.polygon.orient(out, 1.0)


def main():
    ap = argparse.ArgumentParser()
    T2.add_args(ap, 0.5)
    ap.set_defaults(robust=True, rmin=4.0, maxit=300, h=0.5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--post_only", action="store_true", help="跳过优化，读 topo/result.npz 做导出和复核")
    a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    m = T2.Mesh(a.h, True, geom=geometry())
    cons = contact_sets(m)
    kinds = {}
    for c in cons: kinds[(c[2], c[3])] = kinds.get((c[2], c[3]), 0) + 1
    print("接触点:", {f"{k[0][0]}{k[1]}": v for k, v in kinds.items()})
    cur = json.load(open(f"{P.MODEL_DIR}/beam_outline.json"))
    cur_poly = Polygon(cur["exterior"], cur["holes"])
    x_ref = raster(m, cur_poly)
    scs = scenarios()
    ref, ref_f = evaluate(m, x_ref, scs, cons, "现在")
    for r in ref: print("现在", json.dumps(r, ensure_ascii=False), flush=True)
    if a.post_only:
        return post(a, m, cons, scs, cur_poly, ref, ref_f, x_ref, t0)
    E0 = P.E_MAT * P.BEAM_W
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
    cref = np.zeros(len(scs))
    for Fg, free, idx in groups:
        Kf = sp.triu(Kref[free][:, free], format="csr")
        Uf = pypardiso.PyPardisoSolver(mtype=2).solve(Kf, Fg[free]).reshape(free.size, -1)
        for c, i in enumerate(idx): cref[i] = float(Fg[free, c] @ Uf[:, c])
    print("线性化工况下现在横梁的柔度", dict(zip(names, np.round(cref, 4))), flush=True)
    assert np.all(np.isfinite(cref)) and np.all(cref > 0) and np.all(cref < 1e3), "线性化工况可能奇异"
    A_ref = cur_poly.intersection(box(-HALF, -500, HALF, 500)).area
    A_fix = shapely.unary_union(list(m.solids.values())).intersection(box(UE, 0, UE + L, VLEN)).area
    A_act = float(m.active.sum()) * a.h ** 2
    a.vf = (A_ref - A_fix) / A_act
    a.pad_void = True
    print(f"用料：现在杆身 {A_ref:.0f} mm²，固定 {A_fix:.0f}，可设计 {A_act:.0f} → 体积分数 {a.vf:.4f}", flush=True)
    free0 = groups[0][1]; fixed0 = np.setdiff1d(np.arange(m.ndof), free0)
    out_save = a.out; a.out = a.out + "/topo"
    T2.optimize(m, F, names, np.array(imp), free0, fixed0, a, thick=P.BEAM_W, vf_mode="active", fe_groups=groups, t0=t0, cnorm=cref, sym=sym_operator(m))
    a.out = out_save
    return post(a, m, cons, scs, cur_poly, ref, ref_f, x_ref, t0)


def post(a, m, cons, scs, cur_poly, ref, ref_f, x_ref, t0):
    r = np.load(f"{a.out}/topo/result.npz")
    shape = symmetrize(extract(r["xphys"], m))
    outline = with_snaps_v2(shape)
    json.dump(dict(exterior=list(outline.exterior.coords), holes=[list(hh.coords) for hh in outline.interiors], half=HALF),
              open(f"{a.out}/beam_outline.json", "w"))
    x_new = raster(m, outline)
    new, new_f = evaluate(m, x_new, scs, cons, "新")
    info = dict(现在_面积_mm2=round(cur_poly.area, 1), 新_面积_mm2=round(outline.area, 1), 现在_用料_cm3=round(cur_poly.area * P.BEAM_W / 1000, 1),
                新_用料_cm3=round(outline.area * P.BEAM_W / 1000, 1), 开孔数=len(outline.interiors),
                size_mm=[round(outline.bounds[2] - outline.bounds[0], 1), round(outline.bounds[3] - outline.bounds[1], 1)], W_BEAM=W_BEAM)
    json.dump(dict(info=info, ref=ref, new=new), open(f"{a.out}/verify.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))
    for r0, r1 in zip(ref, new):
        print(f"{r0['工况']}: 柔度 {r0['柔度']:.4g} → {r1['柔度']:.4g}（{r1['柔度'] / r0['柔度']:.2f}），最大应力 {r0['最大应力_MPa']:.2f} → {r1['最大应力_MPa']:.2f} MPa")
    np.savez_compressed(f"{a.out}/fields.npz", x_ref=x_ref, x_new=x_new, nelx=m.nelx, nely=m.nely, h=m.h,
                        **{f"ref_{i}": f for i, f in enumerate(ref_f.values())}, **{f"new_{i}": f for i, f in enumerate(new_f.values())})
    print(f"完成，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
