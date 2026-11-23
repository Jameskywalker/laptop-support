"""
面板 / 后横梁的无填充三维拓扑优化（端头是刚性插指 + U 形卡子，params.clip_end，按单向接触建模）

做法（接触建模同 panel_contact.py / beam_contact.py / clip_trial.py，只是换成三维体素）：
  - 网格：立方体素 h（默认 0.5 mm）；a 沿横向（0 = 网格左端，支撑内侧面在 a = UE、UE + L），v 沿面板斜面 / 横梁前后，
    z = 打印方向（0 = 贴床面）。面板 z ∈ [0, 16]；后横梁杆身可加高到 H（默认 32 mm），插指和方槽里那段只有 16 mm。
  - 固定实心：插指（伸进支撑的两根，16 mm 厚，带卡子缺口）+ 台肩条（支撑内侧面往里 ROOT_BAND，|dv| ∈ [FING_IN, BEAM_H/2]）；
    两插指之间离支撑内侧面 0.5 mm 以内、两侧（台肩以外）离支撑内侧面 0.5 mm 以内留空；其余杆身 / 面板都是设计域。
  - 每个接口是一块刚性支撑（6 个自由度），和零件之间只有单向接触（零间隙）：
      台肩（支撑内侧面，横向）、槽长边壁（插指外侧面，v 向）、槽短边壁（插指上下面，z 向，三维新加）、卡子腿（缺口外壁，横向反向）。
  - 工况：二维接触优化的面内工况 + 面外工况（法向 / 上下错动、绕横向轴扭、绕面内另一轴转，正反两个方向），力 10 N、扭矩 2 N·m，
    面内、面外两组的权重和相同。受载支撑只放开受载的那一个自由度，其余自由度和另一侧支撑都固定。
  - 先用现在的插指版零件（clip_trial 的轮廓拉伸 16 mm，按 3 圈墙 + 上下 4 层表皮 + 15% 填充，插指和台肩条按实心）解非线性接触
    （主动集），定下每个工况哪些接触点受力；优化时主动集固定（接触点跟着所在支撑做刚体运动），受载自由度的位移由广义力平衡解出。
    最后用非线性接触复核新旧零件。
  - 目标：各工况柔度相对现在零件的比值加权和。约束：用料（实心，克数按 1.24 g/cm³）；面板另加透空约束：
    每根 h×h 柱子有一个二维的"可以有料"变量 s（过滤 + 投影），三维密度乘上它，投影面积 Σ s̄ 不超过现在面板。
  - 45° 悬空（AM 滤波）、最小厚度（腐蚀设计）、左右 + 前后镜像对称，同 topopt3d_noinfill.py。
用法: STAND_VARIANT=desk40 python link3d.py --part panel|beam [--h 0.5] [--maxit 250] [--check] [--post_only]
      本地 CPU 自检（小网格、直接法）：python link3d.py --part beam --h 2 --cpu --check
"""
import argparse, json, os, sys, time
import numpy as np
CPU = "--cpu" in sys.argv
if CPU:
    import cpu_shim
    cpu_shim.install()
import cupy as cp
import cupyx.scipy.ndimage as cnd
import shapely
from shapely.geometry import Polygon, box
from shapely import affinity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2
import topopt3d as T3
import topopt3d_noinfill as N3

F32 = np.float32
RHO = 1.24                      # g/cm³，同 render.py 的打印重量估算
HALF = P.SPACING / 2 - P.T / 2  # 支撑内侧面 X = ±HALF
L = 2 * HALF
UE = 26.0                       # 网格伸进支撑一侧的长度（盖住插指 24 mm），同 clip_trial.UE_TRIAL
BH2 = P.BEAM_H / 2
YA = P.TENON_H / 2
TS = P.BEAM_W                   # 插指 / 方槽里的厚度：z ∈ [0, 16]
ROOT_BAND = 3.0
F_LOAD = 10.0
M_LOAD = 2000.0                 # N·mm
DOFN = ("ta", "tv", "tz", "ra", "rv", "rz")
WALL, SKIN, INFILL = P.NWALL * P.LINE_W, P.NSKIN * P.LAYER_H, P.INFILL
GMIN = 1e-6


def part_layout(part, H=None, width=None):
    if part == "panel":
        import panel as PN
        lay = dict(VC=[float(v) for v in PN.VC], VLEN=float(PN.VLEN), V0=float(PN.V0), joints=[("L", 0), ("L", 1), ("R", 0), ("R", 1)],
                   H=float(H or P.PANEL_T), ref=f"{P.OUT_DIR}/clip_trial/panel_clip_outline.json", mass_g=83.0)
    else:
        W = float(width or 80.0)
        lay = dict(VC=[W / 2], VLEN=W, V0=W / 2, joints=[("L", 0), ("R", 0)], H=float(H or 32.0),
                   ref=f"{P.OUT_DIR}/clip_trial/beam_clip_outline.json", mass_g=53.0)
    assert lay["H"] >= TS
    return lay


# ======================= 几何 =======================
class Geo:
    def __init__(self, part, h, H=None, width=None):
        lay = part_layout(part, H, width)
        self.part, self.h = part, h
        self.VC, self.VLEN, self.V0, self.joints, self.H = lay["VC"], lay["VLEN"], lay["V0"], lay["joints"], lay["H"]
        self.ref_file, self.mass_g_default = lay["ref"], lay["mass_g"]
        W = L + 2 * UE; self.W = W
        Na, Nv = int(np.ceil(W / h - 1e-9)), int(np.ceil(self.VLEN / h - 1e-9))
        NZ = int(round(self.H / h)); assert abs(NZ * h - self.H) < 1e-6, "H 要是 h 的整数倍"
        self.nzs = int(round(TS / h)); assert abs(self.nzs * h - TS) < 1e-6, "16 mm 要是 h 的整数倍"
        mult, q = 1, NZ
        while q % 2 == 0 and q > 2:
            q //= 2; mult *= 2
        mult = max(mult, 2)
        NX, NY = -(-Na // mult) * mult, -(-Nv // mult) * mult
        i0, j0 = (NX - Na) // 2, (NY - Nv) // 2
        self.NX, self.NY, self.NZ = NX, NY, NZ
        self.a0 = (W - Na * h) / 2 - i0 * h; self.v0 = (self.VLEN - Nv * h) / 2 - j0 * h      # 网格原点（节点 0 的 a、v）
        self.vox = h ** 3
        ia = np.arange(NX); core = (ia >= i0) & (ia < i0 + Na); self.ia_m = np.where(core, 2 * i0 + Na - 1 - ia, ia)
        jv = np.arange(NY); core = (jv >= j0) & (jv < j0 + Nv); self.jv_m = np.where(core, 2 * j0 + Nv - 1 - jv, jv)
        self.ac = self.a0 + (ia + 0.5) * h; self.vc = self.v0 + (jv + 0.5) * h; self.zc = (np.arange(NZ) + 0.5) * h
        self.an = self.a0 + np.arange(NX + 1) * h; self.vn = self.v0 + np.arange(NY + 1) * h; self.zn = np.arange(NZ + 1) * h
        self._build()

    def to_opt(self, g):
        return affinity.translate(g, HALF + UE, self.V0)

    def _build(self):
        VC, VLEN, W = self.VC, self.VLEN, self.W
        parts, cuts = [], []
        for vc in VC:
            for sx in (1, -1):
                p_, c_ = P.clip_end(sx, HALF, vc - self.V0); parts += p_; cuts += c_
        fing = self.to_opt(shapely.unary_union(parts)); cut = self.to_opt(shapely.unary_union(cuts))
        body = box(UE, 0, UE + L, VLEN)
        keep = shapely.unary_union([box(0, vc - BH2, W, vc + BH2) for vc in VC])
        rec_all = shapely.unary_union([box(UE, 0, UE + 0.5, VLEN), box(UE + L - 0.5, 0, UE + L, VLEN)])
        band = shapely.unary_union([box(a0, vc - BH2, a1, vc + BH2).difference(box(a0, vc - P.FING_IN, a1, vc + P.FING_IN))
                                    for vc in VC for a0, a1 in ((UE, UE + ROOT_BAND), (UE + L - ROOT_BAND, UE + L))])
        self.polys = dict(fing=fing, cut=cut, body=body, band=band, rec=rec_all.difference(keep), rec_all=rec_all)
        A, V = np.meshgrid(self.ac, self.vc, indexing="ij")
        cont = lambda g: shapely.contains_xy(g, A, V)
        in_body, in_fing, in_cut = cont(body), cont(fing), cont(cut)
        in_rl, in_ra, in_band = cont(rec_all.difference(keep)), cont(rec_all), cont(band)
        low = (self.zc < TS)[None, None, :]
        d_low = (in_body & ~in_cut & ~in_rl) | (in_fing & ~in_cut)
        d_high = in_body & ~in_cut & ~in_ra
        self.dom3 = (d_low[:, :, None] & low) | (d_high[:, :, None] & ~low)
        self.solid3 = ((in_fing | in_band) & ~in_cut)[:, :, None] & low & self.dom3
        self.design3 = self.dom3 & ~self.solid3
        self.body2 = in_body & ~in_cut                         # 两支撑之间的柱子（透空约束的范围）
        self.colfix = self.body2 & self.solid3.any(2)           # 有固定实心的柱子：s̄ = 1
        self.colvar = self.body2 & ~self.colfix & self.design3.any(2)
        for nm, M in (("design3", self.design3), ("solid3", self.solid3)):
            ok = (M == M[self.ia_m]).all() and (M == M[:, self.jv_m]).all()
            if not ok:
                print(f"警告：{nm} 不严格镜像对称（{int((M != M[self.ia_m]).sum())} / {int((M != M[:, self.jv_m]).sum())} 格）", flush=True)

    # ---------- 接触点 ----------
    def contacts(self):
        h, NX, NY, NZ, tol = self.h, self.NX, self.NY, self.NZ, 1e-6
        FXp = np.pad(self.solid3, 1)

        def side(axis, plus):
            sel = [[0, 1], [0, 1], [0, 1]]; sel[axis] = [1] if plus else [0]
            out = np.zeros((NX + 1, NY + 1, NZ + 1), bool)
            for di in sel[0]:
                for dj in sel[1]:
                    for dk in sel[2]:
                        out |= FXp[di:di + NX + 1, dj:dj + NY + 1, dk:dk + NZ + 1]
            return out
        S = {(ax, pl): side(ax, pl) for ax in range(3) for pl in (False, True)}
        A = self.an[:, None, None]; V = self.vn[None, :, None]; Z = self.zn[None, None, :]
        zin = (Z >= -tol) & (Z <= TS + tol)
        rows = []
        for ji, (s, k) in enumerate(self.joints):
            sx = -1 if s == "L" else 1; af = UE if s == "L" else UE + L
            dv = V - self.VC[k]; u = sx * (A - af)
            bod, sup = S[(0, s == "L")], S[(0, s != "L")]
            sh = bod & ~sup & (np.abs(u) < 0.75 * h) & (np.abs(dv) >= P.SLOT_A / 2 - tol) & (np.abs(dv) <= BH2 + tol) & zin
            rows.append((sh, 0, sx, ji, "台肩"))
            insup = (u >= -tol) & (u <= P.T + tol)
            for sy in (1, -1):
                face = S[(1, sy < 0)] & ~S[(1, sy > 0)] & (np.abs(dv - sy * YA) < 0.75 * h) & insup & zin
                rows.append((face, 1, sy, ji, "槽壁"))
            fz = (np.abs(dv) >= P.FING_IN - tol) & (np.abs(dv) <= YA + tol) & insup
            rows.append((S[(2, False)] & ~S[(2, True)] & (np.abs(Z - TS) < 0.5 * h) & fz, 2, 1, ji, "槽顶"))
            rows.append((S[(2, True)] & ~S[(2, False)] & (np.abs(Z) < 0.5 * h) & fz, 2, -1, ji, "槽底"))
            hk = S[(0, s == "R")] & ~S[(0, s == "L")] & (np.abs(u - P.NOTCH_U1) < 0.75 * h) & \
                (np.abs(dv) >= P.NECK_Y + P.NOTCH_R - tol) & (np.abs(dv) <= YA + tol) & zin
            rows.append((hk, 0, -sx, ji, "卡子"))
        node, ax, n, jj, kind = [], [], [], [], []
        for m, a_, n_, j_, k_ in rows:
            nd = np.flatnonzero(m)
            node.append(nd); ax.append(np.full(nd.size, a_)); n.append(np.full(nd.size, float(n_))); jj.append(np.full(nd.size, j_)); kind += [k_] * nd.size
        node, ax, n, jj = map(np.concatenate, (node, ax, n, jj)); kind = np.array(kind)
        dof = 3 * node + ax
        _, first = np.unique(dof, return_index=True); first = np.sort(first)
        self.c = dict(node=node[first], ax=ax[first], n=n[first], j=jj[first], kind=kind[first], dof=dof[first])
        i, r_ = np.divmod(self.c["node"], (NY + 1) * (NZ + 1)); j, k = np.divmod(r_, NZ + 1)
        self.c["pos"] = np.c_[self.an[i], self.vn[j], self.zn[k]]
        self.c["Rp"] = rigid_rows(self.c["pos"] - np.array([self.W / 2, self.VLEN / 2, TS / 2]), self.c["ax"]) * np.array([1, 1, 1, 0.01, 0.01, 0.01])
        return self.c

    def body_center(self, jidx):
        a = np.mean([UE - P.T / 2 if self.joints[q][0] == "L" else UE + L + P.T / 2 for q in jidx])
        v = np.mean([self.VC[self.joints[q][1]] for q in jidx])
        return np.array([a, v, TS / 2])

    # ---------- 现在的零件（插指版二维轮廓拉伸 16 mm，按打印结构） ----------
    def reference_fields(self, outline=None):
        if outline is None:
            d = json.load(open(self.ref_file)); outline = Polygon(d["exterior"], d["holes"])
        g = affinity.translate(self.to_opt(outline), -self.a0, -self.v0)
        cov, fw = N3.print_columns(g, self.NX, self.NY, self.h, WALL)
        z0 = np.arange(self.NZ) * self.h; z1 = z0 + self.h
        fs = np.clip((np.clip(np.minimum(z1, SKIN) - z0, 0, None) + np.clip(np.minimum(z1, TS) - np.maximum(z0, TS - SKIN), 0, None)) / self.h, 0, 1)
        E, m = N3.print_voxels(cov, fw, fs, INFILL ** 2, INFILL)
        low = (self.zc < TS)[None, None, :]
        E = np.where(self.solid3, 1.0, E * low) * self.dom3; m = np.where(self.solid3, 1.0, m * low) * self.dom3
        proj = float(np.where(self.body2, np.maximum(cov, self.colfix), 0).sum()) * self.h ** 2
        return E, m, proj, outline


def rigid_rows(r, ax):
    """刚体运动 (t, θ) 在点 r（相对转动中心）、方向 ax 上的位移系数：u = t + θ × r"""
    R = np.zeros((r.shape[0], 6)); R[np.arange(r.shape[0]), ax] = 1.0
    a0, a1, a2 = ax == 0, ax == 1, ax == 2
    R[a0, 4], R[a0, 5] = r[a0, 2], -r[a0, 1]
    R[a1, 5], R[a1, 3] = r[a1, 0], -r[a1, 2]
    R[a2, 3], R[a2, 4] = r[a2, 1], -r[a2, 0]
    return R


# ======================= 工况 =======================
class Case:
    def __init__(self, G, name, w, bodies):
        """bodies：[(接口下标列表, {自由度名: 广义力})]，没写的自由度固定"""
        self.name, self.w = name, w
        c = G.c
        self.nb = len(bodies)
        jb = {}
        for b, (jl, ld) in enumerate(bodies):
            for q in jl: jb[q] = b
        assert sorted(jb) == list(range(len(G.joints))), name
        self.cent = np.array([G.body_center(jl) for jl, _ in bodies])
        self.free = [(b, DOFN.index(k), float(v)) for b, (jl, ld) in enumerate(bodies) for k, v in ld.items()]
        self.body = np.array([jb[q] for q in c["j"]])
        self.R = rigid_rows(c["pos"] - self.cent[self.body], c["ax"])
        self.act = None
        self.group = "面内"


def make_cases(G):
    J = {q: i for i, q in enumerate(G.joints)}
    f, M = F_LOAD, M_LOAD
    one = lambda q, **ld: ([J[q]], ld)
    C = []
    if G.part == "panel":
        left, right = [J[("L", 0)], J[("L", 1)]], [J[("R", 0)], J[("R", 1)]]
        C += [Case(G, "侧向错动", 1.0, [one(("L", 0)), one(("R", 0)), one(("L", 1), ta=f / 2), one(("R", 1), ta=f / 2)]),
              Case(G, "撑开", 0.5, [one(("L", 0)), one(("L", 1)), one(("R", 0), ta=f / 2), one(("R", 1), ta=f / 2)]),
              Case(G, "前后错动", 0.5, [one(("L", 0)), one(("R", 0)), one(("L", 1), tv=f / 2), one(("R", 1), tv=-f / 2)])]
        ext = (("绕法线", "rz", M, "转"), ("法向错动", "tz", f, ""), ("绕横向轴", "ra", M, "扭"), ("绕斜面方向轴", "rv", M, "转"))
    else:
        left, right = [J[("L", 0)]], [J[("R", 0)]]
        C += [Case(G, "撑开", 0.5, [(left, {}), (right, dict(ta=f))]),
              Case(G, "压拢", 0.5, [(left, {}), (right, dict(ta=-f))]),
              Case(G, "前后错动", 1.0, [(left, {}), (right, dict(tv=f))])]
        ext = (("绕竖直轴", "rz", M, "转"), ("上下错动", "tz", f, ""), ("绕横向轴", "ra", M, "扭"), ("绕前后轴", "rv", M, "转"))
    for nm, key, val, verb in ext:
        for sg, sn in ((1, "正"), (-1, "反")):
            c = Case(G, f"右支撑{nm}{sn}{verb}", 0.5, [(left, {}), (right, {key: sg * val})])
            c.group = "面内" if key == "rz" else "面外"
            C.append(c)
    return C


# ======================= 有限元（给定约束集、约束处给定位移） =======================
class GPUFE:
    """矩阵无关多重网格（topopt3d.MGCG，去平移 kernel），每个约束集一套层级；残差用双精度、迭代修正"""

    def __init__(self, G, tol=1e-5, inner=1e-2):
        N3.use_centered_kernels()
        self.G = G; self.tol, self.inner = tol, inner
        self.D = T3.material_D(P.E_MAT, P.NU, P.KZ)
        self.nn = (G.NX + 1) * (G.NY + 1) * (G.NZ + 1); self.nel = G.NX * G.NY * G.NZ
        KE = T3.hex_KE(G.h, G.h, G.h, self.D)
        self.KE32 = cp.asarray(KE.ravel(), F32); self.KE64 = cp.asarray(KE.ravel())
        self.ones = cp.ones(3 * self.nn, F32)
        self.dims = (np.int32(G.NX), np.int32(G.NY), np.int32(G.NZ))
        self.hier = {}; self.g = None; self.its = 0

    def set_E(self, E):
        self.g = (F32(GMIN) + F32(1 - GMIN) * cp.asarray(E, F32)).ravel()
        for H in self.hier.values():
            H["dirty"] = True

    def k64(self, v, fm):
        out = cp.empty_like(v)
        N3._kfun("kmul64")(T3._blocks(self.nn), (T3._TB,), (v, out, self.g, self.KE64, fm, *self.dims))
        return out

    def _get(self, key, cdofs):
        if key not in self.hier:
            fm = cp.ones(3 * self.nn, F32); fm[cp.asarray(cdofs)] = 0
            self.hier[key] = dict(mg=T3.MGCG(self.G.NX, self.G.NY, self.G.NZ, self.G.h, self.G.h, self.D, fm), fm=fm, dirty=True, x={})
        H = self.hier[key]
        if H["dirty"]:
            H["mg"].update(self.g); H["dirty"] = False
        return H

    def drop(self, key=None):
        """释放约束集 key 的层级（None = 全部）；cuDSS 的显存不在 CuPy 池里，要先把池里空着的块还回去"""
        for k in (list(self.hier) if key is None else [key]):
            H = self.hier.pop(k, None)
            if H is not None:
                try:
                    H["mg"].coarse.solver.free()
                except Exception:
                    pass
        cp.get_default_memory_pool().free_all_blocks()

    def solve(self, key, cdofs, Gv):
        """约束自由度 cdofs 处给定位移 Gv（ncon × nrhs，其余自由度不受外力）→ [U（float32 全向量）]，约束反力 ncon × nrhs"""
        H = self._get(key, cdofs); fm, mg = H["fm"], H["mg"]
        cd = cp.asarray(cdofs); Us, Rs = [], []
        for c in range(Gv.shape[1]):
            g = cp.zeros(3 * self.nn); g[cd] = cp.asarray(Gv[:, c])
            b = -self.k64(g, fm); bn = float(cp.linalg.norm(b))
            x = H["x"].get(c); x = cp.zeros_like(b) if x is None else x.astype(cp.float64)
            if bn > 0:
                for _ in range(15):
                    r = b - self.k64(x, fm)
                    if float(cp.linalg.norm(r)) <= self.tol * bn:
                        break
                    d, k = mg.solve((r / bn).astype(F32), None, tol=self.inner)
                    x += d.astype(cp.float64) * bn; self.its += k
            H["x"][c] = x.astype(F32)
            U = x + g
            Rs.append(cp.asnumpy(self.k64(U, self.ones)[cd]))
            Us.append(U.astype(F32))
        return Us, np.stack(Rs, 1)

    def energy(self, Us, ws):
        ce = cp.zeros(self.nel, F32)
        for U, w in zip(Us, ws):
            T3._K()["kener"](T3._blocks(self.nel), (T3._TB,), (cp.asarray(U, F32), ce, self.KE32, np.float32(w), *self.dims))
        return ce.reshape(self.G.NX, self.G.NY, self.G.NZ)

    def vm(self, U):
        """体素中心 von Mises（按实心材料）"""
        DB = cp.asarray((self.D @ T3._hex_B(self.G.h, self.G.h, self.G.h, 0, 0, 0)).ravel(), F32)
        sig = cp.empty(6 * self.nel, F32)
        T3._K()["kstress"](T3._blocks(self.nel), (T3._TB,), (cp.asarray(U, F32), sig, cp.ones(self.nel, F32), DB, *self.dims))
        S = sig.reshape(-1, 6)
        v = cp.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2) + 3 * (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2))
        return cp.asnumpy(v).reshape(self.G.NX, self.G.NY, self.G.NZ)


class CPUFE:
    """本地自检：组装稀疏矩阵，pypardiso 直接解（只适合 h ≥ 2 mm 的小网格）"""

    def __init__(self, G, **kw):
        import scipy.sparse as sp
        import pypardiso
        self.sp, self.pypardiso = sp, pypardiso
        self.G = G; self.D = T3.material_D(P.E_MAT, P.NU, P.KZ)
        self.nn = (G.NX + 1) * (G.NY + 1) * (G.NZ + 1); self.nel = G.NX * G.NY * G.NZ
        self.KE = T3.hex_KE(G.h, G.h, G.h, self.D)
        self.edof = T3.grid_edof(G.NX, G.NY, G.NZ).astype(np.int64)
        self.iK = np.repeat(self.edof, 24, axis=1).ravel(); self.jK = np.tile(self.edof, (1, 24)).ravel()
        self.K = None; self.its = 0

    def set_E(self, E):
        g = GMIN + (1 - GMIN) * np.asarray(E, float).ravel()
        self.K = self.sp.csr_matrix(((self.KE.ravel()[None, :] * g[:, None]).ravel(), (self.iK, self.jK)), shape=(3 * self.nn, 3 * self.nn))

    def drop(self, key=None):
        pass

    def solve(self, key, cdofs, Gv):
        n = 3 * self.nn
        free = np.ones(n, bool); free[cdofs] = False; fi = np.flatnonzero(free)
        Kf = self.K[fi]; Kff = self.sp.triu(Kf[:, fi], format="csr"); Kfc = Kf[:, cdofs]
        solver = self.pypardiso.PyPardisoSolver(mtype=2)
        X = solver.solve(Kff, -(Kfc @ Gv)); solver.free_memory()
        X = X.reshape(fi.size, -1)
        Us, Rs = [], []
        for c in range(Gv.shape[1]):
            U = np.zeros(n); U[fi] = X[:, c]; U[cdofs] = Gv[:, c]
            Us.append(U); Rs.append((self.K @ U)[cdofs])
        return Us, np.stack(Rs, 1)

    def energy(self, Us, ws):
        ce = np.zeros(self.nel)
        for U, w in zip(Us, ws):
            ue = np.asarray(U, float)[self.edof]
            ce += w * np.einsum("ei,ij,ej->e", ue, self.KE, ue)
        return ce.reshape(self.G.NX, self.G.NY, self.G.NZ)

    def vm(self, U):
        ue = np.asarray(U, float)[self.edof]
        S = ue @ (self.D @ T3._hex_B(self.G.h, self.G.h, self.G.h, 0, 0, 0)).T
        v = np.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2) + 3 * (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2))
        return v.reshape(self.G.NX, self.G.NY, self.G.NZ)


def solve_case(fe, G, case, act):
    """主动集 act 固定（双向）：受力接触点跟着所在支撑做刚体运动，受载自由度的位移由广义力平衡解出"""
    idx = np.flatnonzero(act)
    cd = G.c["dof"][idx]; bi = case.body[idx]; R = case.R[idx]
    nf = len(case.free)
    Gv = np.zeros((idx.size, nf))
    for j, (b, d, F) in enumerate(case.free):
        Gv[:, j] = np.where(bi == b, R[:, d], 0.0)
    key = (case.name, hash(act.tobytes()))
    Us, Rs = fe.solve(key, cd, Gv)
    S = Gv.T @ Rs; S = 0.5 * (S + S.T)
    Fq = np.array([F for _, _, F in case.free])
    if np.min(np.diag(S)) <= 1e-9 * max(np.abs(S).max(), 1e-300):
        raise RuntimeError(f"{case.name}: 受载自由度没有接触点约束（主动集 {idx.size} 点），广义刚度 {np.diag(S)}")
    q = np.linalg.solve(S, Fq)
    U = Us[0] * q[0]
    for k in range(1, nf):
        U = U + Us[k] * q[k]
    qb = np.zeros((case.nb, 6))
    for j, (b, d, F) in enumerate(case.free):
        qb[b, d] = q[j]
    return dict(U=U, C=float(Fq @ q), q=q, qb=qb, fc=Rs @ q, idx=idx, key=key)


BILATERAL = False             # 自检用：接触全按双向（不做主动集迭代）


def contact_nl(fe, G, case, maxit=40, verbose=False):
    """非线性单向接触（主动集迭代，从全部接触开始）：受拉的放开，穿入的加上"""
    c = G.c; nc = c["dof"].size
    act = np.ones(nc, bool); hist = []; seen = set()
    for it in range(1 if BILATERAL else maxit):
        sol = solve_case(fe, G, case, act)
        Ui = np.asarray(cp.asnumpy(sol["U"][cp.asarray(c["dof"])]), float)
        us = (case.R * sol["qb"][case.body]).sum(1)
        gap = c["n"] * (Ui - us)
        f = np.zeros(nc); f[sol["idx"]] = sol["fc"]
        ten = c["n"] * f
        # 容差要比求解精度（1e-5）松：拉力 / 穿入量在数值零附近的几个点会来回翻
        ft = 1e-4 * max(np.abs(f).max(), 1e-30); gt = 1e-5 * max(np.abs(Ui).max(), 1e-30)
        new = (act & (ten <= ft)) | (~act & (gap > gt))
        new = ensure_rank(G, new, ten)
        hist.append(int(act.sum())); seen.add(hash(act.tobytes()))
        if (new == act).all() or hash(new.tobytes()) in seen:
            break
        fe.drop(sol["key"])
        act = new
    sol.update(act=act, it=it + 1, f=f, gap=gap, hist=hist)
    if verbose:
        print(f"    {case.name}: 主动集迭代 {it + 1} 次，受力接触点 {int(act.sum())}/{nc}，柔度 {sol['C']:.5g}", flush=True)
    return sol


def ensure_rank(G, act, ten, nadd=6):
    """放开受拉接触后，零件的 6 个刚体运动要仍被约束住（比如撑开时插指上下面因泊松收缩都受拉，全放开后 z 向就没约束了，
    矩阵奇异）。缺哪个方向，就把能约束它、拉力最小的几个接触点留下（这些方向本来就几乎不受力）"""
    Rp = G.c["Rp"]; act = act.copy()
    for _ in range(12):
        A = Rp[act]; w, V = np.linalg.eigh(A.T @ A)
        bad = w < 1e-9 * max(w.max(), 1e-300)
        if not bad.any():
            return act
        rel = np.flatnonzero(~act)
        sc = np.abs(Rp[rel] @ V[:, bad]).max(1)
        cand = rel[sc > 0.3 * sc.max()]
        act[cand[np.argsort(ten[cand])[:nadd]]] = True
    return act


def contact_summary(G, case, sol):
    """各接口、各种接触受力的合力（N）"""
    c = G.c; out = {}
    for ji, (s, k) in enumerate(G.joints):
        for kd in ("台肩", "槽壁", "槽顶", "槽底", "卡子"):
            m = (c["j"] == ji) & (c["kind"] == kd) & sol["act"]
            if m.any():
                out[f"{s}{k}{kd}"] = round(float(np.abs(sol["f"][m]).sum()), 3)
    return out


# ======================= 设计变量 → 打印密度 =======================
class Design:
    """x（三维）→ 过滤 → 投影 x̄；s（每根柱子，面板透空用）→ 二维过滤 → 投影 s̄；x̄·s̄ → AM 滤波 → ξ（打印设计）。
    最小厚度时刚度按腐蚀设计 ξ_e = x̄_e·s̄·(ξ / x̄s̄)（同 topopt3d_noinfill.Pipeline）"""

    def __init__(self, G, rmin, tmin, eta_e, cols, rs):
        self.G = G
        self.pipe = N3.Pipeline((G.NX, G.NY, G.NZ), G.h, rmin, ~G.design3, G.solid3.astype(float), pad_void=tmin > 0,
                                eta_e=eta_e if tmin > 0 else None, dz=G.h)
        self.eta_e = eta_e if tmin > 0 else None
        self.d3 = cp.asarray(G.design3)
        self.cols = cols
        if cols:
            self.cvar, self.cfix = cp.asarray(G.colvar), cp.asarray(G.colfix)
            R = int(np.ceil(rs / G.h)) - 1; o = np.arange(-R, R + 1)
            di, dj = np.meshgrid(o, o, indexing="ij")
            self.K2 = cp.asarray(np.maximum(0.0, rs - G.h * np.sqrt(di ** 2 + dj ** 2)))
            self.K2s = cp.maximum(cnd.correlate(cp.asarray(G.body2, float), self.K2, mode="constant", cval=0.0), 1e-12)
            self.body2 = cp.asarray(G.body2)

    def sbar(self, s, beta):
        sf = cp.where(self.cvar, s, cp.where(self.cfix, 1.0, 0.0))
        st = cnd.correlate(sf, self.K2, mode="constant", cval=0.0) / self.K2s
        sb, dsb = T2.project(st, float(beta), 0.5)
        return cp.where(self.cvar, sb, 1.0), cp.where(self.cvar, dsb, 0.0)

    def forward(self, x, s, beta):
        P_ = self.pipe
        xf = cp.where(P_.fixed, P_.fval, x); xt = P_.filt(xf)
        xb, dxb = T2.project(xt, float(beta), 0.5)
        xb = cp.where(P_.fixed, P_.fval, xb); dxb = cp.where(P_.fixed, 0.0, dxb)
        c = dict(dxb=dxb, xb=xb)
        if self.cols:
            sb, dsb = self.sbar(s, beta)
            xbs = cp.where(P_.fixed, P_.fval, xb * sb[:, :, None])
            c.update(sb=sb, dsb=dsb)
        else:
            xbs = xb
        c["xbs"] = xbs
        xi, ca = P_.am_forward(xbs); c.update(ca)
        if self.eta_e is not None:
            xe, dxe = T2.project(xt, float(beta), self.eta_e)
            xe = cp.where(P_.fixed, P_.fval, xe); dxe = cp.where(P_.fixed, 0.0, dxe)
            xes = cp.where(P_.fixed, P_.fval, xe * c["sb"][:, :, None]) if self.cols else xe
            den = cp.maximum(cp.maximum(xbs, xi), 1e-12); s_ = xi / den
            c.update(xe=xe, dxe=dxe, sr=s_, br=xbs > xi, q=xes / den)
            c["xie"] = xes * s_
        return xi, c

    def col_backward(self, gsb, c):
        gs_t = gsb * c["dsb"]
        return cnd.correlate(gs_t / self.K2s, self.K2, mode="constant", cval=0.0) * self.cvar

    def backward(self, gxi, c, gxe=None, gxbs=None):
        """dF/dξ（打印设计，可为 None）+ dF/dξ_e（腐蚀设计）+ dF/d(x̄·s̄)（悬空滤波之前，用料用）→ (dF/dx, dF/ds)"""
        P_ = self.pipe; fr = P_.free
        gdir = 0.0
        if gxe is not None:
            gq = cp.where(c["br"], gxe * c["q"], 0.0)
            gxi = (0.0 if gxi is None else gxi) + gq
            gdir = -gq * c["sr"]
        g_xbs = (P_.am_backward(gxi, c) if gxi is not None else 0.0) + gdir + (0.0 if gxbs is None else gxbs)
        g_xes = gxe * c["sr"] if gxe is not None else None
        gs = None
        if self.cols:
            sb3 = c["sb"][:, :, None]
            g_xb = cp.where(fr, g_xbs * sb3, 0.0)
            gsb = cp.where(fr, g_xbs * c["xb"], 0.0).sum(2)
            if g_xes is not None:
                g_xe = cp.where(fr, g_xes * sb3, 0.0)
                gsb = gsb + cp.where(fr, g_xes * c["xe"], 0.0).sum(2)
            else:
                g_xe = None
            gs = self.col_backward(gsb, c)
        else:
            g_xb, g_xe = g_xbs, g_xes
        gt = g_xb * c["dxb"]
        if g_xe is not None:
            gt = gt + g_xe * c["dxe"]
        gx = cnd.correlate(gt / P_.Ks, P_.K, mode="constant", cval=0.0) * fr
        return gx, gs

    def area(self, c):
        return float((c["sb"] * self.body2).sum()) * self.G.h ** 2


def sym_ops(G):
    ia, jv = cp.asarray(G.ia_m), cp.asarray(G.jv_m)
    s3 = lambda A: 0.25 * (A + A[ia] + A[:, jv] + A[ia][:, jv])
    return s3, s3


# ======================= 优化 =======================
def optimize(a, G, fe, cases, ref_C, target, A_max, out, x0=None, s0=None):
    h, vox = G.h, G.vox
    cols = A_max is not None
    rmin = a.rmin
    if a.tmin > 0:
        rmin, tq, _ = N3.rmin_for_thickness(a.tmin, h, a.eta_e, h)
        print(f"最小厚度 {a.tmin} mm（网格上 {tq:.2f} mm）→ 过滤半径 {rmin:.3f} mm，腐蚀阈值 η_e={a.eta_e}", flush=True)
    Dz = Design(G, rmin, a.tmin, a.eta_e, cols, a.rs or rmin)
    d3 = Dz.d3; nd = int(G.design3.sum()); s3, s2 = sym_ops(G)
    solid = cp.asarray(G.solid3, float)
    fixm = float(G.solid3.sum()) * vox
    vf = (target - fixm) / (nd * vox)
    if cols:
        cvar = Dz.cvar; ncv = int(G.colvar.sum())
        A_var = (A_max - float(G.colfix.sum()) * h ** 2)
        sf0 = float(np.clip(A_var / (ncv * h ** 2), 0.15, 1.0))
        print(f"透空约束：投影面积 ≤ {A_max:.0f} mm²（两支撑之间柱子 {float(G.body2.sum()) * h ** 2:.0f} mm²，固定实心柱 {float(G.colfix.sum()) * h ** 2:.0f}）；s 初值 {sf0:.3f}", flush=True)
    else:
        sf0 = 1.0
    # 均匀初值取 a.x0（默认 0.5，先超重、MMA 往下减）：设计区平均密度很低（后横梁约 0.07）时，均匀 vf 经腐蚀投影后几乎没有刚度
    xi0 = a.x0 if a.x0 > 0 else min(1.0, vf / sf0)
    x = cp.where(d3, xi0, 0.0) if x0 is None else cp.where(d3, cp.asarray(x0), 0.0)
    s = (cp.where(cvar, a.x0 if a.x0 > 0 else sf0, 0.0) if s0 is None else cp.where(cvar, cp.asarray(s0), 0.0)) if cols else None
    print(f"用料目标 {target / 1000:.2f} cm³（{target / 1000 * RHO:.1f} g），固定实心 {fixm / 1000:.2f} cm³，设计区平均密度 {vf:.4f}", flush=True)
    w = np.array([c.w for c in cases]); wf = w / w.sum(); qm = a.pmean
    mma = N3.MMA(2 if cols else 1, move=a.move)
    beta, lb, hist, t0 = a.beta0, 0, [], time.time()
    names = [c.name for c in cases]
    for it in range(1, a.maxit + 1):
        ti = time.time(); its0 = fe.its
        xi, cache = Dz.forward(x, s, beta)
        xa = cache.get("xie", xi)
        fe.set_E(cp.where(d3, xa ** a.penal, solid))
        sols = [solve_case(fe, G, c, c.act) for c in cases]
        comps = np.array([q["C"] for q in sols])
        # 目标：各工况柔度比 r 的 q 次幂平均（q = 1 是加权平均；q 大时最差的工况占的分量重）
        rr = comps / ref_C; mean1 = float(wf @ rr)
        obj = float((wf @ rr ** qm) ** (1.0 / qm))
        we = obj ** (1 - qm) * wf * rr ** (qm - 1) / ref_C
        ce = cp.asarray(fe.energy([q["U"] for q in sols], we), cp.float64)
        del sols
        gE = -ce * (1 - GMIN) * a.penal * xa ** (a.penal - 1) * d3
        gx, gs = Dz.backward(None, cache, gE) if Dz.eta_e is not None else Dz.backward(gE, cache)
        # 用料按悬空滤波之前的密度 x̄·s̄ 算：支撑不住的料照算用料、不给刚度，优化自己去掉它。按打印密度 ξ 算的话，
        # 贴床那层每降一点，它托着的整列都不算料了 → 用料约束先砍贴床层，砍完上面全空，卡死（低体积分数时）
        gxm, gsm = Dz.backward(None, cache, gxbs=d3 * vox)
        mass = (float(cache["xbs"][d3].sum()) + float(G.solid3.sum())) * vox
        mass_p = (float(xi[d3].sum()) + float(G.solid3.sum())) * vox
        fv = [mass / target - 1.0]
        gx = s3(gx); gxm = s3(gxm)
        if cols:
            gs, gsm = s2(gs), s2(gsm)
            Ar = Dz.area(cache); gsA = s2(Dz.col_backward(Dz.body2 * h ** 2, cache))
            fv.append(Ar / A_max - 1.0)
            X = cp.concatenate([x[d3], s[cvar]]); df0 = cp.concatenate([gx[d3], gs[cvar]])
            dfv = [cp.concatenate([gxm[d3], gsm[cvar]]) / target, cp.concatenate([cp.zeros(nd), gsA[cvar]]) / A_max]
        else:
            Ar = None
            X = x[d3]; df0 = gx[d3]; dfv = [gxm[d3] / target]
        Xn = mma.update(X, df0, fv, dfv)
        change = float(cp.abs(Xn - X).max())
        x[d3] = Xn[:nd]; x = cp.where(d3, s3(x), 0.0)
        if cols:
            s[cvar] = Xn[nd:]; s = cp.where(cvar, s2(s), 0.0)
        grey = float((4 * xi * (1 - xi))[d3].mean())
        hist.append(dict(it=it, c=obj, mean=mean1, mass=mass / target, mass_print=mass_p / target, area=None if Ar is None else Ar / A_max, change=change, beta=beta, grey=grey,
                         cg=fe.its - its0, comp=(comps / ref_C).tolist()))
        print(f"it {it:3d} 目标={obj:.4f}（加权平均 {mean1:.4f}）用料/目标={mass / target:.3f}（打印 {mass_p / target:.3f}）" + ("" if Ar is None else f" 投影/现在={Ar / A_max:.3f}") +
              f" ch={change:.3f} β={beta:g} 灰度={grey:.3f} CG={fe.its - its0} {time.time() - ti:.1f}s t={time.time() - t0:.0f}s", flush=True)
        if it % 10 == 1:
            print("      " + "  ".join(f"{n}={r:.3f}" for n, r in zip(names, comps / ref_C)), flush=True)
        lb += 1
        feas = all(f < 0.01 for f in fv)
        if beta < a.beta_max and (lb >= a.beta_every or (change < 0.01 and lb >= 15)):
            beta *= 2; lb = 0; mma.reset()
        elif beta >= a.beta_max and lb >= 30 and feas and (change < 0.01 or abs(hist[-11]["c"] - obj) / obj < 1e-3):
            break
        if (a.save_every and it % a.save_every == 0) or it % 20 == 0:
            save_result(out, G, x, s, xi, cases, ref_C, target, A_max, hist, a, beta)
            fig_layers(G, cp.asnumpy(xi), f"{out}/layers_it{it}.png", f"it={it} β={beta:g} 加权柔度/现在={obj:.3f}")
        if a.time_budget and time.time() - t0 > a.time_budget:
            print(f"到时间上限，第 {it} 轮停下", flush=True)
            break
    xi, cache = Dz.forward(x, s, beta)
    save_result(out, G, x, s, xi, cases, ref_C, target, A_max, hist, a, beta)
    return cp.asnumpy(xi), hist


def save_result(out, G, x, s, xi, cases, ref_C, target, A_max, hist, a, beta):
    np.savez_compressed(f"{out}/result.npz", x=cp.asnumpy(x).astype(np.float32), s=np.zeros(1) if s is None else cp.asnumpy(s).astype(np.float32),
                        xi=cp.asnumpy(xi).astype(np.float32), h=G.h, NX=G.NX, NY=G.NY, NZ=G.NZ, a0=G.a0, v0=G.v0, beta=beta,
                        names=np.array([c.name for c in cases]), ref_C=ref_C, target=target, A_max=-1.0 if A_max is None else A_max,
                        act=np.stack([c.act for c in cases]))
    json.dump(dict(args=vars(a), names=[c.name for c in cases], ref_C=list(map(float, ref_C)), hist=hist), open(f"{out}/history.json", "w"),
              ensure_ascii=False, indent=1)


# ======================= 二值化、复核、导出 =======================
def binarize(G, xi):
    from scipy.ndimage import label
    B = np.where(G.design3, xi > 0.5, G.solid3)
    B1 = N3.am_binary(B, *N3.am_stencil(G.h, G.h, 45.0))
    lab, n = label(B1); sz = np.bincount(lab.ravel()); sz[0] = 0
    B2 = lab == int(sz.argmax())
    return B2, dict(悬空去掉体素=int((B & ~B1).sum()), 碎块去掉体素=int((B1 & ~B2).sum()), 碎块数=int(n - 1))


def evaluate(fe, G, cases, E, solid, tag, verbose=True):
    """非线性接触复核：各工况柔度、受载支撑位移、应力（只统计实心体素：杆身 / 伸进支撑那段）"""
    fe.set_E(E)
    A = G.ac[:, None, None]
    inb = ((A >= UE) & (A <= UE + L)) & solid; outb = ~((A >= UE) & (A <= UE + L)) & solid
    res = []
    for c in cases:
        sol = contact_nl(fe, G, c)
        vm = fe.vm(sol["U"])
        pct = lambda m: float(np.percentile(vm[m], 99.9)) if m.any() else 0.0
        b, d, _ = c.free[0]
        res.append(dict(工况=c.name, 组=c.group, 柔度=sol["C"], 受载位移=float(sol["q"][0]), 主动集迭代=sol["it"],
                        杆身最大应力_MPa=float(vm[inb].max()), 杆身_99_9pct_MPa=pct(inb),
                        插指段最大应力_MPa=float(vm[outb].max()), 插指段_99_9pct_MPa=pct(outb), 接触合力_N=contact_summary(G, c, sol), act=sol["act"]))
        fe.drop(sol["key"])
        if verbose:
            r = res[-1]
            print(f"  [{tag}] {c.name}: 柔度 {r['柔度']:.5g}（迭代 {r['主动集迭代']}），杆身最大 {r['杆身最大应力_MPa']:.2f} / 99.9% {r['杆身_99_9pct_MPa']:.2f} MPa，"
                  f"插指段最大 {r['插指段最大应力_MPa']:.2f} / 99.9% {r['插指段_99_9pct_MPa']:.2f}", flush=True)
    return res


def to_print(G, v):
    """优化坐标 (a, v, z) → 打印坐标 (X, Y, Z)"""
    return v - np.array([HALF + UE, G.V0, 0.0])


def export_mesh(G, B, fn, smooth=0.6):
    import manifold3d as m3d
    import trimesh
    from scipy.ndimage import gaussian_filter
    from skimage import measure

    def polys(g):
        return [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]

    def extrude(geom, z0, z1):
        parts = []
        for p in polys(geom):
            p = shapely.geometry.polygon.orient(p, 1.0)
            rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(i.coords)[:-1] for i in p.interiors]
            parts.append(m3d.Manifold.extrude(m3d.CrossSection(rings, m3d.FillRule.EvenOdd), z1 - z0).translate((0, 0, z0)))
        return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    h = G.h; pz = 3
    F = B.astype(np.float32)
    F = np.concatenate([F[:, :, pz - 1::-1], F, F[:, :, :-pz - 1:-1]], 2)
    if smooth > 0:
        F = gaussian_filter(F, smooth)
    F = np.pad(F, 2)
    v, f, _, _ = measure.marching_cubes(F, 0.5, spacing=(h, h, h), allow_degenerate=False)
    v = v + np.array([G.a0 - 1.5 * h, G.v0 - 1.5 * h, -(pz + 1.5) * h])
    mesh = m3d.Manifold(m3d.Mesh(vert_properties=v.astype(np.float32), tri_verts=f[:, ::-1].astype(np.uint32)))
    if mesh.volume() < 0:
        mesh = m3d.Manifold(m3d.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32)))
    pg = G.polys
    low = pg["body"].difference(pg["cut"]).difference(pg["rec"]); high = pg["body"].difference(pg["cut"]).difference(pg["rec_all"])
    allowed = extrude(low, 0, TS) + (extrude(high, TS, G.H) if G.H > TS else extrude(low, 0, TS))
    fixed = pg["fing"].union(pg["band"]).difference(pg["cut"])
    mesh = (mesh ^ allowed) + extrude(fixed, 0, TS)
    mesh = mesh.translate((-(HALF + UE), -G.V0, 0.0))
    mm = mesh.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
    tm.export(fn)
    return dict(volume_cm3=round(mesh.volume() / 1000, 2), mass_g=round(mesh.volume() / 1000 * RHO, 1), watertight=bool(tm.is_watertight),
                bbox_mm=[round(float(x), 1) for x in (tm.bounds[1] - tm.bounds[0])], genus=int(mesh.genus()))


# ======================= 图 =======================
def fig_layers(G, xi, fn, title):
    ks = np.unique(np.linspace(0, G.NZ - 1, 8).round().astype(int))
    fig, axs = plt.subplots(2, 4, figsize=(22, 4.4 * max(1.0, G.NY / G.NX * 2.2)))
    ext = [G.a0, G.a0 + G.NX * G.h, G.v0, G.v0 + G.NY * G.h]
    for ax, k in zip(axs.ravel(), ks):
        ax.imshow(1 - xi[:, :, k].T, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"第 {k + 1} 层  z = {k * G.h:.1f}–{(k + 1) * G.h:.1f} mm", fontsize=9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.savefig(fn, dpi=80, bbox_inches="tight"); plt.close(fig)


def fig_check(G, cases, ref, fn):
    """几何 + 接触点（自检用）"""
    c = G.c
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    ext = [G.a0, G.a0 + G.NX * G.h, G.v0, G.v0 + G.NY * G.h]
    code = np.where(G.solid3[:, :, 0], 2, np.where(G.design3[:, :, 0], 1, 0))
    for ax, (lo, hi) in zip(axs[:2], ((0, UE + 40), (UE + L - 40, G.W))):
        ax.imshow(code.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
        for kd, col in (("台肩", "r"), ("槽壁", "b"), ("槽顶", "g"), ("槽底", "c"), ("卡子", "m")):
            m = c["kind"] == kd
            ax.scatter(c["pos"][m, 0], c["pos"][m, 1], s=3, c=col, label=kd)
        ax.set_xlim(lo, hi); ax.set_aspect("equal"); ax.legend(fontsize=7, markerscale=3)
    zs = G.design3.sum(2) + G.solid3.sum(2)
    axs[2].imshow(zs.T, origin="lower", extent=ext, cmap="viridis", interpolation="nearest"); axs[2].set_title("可用格数（沿 z）")
    fig.tight_layout(); fig.savefig(fn, dpi=90); plt.close(fig)


# ======================= 主程序 =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("panel", "beam"), required=True)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--H", type=float, default=None, help="杆身最大厚度 mm（面板默认 16，后横梁默认 32）")
    ap.add_argument("--width", type=float, default=None, help="后横梁设计范围前后总宽（默认 80）")
    ap.add_argument("--mass_g", type=float, default=None, help="用料目标 g（面板默认 83，后横梁 53；按 1.24 g/cm³ 实心）")
    ap.add_argument("--no_open", action="store_true", help="面板不加透空约束")
    ap.add_argument("--open_scale", type=float, default=1.0, help="面板投影面积上限 = 现在面板 × 此值")
    ap.add_argument("--rmin", type=float, default=1.2)
    ap.add_argument("--x0", type=float, default=0.5, help="均匀初值（0 = 按用料目标）")
    ap.add_argument("--pmean", type=float, default=1.0, help="目标 = 各工况柔度比的 q 次幂平均（1 = 加权平均）")
    ap.add_argument("--init_tube", default=None, help="后横梁：wo,zw[,t] 尖顶闭口管当初值（见 tube_init；选定版用 17,10）")
    ap.add_argument("--init_npy", default=None, help="从一个布尔体素文件（.npy，同一网格）起步：True 处 x = 1，其余 x = --init_ref（默认 0.1）")
    ap.add_argument("--init_hollow", default=None, help="t,zw：现在零件的轮廓按 45° 掏空当初值（墙 / 底厚 t，空腔从 zw 往上每层收一格成尖顶；宽杆收不拢就开口）")
    ap.add_argument("--init_box", action="store_true", help="从高实心杆起步：杆身 |dv| ≤ BEAM_H/2 的全高 x = 1（配 --init_ref 给其余的值）")
    ap.add_argument("--init_ref", type=float, default=None, help="从现在的零件起步：轮廓内（16 mm 以内）x = 1，其余 x = 这个值")
    ap.add_argument("--rs", type=float, default=None, help="透空变量的二维过滤半径（默认同三维）")
    ap.add_argument("--tmin", type=float, default=1.9)
    ap.add_argument("--eta_e", type=float, default=0.75)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--maxit", type=int, default=250)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta_max", type=float, default=16.0)
    ap.add_argument("--beta_every", type=int, default=40)
    ap.add_argument("--move", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--time_budget", type=float, default=0)
    ap.add_argument("--init", default=None, help="从另一次结果（result.npz，可以是粗网格）插值出初始设计")
    ap.add_argument("--cpu", action="store_true", help="本地 CPU 自检（numpy + pypardiso，小网格）")
    ap.add_argument("--check", action="store_true", help="只建几何、接触，复核现在的零件")
    ap.add_argument("--test_L", type=float, default=None, help="自检用：把两支撑内侧面间距改小（mm），网格小一点")
    ap.add_argument("--bilateral", action="store_true", help="自检用：接触全按双向")
    ap.add_argument("--post_only", action="store_true", help="跳过优化，读 --out/result.npz 做二值化、复核、导出")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global HALF, L, BILATERAL
    if a.test_L:
        L = a.test_L; HALF = L / 2
    BILATERAL = a.bilateral
    out = a.out or f"{P.OUT_DIR}/link3d_{a.part}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    G = Geo(a.part, a.h, a.H, a.width)
    c = G.contacts(); cases = make_cases(G)
    print(f"{a.part}: 网格 {G.NX}×{G.NY}×{G.NZ}（h={G.h} mm，厚 {G.H} mm），{G.NX * G.NY * G.NZ / 1e6:.2f} M 体素；设计 {int(G.design3.sum()) / 1e6:.2f} M，"
          f"固定实心 {G.solid3.sum() * G.vox / 1000:.2f} cm³", flush=True)
    kinds = {}
    for kd, j in zip(c["kind"], c["j"]):
        kinds[f"{G.joints[j][0]}{G.joints[j][1]}{kd}"] = kinds.get(f"{G.joints[j][0]}{G.joints[j][1]}{kd}", 0) + 1
    print("接触点:", kinds, flush=True)
    print("工况:", [(q.name, q.w, q.group) for q in cases], flush=True)
    fe = CPUFE(G) if CPU else GPUFE(G, tol=a.tol)
    E_ref, m_ref, A_ref, _ = G.reference_fields()
    mref = float(m_ref.sum()) * G.vox
    print(f"现在的零件：打印用料 {mref / 1000:.2f} cm³（{mref / 1000 * RHO:.1f} g）" + (f"，两支撑之间投影 {A_ref:.0f} mm²" if a.part == "panel" else ""), flush=True)
    ref = evaluate(fe, G, cases, E_ref, E_ref >= 0.5, "现在")
    for q, r in zip(cases, ref):
        q.act = r.pop("act")
    fe.drop()
    ref_C = np.array([r["柔度"] for r in ref])
    rep = dict(part=a.part, grid=[G.NX, G.NY, G.NZ], h=G.h, H=G.H, ref_mass_g=round(mref / 1000 * RHO, 1), ref_proj_mm2=A_ref, ref=ref,
               contacts=kinds, t_ref_s=round(time.time() - t0, 1))
    if a.check:
        fig_check(G, cases, ref, f"{out}/check.png")
        json.dump(rep, open(f"{out}/check.json", "w"), ensure_ascii=False, indent=1)
        print(f"自检完成 {time.time() - t0:.0f}s", flush=True)
        return
    target = (a.mass_g or G.mass_g_default) / RHO * 1000
    A_max = None if (a.part != "panel" or a.no_open) else A_ref * a.open_scale
    if a.post_only:     # 有 result.npz 就重新二值化；只有 binary.npz（仓库里只提交了它）就直接复核、导出
        xi = np.load(f"{out}/result.npz")["xi"] if os.path.exists(f"{out}/result.npz") else np.where(np.load(f"{out}/binary.npz")["B"], 1.0, 0.0)
    else:
        x0 = s0 = None
        if a.init:
            x0, s0 = interp_init(G, a.init)
        elif a.init_tube:
            assert a.part == "beam", "--init_tube 只给后横梁用"
            v = [float(s) for s in a.init_tube.split(",")]
            x0 = tube_init(G, v[0], v[1], v[2] if len(v) > 2 else 2.0, a.init_ref if a.init_ref is not None else 0.05)
            print(f"初始设计：尖顶闭口管 wo={v[0]} zw={v[1]}，{float((x0 > 0.5).sum()) * G.vox / 1000:.1f} cm³（设计区）", flush=True)
        elif a.init_npy:
            x0 = np.where(np.load(a.init_npy) & G.design3, 1.0, a.init_ref if a.init_ref is not None else 0.1)
            print(f"初始设计取自 {a.init_npy}", flush=True)
        elif a.init_hollow:
            t_, zw_ = map(float, a.init_hollow.split(","))
            x0 = hollow_init(G, E_ref, t_, zw_, a.init_ref if a.init_ref is not None else 0.05)
            s0 = np.where(x0.max(2) > 0.5, 1.0, 0.1) if a.part == "panel" else None
            print(f"初始设计：现在的零件按 45° 掏空（t={t_}, zw={zw_}），{float((x0 > 0.5).sum()) * G.vox / 1000:.1f} cm³", flush=True)
        elif a.init_box:
            dv = np.min(np.abs(G.vc[None, :] - np.array(G.VC)[:, None]), 0)
            x0 = np.where((dv <= BH2)[None, :, None] & G.design3, 1.0, a.init_ref or 0.1)
            s0 = None
            print("初始设计：杆身 |dv| ≤ 20 的全高实心", flush=True)
        elif a.init_ref is not None:
            x0 = np.where(E_ref > 0, 1.0, a.init_ref)
            s0 = np.where(E_ref.max(2) > 0, 1.0, max(a.init_ref, 0.1)) if a.part == "panel" else None
            print(f"初始设计：现在的零件（轮廓内 x = 1，其余 {a.init_ref}）", flush=True)
        xi, hist = optimize(a, G, fe, cases, ref_C, target, A_max, out, x0, s0)
    fe.drop()
    B, binfo = binarize(G, xi)
    mass = float(B.sum()) * G.vox
    proj = float((B.any(2) & G.body2).sum()) * G.h ** 2
    print(f"二值化：{binfo}；用料 {mass / 1000:.2f} cm³（{mass / 1000 * RHO:.1f} g）" + (f"，投影 {proj:.0f} mm²（现在 {A_ref:.0f}）" if a.part == "panel" else ""), flush=True)
    new = evaluate(fe, G, cases, B.astype(float), B, "三维")
    for r in new:
        r.pop("act")
    w = np.array([q.w for q in cases])
    ratio = np.array([n["柔度"] for n in new]) / ref_C
    print("非线性接触复核（柔度比 三维 / 现在）：" + "  ".join(f"{q.name}={r:.3f}" for q, r in zip(cases, ratio)), flush=True)
    print(f"加权 {float(w @ ratio / w.sum()):.4f}；面内 {float(w[[q.group == '面内' for q in cases]] @ ratio[[q.group == '面内' for q in cases]] / w[[q.group == '面内' for q in cases]].sum()):.4f}，"
          f"面外 {float(w[[q.group == '面外' for q in cases]] @ ratio[[q.group == '面外' for q in cases]] / w[[q.group == '面外' for q in cases]].sum()):.4f}", flush=True)
    np.savez_compressed(f"{out}/binary.npz", B=B, h=G.h, a0=G.a0, v0=G.v0)
    rep.update(new=new, ratio=ratio.tolist(), weighted=float(w @ ratio / w.sum()), mass_g=round(mass / 1000 * RHO, 1), proj_mm2=proj, binarize=binfo)
    try:
        rep["stl"] = export_mesh(G, B, f"{out}/{a.part}_3d.stl")
        print("STL:", rep["stl"], flush=True)
    except Exception as e:
        print("导出 STL 失败:", e, flush=True)
    fig_layers(G, B.astype(float), f"{out}/layers_binary.png", f"{a.part} 二值化，加权柔度比 {rep['weighted']:.3f}")
    json.dump(rep, open(f"{out}/report.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"完成 {time.time() - t0:.0f}s", flush=True)


def hollow_init(G, E_ref, t, zw, other):
    """现在零件（拉伸体）→ 满足 45° 的空心件：空腔 = 轮廓往里收 t、z ∈ [t, zw] 直上直下，再往上每层四邻腐蚀一格（尖顶）；
    顶面以下 t·√2 以内不留空腔（宽杆收不拢的地方空腔留到顶，成开口槽）"""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    S = E_ref > 0
    M2 = S.any(2) & G.body2
    d = distance_transform_edt(np.pad(M2, 1))[1:-1, 1:-1] * G.h
    C = d > t
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    cav = np.zeros_like(S)
    top = min(G.H, TS) if G.part == "panel" else TS
    for k in range(G.NZ):
        z = G.zc[k]
        if z < t or z > top:
            continue
        if z > zw:
            C = binary_erosion(C, cross)
        cav[:, :, k] = C
    # 顶面附近：收不拢的地方空腔一直开到顶（开口），收得拢的地方顶上留 t·√2
    open_ = cav[:, :, int(top / G.h) - 1] if int(top / G.h) - 1 < G.NZ else np.zeros_like(M2)
    for k in range(G.NZ):
        if G.zc[k] > top - t * np.sqrt(2):
            cav[:, :, k] &= open_
    B = S & ~cav & (G.design3 | G.solid3)
    return np.where(G.design3, np.where(B, 1.0, other), 0.0)


def tube_init(G, wo, zw, t=2.0, other=0.05):
    """后横梁的初值：一根满足 45° 的尖顶闭口管（外半宽 wo、直墙高 zw、壁厚 t，屋脊 45°），沿杆身全长。
    从均匀或实心起步，优化会长成开口 U 形槽（扭转比现在的方管软 2–3 倍）；从闭口管起步才找得到闭口的解"""
    dv = np.abs(G.vc - G.VC[0])[None, :, None]; z = G.zc[None, None, :]
    wall = (dv <= wo) & (dv >= wo - t) & (z <= zw + (wo - dv))
    bot = (dv <= wo) & (z < t)
    roof = (dv <= wo) & (z <= zw + (wo - dv)) & (z >= zw + (wo - dv) - t * np.sqrt(2))
    B = G.solid3 | ((wall | bot | roof) & G.design3)
    B = N3.am_binary(B, *N3.am_stencil(G.h, G.h, 45.0))
    return np.where(G.design3, np.where(B, 1.0, other), 0.0)


def interp_init(G, fn):
    from scipy.ndimage import map_coordinates
    d = np.load(fn); hc = float(d["h"]); xc = d["x"].astype(float)
    a0c, v0c = float(d["a0"]), float(d["v0"])
    I = (G.ac - a0c) / hc - 0.5; J = (G.vc - v0c) / hc - 0.5; K = G.zc / hc - 0.5
    II, JJ, KK = np.meshgrid(I, J, K, indexing="ij")
    x0 = np.clip(map_coordinates(xc, [II, JJ, KK], order=1, mode="nearest"), 0, 1)
    s0 = None
    if d["s"].ndim == 2:
        I2, J2 = np.meshgrid(I, J, indexing="ij")
        s0 = np.clip(map_coordinates(d["s"].astype(float), [I2, J2], order=1, mode="nearest"), 0, 1)
    print(f"初始设计取自 {fn}（网格 {xc.shape}，h={hc}）", flush=True)
    return x0, s0


if __name__ == "__main__":
    main()
