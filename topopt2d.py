"""
二维 SIMP 拓扑优化（平面应力，Q4 单元，密度过滤 + Heaviside 投影 + β 续进，OC 更新）
多工况加权柔度最小化，体积约束。支撑件侧面轮廓 → 拉伸后打印无悬空。
用法: STAND_VARIANT=desk40 python topopt2d.py [--robust] [--h 1.0] [--vf 0.28] [--rmin 3.0] [--nojoint] [--out 目录]
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import pypardiso
import params as P


def element_stiffness(nu):
    D = 1 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    g = 1 / np.sqrt(3)
    xin = np.array([-1, 1, 1, -1]); etn = np.array([-1, -1, 1, 1])
    KE = np.zeros((8, 8))
    for xi in (-g, g):
        for et in (-g, g):
            dNx = 0.25 * xin * (1 + et * etn) * 2
            dNy = 0.25 * etn * (1 + xi * xin) * 2
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx; B[1, 1::2] = dNy; B[2, 0::2] = dNy; B[2, 1::2] = dNx
            KE += B.T @ D @ B * 0.25
    return KE  # 单位厚度、E=1；方形单元与尺寸无关


class Mesh:
    def __init__(self, h, use_joint):
        self.h = h
        self.domain, self.solids = P.geometry(use_joint)
        minx, miny, maxx, maxy = self.domain.bounds
        self.nelx = int(np.ceil(maxx / h)); self.nely = int(np.ceil(maxy / h))
        nelx, nely = self.nelx, self.nely
        self.nel = nelx * nely
        I, J = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        self.I, self.J = I.ravel(), J.ravel()               # e = i*nely + j
        self.xc = (self.I + 0.5) * h; self.yc = (self.J + 0.5) * h
        n0 = self.I * (nely + 1) + self.J
        n1 = n0 + (nely + 1)
        nodes = np.stack([n0, n1, n1 + 1, n0 + 1], 1)
        self.edof = np.empty((self.nel, 8), dtype=np.int64)
        self.edof[:, 0::2] = 2 * nodes; self.edof[:, 1::2] = 2 * nodes + 1
        self.nn = (nelx + 1) * (nely + 1); self.ndof = 2 * self.nn
        IN, JN = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1), indexing="ij")
        self.xn = IN.ravel() * h; self.yn = JN.ravel() * h
        self.in_dom = shapely.contains_xy(self.domain, self.xc, self.yc)
        self.solid = np.zeros(self.nel, bool)
        for g in self.solids.values():
            self.solid |= shapely.contains_xy(g, self.xc, self.yc)
        self.solid &= self.in_dom
        self.void = ~self.in_dom
        self.active = self.in_dom & ~self.solid
        self.iK = np.kron(self.edof, np.ones((8, 1), dtype=np.int64)).ravel()
        self.jK = np.kron(self.edof, np.ones((1, 8), dtype=np.int64)).ravel()
        # 节点是否属于实体单元（用于加载）
        used = np.zeros(self.nn, bool)
        used[nodes[self.in_dom].ravel()] = True
        self.node_in = used

    def grid(self, v):
        return v.reshape(self.nelx, self.nely).T   # [row=j(y), col=i(x)]


def build_filter(m, rmin):
    R = int(np.ceil(rmin / m.h)) - 1
    rows, cols, vals = [], [], []
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            w = rmin - m.h * np.hypot(di, dj)
            if w <= 0: continue
            i2 = m.I + di; j2 = m.J + dj
            ok = (i2 >= 0) & (i2 < m.nelx) & (j2 >= 0) & (j2 < m.nely)
            e = np.nonzero(ok)[0]
            rows.append(e); cols.append(i2[ok] * m.nely + j2[ok]); vals.append(np.full(e.size, w))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m.nel, m.nel))
    Hs = np.asarray(H.sum(1)).ravel()
    return H, Hs


def load_tools(m):
    """返回加载辅助函数。顶面节点 top、前挡后表面节点 lipback。"""
    s, n = P.xy2sn(m.xn, m.yn)
    ok = m.node_in
    top = ok & (n > -2.6) & (n <= 0.05)
    lipback = ok & (s > -2.6) & (s <= 0.05) & (n >= 0.0) & (n <= P.LIP_H)

    def add(mask, f, Fv):
        idx = np.nonzero(mask)[0]
        assert idx.size > 0, "加载区域无节点"
        Fv[2 * idx] += f[0] / idx.size; Fv[2 * idx + 1] += f[1] / idx.size

    def press(Fv, s0, s1, F):
        """竖直向下压笔记本 F：垂直斜面的分量压在顶面 [s0,s1]，沿斜面分量全部由前挡承担（按无摩擦，偏保守）"""
        add(top & (s >= s0) & (s <= s1), -F * np.cos(P.TH) * P.N, Fv)
        add(lipback, -F * np.sin(P.TH) * P.U, Fv)

    return s, n, top, lipback, add, press


def build_loads(m):
    s, n, top, lipback, add, press = load_tools(m)
    F = []; names = []

    def new():
        return np.zeros(m.ndof)
    # LC1 笔记本自重：机身 70% 均布 + 屏幕 30% 在转轴附近；沿斜面分量由前挡承担
    f = new(); press(f, 5, P.S_END, 0.7 * P.W1); press(f, 180, P.S_END, 0.3 * P.W1); F.append(f); names.append("自重")
    f = new(); press(f, 10, 40, P.F_PRESS); F.append(f); names.append("前压")
    f = new(); press(f, 175, P.S_END, P.F_PRESS); F.append(f); names.append("后压")
    # 向后推屏（摩擦传到顶面后部）
    f = new(); add(top & (s >= 185) & (s <= P.S_END), (P.F_PUSH, 0), f); F.append(f); names.append("后推")
    # 向前推机身（前挡受水平力）
    f = new(); add(lipback, (-P.F_PUSH, 0), f); F.append(f); names.append("前推")
    # 前挡冲击：把笔记本滑进支架时沿斜面撞击前挡
    f = new(); add(lipback, -P.F_LIP_IMPACT * P.U, f); F.append(f); names.append("前挡冲击")
    # 每根横梁一个工况：横梁传来的力（搬动、两侧受力不均），保证每个接口都被结构连上
    for j in P.JOINTS:
        if "joint_" + j["name"] not in m.solids:
            continue
        jr = P.joint_rect(P.BEAM_W, P.BEAM_H, j)
        joint = m.node_in & shapely.contains_xy(jr, m.xn, m.yn)
        f = new(); add(joint, P.F_JOINT, f); F.append(f); names.append("横梁·" + j["cn"])
    weights_imp = {"自重": 2.0, "前压": 1.0, "后压": 1.0, "后推": 0.5, "前推": 0.5, "前挡冲击": 1.0}
    imp = np.array([weights_imp.get(k, 0.5) for k in names])
    return np.stack(F, 1), names, imp


def build_bc(m):
    yn0 = np.isclose(m.yn, 0.0)
    front = np.nonzero(yn0 & (m.xn <= P.FOOT_LEN + 1e-9))[0]
    rear = np.nonzero(yn0 & (m.xn >= P.X_MAX - P.FOOT_LEN - 1e-9) & (m.xn <= P.X_MAX + 1e-9))[0]
    fixed = []
    for nodes, how in ((front, P.FRONT_FIX), (rear, P.REAR_FIX)):
        if "x" in how: fixed.append(2 * nodes)
        if "y" in how: fixed.append(2 * nodes + 1)
    fixed = np.concatenate(fixed)
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    return free, fixed


def project(xt, beta, eta=0.5):
    a = np.tanh(beta * eta); b = np.tanh(beta * (1 - eta))
    xp = (a + np.tanh(beta * (xt - eta))) / (a + b)
    dxp = beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / (a + b)
    return xp, dxp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--vf", type=float, default=P.VOLFRAC)
    ap.add_argument("--rmin", type=float, default=3.0)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--maxit", type=int, default=400)
    ap.add_argument("--beta_max", type=float, default=32.0)
    ap.add_argument("--nojoint", action="store_true")
    ap.add_argument("--robust", action="store_true", help="腐蚀/中间/膨胀三场鲁棒公式，保证实体和孔洞最小尺寸")
    ap.add_argument("--eta_e", type=float, default=0.7)
    ap.add_argument("--eta_d", type=float, default=0.3)
    ap.add_argument("--out", default=P.TOPO_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    m = Mesh(a.h, not a.nojoint)
    H, Hs = build_filter(m, a.rmin)
    F, names, imp = build_loads(m)
    free, fixed = build_bc(m)
    KE = element_stiffness(P.NU)
    E0 = P.E_MAT * P.T; Emin = E0 * 1e-9
    nlc = F.shape[1]
    print(f"网格 {m.nelx}x{m.nely}, 设计域单元 {m.in_dom.sum()}, 可设计 {m.active.sum()}, 自由度 {free.size}, 工况 {names}")

    x = np.where(m.solid, 1.0, np.where(m.void, 0.0, a.vf))
    beta = 1.0; loop_beta = 0; change = 1.0
    nd = m.in_dom.sum()
    solver = pypardiso.PyPardisoSolver(mtype=2)
    hist = []; wts = None

    eta_e = a.eta_e if a.robust else 0.5
    eta_d = a.eta_d if a.robust else 0.5

    def physics(x, beta, eta=0.5):
        xt = (H @ x) / Hs
        xp, dxp = project(xt, beta, eta)
        xp[m.solid] = 1.0; xp[m.void] = 0.0
        dxp[m.solid | m.void] = 0.0
        return xp, dxp

    vd_star = a.vf
    vmask = m.active if P.VF_MODE == "active" else m.in_dom      # 体积约束统计的区域
    nd = int(vmask.sum())
    print(f"体积约束: {P.VF_MODE}，目标 {a.vf}（统计 {nd} 个单元；非设计实体 {int(m.solid.sum())} 个）")
    xe, dxe = physics(x, beta, eta_e)
    for it in range(1, a.maxit + 1):
        sK = (KE.ravel()[None, :] * (Emin + xe[:, None] ** a.penal * (E0 - Emin))).ravel()
        K = sp.csr_matrix((sK, (m.iK, m.jK)), shape=(m.ndof, m.ndof))
        Kf = sp.triu(K[free][:, free], format="csr")
        U = np.zeros((m.ndof, nlc))
        U[free] = solver.solve(Kf, F[free]) if nlc > 1 else solver.solve(Kf, F[free]).reshape(-1, 1)
        solver.free_memory()
        ce = np.zeros((m.nel, nlc))
        for k in range(nlc):
            ue = U[m.edof, k]
            ce[:, k] = np.einsum("ij,jk,ik->i", ue, KE, ue)
        Ek = Emin + xe ** a.penal * (E0 - Emin)
        comp = (Ek[:, None] * ce).sum(0)
        if wts is None:
            wts = imp / comp
        c = float((wts * comp).sum())
        dc_p = -a.penal * xe ** (a.penal - 1) * (E0 - Emin) * (ce @ wts)
        dc = H.T @ (dc_p * dxe / Hs)
        xdd, dxd = physics(x, beta, eta_d)
        dv = H.T @ (vmask * dxd / Hs) / nd
        if a.robust and (it == 1 or it % 10 == 0):
            xi_, _ = physics(x, beta, 0.5)
            vd_star = a.vf * xdd[vmask].mean() / max(xi_[vmask].mean(), 1e-9)
        dc[~m.active] = 0; dc = np.minimum(dc, -1e-30)
        dv = np.maximum(dv, 1e-30)
        move = 0.1 if beta < 8 else 0.05
        l1, l2 = 0.0, 1e12
        xa = x[m.active]
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-5:
            lm = 0.5 * (l1 + l2)
            xn = np.clip(xa * np.sqrt(-dc[m.active] / dv[m.active] / lm), np.maximum(0, xa - move), np.minimum(1, xa + move))
            xt = x.copy(); xt[m.active] = xn
            xpt, _ = physics(xt, beta, eta_d)
            if xpt[vmask].mean() > vd_star: l1 = lm
            else: l2 = lm
        change = float(np.abs(xn - xa).max())
        x[m.active] = xn
        xe, dxe = physics(x, beta, eta_e)
        xp, _ = physics(x, beta, 0.5)
        vol = float(xp[vmask].mean())
        grey = float((4 * xp * (1 - xp))[m.in_dom].mean())
        hist.append(dict(it=it, c=c, vol=vol, change=change, beta=beta, grey=grey, comp=comp.tolist()))
        if it % 10 == 0 or it == 1:
            print(f"it {it:4d} c={c:.4f} vol={vol:.3f} ch={change:.3f} beta={beta:g} grey={grey:.3f} t={time.time()-t0:.0f}s", flush=True)
        loop_beta += 1
        if beta < a.beta_max and (loop_beta >= 50 or (change < 0.01 and loop_beta >= 20)):
            beta *= 2; loop_beta = 0
            xe, dxe = physics(x, beta, eta_e)
            fig_save(m, xp, f"{a.out}/xphys_beta{int(beta/2)}.png", f"beta={beta/2:g} it={it}")
        elif beta >= a.beta_max and loop_beta >= 30:
            c10 = hist[-11]["c"]
            if change < 0.005 or abs(c10 - c) / c < 2e-4:
                break

    xdd, _ = physics(x, beta, eta_d)
    np.savez(f"{a.out}/result.npz", x=x, xphys=xp, xero=xe, xdil=xdd, nelx=m.nelx, nely=m.nely, h=m.h,
             in_dom=m.in_dom, solid=m.solid, comp=comp, names=np.array(names))
    with open(f"{a.out}/history.json", "w") as fh:
        json.dump(dict(args=vars(a), names=names, hist=hist), fh, ensure_ascii=False)
    fig_save(m, xp, f"{a.out}/xphys_final.png", f"final it={it} c={c:.3f} vol={vol:.3f} grey={grey:.3f}")
    fig_setup(m, F, free, fixed, f"{a.out}/setup.png")
    print(f"完成: it={it} c={c:.4f} vol={vol:.3f} grey={grey:.3f} 用时 {time.time()-t0:.0f}s")
    for k, nm in enumerate(names):
        print(f"  工况 {nm}: 柔度 {comp[k]:.4e} N·mm")


def fig_save(m, xp, fn, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    g = m.grid(xp)
    ax.imshow(1 - g, cmap="gray", origin="lower", extent=[0, m.nelx * m.h, 0, m.nely * m.h], vmin=0, vmax=1)
    xs, ys = m.domain.exterior.xy
    ax.plot(xs, ys, "c-", lw=0.6)
    ax.set_title(title); ax.set_aspect("equal")
    fig.savefig(fn, dpi=110, bbox_inches="tight"); plt.close(fig)


def fig_setup(m, F, free, fixed, fn):
    fig, ax = plt.subplots(figsize=(9, 7))
    xs, ys = m.domain.exterior.xy
    ax.fill(xs, ys, color="#dde7f0", lw=0)
    for nm, g in m.solids.items():
        gx, gy = g.exterior.xy; ax.fill(gx, gy, color="#556", alpha=0.8)
    fx = np.unique(fixed // 2)
    ax.plot(m.xn[fx], m.yn[fx] - 1.5, "g^", ms=3)
    cols = plt.cm.tab10(np.arange(F.shape[1]))
    for k in range(F.shape[1]):
        idx = np.nonzero(np.abs(F[0::2, k]) + np.abs(F[1::2, k]))[0]
        fxk = F[2 * idx, k]; fyk = F[2 * idx + 1, k]
        sc = 6.0 / max(np.hypot(fxk, fyk).max(), 1e-12)
        sel = idx[:: max(1, idx.size // 25)]
        ax.quiver(m.xn[sel], m.yn[sel], F[2 * sel, k], F[2 * sel + 1, k], color=cols[k], angles="xy", scale_units="xy", scale=1 / sc / 2, width=0.003)
    ax.set_aspect("equal"); ax.set_title("设计域 / 非设计区 / 约束 / 载荷")
    fig.savefig(fn, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
