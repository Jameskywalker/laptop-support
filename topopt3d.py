"""
三维拓扑优化：二维结果膨胀 + 加厚作初始模型，高度场参数化（只从上表面往下去料，无悬空）

- 网格：面内沿用二维结果的 1 mm 网格（补齐到 8 的倍数），z 向单元高 = 层高 P.LAYER_H。
- 材料：横观各向同性，z 向模量和层间剪切模量 × P.KZ，模拟层间结合弱。
- 设计变量：每根柱子一个高度 t(x, y)，体素密度 = 该层被填充的比例（顶层可部分填充，
  最后取整到层）。比 P.T3_MIN 薄的柱子刚度按 (t/T3_MIN)^3 惩罚。
  横梁接口区厚度固定为 P.T（卡扣尺寸依赖它），其余非设计区厚度 ≥ P.T。
- 求解：GPU 上矩阵无关的几何多重网格预条件共轭梯度（单精度）。z 向先半粗化到单元接近
  立方体，再三向粗化；最粗层用 cuDSS 直接解。
- 体积约束：与二维结果按 P.T 拉伸的体积相同。
用法: STAND_VARIANT=desk40 python topopt3d.py [--maxit 80] [--kz 0.5] [--tmax 19.2] [--dilate 3]
"""
import argparse, functools, json, logging, os, time
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation
import shapely
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2          # 字体设置

F32 = cp.float32


# ======================= 材料与单元 =======================
def material_D(E, nu, kz):
    """横观各向同性弹性矩阵，Voigt 顺序 xx, yy, zz, xy, yz, zx；z 为层叠方向"""
    G = E / (2 * (1 + nu))
    S = np.zeros((6, 6))
    S[0, 0] = S[1, 1] = 1 / E
    S[0, 1] = S[1, 0] = -nu / E
    S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = -nu / E        # ν_xz / E_x = ν_zx / E_z
    S[2, 2] = 1 / (kz * E)
    S[3, 3] = 1 / G
    S[4, 4] = S[5, 5] = 1 / (kz * G)
    D = np.linalg.inv(S)
    assert np.all(np.linalg.eigvalsh(D) > 0)
    return D


_SGN = np.array([[2 * (a & 1) - 1, 2 * ((a >> 1) & 1) - 1, 2 * (a >> 2) - 1] for a in range(8)], float)


def _hex_B(dx, dy, dz, xi, et, ze):
    """八节点六面体应变矩阵；局部节点 a 的偏移为 (a&1, (a>>1)&1, a>>2)"""
    sx, sy, sz = _SGN.T
    dN = np.array([sx * (1 + sy * et) * (1 + sz * ze) / 8 * 2 / dx,
                   sy * (1 + sx * xi) * (1 + sz * ze) / 8 * 2 / dy,
                   sz * (1 + sx * xi) * (1 + sy * et) / 8 * 2 / dz])
    B = np.zeros((6, 24))
    B[0, 0::3] = dN[0]; B[1, 1::3] = dN[1]; B[2, 2::3] = dN[2]
    B[3, 0::3] = dN[1]; B[3, 1::3] = dN[0]
    B[4, 1::3] = dN[2]; B[4, 2::3] = dN[1]
    B[5, 0::3] = dN[2]; B[5, 2::3] = dN[0]
    return B


def hex_KE(dx, dy, dz, D):
    g = 1 / np.sqrt(3); KE = np.zeros((24, 24))
    for xi in (-g, g):
        for et in (-g, g):
            for ze in (-g, g):
                B = _hex_B(dx, dy, dz, xi, et, ze)
                KE += B.T @ D @ B * (dx * dy * dz / 8)
    return KE


def grid_edof(nx, ny, nz):
    i, j, k = [a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")]
    nyn, nzn = ny + 1, nz + 1
    base = (i * nyn + j) * nzn + k
    offs = np.array([((a & 1) * nyn + ((a >> 1) & 1)) * nzn + (a >> 2) for a in range(8)])
    nodes = base[:, None] + offs[None, :]
    return (3 * nodes[:, :, None] + np.arange(3)).reshape(-1, 24)


# ======================= GPU kernel =======================
# 节点编号 n = (i*(ny+1) + j)*(nz+1) + k（z 最快），单元编号 e = (i*ny + j)*nz + k，每节点 3 个自由度交错存放。
_SRC = r'''
#define NODE_IJK(n) const int k = n % nzn; const long r_ = n / nzn; const int j = r_ % nyn; const int i = r_ / nyn;
#define LOAD_KE __shared__ float KE[576]; for (int t = threadIdx.x; t < 576; t += blockDim.x) KE[t] = KEg[t]; __syncthreads();
extern "C" {
// out = diag(fm) * K(g) * u：按节点汇总相邻 8 个单元的贡献，无原子操作
__global__ void kmul(const float* __restrict__ u, float* __restrict__ out, const float* __restrict__ g,
                     const float* __restrict__ KEg, const float* __restrict__ fm, int nx, int ny, int nz) {
    LOAD_KE
    const int nyn = ny + 1, nzn = nz + 1;
    const long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)(nx + 1) * nyn * nzn) return;
    NODE_IJK(n)
    float o0 = 0.f, o1 = 0.f, o2 = 0.f;
    for (int a = 0; a < 8; a++) {
        const int ei = i - (a & 1), ej = j - ((a >> 1) & 1), ek = k - (a >> 2);
        if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
        const float ge = g[((long)ei * ny + ej) * nz + ek];
        const float* Ka = KE + 72 * a;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            const long m = 3 * (((long)(ei + (b & 1)) * nyn + (ej + ((b >> 1) & 1))) * nzn + (ek + (b >> 2)));
            const float u0 = u[m], u1 = u[m + 1], u2 = u[m + 2];
            const int c = 3 * b;
            s0 += Ka[c] * u0 + Ka[c + 1] * u1 + Ka[c + 2] * u2;
            s1 += Ka[24 + c] * u0 + Ka[25 + c] * u1 + Ka[26 + c] * u2;
            s2 += Ka[48 + c] * u0 + Ka[49 + c] * u1 + Ka[50 + c] * u2;
        }
        o0 += ge * s0; o1 += ge * s1; o2 += ge * s2;
    }
    const long q = 3 * n;
    out[q] = o0 * fm[q]; out[q + 1] = o1 * fm[q + 1]; out[q + 2] = o2 * fm[q + 2];
}
__global__ void kdiag(float* __restrict__ d, const float* __restrict__ g, const float* __restrict__ KEg, int nx, int ny, int nz) {
    LOAD_KE
    const int nyn = ny + 1, nzn = nz + 1;
    const long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)(nx + 1) * nyn * nzn) return;
    NODE_IJK(n)
    float d0 = 0.f, d1 = 0.f, d2 = 0.f;
    for (int a = 0; a < 8; a++) {
        const int ei = i - (a & 1), ej = j - ((a >> 1) & 1), ek = k - (a >> 2);
        if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
        const float ge = g[((long)ei * ny + ej) * nz + ek];
        d0 += ge * KE[75 * a]; d1 += ge * KE[75 * a + 25]; d2 += ge * KE[75 * a + 50];
    }
    d[3 * n] = d0; d[3 * n + 1] = d1; d[3 * n + 2] = d2;
}
// ce[e] += w * u_e^T KE u_e
__global__ void kener(const float* __restrict__ u, float* __restrict__ ce, const float* __restrict__ KEg, float w, int nx, int ny, int nz) {
    LOAD_KE
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long)nx * ny * nz) return;
    const int k = e % nz; const long r_ = e / nz; const int j = r_ % ny; const int i = r_ / ny;
    const int nyn = ny + 1, nzn = nz + 1;
    float ue[24];
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const long m = 3 * (((long)(i + (b & 1)) * nyn + (j + ((b >> 1) & 1))) * nzn + (k + (b >> 2)));
        ue[3 * b] = u[m]; ue[3 * b + 1] = u[m + 1]; ue[3 * b + 2] = u[m + 2];
    }
    float s = 0.f;
    for (int p = 0; p < 24; p++) {
        float t = 0.f;
        #pragma unroll
        for (int q = 0; q < 24; q++) t += KE[24 * p + q] * ue[q];
        s += ue[p] * t;
    }
    ce[e] += w * s;
}
// 单元中心应力 sig[6e+c] = g_e * (D B_c u_e)_c
__global__ void kstress(const float* __restrict__ u, float* __restrict__ sig, const float* __restrict__ g,
                        const float* __restrict__ DBg, int nx, int ny, int nz) {
    __shared__ float DB[144]; for (int t = threadIdx.x; t < 144; t += blockDim.x) DB[t] = DBg[t]; __syncthreads();
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long)nx * ny * nz) return;
    const int k = e % nz; const long r_ = e / nz; const int j = r_ % ny; const int i = r_ / ny;
    const int nyn = ny + 1, nzn = nz + 1;
    float ue[24];
    for (int b = 0; b < 8; b++) {
        const long m = 3 * (((long)(i + (b & 1)) * nyn + (j + ((b >> 1) & 1))) * nzn + (k + (b >> 2)));
        ue[3 * b] = u[m]; ue[3 * b + 1] = u[m + 1]; ue[3 * b + 2] = u[m + 2];
    }
    for (int c = 0; c < 6; c++) {
        float s = 0.f;
        for (int q = 0; q < 24; q++) s += DB[24 * c + q] * ue[q];
        sig[6 * e + c] = g[e] * s;
    }
}
}
'''


@functools.cache
def _K():
    mod = cp.RawModule(code=_SRC, options=("--use_fast_math",))
    return {n: mod.get_function(n) for n in ("kmul", "kdiag", "kener", "kstress")}


_jacobi = cp.ElementwiseKernel("float32 b, float32 Au, float32 dinv, float32 w", "float32 u",
                               "u += w * dinv * (b - Au)", "mg_jacobi")
_TB = 256


def _blocks(n):
    return ((n + _TB - 1) // _TB,)


# ======================= 多重网格 =======================
class Level:
    def __init__(self, nx, ny, nz, dx, dy, dz, D, fm):
        self.nx, self.ny, self.nz, self.dx, self.dy, self.dz = nx, ny, nz, dx, dy, dz
        self.shape = (nx + 1, ny + 1, nz + 1, 3)
        self.nn = (nx + 1) * (ny + 1) * (nz + 1); self.nel = nx * ny * nz
        self.KE_np = hex_KE(dx, dy, dz, D)
        self.KE = cp.asarray(self.KE_np.ravel(), F32)
        self.fm = fm
        self.g = None; self.dinv = None; self.omega = None; self._pv = None

    def A(self, u):
        out = cp.empty_like(u)
        _K()["kmul"](_blocks(self.nn), (_TB,), (u, out, self.g, self.KE, self.fm, np.int32(self.nx), np.int32(self.ny), np.int32(self.nz)))
        return out

    def set_g(self, g):
        self.g = g
        d = cp.empty(3 * self.nn, F32)
        _K()["kdiag"](_blocks(self.nn), (_TB,), (d, g, self.KE, np.int32(self.nx), np.int32(self.ny), np.int32(self.nz)))
        self.dinv = self.fm / d

    def estimate_omega(self, rel, iters=12):
        """幂迭代估计 λmax(D⁻¹A)，Jacobi 阻尼取 rel/λmax（λmax > 2/ω 时光滑步会发散）"""
        if self._pv is None:
            self._pv = cp.asarray(np.random.default_rng(0).standard_normal(3 * self.nn), F32) * self.fm
        v = self._pv
        for _ in range(iters):
            w = self.dinv * self.A(v); lam = float(cp.linalg.norm(w) / cp.linalg.norm(v)); v = w / cp.linalg.norm(w)
        self._pv = v; self.lam = lam
        self.omega = np.float32(rel / (1.05 * lam))


def _sl(ax, s, nd=4):
    return tuple(s if d == ax else slice(None) for d in range(nd))


def _prolong1(a, ax):
    shp = list(a.shape); shp[ax] = 2 * a.shape[ax] - 1
    f = cp.empty(shp, a.dtype)
    f[_sl(ax, slice(0, None, 2))] = a
    f[_sl(ax, slice(1, None, 2))] = 0.5 * (a[_sl(ax, slice(0, -1))] + a[_sl(ax, slice(1, None))])
    return f


def _restrict1(f, ax):
    c = f[_sl(ax, slice(0, None, 2))].copy()
    odd = 0.5 * f[_sl(ax, slice(1, None, 2))]
    c[_sl(ax, slice(0, -1))] += odd
    c[_sl(ax, slice(1, None))] += odd
    return c


class CoarseDirect:
    """最粗层：上三角稀疏模式只建一次，每轮 scatter 组装后 cuDSS 分解"""

    def __init__(self, L):
        from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, DirectSolverMatrixViewType
        import cupyx.scipy.sparse as csp
        edof = grid_edof(L.nx, L.ny, L.nz)
        fmh = cp.asnumpy(L.fm) > 0.5
        free = np.nonzero(fmh)[0]; nf = free.size
        fmap = np.full(3 * L.nn, -1, np.int64); fmap[free] = np.arange(nf)
        r = fmap[np.repeat(edof, 24, axis=1).ravel()]; c = fmap[np.tile(edof, (1, 24)).ravel()]
        keep = (r >= 0) & (c >= 0) & (r <= c)
        ukey, pos = np.unique(r[keep] * nf + c[keep], return_inverse=True)
        rows, cols = ukey // nf, ukey % nf
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=nf))])
        self.nnz = ukey.size; self.L = L
        self.keep = cp.asarray(np.nonzero(keep)[0]); self.pos = cp.asarray(pos)
        self.KE = cp.asarray(L.KE_np.ravel())
        self.K = csp.csr_matrix((cp.zeros(self.nnz), cp.asarray(cols, dtype=cp.int32), cp.asarray(indptr, dtype=cp.int32)), shape=(nf, nf))
        self.free = cp.asarray(free); self.b = cp.zeros(nf)
        quiet = logging.getLogger("topopt3d.cudss"); quiet.setLevel(logging.ERROR)
        self.solver = DirectSolver(self.K, self.b, options=dict(sparse_system_type=DirectSolverMatrixType.SPD,
                                                                sparse_system_view=DirectSolverMatrixViewType.UPPER, logger=quiet))
        self.planned = False

    def factor(self, g):
        vals = (self.KE[None, :] * g.astype(cp.float64)[:, None]).ravel()[self.keep]
        self.K.data[...] = cp.bincount(self.pos, weights=vals, minlength=self.nnz)
        if not self.planned:
            self.solver.plan(); self.planned = True
        self.solver.factorize()

    def solve(self, b):
        self.b[...] = b[self.free]
        out = cp.zeros(3 * self.L.nn, F32)
        out[self.free] = self.solver.solve().astype(F32)
        return out


MG_INJECT = False          # True：约束按注入法传到粗层（旧做法，只用来对比）


class MGCG:
    def __init__(self, nx, ny, nz, dx, dz, D, fm, nu_s=2, omega=1.6, coarse_nodes=12000):
        self.levels = [Level(nx, ny, nz, dx, dx, dz, D, fm)]; self.axes = []
        self.nu_s, self.omega = nu_s, omega
        while self.levels[-1].nn > coarse_nodes:
            L = self.levels[-1]
            ev = (L.nx % 2 == 0, L.ny % 2 == 0, L.nz % 2 == 0)
            if L.dz < 0.7 * L.dx and ev[2]:
                ax = (False, False, True)                # z 向半粗化，直到单元接近立方体
            elif all(ev):
                ax = (True, True, True)
            elif ev[0] and ev[1]:
                ax = (True, True, False)
            else:
                break
            f = [2 if a else 1 for a in ax]
            # 约束传到粗层：粗节点 i 对应细节点 2i，它或相邻的 2i±1 受约束就算受约束（只取 2i 的话，约束落在奇数节点上
            # ——比如扁体素补层后厚度 T 处的内侧面——到粗层就丢了，粗网格校正不准，CG 次数会翻几倍）。只影响预条件，不影响解
            a = L.fm.reshape(L.shape)
            for d in range(3):
                if ax[d] and not MG_INJECT:
                    lo = cp.concatenate([cp.ones_like(a[_sl(d, slice(0, 1), 3)]), a[_sl(d, slice(0, -1), 3)]], d)
                    hi = cp.concatenate([a[_sl(d, slice(1, None), 3)], cp.ones_like(a[_sl(d, slice(0, 1), 3)])], d)
                    a = cp.minimum(cp.minimum(lo, a), hi)[_sl(d, slice(0, None, 2), 3)]
            fmc = a if not MG_INJECT else L.fm.reshape(L.shape)[tuple(slice(0, None, f[d]) for d in range(3))]
            self.levels.append(Level(L.nx // f[0], L.ny // f[1], L.nz // f[2], L.dx * f[0], L.dy * f[1], L.dz * f[2], D,
                                     cp.ascontiguousarray(fmc).ravel()))
            self.axes.append(ax)
        self.coarse = CoarseDirect(self.levels[-1])

    def describe(self):
        return " → ".join(f"{L.nx}×{L.ny}×{L.nz}({L.dx:g}×{L.dy:g}×{L.dz:g})" for L in self.levels)

    def update(self, g):
        self.levels[0].set_g(g)
        self.levels[0].estimate_omega(self.omega)
        for l, ax in enumerate(self.axes):
            L = self.levels[l]
            a = L.g.reshape(L.nx, L.ny, L.nz)
            for d in range(3):
                if ax[d]:
                    a = 0.5 * (a[_sl(d, slice(0, None, 2), 3)] + a[_sl(d, slice(1, None, 2), 3)])
            self.levels[l + 1].set_g(cp.ascontiguousarray(a).ravel())
            self.levels[l + 1].estimate_omega(self.omega)
        self.coarse.factor(self.levels[-1].g)

    def vcycle(self, l, b):
        if l == len(self.levels) - 1:
            return self.coarse.solve(b)
        L, Lc, ax = self.levels[l], self.levels[l + 1], self.axes[l]
        w = L.omega
        u = w * L.dinv * b
        for _ in range(self.nu_s - 1):
            _jacobi(b, L.A(u), L.dinv, w, u)
        r = (b - L.A(u)).reshape(L.shape)
        for d in range(3):
            if ax[d]: r = _restrict1(r, d)
        e = self.vcycle(l + 1, cp.ascontiguousarray(r).ravel() * Lc.fm).reshape(Lc.shape)
        for d in (2, 1, 0):
            if ax[d]: e = _prolong1(e, d)
        u += e.ravel() * L.fm
        for _ in range(self.nu_s):
            _jacobi(b, L.A(u), L.dinv, w, u)
        return u

    def solve(self, b, x=None, tol=1e-5, maxit=300):
        A = self.levels[0].A
        bn = float(cp.linalg.norm(b))
        if x is None:
            x = cp.zeros_like(b); r = b.copy()
        else:
            x = x.copy(); r = b - A(x)
        if float(cp.linalg.norm(r)) <= tol * bn:
            return x, 0
        z = self.vcycle(0, r); p = z.copy(); rz = float(cp.dot(r, z))
        for it in range(1, maxit + 1):
            q = A(p)
            alpha = rz / float(cp.dot(p, q))
            x += np.float32(alpha) * p; r -= np.float32(alpha) * q
            if float(cp.linalg.norm(r)) <= tol * bn:
                break
            z = self.vcycle(0, r)
            rz2 = float(cp.dot(r, z)); p = z + np.float32(rz2 / rz) * p; rz = rz2
        return x, it


# ======================= 打印结构 =======================
class PrintModel:
    """把柱高场换算成打印件的材料分布：每层轮廓向内 NWALL 圈墙、每根柱子底部和顶部各 NSKIN 层表皮为实心，
    其余为 gyroid 填充（刚度 ρ²、用料 ρ）。墙用"腐蚀"判断：本层邻域内只要有空柱，该柱就在墙上；
    邻域求平均后做 Heaviside 投影，可求导。面内网格 1 mm：第一圈柱子全是墙，第二圈按 (墙厚 − 1 mm) 计墙。

    体素值 v = o − I·(1 − s)·(1 − q)：o 为该层被占比例，I 为"内部"程度，s 为表皮程度，
    q 为填充相对实心的比例（刚度取 ρ²，用料取 ρ）。"""

    def __init__(self, M, nwall, nskin, infill):
        self.M, self.nskin = M, nskin
        wall = nwall * P.LINE_W
        assert wall >= M.h, "墙厚至少一圈柱子"
        self.b = float(np.clip(wall - M.h, 0, 1)); self.a = 1 - self.b
        self.q = {"E": infill ** 2, "m": infill}
        r1 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        r2 = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if i * i + j * j <= 4]
        self.st = []
        for offs in (r1, r2):
            n = len(offs); eta = 1 - 0.5 / n
            self.st.append((offs, eta, 2.3 / (eta - (n - 1) / n)))      # 缺一个邻居时投影值约 0.02
        NZ = M.NZ
        self.zb = cp.arange(NZ, dtype=F32) * F32(M.dz)
        self.sbot = (cp.arange(NZ) < nskin).astype(F32)

    @staticmethod
    def _avg(A, offs):
        """out[i,j,:] = 平均 A[i+di, j+dj, :]（越界按空）；邻域关于原点对称，所以它也是自己的转置"""
        out = cp.zeros_like(A); NX, NY = A.shape[:2]
        for di, dj in offs:
            i0, i1 = max(0, -di), NX - max(0, di); j0, j1 = max(0, -dj), NY - max(0, dj)
            out[i0:i1, j0:j1] += A[i0 + di:i1 + di, j0 + dj:j1 + dj]
        return out / F32(len(offs))

    @staticmethod
    def _proj(F, beta, eta):
        a, b = np.tanh(beta * eta), np.tanh(beta * (1 - eta))
        th = cp.tanh(F32(beta) * (F - F32(eta)))
        return (F32(a) + th) / F32(a + b), F32(beta) * (1 - th * th) / F32(a + b)

    def forward(self, t):
        M = self.M; dz = F32(M.dz)
        sh = (M.NX, M.NY, M.NZ)
        o = cp.clip((t[:, None] - self.zb[None, :]) / dz, 0, 1).reshape(sh)
        (o1, e1, b1), (o2, e2, b2) = self.st
        i1, d1 = self._proj(self._avg(o, o1), b1, e1)
        i2, d2 = self._proj(self._avg(o, o2), b2, e2)
        I = F32(self.a) * i1 + F32(self.b) * i2
        arg = (F32(self.nskin) * dz - (t[:, None] - (self.zb[None, :] + dz))) / dz
        stop = cp.clip(arg, 0, 1).reshape(sh)
        sb = self.sbot[None, None, :]
        s = 1 - (1 - stop) * (1 - sb)
        c = dict(t=t, o=o, d1=d1, d2=d2, I=I, s=s, sb=sb, arg=arg.reshape(sh))
        return {k: o - I * (1 - s) * F32(1 - q) for k, q in self.q.items()}, c

    def backward(self, gv, c, key):
        """gv: dX/dv（形状 NX×NY×NZ）→ dX/dt（每根柱子）"""
        M = self.M; dz = F32(M.dz); q = F32(self.q[key])
        (o1, _, _), (o2, _, _) = self.st
        gI = -gv * (1 - c["s"]) * (1 - q)
        go = gv + self._avg(gI * F32(self.a) * c["d1"], o1) + self._avg(gI * F32(self.b) * c["d2"], o2)
        gs = gv * c["I"] * (1 - q) * (1 - c["sb"])
        zt = (c["t"][:, None] - self.zb[None, :]).reshape(go.shape)
        do = ((zt > 0) & (zt < dz)).astype(F32) / dz
        ds = -((c["arg"] > 0) & (c["arg"] < 1)).astype(F32) / dz
        return (go * do + gs * ds).sum(2).ravel()


# ======================= 模型 =======================
class Model3D:
    def __init__(self, a):
        r = np.load(f"{P.TOPO_DIR}/result.npz")
        nelx, nely, h = int(r["nelx"]), int(r["nely"]), float(r["h"])
        self.h, self.dz = h, P.LAYER_H
        self.NX, self.NY = -(-nelx // 8) * 8, -(-nely // 8) * 8
        self.NZ = int(round(a.tmax / self.dz)); self.NT = int(round(P.T / self.dz))
        assert self.NZ % 8 == 0 and abs(self.NZ * self.dz - a.tmax) < 1e-9, "层数需能被 8 整除"
        self.tmax, self.tmin = self.NZ * self.dz, a.tmin
        NX, NY, NZ = self.NX, self.NY, self.NZ

        def pad(v):
            g = np.zeros((NX, NY), bool); g[:nelx, :nely] = v.reshape(nelx, nely); return g
        dom, solid = pad(r["in_dom"]), pad(r["solid"])
        shape2 = pad(r["xphys"] > 0.5) & dom
        joint = np.zeros((NX, NY), bool)
        I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
        xc, yc = (I + 0.5) * h, (J + 0.5) * h
        for j in P.JOINTS:
            joint |= shapely.contains_xy(P.joint_rect(2 * P._HB, 2 * P._HA, j), xc, yc)
        slot = shapely.contains_xy(P.joint_slot(), xc, yc)            # 方槽是贯通孔
        dom &= ~slot; solid &= ~slot; shape2 &= ~slot
        joint &= solid
        rr = int(round(a.dilate / h)); yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        dil = binary_dilation(shape2 | solid, xx ** 2 + yy ** 2 <= rr * rr) & dom
        self.cols = dict(dom=dom, solid=solid, joint=joint, shape2=shape2, dil=dil,
                         design=dil & ~solid, thick=solid & ~joint)          # thick：厚度 ≥ T、可往上加厚的非设计区
        c = {k: v.ravel() for k, v in self.cols.items()}
        self.c = c
        self.var = c["design"] | c["thick"]                                  # 设计变量所在的柱子
        self.s_lo = np.where(c["thick"], P.T / self.tmax, 0.0)
        self.s_fix = np.where(c["joint"], P.T / self.tmax, 0.0)              # 非变量柱子的值（参与过滤）
        # 面内密度过滤（柱网格）
        R = int(np.ceil(a.rmin / h)) - 1; rows, colsf, vals = [], [], []
        Ir, Jr = I.ravel(), J.ravel()
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):
                wv = a.rmin - h * np.hypot(di, dj)
                if wv <= 0: continue
                i2, j2 = Ir + di, Jr + dj
                ok = (i2 >= 0) & (i2 < NX) & (j2 >= 0) & (j2 < NY)
                e = np.nonzero(ok)[0]
                rows.append(e); colsf.append(i2[ok] * NY + j2[ok]); vals.append(np.full(e.size, wv))
        self.H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(colsf))), shape=(NX * NY, NX * NY))
        self.Hs = np.asarray(self.H.sum(1)).ravel()
        # 材料、多重网格
        self.D = material_D(P.E_MAT, P.NU, a.kz)
        self.gmin = 1e-6
        self.mode = getattr(a, "model", "solid")
        self.pm = PrintModel(self, a.nwall, a.nskin, a.infill) if self.mode == "print" else None
        self._loads_bc()
        self.mg = MGCG(NX, NY, NZ, h, self.dz, self.D, cp.asarray(self.fm, F32), nu_s=a.nu_s, omega=a.omega)
        self.KE0 = self.mg.levels[0].KE
        self.zb = cp.arange(NZ, dtype=F32) * self.dz

    # ---------- 载荷与约束 ----------
    def reload(self, colmask, misuse=True):
        """按实际几何（厚度 ≥ T 的柱子）重新选加载节点；复核时用，几何里有方槽等细节"""
        self._loads_bc(colmask, misuse)

    def _loads_bc(self, colmask=None, misuse=False):
        NX, NY, NZ, h, dz = self.NX, self.NY, self.NZ, self.h, self.dz
        IN, JN = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing="ij")
        xn, yn = IN * h, JN * h
        sol = np.pad(self.cols["solid"] if colmask is None else colmask.reshape(NX, NY), 1)
        nsol = sol[:-1, :-1] | sol[1:, :-1] | sol[:-1, 1:] | sol[1:, 1:]       # 节点挨着非设计实体柱
        s, n = P.xy2sn(xn, yn)
        top = nsol & (n > -2.6) & (n <= 0.05)
        lipback = nsol & (s > -2.6) & (s <= 0.05) & (n >= 0.0) & (n <= P.LIP_H)
        kz_ok = np.arange(NZ + 1) <= self.NT                                     # 加载/约束只在 0..T 厚度内
        nn = (NX + 1) * (NY + 1) * (NZ + 1)

        def m3(m2):
            return (m2[:, :, None] & kz_ok[None, None, :]).ravel()
        F = []; names = []

        def add(Fv, m2, f):
            idx = np.nonzero(m3(m2))[0]; assert idx.size > 0
            for d in range(3):
                Fv[3 * idx + d] += f[d] / idx.size

        def press(Fv, s0, s1, Fm):
            fn = -Fm * np.cos(P.TH) * P.N; fu = -Fm * np.sin(P.TH) * P.U
            add(Fv, top & (s >= s0) & (s <= s1), (fn[0], fn[1], 0.0))
            add(Fv, lipback, (fu[0], fu[1], 0.0))
        new = lambda: np.zeros(3 * nn)
        f = new(); press(f, 5, P.S_END, 0.7 * P.W1); press(f, 180, P.S_END, 0.3 * P.W1); F.append(f); names.append("自重")
        f = new(); press(f, 10, 40, P.F_PRESS); F.append(f); names.append("前压")
        f = new(); press(f, 175, P.S_END, P.F_PRESS); F.append(f); names.append("后压")
        f = new(); add(f, top & (s >= 185) & (s <= P.S_END), (P.F_PUSH, 0, 0)); F.append(f); names.append("后推")
        f = new(); add(f, lipback, (-P.F_PUSH, 0, 0)); F.append(f); names.append("前推")
        f = new(); fl = -P.F_LIP_IMPACT * P.U; add(f, lipback, (fl[0], fl[1], 0)); F.append(f); names.append("前挡冲击")
        f = new(); add(f, top & (s >= 0) & (s <= P.S_END), (0, 0, P.F_SIDE)); F.append(f); names.append("侧推")
        # 横梁：面内力经卡扣臂顶在受压一侧的方槽槽壁上（沿厚度 0..T 均布）；轴向力由卡钩扣在外侧面（z=0）的搭接带上
        k0 = np.arange(NZ + 1) == 0
        for j in P.JOINTS:
            w2 = P.slot_bearing(j, xn, yn, P.F_JOINT, 0.75 * h) * nsol
            w3 = (w2[:, :, None] * kz_ok[None, None, :]).ravel(); assert w3.sum() > 0
            f = new()
            for d in range(2):
                f[d::3] += P.F_JOINT[d] * w3 / w3.sum()
            hk = (nsol & P.hook_strip(j, xn, yn, 0.75 * h))[:, :, None] & k0[None, None, :]
            idx = np.nonzero(hk.ravel())[0]; assert idx.size > 0
            f[3 * idx + 2] += P.F_JOINT_Z / idx.size
            F.append(f); names.append("横梁·" + j["cn"])
        if misuse:   # 与二维复核相同的误用工况：手肘撑在前部 / 转轴处 150 N，只看应力，不计入加权柔度
            f = new(); press(f, 10, 40, 150.0); F.append(f); names.append("前部误用150N")
            f = new(); press(f, 175, P.S_END, 150.0); F.append(f); names.append("后部误用150N")
        imp = {"自重": 2.0, "前压": 1.0, "后压": 1.0, "后推": 0.5, "前推": 0.5, "前挡冲击": 1.0, "侧推": 1.0,
               "前部误用150N": 0.0, "后部误用150N": 0.0}
        self.imp = np.array([imp.get(k, 0.5) for k in names])
        self.names = names
        self.F = [cp.asarray(v, F32) for v in F]
        self.U = [None] * len(self.F)
        # 约束：前脚只给竖向、后脚铰支（与二维相同），两脚 z 向靠摩擦约束
        fixed = np.zeros(3 * nn, bool)
        y0 = JN == 0
        for m2, how in ((y0 & (xn <= P.FOOT_LEN + 1e-9), P.FRONT_FIX), (y0 & (xn >= P.X_MAX - P.FOOT_LEN - 1e-9) & (xn <= P.X_MAX + 1e-9), P.REAR_FIX)):
            idx = np.nonzero(m3(m2))[0]
            for d, ch in enumerate("xy"):
                if ch in how: fixed[3 * idx + d] = True
            fixed[3 * idx + 2] = True
        self.fm = (~fixed).astype(np.float32)
        for v in self.F:
            v *= cp.asarray(self.fm)

    # ---------- 设计变量 → 柱高 → 体素刚度 ----------
    def heights(self, s):
        """s: 所有柱子的设计变量（numpy）；返回柱高 t 与 dt/ds̃"""
        sf = np.where(self.var, s, self.s_fix)
        st = (self.H @ sf) / self.Hs
        t = np.zeros_like(st); dt = np.zeros_like(st)
        c = self.c
        t[c["design"]] = self.tmax * st[c["design"]]; dt[c["design"]] = self.tmax
        th = c["thick"]
        t[th] = np.maximum(P.T, self.tmax * st[th]); dt[th] = np.where(self.tmax * st[th] > P.T, self.tmax, 0.0)
        t[c["joint"]] = P.T
        return t, dt

    @property
    def pen_g(self):
        if not hasattr(self, "_pen_g"):
            self._pen_g = cp.asarray(self.c["design"])
        return self._pen_g

    def fields(self, t):
        """柱高（cupy）→ 体素刚度系数 g 及反传用的中间量。
        solid：体素按填充比例 ρ 计实心；print：按墙/表皮/填充（PrintModel）。薄于 tmin 的设计柱再乘 φ=(t/tmin)³。"""
        q = cp.minimum(1, t / F32(self.tmin))
        phi = cp.where(self.pen_g, q ** 3, F32(1))
        dphi = cp.where(self.pen_g & (t < self.tmin), 3 * q ** 2 / F32(self.tmin), F32(0))
        if self.pm is not None:
            v, c = self.pm.forward(t)
            vE, vm = v["E"].reshape(-1, self.NZ), v["m"]
        else:
            vE = cp.clip((t[:, None] - self.zb[None, :]) / F32(self.dz), 0, 1); vm = vE; c = None
        g = (F32(self.gmin) + F32(1 - self.gmin) * phi[:, None] * vE).astype(F32)
        return dict(t=t, g=g, vE=vE, vm=vm, phi=phi, dphi=dphi, c=c)

    def _dvdt(self, f, gv, key):
        if self.pm is not None:
            return self.pm.backward(gv.reshape(self.NX, self.NY, self.NZ), f["c"], key)
        zt = f["t"][:, None] - self.zb[None, :]
        return (gv.reshape(-1, self.NZ) * ((zt > 0) & (zt < self.dz)).astype(F32) / F32(self.dz)).sum(1)

    def mass(self, t_np):
        """约束量：print 为打印用料（实心等效体积 mm³），solid 为几何体积"""
        if self.pm is None:
            return float(t_np[self.var].sum() + t_np[self.c["joint"]].sum()) * self.h ** 2
        f = self.fields(cp.asarray(t_np, F32))
        return float(f["vm"].sum(dtype=cp.float64)) * self.h ** 2 * self.dz

    def dmass_dt(self, t_np):
        if self.pm is None:
            return np.where(self.var, self.h ** 2, 0.0)
        f = self.fields(cp.asarray(t_np, F32))
        gv = cp.full(f["vm"].shape, F32(self.h ** 2 * self.dz))
        return cp.asnumpy(self._dvdt(f, gv, "m")).astype(np.float64)

    def solve_g(self, g, tol=1e-5):
        self.mg.update(g)
        comps = np.zeros(len(self.F)); its = []
        for l, Fv in enumerate(self.F):
            self.U[l], it = self.mg.solve(Fv, self.U[l], tol=tol)
            comps[l] = float(cp.dot(Fv, self.U[l])); its.append(it)
        return comps, its

    def analyze(self, t_np, tol=1e-5, with_sens=True, weights=None):
        f = self.fields(cp.asarray(t_np, F32))
        comps, its = self.solve_g(f["g"].ravel(), tol)
        if not with_sens:
            return comps, its, None
        w = weights if weights is not None else self.imp / comps
        ce = cp.zeros(self.NX * self.NY * self.NZ, F32)
        L0 = self.mg.levels[0]
        for l in range(len(self.F)):
            _K()["kener"](_blocks(L0.nel), (_TB,), (self.U[l], ce, self.KE0, np.float32(w[l]), np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
        dcdg = -ce.reshape(-1, self.NZ) * F32(1 - self.gmin)
        dcdt = self._dvdt(f, dcdg * f["phi"][:, None], "E") + (dcdg * f["vE"]).sum(1) * f["dphi"]
        return comps, its, cp.asnumpy(dcdt).astype(np.float64)

    def stress_fields(self, t_np):
        """应力统计用：solid 模式按 SIMP 应力、只看实心体素；print 模式看墙和表皮里实心材料的应力"""
        f = self.fields(cp.asarray(t_np, F32))
        if self.pm is None:
            return f["g"].ravel(), f["g"].ravel() > 0.99
        return cp.ones(self.NX * self.NY * self.NZ, F32), f["vE"].ravel() > 0.5

    def stresses(self, gs, mask):
        """各工况体素中心应力 gs·D·B·u，只统计 mask 内的体素"""
        L0 = self.mg.levels[0]
        DB = cp.asarray((self.D @ _hex_B(self.h, self.h, self.dz, 0, 0, 0)).ravel(), F32)
        sig = cp.empty(6 * L0.nel, F32)
        out = []
        for l, nm in enumerate(self.names):
            _K()["kstress"](_blocks(L0.nel), (_TB,), (self.U[l], sig, gs, DB, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
            S = sig.reshape(-1, 6)[mask]
            vm = cp.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2) + 3 * (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2))
            tz = cp.sqrt(S[:, 4] ** 2 + S[:, 5] ** 2)
            u = self.U[l].reshape(-1, 3)
            pct = lambda v: float(cp.percentile(v, 99.9))
            out.append(dict(工况=nm, 最大位移_mm=float(cp.sqrt((u ** 2).sum(1)).max()),
                            面内位移_mm=float(cp.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2).max()), z向位移_mm=float(cp.abs(u[:, 2]).max()),
                            VM_99_9pct_MPa=pct(vm), 层间拉应力_99_9pct_MPa=pct(cp.maximum(S[:, 2], 0)),
                            层间剪应力_99_9pct_MPa=pct(tz), VM_max_MPa=float(vm.max()),
                            层间拉应力_max_MPa=float(S[:, 2].max()), 层间剪应力_max_MPa=float(tz.max())))
        return out


def fig_thickness(M, t, fn, title):
    tg = np.where(t > 0.05, t, np.nan).reshape(M.NX, M.NY).T
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(tg, origin="lower", extent=[0, M.NX * M.h, 0, M.NY * M.h], cmap="viridis", vmin=0, vmax=M.tmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label="厚度 (mm)")
    ax.set_aspect("equal"); ax.set_title(title)
    fig.savefig(fn, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=P.T3_MAX)
    ap.add_argument("--dilate", type=float, default=P.T3_DILATE)
    ap.add_argument("--tmin", type=float, default=P.T3_MIN)
    ap.add_argument("--kz", type=float, default=P.KZ)
    ap.add_argument("--rmin", type=float, default=3.0)
    ap.add_argument("--maxit", type=int, default=80)
    ap.add_argument("--move", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--nu_s", type=int, default=2)
    ap.add_argument("--omega", type=float, default=1.6, help="Jacobi 阻尼 = omega / λmax(D⁻¹A)")
    ap.add_argument("--model", choices=("print", "solid"), default="print", help="print：按墙+表皮+填充计算刚度和用料；solid：按实心")
    ap.add_argument("--nwall", type=int, default=P.NWALL)
    ap.add_argument("--nskin", type=int, default=P.NSKIN)
    ap.add_argument("--infill", type=float, default=P.INFILL)
    ap.add_argument("--out", default=f"{P.OUT_DIR}/topo3d")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    M = Model3D(a)
    print(f"网格 {M.NX}×{M.NY}×{M.NZ}（面内 {M.h:g} mm，层高 {M.dz:g} mm），体素 {M.NX*M.NY*M.NZ/1e6:.2f} M，自由度 {3*M.mg.levels[0].nn/1e6:.2f} M")
    print("多重网格:", M.mg.describe())
    print(f"柱子：二维形状 {M.c['shape2'].sum()}，膨胀后 {M.c['dil'].sum()}，设计 {M.c['design'].sum()}，可加厚非设计 {M.c['thick'].sum()}，接口 {M.c['joint'].sum()}")
    # 基准：二维结果按 T 拉伸；约束 = 与基准相同的用料（print）或体积（solid）
    t_base = np.where(M.c["shape2"], P.T, 0.0)
    target = M.mass(t_base)
    what = f"打印用料（{a.nwall} 圈墙、上下各 {a.nskin} 层、{a.infill*100:.0f}% 填充）" if M.pm else "体积"
    print(f"工况 {M.names}；{what}目标 {target/1000:.1f} cm³ 实心等效（二维结果拉伸 {P.T:g} mm）", flush=True)
    tb = time.time()
    comps_b, its_b, _ = M.analyze(t_base, tol=a.tol, with_sens=False)
    print(f"基准（二维拉伸）柔度 {np.round(comps_b, 5).tolist()}，CG 迭代 {its_b}，{time.time()-tb:.1f}s", flush=True)
    stress_b = M.stresses(*M.stress_fields(t_base))
    M.U = [None] * len(M.F)

    s = np.where(M.var, 1.0, 0.0)
    w = M.imp / comps_b                       # 用基准柔度归一化，加权柔度直接表示相对二维方案的比值
    hist = []
    for it in range(1, a.maxit + 1):
        ti = time.time()
        t, dt = M.heights(s)
        comps, its, dcdt = M.analyze(t, tol=a.tol, weights=w)
        c = float((w * comps).sum()) / w.dot(comps_b)
        dc = M.H.T @ (dcdt * dt / M.Hs)
        dv = M.H.T @ (M.dmass_dt(t) * np.where(M.var, dt, 0.0) / M.Hs)
        dc = np.minimum(dc, -1e-30); dv = np.maximum(dv, 1e-30)
        v = M.var
        sa = s[v]; lo = np.maximum(M.s_lo[v], sa - a.move); hi = np.minimum(1.0, sa + a.move)
        q = -dc[v] / dv[v]
        l1, l2 = 0.0, 1e12
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-6:
            lm = 0.5 * (l1 + l2)
            sn = np.clip(sa * np.sqrt(q / lm), lo, hi)
            st = s.copy(); st[v] = sn
            if M.mass(M.heights(st)[0]) > target: l1 = lm
            else: l2 = lm
        change = float(np.abs(sn - sa).max())
        s[v] = sn
        V = M.mass(t)
        hist.append(dict(it=it, c=c, vol=V / target, change=change, cg=its, comp=comps.tolist()))
        print(f"it {it:3d} 加权柔度/基准={c:.4f} 用料/目标={V/target:.3f} ch={change:.3f} CG={sum(its)} "
              f"{time.time()-ti:.1f}s t={time.time()-t0:.0f}s", flush=True)
        if it % 10 == 0:
            fig_thickness(M, t, f"{a.out}/thickness_it{it}.png", f"it={it} 加权柔度/基准={c:.3f}")
        if it >= 20 and change < 0.01:
            break

    # 取整到层：太薄的柱子去掉，其余不低于 T3_MIN
    t, _ = M.heights(s)
    tq = np.round(t / M.dz) * M.dz
    thin = M.c["design"] & (tq < M.tmin)
    tq[thin & (tq < M.tmin / 2)] = 0.0
    tq[thin & (tq >= M.tmin / 2)] = M.tmin
    comps_q, its_q, _ = M.analyze(tq, tol=a.tol, with_sens=False)
    stress_q = M.stresses(*M.stress_fields(tq))
    Vq = M.mass(tq)
    rel = comps_q / comps_b
    print(f"取整后：用料/目标 {Vq/target:.3f}，加权柔度/基准 {float((w*comps_q).sum())/w.dot(comps_b):.4f}")
    for k, nm in enumerate(M.names):
        print(f"  {nm}: 柔度 3D/2D = {rel[k]:.3f}")
    np.savez(f"{a.out}/result3d.npz", s=s, t=t, tq=tq, t_base=t_base, NX=M.NX, NY=M.NY, NZ=M.NZ, h=M.h, dz=M.dz,
             **{"col_" + k: v for k, v in M.c.items()}, comps_b=comps_b, comps_q=comps_q, names=np.array(M.names))
    with open(f"{a.out}/history3d.json", "w") as fh:
        json.dump(dict(args=vars(a), names=M.names, hist=hist, base=dict(comp=comps_b.tolist(), stress=stress_b),
                       final=dict(comp=comps_q.tolist(), rel=rel.tolist(), vol_ratio=Vq / target, stress=stress_q), target_mm3=target),
                  fh, ensure_ascii=False, indent=1)
    fig_thickness(M, tq, f"{a.out}/thickness_final.png", f"三维优化厚度分布（取整到层，{'按打印结构' if M.pm else '按实心'}）  用料/二维 = {Vq/target:.3f}")
    print(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
