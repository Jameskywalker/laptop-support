"""
无填充三维拓扑优化（试验）：直接在三维体素上优化（不从二维结果开始），用 AM 滤波约束悬空

思路：零件不用切片软件的稀疏填充，内部结构（墙、腹板、空腔和封顶的拱）全部由优化给出，模型里有材料的地方
都按实心打印。空腔可以封闭在内部（FDM 不用清粉），但每一层都要落在下一层上：体素 (i,j,k) 有材料时，
它正下方 (i,j,k-1) 及其四邻里至少要有一个有材料 → 悬空不超过 45°，空腔的顶自然收成尖拱。
- 网格：立方体素，边长 h = T/nz（默认 nz=32，h≈0.47 mm）；包络与二维方案相同（厚 T=15 mm，z=0 贴床 = 外侧面）。
- 设计变量：每个体素一个密度 → 三维密度过滤 → Heaviside 投影（β 续进）→ AM 滤波（Langelaar 2017，从床面逐层
  往上：ξ_k = smin(x̄_k, smax(ξ_{k-1} 在下方十字邻域)））→ 打印密度 ξ。支撑不住的材料不计刚度却照算用料，
  优化会自己把它去掉，或者在下面补上支撑。
- 材料：实心，横观各向同性（z 向模量、层间剪切模量 × KZ）；SIMP p=3。
- 固定区（全厚度）：顶面接触条、前挡后表面、前后脚垫，宽 --skin（实心）；横梁接口区（26×50 mm 去掉方槽）
  暂时保留原来的墙 + 上下表皮 + 稀疏填充，厚度 T、卡扣尺寸不变；方槽是贯通孔。
- 用料约束：与二维方案（profile.json 拉伸 T）按 3 圈墙 + 上下 4 层表皮 + 15% 填充打印的用料相同。
- 目标：多工况加权柔度（工况与权重同 topopt3d.py），用二维方案的柔度归一化，数值即相对二维方案的比值。
  面内工况（自重、按压、推、前挡冲击、误用）按装配状态算：面板和横梁拉住接口，卡钩搭接带和梁身台肩处 z 向不动；
  侧推、横梁工况仍按单片（接口自由）。--free-joints 全按单片；--warp κ 另加翘曲惩罚（默认不加）。
- 数值：GPU 多重网格 CG。单精度 kernel 先减去单元平均位移（去平移）再乘单元刚度，残差用双精度算并迭代修正，
  整体面外弯曲的工况才算得准。灵敏度（AM 滤波、z 向体力、翘曲项）都用有限差分核对过。
用法: STAND_VARIANT=desk40 python topopt3d_noinfill.py [--nz 24] [--rmin 1.4] [--init 粗网格/result.npz --beta0 4]
"""
import argparse, json, os, time
import numpy as np
import shapely
from shapely.geometry import Polygon, box
from scipy.ndimage import distance_transform_edt, label
import cupy as cp
import cupyx.scipy.ndimage as cnd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import topopt2d as T2          # 字体设置、project()
import topopt3d as T3          # 单元、GPU 多重网格

F32 = cp.float32
OUT = f"{P.OUT_DIR}/noinfill"
IMP = {"自重": 2.0, "前压": 1.0, "后压": 1.0, "后推": 0.5, "前推": 0.5, "前挡冲击": 1.0, "侧推": 1.0, "z向体力": 1.0,
       "侧推后段": 0.0, "前部误用150N": 0.0, "后部误用150N": 0.0}
WARP_CASES = ("自重", "前压", "后压", "后推", "前推", "前挡冲击")   # 面内载荷工况：量它们引起的顶面 z 向位移（翘曲）
# 按装配状态求解的工况（左右对称受力，面板和横梁拉住接口 z 向）；侧推、横梁工况、z 向体力仍是单片、接口自由
ASM_CASES = WARP_CASES + ("前部误用150N", "后部误用150N")
F_LAT = 10.0              # z 向体力的合力（按二维方案的用料折算单位体积的力）；柔度用基准归一，大小不影响结果


# ======================= 几何刚度（线性屈曲复核） =======================
_GSRC = r'''
extern "C" __global__ void KNAME(const REAL* __restrict__ u, REAL* __restrict__ out, const REAL* __restrict__ sig,
                                 const REAL* __restrict__ Mg, const float* __restrict__ fm, int nx, int ny, int nz) {
    __shared__ REAL M[384]; for (int t = threadIdx.x; t < 384; t += blockDim.x) M[t] = Mg[t]; __syncthreads();
    const int nyn = ny + 1, nzn = nz + 1;
    const long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)(nx + 1) * nyn * nzn) return;
    const int k = n % nzn; const long r_ = n / nzn; const int j = r_ % nyn; const int i = r_ / nyn;
    REAL o0 = 0, o1 = 0, o2 = 0;
    for (int a = 0; a < 8; a++) {
        const int ei = i - (a & 1), ej = j - ((a >> 1) & 1), ek = k - (a >> 2);
        if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
        const REAL* s = sig + 6 * (((long)ei * ny + ej) * nz + ek);
        const REAL s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4], s5 = s[5];
        if (s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 && s5 == 0) continue;
        for (int b = 0; b < 8; b++) {
            const int q = 8 * a + b;
            const REAL hab = s0 * M[q] + s1 * M[64 + q] + s2 * M[128 + q] + s3 * M[192 + q] + s4 * M[256 + q] + s5 * M[320 + q];
            const long m = 3 * (((long)(ei + (b & 1)) * nyn + (ej + ((b >> 1) & 1))) * nzn + (ek + (b >> 2)));
            o0 += hab * u[m]; o1 += hab * u[m + 1]; o2 += hab * u[m + 2];
        }
    }
    const long q3 = 3 * n;
    out[q3] = o0 * fm[q3]; out[q3 + 1] = o1 * fm[q3 + 1]; out[q3 + 2] = o2 * fm[q3 + 2];
}
'''
# 单精度 kernel 的"去平移"版本：单元位移先减去 8 个节点的平均值再乘单元刚度（KE、B 对刚体平移为零，结果不变）。
# 整体弯曲时位移场以平移为主，直接乘的话相消误差可达百分之几；去平移后误差降到 1e-5 量级
_CSRC = r'''
#define NODE_IJK(n) const int k = n % nzn; const long r_ = n / nzn; const int j = r_ % nyn; const int i = r_ / nyn;
#define LOAD_KE __shared__ float KE[576]; for (int t = threadIdx.x; t < 576; t += blockDim.x) KE[t] = KEg[t]; __syncthreads();
#define GATHER(ei, ej, ek) \
    float ue[24]; float m0 = 0.f, m1 = 0.f, m2 = 0.f; \
    for (int b = 0; b < 8; b++) { \
        const long m = 3 * (((long)(ei + (b & 1)) * nyn + (ej + ((b >> 1) & 1))) * nzn + (ek + (b >> 2))); \
        ue[3 * b] = u[m]; ue[3 * b + 1] = u[m + 1]; ue[3 * b + 2] = u[m + 2]; m0 += ue[3 * b]; m1 += ue[3 * b + 1]; m2 += ue[3 * b + 2]; } \
    m0 *= 0.125f; m1 *= 0.125f; m2 *= 0.125f; \
    for (int b = 0; b < 8; b++) { ue[3 * b] -= m0; ue[3 * b + 1] -= m1; ue[3 * b + 2] -= m2; }
extern "C" {
__global__ void kmulc(const float* __restrict__ u, float* __restrict__ out, const float* __restrict__ g,
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
        GATHER(ei, ej, ek)
        const float* Ka = KE + 72 * a;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        #pragma unroll
        for (int c = 0; c < 24; c++) { s0 += Ka[c] * ue[c]; s1 += Ka[24 + c] * ue[c]; s2 += Ka[48 + c] * ue[c]; }
        o0 += ge * s0; o1 += ge * s1; o2 += ge * s2;
    }
    const long q = 3 * n;
    out[q] = o0 * fm[q]; out[q + 1] = o1 * fm[q + 1]; out[q + 2] = o2 * fm[q + 2];
}
__global__ void kenerc(const float* __restrict__ u, float* __restrict__ ce, const float* __restrict__ KEg, float w, int nx, int ny, int nz) {
    LOAD_KE
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long)nx * ny * nz) return;
    const int k = e % nz; const long r_ = e / nz; const int j = r_ % ny; const int i = r_ / ny;
    const int nyn = ny + 1, nzn = nz + 1;
    GATHER(i, j, k)
    float s = 0.f;
    for (int p = 0; p < 24; p++) {
        float t = 0.f;
        #pragma unroll
        for (int q = 0; q < 24; q++) t += KE[24 * p + q] * ue[q];
        s += ue[p] * t;
    }
    ce[e] += w * s;
}
__global__ void kstressc(const float* __restrict__ u, float* __restrict__ sig, const float* __restrict__ g,
                         const float* __restrict__ DBg, int nx, int ny, int nz) {
    __shared__ float DB[144]; for (int t = threadIdx.x; t < 144; t += blockDim.x) DB[t] = DBg[t]; __syncthreads();
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long)nx * ny * nz) return;
    const int k = e % nz; const long r_ = e / nz; const int j = r_ % ny; const int i = r_ / ny;
    const int nyn = ny + 1, nzn = nz + 1;
    GATHER(i, j, k)
    for (int c = 0; c < 6; c++) {
        float s = 0.f;
        for (int q = 0; q < 24; q++) s += DB[24 * c + q] * ue[q];
        sig[6 * e + c] = g[e] * s;
    }
}
}
'''


def use_centered_kernels():
    """把 topopt3d 的单精度 kmul / kener / kstress 换成去平移版（同一进程内，MGCG 各层也跟着用）"""
    K = T3._K()
    if K.get("_centered"):
        return
    mod = cp.RawModule(code=_CSRC, options=("--use_fast_math",))
    K["kmul"], K["kener"], K["kstress"] = mod.get_function("kmulc"), mod.get_function("kenerc"), mod.get_function("kstressc")
    K["_centered"] = True


# 双精度矩阵乘：屈曲模态很光滑，单精度下 K·φ、K_G·φ 的相消误差能到百分之几，Rayleigh–Ritz 一步用双精度
_KSRC64 = r'''
extern "C" __global__ void kmul64(const double* __restrict__ u, double* __restrict__ out, const float* __restrict__ g,
                                  const double* __restrict__ KEg, const float* __restrict__ fm, int nx, int ny, int nz) {
    __shared__ double KE[576]; for (int t = threadIdx.x; t < 576; t += blockDim.x) KE[t] = KEg[t]; __syncthreads();
    const int nyn = ny + 1, nzn = nz + 1;
    const long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)(nx + 1) * nyn * nzn) return;
    const int k = n % nzn; const long r_ = n / nzn; const int j = r_ % nyn; const int i = r_ / nyn;
    double o0 = 0., o1 = 0., o2 = 0.;
    for (int a = 0; a < 8; a++) {
        const int ei = i - (a & 1), ej = j - ((a >> 1) & 1), ek = k - (a >> 2);
        if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
        const double ge = g[((long)ei * ny + ej) * nz + ek];
        const double* Ka = KE + 72 * a;
        double s0 = 0., s1 = 0., s2 = 0.;
        for (int b = 0; b < 8; b++) {
            const long m = 3 * (((long)(ei + (b & 1)) * nyn + (ej + ((b >> 1) & 1))) * nzn + (ek + (b >> 2)));
            const double u0 = u[m], u1 = u[m + 1], u2 = u[m + 2];
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
'''
_kern = {}


def _kfun(name):
    if name not in _kern:
        src = {"kgeo": "#define REAL float\n#define KNAME kgeo\n" + _GSRC,
               "kgeo64": "#define REAL double\n#define KNAME kgeo64\n" + _GSRC, "kmul64": _KSRC64}[name]
        _kern[name] = cp.RawModule(code=src).get_function(name)
    return _kern[name]


def geo_mats(h):
    """立方体单元的 ∫ ∂N_a/∂x_k ∂N_b/∂x_l dV（2×2×2 高斯点），按应力分量 xx, yy, zz, xy, yz, zx 组合（剪切项含 kl+lk）"""
    g = 1 / np.sqrt(3); sx, sy, sz = T3._SGN.T
    Mk = np.zeros((3, 3, 8, 8))
    for xi in (-g, g):
        for et in (-g, g):
            for ze in (-g, g):
                dN = np.array([sx * (1 + sy * et) * (1 + sz * ze) / 8 * 2 / h,
                               sy * (1 + sx * xi) * (1 + sz * ze) / 8 * 2 / h,
                               sz * (1 + sx * xi) * (1 + sy * et) / 8 * 2 / h])
                Mk += np.einsum("ka,lb->klab", dN, dN) * (h ** 3 / 8)
    out = [Mk[0, 0], Mk[1, 1], Mk[2, 2], Mk[0, 1] + Mk[1, 0], Mk[1, 2] + Mk[2, 1], Mk[2, 0] + Mk[0, 2]]
    return cp.asarray(np.stack(out).ravel())


# ======================= 打印结构（墙 + 表皮 + 填充） =======================
def print_columns(geom, NX, NY, h, wall, ss=5):
    """每根 h×h 柱子里：geom 覆盖的比例 cov、其中离轮廓 ≤ wall 的墙所占比例 fw（都相对柱子面积）。子网格 h/ss 上算距离"""
    hs = h / ss
    X, Y = np.meshgrid((np.arange(NX * ss) + 0.5) * hs, (np.arange(NY * ss) + 0.5) * hs, indexing="ij")
    ins = shapely.contains_xy(geom, X, Y)
    d = distance_transform_edt(np.pad(ins, 1))[1:-1, 1:-1] * hs - 0.5 * hs
    bm = lambda a: a.reshape(NX, ss, NY, ss).mean((1, 3))
    return bm(ins.astype(float)), bm((ins & (d <= wall)).astype(float))


def skin_fraction(NZ, dz, skin):
    """每层体素里上下表皮（各厚 skin）所占比例"""
    z0 = np.arange(NZ) * dz; z1 = z0 + dz
    ov = np.clip(np.minimum(z1, skin) - z0, 0, None) + np.clip(z1 - np.maximum(z0, P.T - skin), 0, None)
    return np.clip(ov / dz, 0, 1)


def print_voxels(cov, fw, fs, qE, qm):
    """墙 + 表皮 + 填充 → 体素刚度系数 E、用料系数 m（实心 = 1）"""
    fi = (cov - fw)[:, :, None]; fw = fw[:, :, None]; fs = fs[None, None, :]
    return fw + fi * (fs + (1 - fs) * qE), fw + fi * (fs + (1 - fs) * qm)


def profile2d():
    prof = json.load(open(f"{P.MODEL_DIR}/profile.json"))
    return Polygon(prof["exterior"], prof["holes"])


# ======================= 密度过滤 + 投影 + AM 滤波 =======================
def plus_sum(A):
    """十字邻域（自身 + 面内四邻）求和，越界按 0；邻域对称，所以也是自己的转置"""
    out = A.copy()
    out[1:] += A[:-1]; out[:-1] += A[1:]; out[:, 1:] += A[:, :-1]; out[:, :-1] += A[:, 1:]
    return out


class Pipeline:
    """x（设计变量）→ x̃（密度过滤）→ x̄（投影）→ ξ（AM 滤波后的打印密度）。全部 float64，放在显卡上。
    fixed：固定体素（实心或空）的掩码，fval 是它们的值；固定实心都是从床面起的整柱，天然有支撑。"""

    def __init__(self, shape, h, rmin, fixed, fval, p_am=40.0, eps=1e-6, xi0=0.5):
        R = int(np.ceil(rmin / h)) - 1
        o = np.arange(-R, R + 1)
        di, dj, dk = np.meshgrid(o, o, o, indexing="ij")
        K = np.maximum(0.0, rmin - h * np.sqrt(di ** 2 + dj ** 2 + dk ** 2))
        self.K = cp.asarray(K)
        self.Ks = cnd.correlate(cp.ones(shape), self.K, mode="constant", cval=0.0)   # 只按网格内的邻居归一（z 两面不吃料）
        self.fixed = cp.asarray(fixed); self.fval = cp.asarray(fval, cp.float64)
        self.free = ~self.fixed
        self.P = p_am; self.Q = p_am + np.log(5) / np.log(xi0); self.eps = eps

    def filt(self, x):
        return cnd.correlate(x, self.K, mode="constant", cval=0.0) / self.Ks

    def forward(self, x, beta):
        xf = cp.where(self.fixed, self.fval, x)
        xb, dxb = T2.project(self.filt(xf), float(beta), 0.5)
        xb = cp.where(self.fixed, self.fval, xb); dxb = cp.where(self.fixed, 0.0, dxb)
        xi, c = self.am_forward(xb)
        c["dxb"] = dxb
        return xi, c

    def am_forward(self, xb):
        Pp, Q, eps = self.P, self.Q, self.eps
        NZ = xb.shape[2]
        xi = cp.empty_like(xb); S = cp.zeros_like(xb); dsx = cp.zeros_like(xb); dsS = cp.zeros_like(xb)
        xi[:, :, 0] = xb[:, :, 0]; dsx[:, :, 0] = 1.0
        fr = self.free[:, :, 0]
        for k in range(1, NZ):
            Sk = cp.maximum(plus_sum(xi[:, :, k - 1] ** Pp), 1e-280)
            Sg = Sk ** (1.0 / Q)
            x = xb[:, :, k]; d = x - Sg; r = cp.sqrt(d * d + eps)
            v = 0.5 * (x + Sg - r + np.sqrt(eps))
            xi[:, :, k] = cp.where(fr, v, xb[:, :, k])
            S[:, :, k] = Sk
            dsx[:, :, k] = cp.where(fr, 0.5 * (1 - d / r), 0.0)
            dsS[:, :, k] = cp.where(fr, 0.5 * (1 + d / r), 0.0)
        dsx[:, :, 0] = cp.where(fr, 1.0, 0.0)
        return xi, dict(xi=xi, S=S, dsx=dsx, dsS=dsS)

    def am_backward(self, gxi, c):
        """dF/dξ → dF/dx̄（从顶层往下传）"""
        Pp, Q = self.P, self.Q
        g = gxi.copy(); gxb = cp.zeros_like(gxi)
        for k in range(gxi.shape[2] - 1, 0, -1):
            gk = g[:, :, k]
            gxb[:, :, k] = gk * c["dsx"][:, :, k]
            A = gk * c["dsS"][:, :, k] * (1.0 / Q) * c["S"][:, :, k] ** (1.0 / Q - 1.0)
            g[:, :, k - 1] += Pp * c["xi"][:, :, k - 1] ** (Pp - 1) * plus_sum(A)
        gxb[:, :, 0] = g[:, :, 0] * c["dsx"][:, :, 0]
        return gxb

    def backward(self, gxi, c):
        gt = self.am_backward(gxi, c) * c["dxb"]
        return cnd.correlate(gt / self.Ks, self.K, mode="constant", cval=0.0) * self.free


def am_binary(B):
    """二值模型的悬空检查/修整：从床面往上，没有支撑的体素去掉"""
    B = B.copy()
    for k in range(1, B.shape[2]):
        B[:, :, k] &= plus_sum(B[:, :, k - 1].astype(np.uint8)) > 0
    return B


# ======================= 模型 =======================
class Free3D:
    def __init__(self, nz, kz=P.KZ, skin=1.5, nwall=P.NWALL, nskin=P.NSKIN, infill=P.INFILL, misuse=False, lateral=True, warp=True, asm_z=True):
        self.lateral, self.warp, self.asm_z = lateral, warp, asm_z
        self.NZ = NZ = nz; self.h = h = P.T / nz; self.dz = h
        domain, solids = P.geometry(True)                       # 设计域、实体区都已经挖掉方槽
        mult, q = 1, nz                                         # 多重网格：z 能对半分几次，x、y 也要能分几次
        while q % 2 == 0 and q > 2:
            q //= 2; mult *= 2
        mult = max(mult, 2)
        self.NX = NX = int(-(-int(np.ceil(domain.bounds[2] / h)) // mult) * mult)
        self.NY = NY = int(-(-int(np.ceil(domain.bounds[3] / h)) // mult) * mult)
        I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
        xc, yc = (I + 0.5) * h, (J + 0.5) * h
        dom = shapely.contains_xy(domain, xc, yc)
        w = max(skin, 1.5 * h); self.skin_w = w
        regs = dict(top=P.sn_rect(-w, P.S_END, -w, 0.0), lip=P.sn_rect(-w, 0.0, 0.0, P.LIP_H),
                    feet=shapely.union(box(0, 0, P.FOOT_LEN, w), box(P.X_MAX - P.FOOT_LEN, 0, P.X_MAX, w)))
        jg = shapely.unary_union([solids["joint_" + j["name"]] for j in P.JOINTS])
        self.joint_geom = jg
        jnt = shapely.contains_xy(jg, xc, yc) & dom
        cols = {k: shapely.contains_xy(g, xc, yc) & dom & ~jnt for k, g in regs.items()}
        fixs = cols["top"] | cols["lip"] | cols["feet"]
        self.cols = dict(dom=dom, jnt=jnt, fixs=fixs, design=dom & ~fixs & ~jnt, **cols)
        self.fixed_regions = dict(regs, joint=jg)
        # 固定体素的刚度/用料系数：实心条 = 1；接口区 = 墙 + 表皮 + 填充（墙沿接口区外边和方槽）
        self.print_set = dict(nwall=nwall, nskin=nskin, infill=infill)
        self.fs = skin_fraction(NZ, h, nskin * P.LAYER_H)
        cov, fw = print_columns(jg, NX, NY, h, nwall * P.LINE_W)
        fwn = np.where(jnt, np.clip(fw / np.maximum(cov, 1e-9), 0, 1), 0.0)
        Ej, mj = print_voxels(jnt.astype(float), fwn, self.fs, infill ** 2, infill)
        f3 = np.broadcast_to(fixs[:, :, None], (NX, NY, NZ))
        self.E_fix = np.where(f3, 1.0, Ej); self.m_fix = np.where(f3, 1.0, mj)
        self.design3 = np.broadcast_to(self.cols["design"][:, :, None], (NX, NY, NZ))
        use_centered_kernels()
        self.D = T3.material_D(P.E_MAT, P.NU, kz); self.kz = kz
        self.gmin = 1e-6
        self._loads_bc(misuse)
        self.mg = T3.MGCG(NX, NY, NZ, h, h, self.D, cp.asarray(self.fm, F32))
        self.KE0 = self.mg.levels[0].KE
        self.fm_d = cp.asarray(self.fm, F32)
        # 面内工况按装配状态：面板 / 横梁拉住接口，接口 z 向不动（另一套约束、另一个求解器）
        self.mgs = [self.mg]; self.fms = [self.fm_d]
        if asm_z:
            self.fms.append(cp.asarray(self.fm_asm, F32))
            self.mgs.append(T3.MGCG(NX, NY, NZ, h, h, self.D, self.fms[1]))
        self.case_bc = [1 if (asm_z and nm in ASM_CASES) else 0 for nm in self.names]
        for l in range(len(self.F)):
            self.F[l] = self.F[l] * self.fms[self.case_bc[l]]
        self.lat = self.names.index("z向体力") if lateral else None
        self.q_lat = None

    # ---------- z 向体力 ----------
    def set_mass(self, m):
        """m：体素用料系数（实心 = 1）→ z 向体力（每个单元的力平分到 8 个节点）。第一次调用时定下单位体积的力"""
        if self.lat is None:
            return
        m = cp.asarray(m, F32)
        if self.q_lat is None:
            self.q_lat = F_LAT / (float(m.sum()) * self.h ** 3)
        e = m * F32(self.q_lat * self.h ** 3 / 8)
        nf = cp.zeros((self.NX + 1, self.NY + 1, self.NZ + 1), F32)
        for a in range(8):
            di, dj, dk = a & 1, (a >> 1) & 1, a >> 2
            nf[di:di + self.NX, dj:dj + self.NY, dk:dk + self.NZ] += e
        f = cp.zeros(3 * nf.size, F32); f[2::3] = nf.ravel()
        self.F[self.lat] = f * self.fm_d

    def body_sens(self):
        """d(F_lat·U_lat)/dm 中载荷随设计变化的那一项的一半：每个单元 8 个节点 z 向位移之和 × 单元力系数"""
        uz = self.U[self.lat][2::3].reshape(self.NX + 1, self.NY + 1, self.NZ + 1)
        s = cp.zeros((self.NX, self.NY, self.NZ), F32)
        for a in range(8):
            di, dj, dk = a & 1, (a >> 1) & 1, a >> 2
            s += uz[di:di + self.NX, dj:dj + self.NY, dk:dk + self.NZ]
        return s * F32(self.q_lat * self.h ** 3 / 8)

    # ---------- 翘曲 ----------
    def warp_measures(self):
        """面内工况 l 引起的顶面平均 z 位移：全长 W_a = L_side·U_l / F_side，后半段 W_r = L_rear·U_l / F_side（互易定理）"""
        ia, ir = self.names.index("侧推"), self.names.index("侧推后段")
        out = {}
        for nm in WARP_CASES:
            l = self.names.index(nm)
            Ul = self.U[l].astype(cp.float64)
            out[nm] = (float(cp.dot(self.F[ia].astype(cp.float64), Ul)) / P.F_SIDE, float(cp.dot(self.F[ir].astype(cp.float64), Ul)) / P.F_SIDE)
        return out

    def mutual_energy(self, a, b):
        """每个体素的 a_eᵀ KE b_e（用 kener 算 (a+b)、(a−b) 两次）"""
        L0 = self.mg.levels[0]
        ce = cp.zeros(L0.nel, F32)
        args = (self.KE0, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz))
        T3._K()["kener"](T3._blocks(L0.nel), (T3._TB,), (a + b, ce, args[0], np.float32(0.25), *args[1:]))
        T3._K()["kener"](T3._blocks(L0.nel), (T3._TB,), (a - b, ce, args[0], np.float32(-0.25), *args[1:]))
        return ce.reshape(self.NX, self.NY, self.NZ)

    # ---------- 线性屈曲 ----------
    def kgeo(self, sig, double=False, fm=None):
        """返回 v ↦ K_G v（sig：每个单元 6 个应力分量，为 0 的单元跳过）。double=True 时输入输出都是 float64"""
        Mg = geo_mats(self.h); L0 = self.mg.levels[0]; fm = self.fm_d if fm is None else fm
        if double:
            sig = sig.astype(cp.float64)
        else:
            Mg = Mg.astype(F32)
        kf = _kfun("kgeo64" if double else "kgeo")

        def apply(v):
            out = cp.empty_like(v)
            kf(T3._blocks(L0.nn), (T3._TB,), (v, out, sig, Mg, fm, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
            return out
        return apply

    def kmul64(self, fm=None):
        """当前刚度场的双精度 v ↦ K v（fm：约束掩码，默认单片约束）"""
        L0 = self.mg.levels[0]; KE = cp.asarray(L0.KE_np.ravel())
        kf = _kfun("kmul64"); fm = self.fm_d if fm is None else fm

        def apply(v):
            out = cp.empty_like(v)
            kf(T3._blocks(L0.nn), (T3._TB,), (v, out, L0.g, KE, fm, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
            return out
        return apply

    def element_stress(self, E, l, emin=0.01):
        """工况 l 的单元中心应力 E·D·B·u；刚度系数小于 emin 的单元（空）置 0，避免空区里的假屈曲模态"""
        L0 = self.mg.levels[0]
        DB = cp.asarray((self.D @ T3._hex_B(self.h, self.h, self.h, 0, 0, 0)).ravel(), F32)
        Eg = cp.asarray(E, F32).ravel()
        sig = cp.empty(6 * L0.nel, F32)
        T3._K()["kstress"](T3._blocks(L0.nel), (T3._TB,), (self.U[l], sig, Eg, DB, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
        sig = sig.reshape(-1, 6); sig[Eg < emin] = 0
        return sig.ravel()

    def buckling(self, E, l, nvec=6, iters=25, tol=1e-3, seed=0):
        """线性屈曲：K φ = λ (−K_G) φ 的最小正 λ（子空间迭代求 μ = 1/λ 的最大值）。需先 solve(E) 得到 U[l]。
        子空间用单精度多重网格求解生成，Rayleigh–Ritz 用双精度矩阵乘（Ritz 值是真实的 Rayleigh 商）"""
        sig = self.element_stress(E, l)
        b = self.case_bc[l]; fm, mg = self.fms[b], self.mgs[b]
        KG = self.kgeo(sig, fm=fm); KG64 = self.kgeo(sig, double=True, fm=fm); A64 = self.kmul64(fm)
        rng = np.random.default_rng(seed)
        n = self.F[0].size
        X = [cp.asarray(rng.standard_normal(n), F32) * fm for _ in range(nvec)]
        mu_old = None; shift = 0.0
        for it in range(iters):
            # 生成子空间用 K⁻¹(−K_G) + shift·I：负特征值（受拉为主的模态）绝对值更大时，平移让正的那端占优
            Z = [mg.solve(-KG(x), None, tol=1e-4)[0] + F32(shift) * x for x in X]
            G = cp.stack(Z)                                   # nvec × n（单精度存）
            Am = np.zeros((nvec, nvec)); Bm = np.zeros((nvec, nvec))
            for j in range(nvec):
                zj = Z[j].astype(cp.float64)
                ga, kz = -KG64(zj), A64(zj)
                for i in range(nvec):
                    zi = Z[i].astype(cp.float64)
                    Am[i, j] = float(cp.dot(zi, ga)); Bm[i, j] = float(cp.dot(zi, kz))
            Am = 0.5 * (Am + Am.T); Bm = 0.5 * (Bm + Bm.T)
            Lb = np.linalg.cholesky(Bm); Li = np.linalg.inv(Lb)
            mu, V = np.linalg.eigh(Li @ Am @ Li.T)
            C = Li.T @ V
            order = np.argsort(-np.abs(mu)); mu = mu[order]; C = C[:, order]
            X = list(cp.asarray(C.T, F32) @ G)
            X = [x / cp.linalg.norm(x) for x in X]
            pos = mu[mu > 0]
            shift = max(shift, 1.05 * float(-mu.min())) if mu.min() < 0 else shift
            if mu_old is not None and pos.size and abs(pos.max() - mu_old) < tol * abs(pos.max()):
                break
            mu_old = pos.max() if pos.size else None
        pos = mu[mu > 0]
        lam = 1.0 / pos.max() if pos.size else float("inf")
        k = int(np.argmax(np.where(mu > 0, mu, -np.inf)))
        return float(lam), X[k], it + 1

    def _loads_bc(self, misuse):
        NX, NY, NZ, h = self.NX, self.NY, self.NZ, self.h
        IN, JN = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing="ij")
        xn, yn = IN * h, JN * h

        def adj(c):
            s = np.pad(c, 1)
            return s[:-1, :-1] | s[1:, :-1] | s[:-1, 1:] | s[1:, 1:]
        C = self.cols
        s, n = P.xy2sn(xn, yn)
        top, lipback, jn = adj(C["top"]), adj(C["lip"]), adj(C["jnt"])
        nn = (NX + 1) * (NY + 1) * (NZ + 1)
        m3 = lambda m2: np.broadcast_to(m2[:, :, None], (NX + 1, NY + 1, NZ + 1)).ravel()
        F, names = [], []
        new = lambda: np.zeros(3 * nn)

        def add(Fv, m2, f):
            idx = np.nonzero(m3(m2))[0]; assert idx.size > 0
            for d in range(3):
                Fv[3 * idx + d] += f[d] / idx.size

        def press(Fv, s0, s1, Fm):
            fn = -Fm * np.cos(P.TH) * P.N; fu = -Fm * np.sin(P.TH) * P.U
            add(Fv, top & (s >= s0) & (s <= s1), (fn[0], fn[1], 0.0))
            add(Fv, lipback, (fu[0], fu[1], 0.0))
        f = new(); press(f, 5, P.S_END, 0.7 * P.W1); press(f, 180, P.S_END, 0.3 * P.W1); F.append(f); names.append("自重")
        f = new(); press(f, 10, 40, P.F_PRESS); F.append(f); names.append("前压")
        f = new(); press(f, 175, P.S_END, P.F_PRESS); F.append(f); names.append("后压")
        f = new(); add(f, top & (s >= 185) & (s <= P.S_END), (P.F_PUSH, 0, 0)); F.append(f); names.append("后推")
        f = new(); add(f, lipback, (-P.F_PUSH, 0, 0)); F.append(f); names.append("前推")
        f = new(); fl = -P.F_LIP_IMPACT * P.U; add(f, lipback, (fl[0], fl[1], 0)); F.append(f); names.append("前挡冲击")
        f = new(); add(f, top & (s >= 0) & (s <= P.S_END), (0, 0, P.F_SIDE)); F.append(f); names.append("侧推")
        if self.warp:      # 只作伴随：顶面后半段的 z 向力，用来量顶面后半段的平均 z 位移（翘曲）
            f = new(); add(f, top & (s >= P.S_END / 2) & (s <= P.S_END), (0, 0, P.F_SIDE)); F.append(f); names.append("侧推后段")
        if self.lateral:   # z 向体力：随材料分布（侧向磕碰 / 惯性），每轮按当前设计重新组装（set_mass）
            F.append(new()); names.append("z向体力")
        # 横梁：面内力经卡扣臂顶在受压一侧的方槽槽壁上（全厚度）；轴向力由卡钩扣在外侧面（z=0）的搭接带上
        for j in P.JOINTS:
            w2 = P.slot_bearing(j, xn, yn, P.F_JOINT, 0.75 * h) * jn
            w3 = m3(w2); assert w3.sum() > 0
            f = new()
            for d in range(2):
                f[d::3] += P.F_JOINT[d] * w3 / w3.sum()
            hk = np.zeros((NX + 1, NY + 1, NZ + 1), bool); hk[:, :, 0] = jn & P.hook_strip(j, xn, yn, 0.75 * h)
            idx = np.nonzero(hk.ravel())[0]; assert idx.size > 0
            f[3 * idx + 2] += P.F_JOINT_Z / idx.size
            F.append(f); names.append("横梁·" + j["cn"])
        if misuse:   # 与二维复核相同的误用工况：手肘撑在前部 / 转轴处 150 N，只看应力，不计入加权柔度
            f = new(); press(f, 10, 40, 150.0); F.append(f); names.append("前部误用150N")
            f = new(); press(f, 175, P.S_END, 150.0); F.append(f); names.append("后部误用150N")
        self.names = names
        self.imp = np.array([IMP.get(k, 0.5) for k in names])
        self.F = [cp.asarray(v, F32) for v in F]
        self.U = [None] * len(F)
        # 约束（与二维相同）：前脚只给竖向、后脚铰支；两脚 z 向靠摩擦约束
        fixed = np.zeros(3 * nn, bool)
        pad = adj(C["feet"] | C["jnt"]) & (JN == 0)
        for m2, how in ((pad & (xn <= P.FOOT_LEN + 1e-9), P.FRONT_FIX),
                        (pad & (xn >= P.X_MAX - P.FOOT_LEN - 1e-9) & (xn <= P.X_MAX + 1e-9), P.REAR_FIX)):
            idx = np.nonzero(m3(m2))[0]; assert idx.size > 0
            for d, ch in enumerate("xy"):
                if ch in how: fixed[3 * idx + d] = True
            fixed[3 * idx + 2] = True
        self.fm = (~fixed).astype(np.float32)
        for v in self.F:
            v *= cp.asarray(self.fm)
        # 装配约束：卡钩搭接带（外侧面 z=0）和梁身台肩（内侧面 z=T，梁身截面内、方槽外）上的节点 z 向固定
        fa = fixed.reshape(NX + 1, NY + 1, NZ + 1, 3).copy()
        for j in P.JOINTS:
            hk = jn & P.hook_strip(j, xn, yn, 0.75 * h)
            sh = jn & shapely.contains_xy(P.joint_rect(P.BEAM_W, P.BEAM_H, j), xn, yn) & ~shapely.contains_xy(P.joint_slot(j), xn, yn)
            fa[hk, 0, 2] = True; fa[sh, NZ, 2] = True
        self.fm_asm = (~fa.ravel()).astype(np.float32)

    # ---------- 有限元 ----------
    def solve(self, E, tol=1e-5, inner=1e-2):
        """E：体素刚度系数（0..1，形状 NX×NY×NZ）→ 各工况柔度"""
        g = (F32(self.gmin) + F32(1 - self.gmin) * cp.asarray(E, F32)).ravel()
        for mg in self.mgs:
            mg.update(g)
        A64s = [self.kmul64(fm) for fm in self.fms]
        comps = np.zeros(len(self.F)); its = []
        for l, Fv in enumerate(self.F):
            mg, A64 = self.mgs[self.case_bc[l]], A64s[self.case_bc[l]]
            # 混合精度迭代修正：残差用双精度算，修正量用单精度多重网格 CG 解。整体面外弯曲（侧推、z 向体力）的位移场很光滑，
            # 单精度矩阵乘相消误差大，只用单精度时 CG 自报收敛、实际残差可达百分之几
            b = Fv.astype(cp.float64); bn = float(cp.linalg.norm(b))
            x = cp.zeros_like(b) if self.U[l] is None else self.U[l].astype(cp.float64)
            it = 0
            for _ in range(12):
                r = b - A64(x)
                if float(cp.linalg.norm(r)) <= tol * bn:
                    break
                d, k = mg.solve(r.astype(F32), None, tol=inner)
                x += d; it += k
            self.U[l] = x.astype(F32)
            comps[l] = float(cp.dot(Fv.astype(cp.float64), self.U[l].astype(cp.float64))); its.append(it)   # 体力工况遍布全部节点，单精度累加误差大
        return comps, its

    def energy(self, w):
        """Σ_l w_l u_e^T KE u_e（每个体素）"""
        L0 = self.mg.levels[0]
        ce = cp.zeros(L0.nel, F32)
        for l in range(len(self.F)):
            T3._K()["kener"](T3._blocks(L0.nel), (T3._TB,), (self.U[l], ce, self.KE0, np.float32(w[l]), np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
        return ce.reshape(self.NX, self.NY, self.NZ)

    def stresses(self, mask):
        """各工况实心材料里的应力（D·B·u，体素中心），只统计 mask 内的体素；另给位移"""
        L0 = self.mg.levels[0]
        DB = cp.asarray((self.D @ T3._hex_B(self.h, self.h, self.h, 0, 0, 0)).ravel(), F32)
        sig = cp.empty(6 * L0.nel, F32); ones = cp.ones(L0.nel, F32)
        mk = cp.asarray(mask.ravel())
        out = []
        for l, nm in enumerate(self.names):
            T3._K()["kstress"](T3._blocks(L0.nel), (T3._TB,), (self.U[l], sig, ones, DB, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
            S = sig.reshape(-1, 6)[mk]
            vm = cp.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2) + 3 * (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2))
            tz = cp.sqrt(S[:, 4] ** 2 + S[:, 5] ** 2)
            u = self.U[l].reshape(-1, 3)
            pct = lambda v: float(cp.percentile(v, 99.9))
            out.append(dict(工况=nm, 最大位移_mm=float(cp.sqrt((u ** 2).sum(1)).max()),
                            面内位移_mm=float(cp.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2).max()), z向位移_mm=float(cp.abs(u[:, 2]).max()),
                            VM_99_9pct_MPa=pct(vm), VM_max_MPa=float(vm.max()),
                            层间拉应力_99_9pct_MPa=pct(cp.maximum(S[:, 2], 0)), 层间剪应力_99_9pct_MPa=pct(tz)))
        return out

    # ---------- 用料 ----------
    def mass(self, xi):
        """实心等效体积 mm³：设计体素按 ξ，固定区按各自系数"""
        return (float(cp.asarray(xi)[cp.asarray(self.design3)].sum()) + float(self.m_fix.sum())) * self.h ** 3

    # ---------- 对比用的体素场 ----------
    def baseline_fields(self, geom=None):
        """二维方案（profile.json 拉伸 T）按墙 + 表皮 + 填充打印 → (E, m, 实心掩码)"""
        geom = geom or profile2d()
        ps = self.print_set
        cov, fw = print_columns(geom, self.NX, self.NY, self.h, ps["nwall"] * P.LINE_W)
        E, m = print_voxels(cov, fw, self.fs, ps["infill"] ** 2, ps["infill"])
        solid = (fw[:, :, None] + (cov - fw)[:, :, None] * self.fs[None, None, :]) > 0.5
        return E, m, solid

    def design_fields(self, xi, p=3.0):
        """优化结果（ξ 或二值）+ 固定区 → (E, m, 实心掩码)"""
        xi = np.asarray(xi, float)
        d3 = self.design3
        E = np.where(d3, xi ** p, self.E_fix); m = np.where(d3, xi, self.m_fix)
        solid = np.where(d3, xi > 0.5, (self.E_fix > 0.5))
        return E, m, solid


# ======================= 图 =======================
def fig_layers(M, xi, fn, title):
    ks = np.unique(np.linspace(0, M.NZ - 1, 8).round().astype(int))
    fig, axs = plt.subplots(2, 4, figsize=(22, 7.5))
    for ax, k in zip(axs.ravel(), ks):
        ax.imshow(1 - xi[:, :, k].T, cmap="gray", origin="lower", extent=[0, M.NX * M.h, 0, M.NY * M.h], vmin=0, vmax=1)
        ax.set_title(f"第 {k+1} 层体素  z = {k*M.h:.1f}–{(k+1)*M.h:.1f} mm"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.savefig(fn, dpi=90, bbox_inches="tight"); plt.close(fig)


def fig_sections(M, xi, fn, title, xs=(30, 60, 95, 130, 165, 200, 235)):
    """竖直剖面（x = 常数，横轴 y、纵轴 z = 打印方向）：看杆件截面的空腔和封顶"""
    fig, axs = plt.subplots(len(xs), 1, figsize=(16, 1.35 * len(xs) + 1))
    for ax, x in zip(axs, xs):
        i = min(M.NX - 1, int(x / M.h))
        ax.imshow(1 - xi[i].T, cmap="gray", origin="lower", extent=[0, M.NY * M.h, 0, M.NZ * M.h], vmin=0, vmax=1, interpolation="nearest")
        ax.set_ylabel(f"x={x}", rotation=0, labelpad=22); ax.set_yticks([0, P.T]); ax.set_aspect("equal")
    axs[-1].set_xlabel("y (mm)    剖面纵轴为打印方向 z（下面贴床）")
    fig.suptitle(title, fontsize=13)
    fig.savefig(fn, dpi=100, bbox_inches="tight"); plt.close(fig)


# ======================= 优化 =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=32, help="厚度方向体素数（立方体素，h = T/nz）")
    ap.add_argument("--rmin", type=float, default=1.2, help="密度过滤半径 mm")
    ap.add_argument("--skin", type=float, default=1.5, help="顶面、前挡后表面、脚垫固定实心条的宽度 mm")
    ap.add_argument("--kz", type=float, default=P.KZ)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--maxit", type=int, default=220)
    ap.add_argument("--beta_max", type=float, default=16.0)
    ap.add_argument("--beta_every", type=int, default=40)
    ap.add_argument("--move", type=float, default=0.1)
    ap.add_argument("--mass_scale", type=float, default=1.0, help="用料目标 = 二维方案打印用料 × 此值")
    ap.add_argument("--tol", type=float, default=1e-4, help="求解相对残差（双精度残差）")
    ap.add_argument("--lateral", action="store_true", help="优化时加 z 向体力工况（随材料分布的侧向力）")
    ap.add_argument("--warp", type=float, default=0.0, help="翘曲惩罚系数 κ：κ·Σ imp·(顶面平均 z 位移 / 同一设计沿载荷方向的平均位移)²；0 = 不惩罚")
    ap.add_argument("--free-joints", dest="asm_z", action="store_false", help="面内工况也按单片计算（接口 z 向自由），默认按装配状态约束接口 z 向")
    ap.add_argument("--init", default=None, help="从另一次结果（result.npz，可以是粗网格）插值出初始设计")
    ap.add_argument("--beta0", type=float, default=1.0, help="起始 β（从粗网格结果接着算时可以直接从 4 开始）")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    M = Free3D(a.nz, kz=a.kz, skin=a.skin, lateral=a.lateral, warp=a.warp > 0, asm_z=a.asm_z)
    NX, NY, NZ, h = M.NX, M.NY, M.NZ, M.h
    nd = int(M.design3.sum())
    print(f"网格 {NX}×{NY}×{NZ}（h={h:.4f} mm），体素 {NX*NY*NZ/1e6:.2f} M，自由度 {3*M.mg.levels[0].nn/1e6:.2f} M；设计体素 {nd/1e6:.2f} M", flush=True)
    print("多重网格:", M.mg.describe())
    E2, m2, _ = M.baseline_fields()
    target = float(m2.sum()) * h ** 3 * a.mass_scale
    fixm = float(M.m_fix.sum()) * h ** 3
    vf = (target - fixm) / (nd * h ** 3)
    print(f"用料目标 {target/1000:.2f} cm³ 实心等效（二维方案 × {a.mass_scale:g}；PETG 约 {target/1000*1.27:.0f} g）；"
          f"固定区 {fixm/1000:.2f} cm³（接口区 {float(M.m_fix[M.cols['jnt']].sum())*h**3/1000:.2f}）；设计区平均密度 {vf:.4f}", flush=True)
    tb = time.time()
    M.set_mass(m2)
    comps_b, its_b = M.solve(E2, a.tol)
    print(f"基准（二维方案按打印结构）柔度 {np.round(comps_b, 5).tolist()}，CG {its_b}，{time.time()-tb:.1f}s", flush=True)
    fres = {nm: float(np.linalg.norm(cp.asnumpy(M.F[M.names.index(nm)].reshape(-1, 3).sum(0)).astype(float))) for nm in WARP_CASES}
    if M.warp:
        wb = M.warp_measures()
        print("基准翘曲比（顶面平均 z 位移 / 沿载荷方向平均位移 c/|F|，全长、后半段）:",
              {k: (round(v[0] * fres[k] / comps_b[M.names.index(k)], 4), round(v[1] * fres[k] / comps_b[M.names.index(k)], 4)) for k, v in wb.items()}, flush=True)
    M.U = [None] * len(M.F)

    fixed = ~M.design3
    fval = np.where(M.E_fix > 0, 1.0, 0.0)                      # 固定实心（含接口区）算作有材料，可以支撑上层
    pipe = Pipeline((NX, NY, NZ), h, a.rmin, fixed, fval)
    d3 = cp.asarray(M.design3)
    w = M.imp / comps_b
    x = cp.where(d3, vf, 0.0)
    if a.init:
        from scipy.ndimage import map_coordinates
        r0 = np.load(a.init); xc, hc = r0["x"].astype(float), float(r0["h"])
        I, J, K = np.meshgrid(*[(np.arange(n) + 0.5) * h / hc - 0.5 for n in (NX, NY, NZ)], indexing="ij")
        x0 = map_coordinates(xc, [I, J, K], order=1, mode="nearest")
        x = cp.where(d3, cp.asarray(np.clip(x0, 0, 1)), 0.0)
        print(f"初始设计取自 {a.init}（网格 {xc.shape}，h={hc:.4f}）", flush=True)
    beta, lb = a.beta0, 0
    hist = []
    for it in range(1, a.maxit + 1):
        ti = time.time()
        xi, cache = pipe.forward(x, beta)
        Ef = cp.where(d3, xi ** a.penal, cp.asarray(M.E_fix))
        M.set_mass(cp.where(d3, xi, cp.asarray(M.m_fix)))
        comps, its = M.solve(Ef, a.tol)
        c = float((w * comps).sum() / w.dot(comps_b))
        # 翘曲比 r = W·|F|/c（W：顶面平均 z 位移）。惩罚 κ·imp·(r_a² + r_r²) 对 c 的那部分导数折进柔度权重：
        # ∂/∂c [k·W²/c²] = −2k·W²/c³，而 dc/dξ 与柔度项同样来自 u_lᵀ KE u_l
        wE = w.copy(); pen = 0.0
        if M.warp:
            wm = M.warp_measures()
            Aa = cp.zeros_like(M.U[0]); Ar = cp.zeros_like(M.U[0])
            for nm, (Wa, Wr) in wm.items():
                l = M.names.index(nm); k = a.warp * M.imp[l] * fres[nm] ** 2 / comps[l] ** 2
                pen += k * (Wa ** 2 + Wr ** 2)
                wE[l] -= 2 * k * (Wa ** 2 + Wr ** 2) / comps[l]
                Aa += F32(2 * k * Wa / P.F_SIDE) * M.U[l]; Ar += F32(2 * k * Wr / P.F_SIDE) * M.U[l]
            c = (float((w * comps).sum()) + pen) / w.dot(comps_b)
        ce = M.energy(wE).astype(cp.float64)
        gE = -ce * (1 - M.gmin) * a.penal * xi ** (a.penal - 1) * d3
        if M.lat is not None:                                     # 体力随设计变化：d(F·U)/dξ 多一项 2·Uᵀ·dF/dξ
            gE = gE + 2 * w[M.lat] * M.body_sens().astype(cp.float64) * d3
        if M.warp:
            # dW_l/dξ = −(1/F_side)·(1−gmin)·p·ξ^(p−1)·u_sideᵀ KE u_l（互易定理：伴随就是侧推、侧推后段的解）
            mut = M.mutual_energy(M.U[M.names.index("侧推")], Aa) + M.mutual_energy(M.U[M.names.index("侧推后段")], Ar)
            gE = gE - mut.astype(cp.float64) * (1 - M.gmin) * a.penal * xi ** (a.penal - 1) * d3
        dc = pipe.backward(gE, cache)
        dm = pipe.backward(d3 * h ** 3, cache)
        dm = cp.maximum(dm, 1e-3 * float(dm[d3].mean()))          # 支撑不住的体素 dm≈0：给个下限，让它们往下走
        dc = cp.minimum(dc, 0.0)
        xa = x[d3]; q = -dc[d3] / dm[d3]
        lo = cp.maximum(0.0, xa - a.move); hi = cp.minimum(1.0, xa + a.move)
        l1, l2 = 0.0, float(q.max()) * 4 + 1e-30
        for _ in range(60):
            lm = 0.5 * (l1 + l2)
            xn = cp.clip(xa * cp.sqrt(q / lm), lo, hi)
            xt = x.copy(); xt[d3] = xn
            if M.mass(pipe.forward(xt, beta)[0]) > target: l1 = lm
            else: l2 = lm
            if (l2 - l1) / (l1 + l2) < 1e-5: break
        change = float(cp.abs(xn - xa).max())
        x[d3] = xn
        mratio = M.mass(xi) / target
        grey = float((4 * xi * (1 - xi))[d3].mean())
        hist.append(dict(it=it, c=c, pen=pen / w.dot(comps_b), mass=mratio, change=change, beta=beta, grey=grey, cg=its, comp=comps.tolist()))
        print(f"it {it:3d} 目标/二维={c:.4f}（翘曲项 {pen / w.dot(comps_b):.4f}）用料/目标={mratio:.3f} ch={change:.3f} β={beta:g} 灰度={grey:.3f} "
              f"CG={sum(its)} {time.time()-ti:.1f}s t={time.time()-t0:.0f}s", flush=True)
        if it % 20 == 0:
            fig_layers(M, cp.asnumpy(xi), f"{a.out}/layers_it{it}.png", f"it={it} β={beta:g} 加权柔度/二维={c:.3f}")
        lb += 1
        if beta < a.beta_max and (lb >= a.beta_every or (change < 0.01 and lb >= 15)):
            beta *= 2; lb = 0
        elif beta >= a.beta_max and lb >= 30 and (change < 0.01 or abs(hist[-11]["c"] - c) / c < 1e-3):
            break
    xi, _ = pipe.forward(x, beta)
    xi = cp.asnumpy(xi)
    np.savez_compressed(f"{a.out}/result.npz", x=cp.asnumpy(x).astype(np.float32), xi=xi.astype(np.float32), NX=NX, NY=NY, NZ=NZ, h=h,
                        comps_b=comps_b, names=np.array(M.names), target=target, **{"col_" + k: v for k, v in M.cols.items()})
    json.dump(dict(args=vars(a), names=M.names, hist=hist, base=dict(comp=comps_b.tolist()), target_mm3=target),
              open(f"{a.out}/history.json", "w"), ensure_ascii=False, indent=1)
    fig_layers(M, xi, f"{a.out}/layers_final.png", f"最终 β={beta:g}")
    fig_sections(M, xi, f"{a.out}/sections_final.png", "剖面（x = 常数）")
    print(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
