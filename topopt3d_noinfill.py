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
# 屈曲灵敏度用：kgq —— 每个单元 6 个二次型 g_ek = Σ_ab M_k[a,b]·(φ_a·φ_b)（φᵀK_Gφ = Σ_e σ_e·g_e），带权累加；
# kstrT —— 单元上的 6 个应力"对偶量" t_e 经 (D·B)ᵀ 组装成节点力（σ_e = D·B·u_e 对 u 的转置）
_BSRC = _CSRC[:_CSRC.index('extern "C"')] + r'''
extern "C" {
__global__ void kgq(const float* __restrict__ u, float* __restrict__ out, const float* __restrict__ Mg, float w, int nx, int ny, int nz) {
    __shared__ float M[384]; for (int t = threadIdx.x; t < 384; t += blockDim.x) M[t] = Mg[t]; __syncthreads();
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long)nx * ny * nz) return;
    const int k = e % nz; const long r_ = e / nz; const int j = r_ % ny; const int i = r_ / ny;
    const int nyn = ny + 1, nzn = nz + 1;
    GATHER(i, j, k)
    float s[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int a = 0; a < 8; a++)
        for (int b = 0; b < 8; b++) {
            const float d = ue[3 * a] * ue[3 * b] + ue[3 * a + 1] * ue[3 * b + 1] + ue[3 * a + 2] * ue[3 * b + 2];
            const int q = 8 * a + b;
            #pragma unroll
            for (int c = 0; c < 6; c++) s[c] += M[64 * c + q] * d;
        }
    for (int c = 0; c < 6; c++) out[6 * e + c] += w * s[c];
}
__global__ void kstrT(const float* __restrict__ t, float* __restrict__ out, const float* __restrict__ DBg, const float* __restrict__ fm,
                      int nx, int ny, int nz) {
    __shared__ float DB[144]; for (int q = threadIdx.x; q < 144; q += blockDim.x) DB[q] = DBg[q]; __syncthreads();
    const int nyn = ny + 1, nzn = nz + 1;
    const long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)(nx + 1) * nyn * nzn) return;
    NODE_IJK(n)
    float o0 = 0.f, o1 = 0.f, o2 = 0.f;
    for (int a = 0; a < 8; a++) {
        const int ei = i - (a & 1), ej = j - ((a >> 1) & 1), ek = k - (a >> 2);
        if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
        const float* te = t + 6 * (((long)ei * ny + ej) * nz + ek);
        for (int c = 0; c < 6; c++) {
            const float tc = te[c];
            o0 += DB[24 * c + 3 * a] * tc; o1 += DB[24 * c + 3 * a + 1] * tc; o2 += DB[24 * c + 3 * a + 2] * tc;
        }
    }
    const long q3 = 3 * n;
    out[q3] = o0 * fm[q3]; out[q3 + 1] = o1 * fm[q3 + 1]; out[q3 + 2] = o2 * fm[q3 + 2];
}
}
'''
_kern = {}


def _kfun(name):
    if name not in _kern:
        src = {"kgeo": "#define REAL float\n#define KNAME kgeo\n" + _GSRC,
               "kgeo64": "#define REAL double\n#define KNAME kgeo64\n" + _GSRC, "kmul64": _KSRC64,
               "kgq": _BSRC, "kstrT": _BSRC}[name]
        _kern[name] = cp.RawModule(code=src).get_function(name)
    return _kern[name]


def geo_mats(h, hy=None, hz=None):
    """长方体单元（h × hy × hz，默认立方）的 ∫ ∂N_a/∂x_k ∂N_b/∂x_l dV（2×2×2 高斯点），按应力分量 xx, yy, zz, xy, yz, zx 组合（剪切项含 kl+lk）"""
    hy = hy or h; hz = hz or h
    g = 1 / np.sqrt(3); sx, sy, sz = T3._SGN.T
    Mk = np.zeros((3, 3, 8, 8))
    for xi in (-g, g):
        for et in (-g, g):
            for ze in (-g, g):
                dN = np.array([sx * (1 + sy * et) * (1 + sz * ze) / 8 * 2 / h,
                               sy * (1 + sx * xi) * (1 + sz * ze) / 8 * 2 / hy,
                               sz * (1 + sx * xi) * (1 + sy * et) / 8 * 2 / hz])
                Mk += np.einsum("ka,lb->klab", dN, dN) * (h * hy * hz / 8)
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


def am_stencil(h, dz, angle=45.0):
    """悬空角 angle（允许的最缓斜面与床面的夹角，度；45 = 常规 45° 规则）→ (m, 面内支撑核)。
    每层允许外扩 g = dz/tan(angle) mm，即 g/h 格：
    g/h ≤ 1（陡）：十字核，每层相对下面 1..m 层都最多外扩一格，m = round(h·tan(angle)/dz)，平均每 m 层外扩一格；
    g/h > 1（缓）：m = 1，核取半径 g/h 格的圆盘（下一层这个圆盘里有材料就算托住）。核为 None 表示十字邻域"""
    gc = dz / np.tan(np.deg2rad(angle)) / h
    if gc <= 1.0 + 1e-9:
        return max(1, int(round(1.0 / gc))), None
    R = int(np.floor(gc + 1e-9)); o = np.arange(-R, R + 1)
    di, dj = np.meshgrid(o, o, indexing="ij")
    return 1, (di ** 2 + dj ** 2 <= gc ** 2 + 1e-9)


class SupportEst:
    """支撑用料估算（可微，单位 mm³）。第 k 层超出下一层支撑邻域的部分算悬空：u = x_k − smin(x_k, S_k)，
    S_k = (邻域 Σ x_{k−1}^P)^{1/Q}，邻域按切片软件的支撑阈值角 angle（与床面夹角，见 am_stencil）。
    每 mm² 悬空要的树状支撑约 c·z^e mm³（z = 悬空面离床面的高度）。e = 0.47 来自 Bambu 切片的高度拟合；c = 2.0 让不限悬空那版
    的估算（体素上）对上 Bambu 实切的 11.2 cm³（阈值 30°、树状支撑）。短桥下的支撑体素上看不到（斜面版实切 1.6 cm³，估算 ≈ 0）。
    xi0 取得低（0.1）：灰度阶段均匀的低密度算"托得住"，不会凭空算出一大片支撑"""

    def __init__(self, h, dz, NT, angle=30.0, P=40.0, xi0=0.1, eps=1e-8, c=2.0, e=0.47, r_mm=0.0, pd=16.0):
        """r_mm > 0：悬空先在面内膨胀 r_mm（p-范数近似的最大值，pd 次）再计面积 —— 树状支撑每个悬空点都要一根枝，
        按面积计费时优化器会把悬空打碎成很多一格的小点（面积小、实切支撑反而多），膨胀后小点不再便宜"""
        m, ker = am_stencil(h, dz, angle)
        self.r, self.pd = r_mm, pd
        if r_mm > 0:
            R = int(np.floor(r_mm / h + 1e-9)); o = np.arange(-R, R + 1); di, dj = np.meshgrid(o, o, indexing="ij")
            self.disk = cp.asarray((((di * di + dj * dj) * h * h) <= r_mm * r_mm + 1e-9).astype(np.float64)[:, :, None])
        self.ker = None if ker is None else cp.asarray(ker[:, :, None].astype(np.float64))
        nst = 5 if ker is None else int(ker.sum())
        self.P, self.Q, self.eps, self.NT = P, P + np.log(nst) / np.log(xi0), eps, NT
        z = np.arange(1, NT) * dz
        self.w = cp.asarray(c * h * h * z ** e)[None, None, :]

    def _ss(self, A):
        return plus_sum(A) if self.ker is None else cnd.correlate(A, self.ker, mode="constant", cval=0.0)

    def __call__(self, x):
        """x：打印设计（体素密度，第三维是层），返回 (支撑体积 mm³, ∂/∂x)"""
        NT, Pp, Q = self.NT, self.P, self.Q
        lo, up = x[:, :, :NT - 1], x[:, :, 1:NT]
        Ssum = cp.maximum(self._ss(lo ** Pp), 1e-280); S = Ssum ** (1.0 / Q)
        d = up - S; r = cp.sqrt(d * d + self.eps)
        u = 0.5 * (d + r - np.sqrt(self.eps))
        du = 0.5 * (1 + d / r)
        if self.r > 0:
            up_ = cp.maximum(u, 0.0); pd = self.pd
            Tsum = cp.maximum(cnd.correlate(up_ ** pd, self.disk, mode="constant", cval=0.0), 1e-300)
            V = float((self.w * Tsum ** (1.0 / pd)).sum())
            gu = up_ ** (pd - 1) * cnd.correlate(self.w * Tsum ** (1.0 / pd - 1.0), self.disk, mode="constant", cval=0.0)
        else:
            V = float((self.w * u).sum()); gu = self.w
        g = cp.zeros_like(x)
        g[:, :, 1:NT] += gu * du
        g[:, :, :NT - 1] += Pp * lo ** (Pp - 1) * self._ss(-gu * du * (1.0 / Q) * Ssum ** (1.0 / Q - 1.0))
        return V, g


class Pipeline:
    """x（设计变量）→ x̃（密度过滤）→ x̄（投影）→ ξ（AM 滤波后的打印密度）。全部 float64，放在显卡上。
    fixed：固定体素（实心或空）的掩码，fval 是它们的值；固定实心都是从床面起的整柱，天然有支撑。
    最小厚度（eta_e 不为 None）：同一个过滤场再按更高的阈值 η_e 投影，得到"腐蚀"设计 ξ_e = ξ·x̄_e（乘上 ξ 带上支撑情况）。
    刚度、屈曲都按 ξ_e 算、用料按 ξ 算：比最小厚度薄的杆件 / 板腐蚀后就没了，只吃料不出力，优化自然不要它们。
    pad_void：过滤时网格外（床面以下、顶面以上）按空，不按网格内的邻居归一 —— 否则贴床的板按镜像算，厚度只要求一半。"""

    def __init__(self, shape, h, rmin, fixed, fval, p_am=40.0, eps=1e-6, xi0=0.5, pad_void=False, eta_e=None, dz=None, no_am=False, am_angle=45.0, ground=False, ground_r=0.0, island_area=0.0, island_beta=0.0, island_soft=0.0, island_connect=False):
        dz = dz or h
        R, Rz = int(np.ceil(rmin / h)) - 1, int(np.ceil(rmin / dz)) - 1
        o, oz = np.arange(-R, R + 1), np.arange(-Rz, Rz + 1)
        di, dj, dk = np.meshgrid(o, o, oz, indexing="ij")
        K = np.maximum(0.0, rmin - np.sqrt((h * di) ** 2 + (h * dj) ** 2 + (dz * dk) ** 2))     # 按物理距离的锥形核（体素扁时是椭球）
        self.K = cp.asarray(K)
        if pad_void:
            self.Ks = float(K.sum())
        else:
            self.Ks = cnd.correlate(cp.ones(shape), self.K, mode="constant", cval=0.0)   # 只按网格内的邻居归一（z 两面不吃料）
        self.eta_e = eta_e
        self.fixed = cp.asarray(fixed); self.fval = cp.asarray(fval, cp.float64)
        self.free = ~self.fixed
        # 悬空：每层最多外扩 dz/tan(angle)（mm），见 am_stencil。45°、体素宽 h = m·dz 时，每层相对下面 1..m 层都最多外扩一格
        # （十字邻域）：平均每 m 层外扩一格 = 45°，单层最多伸出一格。m = 1（立方体素）就是原来的做法
        self.am_angle = am_angle
        self.m_am, ker = am_stencil(h, dz, am_angle)
        self.am_ker = None if ker is None else cp.asarray(ker.astype(np.float64))
        nst = 5 if ker is None else int(ker.sum())
        self.P = p_am; self.Q = p_am + np.log(nst) / np.log(xi0); self.eps = eps     # 邻域格数 nst：均匀 ξ0 时 smax = ξ0
        self.no_am = no_am                                       # True：不加悬空限制（对照：打印时靠支撑）
        self.ground = ground or ground_r > 0                     # True：接地，见 ground_forward
        self.h_xy = h; self.isl_A = island_area; self.isl_b = island_beta      # 底面孤岛最小面积 mm²（0 = 不限），见 island_forward
        self.isl_soft = island_soft; self.isl_conn = island_connect            # 可导近似：局部面积（island_soft_forward）/ 连到主框（island_connect_forward）
        if island_soft > 0:
            Rs_ = int(np.floor(island_soft / h + 1e-9)); os_ = np.arange(-Rs_, Rs_ + 1)
            si, sj = np.meshgrid(os_, os_, indexing="ij")
            self.isl_k = cp.asarray(((si * h) ** 2 + (sj * h) ** 2 <= island_soft ** 2 + 1e-9).astype(np.float64))
        self.seed0 = self.fixed[:, :, 0] & (self.fval[:, :, 0] > 0.5)           # 主框的种子：贴床那层的固定实心格（接口区、固定条）
        self.isl_maxit = 5000
        self.g_ker = None                                        # ground_r > 0：床面那层先在面内膨胀（圆盘，半径 ground_r mm）
        if ground_r > 0:
            Rg = int(np.floor(ground_r / h + 1e-9)); og = np.arange(-Rg, Rg + 1)
            gi, gj = np.meshgrid(og, og, indexing="ij")
            kd = ((gi * h) ** 2 + (gj * h) ** 2 <= ground_r ** 2 + 1e-9).astype(np.float64)
            self.g_ker = cp.asarray(kd)
            self.gQ = p_am + np.log(int(kd.sum())) / np.log(xi0)          # 同 AM：均匀 ξ0 时 smax = ξ0

    def filt(self, x):
        return cnd.correlate(x, self.K, mode="constant", cval=0.0) / self.Ks

    def _ss(self, A):
        """支撑邻域求和（十字或圆盘，都对称，所以也是自己的转置）"""
        return plus_sum(A) if self.am_ker is None else cnd.correlate(A, self.am_ker, mode="constant", cval=0.0)

    def forward(self, x, beta):
        xf = cp.where(self.fixed, self.fval, x)
        xt = self.filt(xf)
        xb, dxb = T2.project(xt, float(beta), 0.5)
        xb = cp.where(self.fixed, self.fval, xb); dxb = cp.where(self.fixed, 0.0, dxb)
        xa, ic = xb, None
        if (self.isl_A > 0 or self.isl_conn) and beta >= self.isl_b:
            xa, ic = (self.island_connect_forward(xa) if self.isl_conn else
                      self.island_soft_forward(xa) if self.isl_soft > 0 else self.island_forward(xa))
        if self.ground:
            xg, gc = self.ground_forward(xa)
            xi, c = self.am_forward(xg)
            c.update(gc)
        else:
            xi, c = self.am_forward(xa)
        c["isl"] = ic
        c["dxb"] = dxb
        if self.eta_e is not None:
            # 腐蚀设计 ξ_e = x̄_e·s，s = ξ/x̄ 是打印设计里"被下层托住"的比例（AM 滤波只对打印设计做一次：
            # 腐蚀设计在床面那层会被吃掉，单独做 AM 滤波的话上面全塌）。灰度阶段 ξ_e ≈ x̄_e，不会两个灰度相乘塌成空
            xe, dxe = T2.project(xt, float(beta), self.eta_e)
            xe = cp.where(self.fixed, self.fval, xe); dxe = cp.where(self.fixed, 0.0, dxe)
            den = cp.maximum(cp.maximum(xb, xi), 1e-12)
            s = xi / den
            c.update(xe=xe, dxe=dxe, s=s, br=xb > xi, q=xe / den)
            c["xie"] = xe * s
        return xi, c

    def island_forward(self, xb):
        """底面孤岛：贴床那层（x̄ > 0.5，8 连通）面积小于 isl_A mm² 的孤岛整块不算打印（ξ_0 = 0），
        上面只靠它托着的部分随之被 AM 滤波去掉 —— 尖刺的尖落在床面上就是这种小孤岛。
        连通性不可导：掩码每轮按当前设计重算，反传时当常数（被去掉的格子梯度为 0）"""
        b0 = xb[:, :, 0]
        lab, n = cnd.label(b0 > 0.5, structure=cp.ones((3, 3), dtype=bool))
        n = int(n)
        if n == 0:
            return xb, dict(kind="label", m=None, n=0, ns=0, msg=" 小孤岛=0/0")
        area = cp.bincount(lab.ravel(), minlength=n + 1) * (self.h_xy ** 2)
        small = area < self.isl_A; small[0] = False
        m = small[lab] & self.free[:, :, 0]
        xa = xb.copy(); xa[:, :, 0] = cp.where(m, 0.0, b0)
        return xa, dict(kind="label", m=m, n=n, ns=int(small.sum()), msg=f" 小孤岛={int(small.sum())}/{n}")

    def island_soft_forward(self, xb):
        """底面孤岛的可导近似①（局部面积）：贴床每格看半径 isl_soft 内贴床材料的面积 A_loc（按自己的密度归一：u = A_loc/(x̄_0+e)，
        均匀灰度时 u = 整个圆盘面积，不偏），u 不到 isl_A 的格子按 g = ½(1+tanh(4(u/A−1))) 关掉：ξ_0 = x̄_0·g。
        几个小点挨得近，彼此的面积算在一起 —— 梯度会鼓励把它们连成一片"""
        b0 = xb[:, :, 0]; A = self.isl_A; kap = 4.0
        Aloc = cnd.correlate(b0, self.isl_k, mode="constant", cval=0.0) * self.h_xy ** 2
        den = b0 + 1e-3; u = Aloc / den
        th = cp.tanh(kap * (u / A - 1.0)); gt_ = 0.5 * (1 + th); gp = 0.5 * kap / A * (1 - th * th)
        free0 = self.free[:, :, 0]
        xa = xb.copy(); xa[:, :, 0] = cp.where(free0, b0 * gt_, b0)
        nb = (b0 > 0.5) & free0
        ns = int((nb & (gt_ < 0.5)).sum())
        return xa, dict(kind="soft", b0=b0, g=gt_, gp=gp, den=den, u=u, msg=f" 关掉的床面格={ns}/{int(nb.sum())}")

    def island_connect_forward(self, xb):
        """底面孤岛的可导近似②（连到主框）：从贴床那层的固定实心格（种子）出发，按十字邻域生长：
        R ← min(x̄_0, max(R 自己和四邻))，收敛后 R(p) = 从主框沿材料走到 p 的所有路径里"最细处密度"的最大值。
        二值时连得上 = 1、连不上 = 0；均匀灰度时 R = x̄_0，不偏。ξ_0 = R。
        R 对 x̄_0 连续、分段线性（同 max/min），导数只落在路径最细的那一格 src(p) 上 —— 缺口格子有梯度：补上它，整串点都被托住。
        （软最大值不行：偏低的话灰度场里每格衰减，偏高的话一块里的格子互相抬高，孤点也会"长"成连上）"""
        b0 = xb[:, :, 0]; seed = self.seed0; NX, NY = b0.shape
        idx = cp.arange(NX * NY, dtype=cp.int64).reshape(NX, NY)
        R = cp.where(seed, 1.0, -1.0); src = cp.where(seed, -1, -2).astype(cp.int64)    # -1：种子（常数），-2：还没到
        nit = 0
        for it in range(self.isl_maxit):
            M = R.copy(); Ms = src.copy()
            for ax, s in ((0, 1), (0, -1), (1, 1), (1, -1)):
                C = cp.full_like(R, -1.0); Cs = cp.full_like(src, -2)
                if ax == 0:
                    if s == 1: C[1:] = R[:-1]; Cs[1:] = src[:-1]
                    else: C[:-1] = R[1:]; Cs[:-1] = src[1:]
                else:
                    if s == 1: C[:, 1:] = R[:, :-1]; Cs[:, 1:] = src[:, :-1]
                    else: C[:, :-1] = R[:, 1:]; Cs[:, :-1] = src[:, 1:]
                bt = C > M
                M = cp.where(bt, C, M); Ms = cp.where(bt, Cs, Ms)
            reach = M >= 0
            Rn = cp.where(seed, 1.0, cp.where(reach, cp.minimum(b0, M), -1.0))
            srcn = cp.where(seed, -1, cp.where(reach, cp.where(b0 <= M, idx, Ms), -2))
            nit = it + 1
            if it % 10 == 9 and bool((Rn == R).all()):
                R, src = Rn, srcn; break
            R, src = Rn, srcn
        Rf = cp.maximum(R, 0.0)
        free0 = self.free[:, :, 0]
        xa = xb.copy(); xa[:, :, 0] = cp.where(free0, Rf, b0)
        nb = (b0 > 0.5) & free0
        ns = int((nb & (Rf < 0.5)).sum())
        return xa, dict(kind="connect", src=src, msg=f" 连不上主框的床面格={ns}/{int(nb.sum())}（{nit} 步）")

    def island_backward(self, g, ic):
        """dF/dξ_0 → dF/dx̄_0（各种底面孤岛处理）"""
        if ic is None or ic.get("kind") == "label" and ic["m"] is None:
            return g
        g = g.copy(); G = g[:, :, 0]; free0 = self.free[:, :, 0]
        if ic["kind"] == "label":
            g[:, :, 0] = cp.where(ic["m"], 0.0, G)
        elif ic["kind"] == "soft":
            Gf = cp.where(free0, G, 0.0)
            W = Gf * ic["b0"] * ic["gp"] / ic["den"]
            g[:, :, 0] = cp.where(free0, Gf * ic["g"], G) + cnd.correlate(W, self.isl_k, mode="constant", cval=0.0) * self.h_xy ** 2 - W * ic["u"]
        else:                                   # ξ_0(p) = x̄_0(src(p))：梯度加到最细处那一格
            src = ic["src"]; Gf = cp.where(free0, G, 0.0); ok = src >= 0
            gb0 = cp.bincount(src[ok].ravel(), weights=Gf[ok].ravel(), minlength=G.size).reshape(G.shape)
            g[:, :, 0] = cp.where(free0, 0.0, G) + gb0
        return g

    def ground_forward(self, xb):
        """接地：底面投影非空的地方才能有材料 —— 每一柱里各层的密度不超过床面那层，x̄_g,k = smin(x̄_k, x̄_0)。
        于是离床的东西（连同 45° 龙骨的斜面）正下方的床面上都得有料：单独伸到空床面上方的尖刺不行，
        要么底下铺板（管道、尖拱），要么下面再长一个倒尖刺接上（沙漏形）。先接地再做 AM 滤波：AM 只会去料、床面那层不动，结果仍接地。
        g_ker 不为 None（--ground_r）时"底面"是床面那层在面内膨胀后的形状：G = smax_圆盘(x̄_0)（P 次和开 Q 次方，同 AM 滤波），
        离床面材料 ground_r 以内的地方都能有料 —— 竖墙可以按 45° 往外扩到 ground_r，再宽的龙骨 / 尖刺仍然没有落脚处。
        返回 x̄_g 和反传要用的量"""
        eps = self.eps; se = np.sqrt(eps)
        b0 = xb[:, :, :1]; gS = None
        if self.g_ker is not None:
            gS = cp.maximum(cnd.correlate(b0[:, :, 0] ** self.P, self.g_ker, mode="constant", cval=0.0), 1e-280)
            b0 = (gS ** (1.0 / self.gQ))[:, :, None]
        d = xb - b0; r = cp.sqrt(d * d + eps)
        xg = 0.5 * (xb + b0 - r + se)
        ga = 0.5 * (1 - d / r); gb = 0.5 * (1 + d / r)
        xg[:, :, 0] = xb[:, :, 0]; ga[:, :, 0] = 1.0; gb[:, :, 0] = 0.0
        xg = cp.where(self.fixed, self.fval, xg)
        ga = cp.where(self.fixed, 0.0, ga); gb = cp.where(self.fixed, 0.0, gb)
        return xg, dict(ga=ga, gb=gb, gS=gS, gx0=xb[:, :, 0])

    def ground_backward(self, g, c):
        """dF/dx̄_g → dF/dx̄（床面那层收各层传下来的那份）"""
        if not self.ground:
            return g
        out = g * c["ga"]
        gG = (g * c["gb"]).sum(axis=2)                           # dF/d(底面形状)
        if self.g_ker is None:
            out[:, :, 0] += gG
        else:
            A = gG * (1.0 / self.gQ) * c["gS"] ** (1.0 / self.gQ - 1.0)
            out[:, :, 0] += self.P * c["gx0"] ** (self.P - 1) * cnd.correlate(A, self.g_ker, mode="constant", cval=0.0)
        return out

    def am_forward(self, xb):
        """ξ_k = smin(x̄_k, Σ_k)，Σ_k = smin_j S_j，S_j = (十字邻域 Σ ξ_{k−j}^P)^{1/Q}（j = 1..m；k−j < 0 时是床面，S_j = 1）"""
        if self.no_am:
            return xb, dict(xi=xb)
        Pp, Q, eps, m = self.P, self.Q, self.eps, self.m_am
        se = np.sqrt(eps)
        NZ = xb.shape[2]
        xi = cp.empty_like(xb); dsx = cp.zeros_like(xb); dsS = cp.zeros_like(xb)
        S = [cp.ones_like(xb) for _ in range(m)]; W = [cp.zeros_like(xb) for _ in range(m)]    # S_j 的 P 次和、∂Σ/∂S_j
        xi[:, :, 0] = xb[:, :, 0]
        for k in range(1, NZ):
            fr = self.free[:, :, k]
            Sig = None; wj = []
            for j in range(1, m + 1):
                if k - j < 0:
                    continue
                Sk = cp.maximum(self._ss(xi[:, :, k - j] ** Pp), 1e-280); S[j - 1][:, :, k] = Sk
                Sg = Sk ** (1.0 / Q)
                if Sig is None:
                    Sig = Sg; wj = [(j, cp.ones_like(Sg))]
                else:                                           # Σ ← smin(Σ, S_j)，之前各项的 ∂Σ/∂S 跟着乘
                    d = Sig - Sg; r = cp.sqrt(d * d + eps)
                    a1, a2 = 0.5 * (1 - d / r), 0.5 * (1 + d / r)
                    Sig = 0.5 * (Sig + Sg - r + se)
                    wj = [(jj, w * a1) for jj, w in wj] + [(j, a2)]
            x = xb[:, :, k]; d = x - Sig; r = cp.sqrt(d * d + eps)
            v = 0.5 * (x + Sig - r + se)
            xi[:, :, k] = cp.where(fr, v, x)
            dsx[:, :, k] = cp.where(fr, 0.5 * (1 - d / r), 0.0)
            dsS[:, :, k] = cp.where(fr, 0.5 * (1 + d / r), 0.0)
            for jj, w in wj:
                W[jj - 1][:, :, k] = w
        dsx[:, :, 0] = cp.where(self.free[:, :, 0], 1.0, 0.0)
        return xi, dict(xi=xi, S=S, W=W, dsx=dsx, dsS=dsS)

    def am_backward(self, gxi, c):
        """dF/dξ → dF/dx̄（从顶层往下传；第 k 层的梯度只来自上面各层，处理到它时已经齐了）"""
        if self.no_am:
            return gxi
        Pp, Q, m = self.P, self.Q, self.m_am
        g = gxi.copy(); gxb = cp.zeros_like(gxi)
        for k in range(gxi.shape[2] - 1, 0, -1):
            gk = g[:, :, k]
            gxb[:, :, k] = gk * c["dsx"][:, :, k]
            gS = gk * c["dsS"][:, :, k]
            for j in range(1, m + 1):
                if k - j < 0:
                    continue
                A = gS * c["W"][j - 1][:, :, k] * (1.0 / Q) * c["S"][j - 1][:, :, k] ** (1.0 / Q - 1.0)
                g[:, :, k - j] += Pp * c["xi"][:, :, k - j] ** (Pp - 1) * self._ss(A)
        gxb[:, :, 0] = g[:, :, 0] * c["dsx"][:, :, 0]
        return gxb

    def backward(self, gxi, c, gxe=None):
        """dF/dξ（打印设计，可为 None）+ dF/dξ_e（腐蚀设计）→ dF/dx"""
        gdir = 0.0
        if gxe is not None:                     # ξ_e = x̄_e·ξ/x̄（x̄ > ξ 时；否则 ξ_e = x̄_e）
            gq = cp.where(c["br"], gxe * c["q"], 0.0)
            gxi = (0.0 if gxi is None else gxi) + gq
            gdir = -gq * c["s"]
        g = self.island_backward(self.ground_backward(self.am_backward(gxi, c), c), c.get("isl"))
        gt = (g + gdir) * c["dxb"]
        if gxe is not None:
            gt = gt + gxe * c["s"] * c["dxe"]
        return cnd.correlate(gt / self.Ks, self.K, mode="constant", cval=0.0) * self.free


def rmin_for_thickness(tmin, h, eta_e=0.75, dz=None):
    """最小厚度 → 过滤半径（mm）。无限大的板（厚 n 个体素，网格外按空）过滤后，腐蚀设计（x̃ ≥ η_e）还剩下东西所需的最小 n
    随半径成台阶变化；取 n = round(tmin/h) 那一段半径的中点（离两头的阈值都远，有限 β 下也稳）。
    体素扁（dz ≠ h）时竖立的板（法向在面内，步长 h）和平躺的板（法向沿 z，步长 dz）分开算，取两者都满足的半径；
    没有交集时优先满足面内，返回实际的平躺板最小厚度。返回 (半径, 面内最小厚度, 层向最小厚度)"""
    dz = dz or h

    def n_min(r, axis_h, axis):
        R, Rz = int(np.ceil(r / h)) - 1, int(np.ceil(r / dz)) - 1
        o, oz = np.arange(-R, R + 1), np.arange(-Rz, Rz + 1)
        di, dj, dk = np.meshgrid(o, o, oz, indexing="ij")
        K = np.maximum(0.0, r - np.sqrt((h * di) ** 2 + (h * dj) ** 2 + (dz * dk) ** 2))
        prof = K.sum(tuple(a for a in range(3) if a != axis)); prof = prof / prof.sum()
        Rp = (len(prof) - 1) // 2
        for n in range(1, 200):
            x = np.zeros(n + 2 * Rp + 4); x[Rp + 2:Rp + 2 + n] = 1
            if np.convolve(x, prof, mode="same").max() >= eta_e:
                return n
        return None
    nt_x, nt_z = max(1, int(round(tmin / h))), max(1, int(round(tmin / dz)))
    rs = np.arange(1.0, 8.0, 0.02) * max(h, dz)
    okx = [r for r in rs if n_min(r, h, 0) == nt_x]
    if not okx:
        raise ValueError(f"最小厚度 {tmin} mm 在 h={h} 下找不到合适的过滤半径")
    if dz == h:
        r = 0.5 * (min(okx) + max(okx)); return r, nt_x * h, nt_x * h
    both = [r for r in okx if n_min(r, dz, 2) == nt_z]
    pick = both or okx
    r = 0.5 * (min(pick) + max(pick))
    return r, nt_x * h, n_min(r, dz, 2) * dz


def am_binary(B, m=1, ker=None):
    """二值模型的悬空检查/修整：从床面往上，没有支撑的体素去掉（每层相对下面 1..m 层的支撑邻域里要有材料，见 am_stencil；
    ker 为 None 时是十字邻域）"""
    B = B.copy()
    if ker is not None:
        from scipy.ndimage import correlate
        kk = ker.astype(np.int32)
    for k in range(1, B.shape[2]):
        for j in range(1, m + 1):
            if k - j >= 0:
                lo = B[:, :, k - j]
                s = plus_sum(lo.astype(np.uint8)) if ker is None else correlate(lo.astype(np.int32), kk, mode="constant", cval=0)
                B[:, :, k] &= s > 0
    return B


# ======================= MMA =======================
class MMA:
    """Svanberg 的移动渐近线法（MMA，1987；参数取 2007 年版 mmasub 的默认值），变量在显卡上，x ∈ [0, 1]。
    约束 f_i ≤ 0，个数 m 很小：子问题用对偶法解（对偶变量只有 m 个，投影牛顿 + 回溯）。
    人工变量 y_i（系数 c 很大）只在约束一时满足不了时出现，保证子问题总有解。"""

    def __init__(self, m, move=0.1, asyinit=0.5, asyincr=1.2, asydecr=0.7, albefa=0.1, c=1e3):
        self.m, self.move, self.c = m, move, c
        self.asyinit, self.asyincr, self.asydecr, self.albefa = asyinit, asyincr, asydecr, albefa
        self.reset()

    def reset(self):
        """β 变了以后从头来（渐近线回到初始宽度）"""
        self.k = 0; self.x1 = self.x2 = self.low = self.upp = None; self.lam = np.ones(self.m)

    def build(self, x, df0, fv, dfv):
        self.k += 1
        if self.k <= 2:
            low, upp = x - self.asyinit, x + self.asyinit
        else:
            s = (x - self.x1) * (self.x1 - self.x2)
            fac = cp.where(s > 0, self.asyincr, cp.where(s < 0, self.asydecr, 1.0))
            low = cp.clip(x - fac * (self.x1 - self.low), x - 10.0, x - 0.01)
            upp = cp.clip(x + fac * (self.upp - self.x1), x + 0.01, x + 10.0)
        alpha = cp.maximum(cp.maximum(0.0, low + self.albefa * (x - low)), x - self.move)
        beta = cp.minimum(cp.minimum(1.0, upp - self.albefa * (upp - x)), x + self.move)
        ux, xl = upp - x, x - low

        def pq(d):
            dp, dn = cp.maximum(d, 0.0), cp.maximum(-d, 0.0)
            r = 0.001 * (dp + dn) + 1e-5
            return (dp + r) * ux ** 2, (dn + r) * xl ** 2
        p0, q0 = pq(df0)
        PQ = [pq(d) for d in dfv]
        b = np.array([float((Pi / ux + Qi / xl).sum()) for Pi, Qi in PQ]) - np.asarray(fv, float)
        return p0, q0, PQ, b, low, upp, alpha, beta

    def update(self, x, df0, fv, dfv):
        """x：当前设计（cp 一维），df0：目标的梯度，fv：约束值（m 个），dfv：约束梯度 → 新设计"""
        p0, q0, PQ, b, low, upp, alpha, beta = sub = self.build(x, df0, fv, dfv)
        xn, self.lam, self.info = self.dual(*sub, self.lam)
        self.x2, self.x1, self.low, self.upp = self.x1, x.copy(), low, upp
        return xn

    def dual(self, p0, q0, PQ, b, low, upp, alpha, beta, lam0):
        m, c = self.m, self.c

        def ev(lam, hess=True):
            p, q = p0.copy(), q0.copy()
            for i, (Pi, Qi) in enumerate(PQ):
                p += lam[i] * Pi; q += lam[i] * Qi
            sp, sq = cp.sqrt(p), cp.sqrt(q)
            x = cp.clip((sp * low + sq * upp) / (sp + sq), alpha, beta)
            ux, xl = upp - x, x - low
            y = np.maximum(0.0, lam - c)
            gi = np.array([float((Pi / ux + Qi / xl).sum()) for Pi, Qi in PQ]) - b
            W = float((p / ux + q / xl).sum()) + float((c * y + 0.5 * y * y - lam * y).sum()) - float(lam @ b)
            H = None
            if hess:
                fr = (x > alpha) & (x < beta)
                d2 = 2 * p / ux ** 3 + 2 * q / xl ** 3
                dg = [cp.where(fr, Pi / ux ** 2 - Qi / xl ** 2, 0.0) for Pi, Qi in PQ]
                H = np.zeros((m, m))
                for i in range(m):
                    for k in range(i, m):
                        H[i, k] = H[k, i] = -float((dg[i] * dg[k] / d2).sum())
                H -= np.diag((lam > c).astype(float))
            return W, gi - y, H, x

        lam = np.maximum(np.asarray(lam0, float), 0.0)
        for it in range(200):
            W, g, H, x = ev(lam)
            act = (lam <= 0) & (g <= 0)
            if np.abs(np.where(act, 0.0, g)).max() < 1e-9:
                break
            fr = ~act
            Hf = H[np.ix_(fr, fr)]
            dH = float(np.abs(np.diag(Hf)).max())
            step = np.zeros(m); ok = False
            if dH > 1e-200:                                      # 牛顿步（W 是凹的，H 半负定）
                try:
                    step[fr] = np.linalg.solve(Hf - 1e-10 * dH * np.eye(int(fr.sum())), -g[fr])
                    ok = bool(np.all(np.isfinite(step))) and step @ g > 0
                except np.linalg.LinAlgError:
                    pass
            if not ok:                                           # x 全卡在上下限时 W 局部是线性的：改走梯度方向，靠回溯定步长
                step = np.where(fr, g, 0.0) * (1.0 + np.abs(lam).max()) / max(float(np.abs(g).max()), 1e-300)
            sm = np.abs(step).max()
            if sm > 1e3 * (1 + np.abs(lam).max()):               # 某个约束几乎不受 x 影响时牛顿步会很大，先截一下
                step *= 1e3 * (1 + np.abs(lam).max()) / sm
            t, moved = 1.0, False
            while t > 1e-14:
                ln = np.maximum(lam + t * step, 0.0)
                if ev(ln, False)[0] >= W - 1e-14 * abs(W):
                    moved = True
                    break
                t *= 0.5
            if not moved or np.abs(ln - lam).max() <= 1e-15 * (1 + np.abs(lam).max()):
                break
            lam = ln
        W, g, _, x = ev(lam, False)
        return x, lam, dict(it=it, grad=g.tolist())


# ======================= 模型 =======================
class Free3D:
    def __init__(self, nz, kz=P.KZ, skin=1.5, nwall=P.NWALL, nskin=P.NSKIN, infill=P.INFILL, misuse=False, lateral=True, warp=True, asm_z=True,
                 hxy=None, dz=None):
        """nz：立方体素时厚度方向的格数（h = T/nz）。给了 dz 时按扁体素：层向 dz（= 打印层高），面内 hxy；
        厚度方向 NT = T/dz 层，网格补到 NZ（能被 16 整除，多重网格要逐级对半分），T 以上的层是固定的空"""
        self.lateral, self.warp, self.asm_z = lateral, warp, asm_z
        if dz is None:
            NT = NZ = nz; h = dz = P.T / nz
            mult, q = 1, nz                                     # 多重网格：z 能对半分几次，x、y 也要能分几次
            while q % 2 == 0 and q > 2:
                q //= 2; mult *= 2
            mult = max(mult, 2)
        else:
            NT = int(round(P.T / dz)); assert abs(NT * dz - P.T) < 1e-6, f"层高 {dz} 不能整分厚度 {P.T}"
            h = hxy or dz
            NZ = -(-NT // 16) * 16
            mult = 32                                           # 层向先单独对半到接近立方，之后三向一起 / 只面内再分几次
        self.NZ, self.NT, self.h, self.dz = NZ, NT, h, dz
        self.vox = h * h * dz                                   # 体素体积 mm³
        zr = np.arange(NZ) < NT; self.zr = zr                   # 厚度 T 以内的层
        domain, solids = P.geometry(True)                       # 设计域、实体区都已经挖掉方槽
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
        self.fs = np.concatenate([skin_fraction(NT, dz, nskin * P.LAYER_H), np.zeros(NZ - NT)])
        cov, fw = print_columns(jg, NX, NY, h, nwall * P.LINE_W)
        fwn = np.where(jnt, np.clip(fw / np.maximum(cov, 1e-9), 0, 1), 0.0)
        Ej, mj = print_voxels(jnt.astype(float), fwn, self.fs, infill ** 2, infill)
        f3 = np.broadcast_to(fixs[:, :, None], (NX, NY, NZ))
        self.E_fix = np.where(f3, 1.0, Ej) * zr; self.m_fix = np.where(f3, 1.0, mj) * zr
        self.design3 = self.cols["design"][:, :, None] & zr[None, None, :]
        use_centered_kernels()
        self.D = T3.material_D(P.E_MAT, P.NU, kz); self.kz = kz
        self.gmin = 1e-6
        self._loads_bc(misuse)
        self.mg = T3.MGCG(NX, NY, NZ, h, dz, self.D, cp.asarray(self.fm, F32))
        self.KE0 = self.mg.levels[0].KE
        self.fm_d = cp.asarray(self.fm, F32)
        # 面内工况按装配状态：面板 / 横梁拉住接口，接口 z 向不动（另一套约束、另一个求解器）
        self.mgs = [self.mg]; self.fms = [self.fm_d]
        if asm_z:
            self.fms.append(cp.asarray(self.fm_asm, F32))
            self.mgs.append(T3.MGCG(NX, NY, NZ, h, dz, self.D, self.fms[1]))
        self.case_bc = [1 if (asm_z and nm in ASM_CASES) else 0 for nm in self.names]
        for l in range(len(self.F)):
            self.F[l] = self.F[l] * self.fms[self.case_bc[l]]
        self.lat = self.names.index("z向体力") if lateral else None
        self.q_lat = None
        # 只差一个倍数的工况（误用 150 N 就是前压 / 后压按比例放大：载荷区、方向、约束都一样）：只解一次，位移按比例给
        self.alias = {}
        for l in range(len(self.F)):
            fl_ = self.F[l].astype(cp.float64); nl = float(cp.linalg.norm(fl_))
            if l == self.lat or nl == 0.0:
                continue
            for m in range(l):
                if m == self.lat or m in self.alias or self.case_bc[m] != self.case_bc[l]:
                    continue
                fm_ = self.F[m].astype(cp.float64); k = float(fl_ @ fm_) / float(fm_ @ fm_)
                if float(cp.linalg.norm(fl_ - k * fm_)) <= 1e-6 * nl:
                    self.alias[l] = (m, k); break

    # ---------- z 向体力 ----------
    def set_mass(self, m):
        """m：体素用料系数（实心 = 1）→ z 向体力（每个单元的力平分到 8 个节点）。第一次调用时定下单位体积的力"""
        if self.lat is None:
            return
        m = cp.asarray(m, F32)
        if self.q_lat is None:
            self.q_lat = F_LAT / (float(m.sum()) * self.vox)
        e = m * F32(self.q_lat * self.vox / 8)
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
        return s * F32(self.q_lat * self.vox / 8)

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
        Mg = geo_mats(self.h, self.h, self.dz); L0 = self.mg.levels[0]; fm = self.fm_d if fm is None else fm
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
        DB = cp.asarray((self.D @ T3._hex_B(self.h, self.h, self.dz, 0, 0, 0)).ravel(), F32)
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

    # ---------- 屈曲约束（优化用） ----------
    def _DB(self):
        if not hasattr(self, "_db"):
            self._db = cp.asarray((self.D @ T3._hex_B(self.h, self.h, self.dz, 0, 0, 0)).ravel(), F32)
        return self._db

    def _dims(self):
        L0 = self.mg.levels[0]
        return np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)

    def solve_rhs(self, b, bc, tol=1e-4, inner=1e-2):
        """当前刚度下解 K x = b（约束组 bc），混合精度迭代修正，同 solve()"""
        mg = self.mgs[bc]; A64 = self.kmul64(self.fms[bc])
        b = b.astype(cp.float64); bn = float(cp.linalg.norm(b)); x = cp.zeros_like(b)
        if bn == 0.0:
            return x.astype(F32)
        b = b / bn                                               # 先归一：右端很小时单精度会下溢
        for _ in range(12):
            r = b - A64(x)
            if float(cp.linalg.norm(r)) <= tol:
                break
            d, _ = mg.solve(r.astype(F32), None, tol=inner); x += d
        return (x * bn).astype(F32)

    def unit_stress(self, l):
        """工况 l 的单元中心应力 D·B·u（单位模量，nel×6 摊平）"""
        L0 = self.mg.levels[0]
        sig = cp.empty(6 * L0.nel, F32)
        T3._K()["kstress"](T3._blocks(L0.nel), (T3._TB,), (self.U[l], sig, cp.ones(L0.nel, F32), self._DB(), *self._dims()))
        return sig

    def buck_init(self, cases, nb=8, seed=0):
        """每个屈曲工况一个子空间（nb 个向量），优化过程中接着上一轮迭代"""
        rng = np.random.default_rng(seed); n = self.F[0].size
        self.bk = {l: dict(X=[cp.asarray(rng.standard_normal(n), F32) * self.fms[self.case_bc[l]] for _ in range(nb)], shift=0.0)
                   for l in cases}

    def buck_step(self, l, sig, nit=2, tol=1e-3):
        """子空间迭代 nit 步：(−K_G) φ = μ K φ，μ = 1/λ。返回 μ（从大到小）和 K 归一化的模态（φᵀKφ = 1）。
        同 buckling()：子空间用单精度多重网格生成，Rayleigh–Ritz 用双精度矩阵乘；负 μ 占优时加平移"""
        st = self.bk[l]; b = self.case_bc[l]; fm, mg = self.fms[b], self.mgs[b]
        KG = self.kgeo(sig, fm=fm); KG64 = self.kgeo(sig, double=True, fm=fm); A64 = self.kmul64(fm)
        X = st["X"]; nb = len(X); mu0 = st.get("mu")
        for _ in range(nit):
            # CG 初值 μ_j·x_j：x_j 是上一步的 Ritz 向量（μ_j 同序），收敛后 K⁻¹(−K_G)x_j ≈ μ_j·x_j，只剩一点要修正。
            # 容差不变（相对 ‖b‖），所以结果和从零开始一样准，只是 CG 少走几步
            G = cp.stack([mg.solve(-KG(x), None if mu0 is None else F32(mu0[j]) * x, tol=tol)[0] + F32(st["shift"]) * x
                          for j, x in enumerate(X)])
            del X
            Am = np.zeros((nb, nb)); Bm = np.zeros((nb, nb))
            for j in range(nb):
                zj = G[j].astype(cp.float64)
                ga, kz = -KG64(zj), A64(zj)
                for i in range(j, nb):
                    zi = G[i].astype(cp.float64) if i != j else zj
                    Am[i, j] = Am[j, i] = float(cp.dot(zi, ga)); Bm[i, j] = Bm[j, i] = float(cp.dot(zi, kz))
            Li = np.linalg.inv(np.linalg.cholesky(Bm))
            mu, V = np.linalg.eigh(Li @ Am @ Li.T)
            C = Li.T @ V                                          # Cᵀ Bm C = I → Φ = Gᵀ C 是 K 归一化的
            Phi = cp.asarray(C.T, F32) @ G
            del G
            X = [p / cp.linalg.norm(p) for p in Phi]
            mu0 = mu
            if mu.min() < 0:
                st["shift"] = max(st["shift"], 1.05 * float(-mu.min()))
        st["X"] = X; st["mu"] = mu0
        o = np.argsort(-mu)
        return mu[o], [Phi[i] for i in o]

    def buck_sens(self, l, EG, sig0, mus, Phis, om, tol=1e-4):
        """G = Σ_j om_j μ_j 对单元刚度系数的导数，拆成三部分（每个单元一个数）：
          A  = Σ om_j μ_j φ_jᵀ KE φ_j          （K 随设计变）
          Bq = Σ_k gs_k σ0_k，gs = Σ om_j g(φ_j)（K_G 里的应力系数 E_G 随设计变）
          Cw = wᵀ KE u，K w = Σ_e E_G,e (D·B)ᵀ gs_e   （应力随位移变：伴随，一次求解）
        dG/dξ_e = −E_G'·Bq + E_K'·(Cw − A)，其中 E_K = gmin + (1−gmin)E、E_G = E（空的单元不受力）"""
        L0 = self.mg.levels[0]; nel = L0.nel; b = self.case_bc[l]; dims = self._dims()
        A = cp.zeros(nel, F32); gs = cp.zeros(6 * nel, F32)
        Mg = geo_mats(self.h, self.h, self.dz).astype(F32)
        for o, mu, ph in zip(om, mus, Phis):
            T3._K()["kener"](T3._blocks(nel), (T3._TB,), (ph, A, self.KE0, np.float32(o * mu), *dims))
            _kfun("kgq")(T3._blocks(nel), (T3._TB,), (ph, gs, Mg, np.float32(o), *dims))
        gs2 = gs.reshape(-1, 6)
        Bq = (gs2 * sig0.reshape(-1, 6)).sum(1)
        t = (gs2 * EG.reshape(-1, 1)).ravel()
        a = cp.empty(3 * L0.nn, F32)
        _kfun("kstrT")(T3._blocks(L0.nn), (T3._TB,), (t, a, self._DB(), self.fms[b], *dims))
        w = self.solve_rhs(a, b, tol)
        s = float(cp.linalg.norm(self.U[l])) / max(float(cp.linalg.norm(w)), 1e-30)    # 两个向量量级拉平再算互能，免得单精度相消
        Cw = self.mutual_energy(w * F32(s), self.U[l]).ravel() / F32(s)
        return A, Bq, Cw

    def mode_energy_split(self, ph, low):
        """模态 φ 的应变能里落在 low（低密度单元）的比例：看有没有空区里的假模态"""
        L0 = self.mg.levels[0]
        e = cp.zeros(L0.nel, F32)
        T3._K()["kener"](T3._blocks(L0.nel), (T3._TB,), (ph, e, self.KE0, np.float32(1.0), *self._dims()))
        e = e * L0.g
        return float(e[low.ravel()].sum() / e.sum())

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
        zn = np.arange(NZ + 1) <= self.NT                       # 加载 / 约束只在厚度 T 以内的节点上（扁体素时网格补高了）
        m3 = lambda m2: (m2[:, :, None] & zn[None, None, :]).ravel() if m2.dtype == bool else (m2[:, :, None] * zn[None, None, :]).ravel()
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
            fa[hk, 0, 2] = True; fa[sh, self.NT, 2] = True
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
            if l in self.alias:                                 # 某个前面工况的倍数：U = k·U_m，柔度 = k²·c_m
                m, k = self.alias[l]
                self.U[l] = self.U[m] * F32(k); comps[l] = k * k * comps[m]; its.append(0)
                continue
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
        DB = cp.asarray((self.D @ T3._hex_B(self.h, self.h, self.dz, 0, 0, 0)).ravel(), F32)
        sig = cp.empty(6 * L0.nel, F32); ones = cp.ones(L0.nel, F32)
        mk = cp.asarray(mask.ravel())
        # 位移只统计实心体素的节点（空单元里的节点只靠极软的空材料连着，位移没有意义；扁体素时 T 以上补的层也是空）
        m3 = cp.asarray(mask, cp.bool_)
        nm_ = cp.zeros((self.NX + 1, self.NY + 1, self.NZ + 1), cp.bool_)
        for a in range(8):
            di, dj, dk = a & 1, (a >> 1) & 1, a >> 2
            nm_[di:di + self.NX, dj:dj + self.NY, dk:dk + self.NZ] |= m3
        nm_ = nm_.ravel()
        out = []
        for l, nm in enumerate(self.names):
            T3._K()["kstress"](T3._blocks(L0.nel), (T3._TB,), (self.U[l], sig, ones, DB, np.int32(L0.nx), np.int32(L0.ny), np.int32(L0.nz)))
            S = sig.reshape(-1, 6)[mk]
            vm = cp.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2) + 3 * (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2))
            tz = cp.sqrt(S[:, 4] ** 2 + S[:, 5] ** 2)
            u = self.U[l].reshape(-1, 3)[nm_]
            pct = lambda v: float(cp.percentile(v, 99.9))
            out.append(dict(工况=nm, 最大位移_mm=float(cp.sqrt((u ** 2).sum(1)).max()),
                            面内位移_mm=float(cp.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2).max()), z向位移_mm=float(cp.abs(u[:, 2]).max()),
                            VM_99_9pct_MPa=pct(vm), VM_max_MPa=float(vm.max()),
                            层间拉应力_99_9pct_MPa=pct(cp.maximum(S[:, 2], 0)), 层间剪应力_99_9pct_MPa=pct(tz)))
        return out

    # ---------- 用料 ----------
    def mass(self, xi):
        """实心等效体积 mm³：设计体素按 ξ，固定区按各自系数"""
        return (float(cp.asarray(xi)[cp.asarray(self.design3)].sum()) + float(self.m_fix.sum())) * self.vox

    # ---------- 对比用的体素场 ----------
    def baseline_fields(self, geom=None):
        """二维方案（profile.json 拉伸 T）按墙 + 表皮 + 填充打印 → (E, m, 实心掩码)"""
        geom = geom or profile2d()
        ps = self.print_set
        cov, fw = print_columns(geom, self.NX, self.NY, self.h, ps["nwall"] * P.LINE_W)
        E, m = print_voxels(cov, fw, self.fs, ps["infill"] ** 2, ps["infill"])
        solid = (fw[:, :, None] + (cov - fw)[:, :, None] * self.fs[None, None, :]) > 0.5
        # 顶面、前挡后表面、脚垫那几条固定实心条按实心（与三维设计相同）：载荷就加在这几条的节点上，二维轮廓在前挡顶、
        # 顶面后端是圆角，网格细的时候最外一排加载节点会落在轮廓外的空单元里，力加在空节点上，柔度被放大好几倍。
        # 二维方案在这里本来就是墙（实心），差别只在圆角那一点点
        fx = self.cols["fixs"][:, :, None] & self.zr[None, None, :]
        E = np.where(fx, 1.0, E); m = np.where(fx, 1.0, m); solid = solid | fx
        return E * self.zr, m * self.zr, solid & self.zr

    def design_fields(self, xi, p=3.0):
        """优化结果（ξ 或二值）+ 固定区 → (E, m, 实心掩码)"""
        xi = np.asarray(xi, float)
        d3 = self.design3
        E = np.where(d3, xi ** p, self.E_fix); m = np.where(d3, xi, self.m_fix)
        solid = np.where(d3, xi > 0.5, (self.E_fix > 0.5))
        return E, m, solid


# ======================= 图 =======================
def fig_layers(M, xi, fn, title):
    ks = np.unique(np.linspace(0, M.NT - 1, 8).round().astype(int))
    fig, axs = plt.subplots(2, 4, figsize=(22, 7.5))
    for ax, k in zip(axs.ravel(), ks):
        ax.imshow(1 - xi[:, :, k].T, cmap="gray", origin="lower", extent=[0, M.NX * M.h, 0, M.NY * M.h], vmin=0, vmax=1)
        ax.set_title(f"第 {k+1} 层体素  z = {k*M.dz:.1f}–{(k+1)*M.dz:.1f} mm"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.savefig(fn, dpi=90, bbox_inches="tight"); plt.close(fig)


def fig_sections(M, xi, fn, title, xs=(30, 60, 95, 130, 165, 200, 235)):
    """竖直剖面（x = 常数，横轴 y、纵轴 z = 打印方向）：看杆件截面的空腔和封顶"""
    fig, axs = plt.subplots(len(xs), 1, figsize=(16, 1.35 * len(xs) + 1))
    for ax, x in zip(axs, xs):
        i = min(M.NX - 1, int(x / M.h))
        ax.imshow(1 - xi[i, :, :M.NT].T, cmap="gray", origin="lower", extent=[0, M.NY * M.h, 0, M.NT * M.dz], vmin=0, vmax=1, interpolation="nearest")
        ax.set_ylabel(f"x={x}", rotation=0, labelpad=22); ax.set_yticks([0, P.T]); ax.set_aspect("equal")
    axs[-1].set_xlabel("y (mm)    剖面纵轴为打印方向 z（下面贴床）")
    fig.suptitle(title, fontsize=13)
    fig.savefig(fn, dpi=100, bbox_inches="tight"); plt.close(fig)


# ======================= 优化 =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=32, help="厚度方向体素数（立方体素，h = T/nz）；给了 --dz 时不用")
    ap.add_argument("--dz", type=float, default=None, help="扁体素：层向尺寸 mm（= 打印层高，如 0.2）")
    ap.add_argument("--hxy", type=float, default=None, help="扁体素：面内尺寸 mm（如 0.4；45° 悬空按每 hxy/dz 层外扩一格）")
    ap.add_argument("--no_am", action="store_true", help="不加悬空限制（对照：悬空的地方打印时加支撑）")
    ap.add_argument("--sup_mu", type=float, default=0.0, help="支撑惩罚：目标 += μ·(估算支撑体积 / sup_ref)；一般配 --no_am")
    ap.add_argument("--sup_ref", type=float, default=10000.0, help="支撑惩罚的参考体积 mm³（10 cm³）")
    ap.add_argument("--sup_angle", type=float, default=30.0, help="估算支撑时的阈值角（与床面夹角）。立方体素上建议 45：30° 的 3×3 邻域把斜对角的台阶也算托住，切片时仍要支撑")
    ap.add_argument("--sup_c", type=float, default=2.0, help="每 mm² 悬空的支撑体积系数（× z^e mm³），按 Bambu 实切标定")
    ap.add_argument("--sup_e", type=float, default=0.47, help="支撑体积随离床高度的指数")
    ap.add_argument("--sup_r", type=float, default=0.0, help="悬空面内膨胀半径 mm（> 0 时按膨胀后的面积计，见 SupportEst）")
    ap.add_argument("--ground", action="store_true", help="接地：底面投影非空才能有材料（每柱各层不超过床面那层）。配 45°：悬空龙骨不能伸到空床面上方，逼出尖拱管道 / 沙漏形；配 --no_am：悬空只能在有底板的地方")
    ap.add_argument("--island_area", type=float, default=0.0, help="底面孤岛最小面积 mm²：贴床那层面积更小的连通块不算打印（尖刺落在床面上的尖）；0 = 不限")
    ap.add_argument("--island_soft", type=float, default=0.0, help="底面孤岛的可导近似①：按半径 r mm 内的贴床面积算（阈值用 --island_area）；导出时仍按精确的最小面积去掉")
    ap.add_argument("--island_connect", action="store_true", help="底面孤岛的可导近似②：贴床那层要和主框（接口区、固定条）连通，软生长计算；导出时去掉连不上的块")
    ap.add_argument("--island_beta", type=float, default=0.0, help="β 达到这个值以后才限制底面孤岛（0 = 从头开始）")
    ap.add_argument("--ground_r", type=float, default=0.0, help="接地放宽：床面那层先在面内膨胀 r mm 再当底面（竖墙可按 45° 外扩到 r）；> 0 时自动打开 --ground")
    ap.add_argument("--am_angle", type=float, default=45.0, help="悬空角：允许的最缓斜面与床面的夹角（度）。45 = 常规；大 = 更陡更严，小 = 更缓更松")
    ap.add_argument("--time_budget", type=float, default=0, help="优化循环的时间上限（秒，0 = 不限）：到了就停下来照常存盘")
    ap.add_argument("--save_every", type=int, default=0, help="每隔几轮存一次 result.npz（租的机器上防止中途断掉白跑）")
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
    ap.add_argument("--tmin", type=float, default=0.0, help="最小厚度 mm（> 0：刚度和屈曲按腐蚀设计算，过滤半径由它定，网格外按空）")
    ap.add_argument("--eta_e", type=float, default=0.75, help="腐蚀设计的投影阈值")
    ap.add_argument("--buck", type=float, default=0.0, help="150 N 误用工况（前、后）的线性屈曲载荷系数下限；0 = 不约束")
    ap.add_argument("--buck_nb", type=int, default=8, help="屈曲子空间维数")
    ap.add_argument("--buck_nk", type=int, default=4, help="每个工况取前几个模态做 KS 聚合")
    ap.add_argument("--buck_nit", type=int, default=2, help="每轮设计迭代的子空间迭代步数（第一轮 10 步）")
    ap.add_argument("--ks", type=float, default=40.0, help="KS 聚合系数（作用在 λ_目标/λ 上）")
    ap.add_argument("--opt", choices=("oc", "mma"), default=None, help="优化器；默认有最小厚度或屈曲约束时用 MMA，否则 OC")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    M = Free3D(a.nz, kz=a.kz, skin=a.skin, lateral=a.lateral, warp=a.warp > 0, asm_z=a.asm_z, misuse=a.buck > 0, hxy=a.hxy, dz=a.dz)
    NX, NY, NZ, h = M.NX, M.NY, M.NZ, M.h
    vox = M.vox
    nd = int(M.design3.sum())
    print(f"网格 {NX}×{NY}×{NZ}（面内 {h:.4f} mm，层向 {M.dz:.4f} mm，厚度内 {M.NT} 层），体素 {NX*NY*NZ/1e6:.2f} M，"
          f"自由度 {3*M.mg.levels[0].nn/1e6:.2f} M；设计体素 {nd/1e6:.2f} M", flush=True)
    print("多重网格:", M.mg.describe())
    E2, m2, _ = M.baseline_fields()
    target = float(m2.sum()) * vox * a.mass_scale
    fixm = float(M.m_fix.sum()) * vox
    vf = (target - fixm) / (nd * vox)
    print(f"用料目标 {target/1000:.2f} cm³ 实心等效（二维方案 × {a.mass_scale:g}；PETG 约 {target/1000*1.27:.0f} g）；"
          f"固定区 {fixm/1000:.2f} cm³（接口区 {float(M.m_fix[M.cols['jnt']].sum())*vox/1000:.2f}）；设计区平均密度 {vf:.4f}", flush=True)
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
    rmin = a.rmin
    if a.tmin > 0:
        rmin, tq, tqz = rmin_for_thickness(a.tmin, h, a.eta_e, M.dz)
        print(f"最小厚度 {a.tmin} mm（网格上竖板 {tq:.3f}、平板 {tqz:.3f} mm）→ 过滤半径 {rmin:.3f} mm，腐蚀阈值 η_e={a.eta_e}；刚度、屈曲按腐蚀设计算", flush=True)
    a.rmin_used = rmin
    pipe = Pipeline((NX, NY, NZ), h, rmin, fixed, fval, pad_void=a.tmin > 0, eta_e=a.eta_e if a.tmin > 0 else None, dz=M.dz, no_am=a.no_am, am_angle=a.am_angle, ground=a.ground, ground_r=a.ground_r, island_area=a.island_area, island_beta=a.island_beta, island_soft=a.island_soft, island_connect=a.island_connect)
    if a.island_connect:
        print("底面孤岛（可导近似）：贴床那层要和主框连通（从固定实心格软生长）", flush=True)
    elif a.island_soft > 0:
        print(f"底面孤岛（可导近似）：半径 {a.island_soft:g} mm 内贴床面积不到 {a.island_area:g} mm² 的格子逐渐关掉", flush=True)
    elif a.island_area > 0:
        print(f"底面孤岛：贴床那层面积小于 {a.island_area:g} mm² 的连通块不算打印" + (f"（β ≥ {a.island_beta:g} 起）" if a.island_beta > 0 else ""), flush=True)
    if a.ground or a.ground_r > 0:
        print("接地：每柱各层的密度不超过床面那层" + (f"在面内膨胀 {a.ground_r:g} mm 后的形状" if a.ground_r > 0 else "（底面投影非空才能有材料）"), flush=True)
    if not a.no_am and pipe.am_ker is not None:
        print(f"悬空 {a.am_angle:g}°（与床面夹角）：下一层半径 {M.dz/np.tan(np.deg2rad(a.am_angle)):.2f} mm（{int(cp.asnumpy(pipe.am_ker).sum())} 格）的圆盘里有材料就算托住", flush=True)
    elif not a.no_am and (pipe.m_am > 1 or a.am_angle != 45.0):
        print(f"悬空 {a.am_angle:g}°（与床面夹角）：每层相对下面 1..{pipe.m_am} 层都最多外扩一格（{h:g} mm）→ 实际约 {np.degrees(np.arctan(pipe.m_am * M.dz / h)):.1f}°，单层最多伸出 {h:g} mm", flush=True)
    supest = SupportEst(h, M.dz, M.NT, angle=a.sup_angle, c=a.sup_c, e=a.sup_e, r_mm=a.sup_r) if a.sup_mu > 0 else None
    if supest is not None:
        print(f"支撑惩罚 μ={a.sup_mu:g}：每 {a.sup_ref/1000:g} cm³ 估算支撑 = 目标里 {a.sup_mu:g}（阈值 {a.sup_angle:g}°）", flush=True)
    d3 = cp.asarray(M.design3)
    use_mma = a.opt == "mma" or (a.opt is None and (a.tmin > 0 or a.buck > 0))
    bcases = [M.names.index(n) for n in ("前部误用150N", "后部误用150N")] if a.buck > 0 else []
    if bcases:
        M.buck_init(bcases, nb=a.buck_nb)
    mma = MMA(1 + len(bcases[:1]), move=a.move) if use_mma else None
    bstate = dict(first=True)

    def buck_con(xa_, it):
        """屈曲约束 KS(λ_目标·μ) − 1 ≤ 0（μ = 1/λ，两个工况各取前 buck_nk 个正的 μ），及其对分析密度的导数"""
        EG = cp.where(d3, xa_ ** a.penal, cp.asarray(M.E_fix)).astype(F32).ravel()
        per = {}
        for l in bcases:
            s0 = M.unit_stress(l)
            mu, Ph = M.buck_step(l, (s0.reshape(-1, 6) * EG[:, None]).ravel(), nit=10 if bstate["first"] else a.buck_nit)
            sel = [j for j in range(len(mu)) if mu[j] > 0][:a.buck_nk]
            per[l] = (s0, mu[sel], [Ph[j] for j in sel])
        bstate["first"] = False
        mh = a.buck * np.concatenate([per[l][1] for l in bcases])
        mx = mh.max(); KS = mx + np.log(np.exp(a.ks * (mh - mx)).sum()) / a.ks
        om = a.buck * np.exp(a.ks * (mh - KS))
        gG = cp.zeros(int(d3.size), F32); o = 0
        for l in bcases:
            s0, mus, Phs = per[l]
            ol = om[o:o + len(mus)]; o += len(mus)
            keep = ol > 1e-8 * om.max()                          # KS 权重可以忽略的模态不算（也免得单精度下溢）
            if not keep.any():
                continue
            A, Bq, Cw = M.buck_sens(l, EG, s0, mus[keep], [p for p, k in zip(Phs, keep) if k], ol[keep])
            gG += -Bq + F32(1 - M.gmin) * (Cw - A)
        gG = gG.reshape(NX, NY, NZ).astype(cp.float64) * a.penal * xa_ ** (a.penal - 1) * d3
        info = {M.names[l]: (round(1.0 / per[l][1][0], 3) if len(per[l][1]) else float("inf")) for l in bcases}
        if it % 10 == 1:                      # 低密度区里的应变能比例：大的话是空区假模态
            low = (xa_ < 0.5) & d3
            info["假模态比例"] = [round(M.mode_energy_split(per[l][2][0], low), 3) for l in bcases if per[l][2]]
        return float(KS - 1.0), gG, info
    w = M.imp / comps_b
    x = cp.where(d3, vf, 0.0)
    if a.init:
        from scipy.ndimage import map_coordinates
        r0 = np.load(a.init); xc, hc = r0["x"].astype(float), float(r0["h"])
        dzc = float(r0["dz"]) if "dz" in r0.files else hc
        I, J = np.meshgrid(*[(np.arange(n) + 0.5) * h / hc - 0.5 for n in (NX, NY)], indexing="ij")
        I, J, K = (np.broadcast_to(v[:, :, None], (NX, NY, NZ)) for v in (I, J, np.zeros((NX, NY))))
        K = np.broadcast_to(((np.arange(NZ) + 0.5) * M.dz / dzc - 0.5)[None, None, :], (NX, NY, NZ))
        x0 = map_coordinates(xc, [I, J, K], order=1, mode="nearest")
        x = cp.where(d3, cp.asarray(np.clip(x0, 0, 1)), 0.0)
        print(f"初始设计取自 {a.init}（网格 {xc.shape}，面内 {hc:.4f}、层向 {dzc:.4f} mm）", flush=True)
    beta, lb = a.beta0, 0
    hist = []
    for it in range(1, a.maxit + 1):
        ti = time.time()
        xi, cache = pipe.forward(x, beta)
        xa_ = cache.get("xie", xi)                                  # 分析用的密度（有最小厚度时是腐蚀设计）
        Ef = cp.where(d3, xa_ ** a.penal, cp.asarray(M.E_fix))
        M.set_mass(cp.where(d3, xa_, cp.asarray(M.m_fix)))
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
        gE = -ce * (1 - M.gmin) * a.penal * xa_ ** (a.penal - 1) * d3
        if M.lat is not None:                                     # 体力随设计变化：d(F·U)/dξ 多一项 2·Uᵀ·dF/dξ
            gE = gE + 2 * w[M.lat] * M.body_sens().astype(cp.float64) * d3
        if M.warp:
            # dW_l/dξ = −(1/F_side)·(1−gmin)·p·ξ^(p−1)·u_sideᵀ KE u_l（互易定理：伴随就是侧推、侧推后段的解）
            mut = M.mutual_energy(M.U[M.names.index("侧推")], Aa) + M.mutual_energy(M.U[M.names.index("侧推后段")], Ar)
            gE = gE - mut.astype(cp.float64) * (1 - M.gmin) * a.penal * xa_ ** (a.penal - 1) * d3
        robust = "xie" in cache
        dc = pipe.backward(None, cache, gE) if robust else pipe.backward(gE, cache)
        sup = 0.0; ctot = c
        if supest is not None:
            sup, gS = supest(xi)                                      # 按打印设计 ξ 估算
            dc = dc + (a.sup_mu / a.sup_ref) * pipe.backward(gS, cache)
            ctot = c + a.sup_mu * sup / a.sup_ref
        dm = pipe.backward(d3 * vox, cache)                       # 用料按打印设计 ξ
        dm = cp.maximum(dm, 1e-3 * float(dm[d3].mean()))          # 支撑不住的体素 dm≈0：给个下限，让它们往下走
        xa = x[d3]; binfo = {}; fG = 0.0
        if use_mma:
            fv, dfv = [M.mass(xi) / target - 1.0], [dm[d3] / target]
            if bcases:
                fG, gG, binfo = buck_con(xa_, it)
                dG = pipe.backward(None, cache, gG) if robust else pipe.backward(gG, cache)
                fv.append(fG); dfv.append(dG[d3])
            xn = mma.update(xa, dc[d3], fv, dfv)
        else:
            dc = cp.minimum(dc, 0.0)
            q = -dc[d3] / dm[d3]
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
        hist.append(dict(it=it, c=c, sup=sup, ctot=ctot, pen=pen / w.dot(comps_b), mass=mratio, change=change, beta=beta, grey=grey, cg=its, comp=comps.tolist(),
                         buck=binfo, ks=fG))
        bs = (" 屈曲 λ " + " / ".join(f"{v:.1f}" for k, v in binfo.items() if k != "假模态比例") + f" KS−1={fG:+.3f}"
              + (f" 假模态{binfo['假模态比例']}" if "假模态比例" in binfo else "")) if binfo else ""
        print(f"it {it:3d} 目标/二维={c:.4f}（翘曲项 {pen / w.dot(comps_b):.4f}）用料/目标={mratio:.3f}{bs}{f" 支撑≈{sup/1000:.2f}cm³ 总目标={ctot:.4f}" if supest is not None else ""}{cache['isl']['msg'] if cache.get("isl") else ""} ch={change:.3f} β={beta:g} 灰度={grey:.3f} "
              f"CG={sum(its)} {time.time()-ti:.1f}s t={time.time()-t0:.0f}s", flush=True)
        if it % 20 == 0:
            fig_layers(M, cp.asnumpy(xi), f"{a.out}/layers_it{it}.png", f"it={it} β={beta:g} 加权柔度/二维={c:.3f}")
        lb += 1
        feas = mratio < 1.01 and fG < 0.02
        if beta < a.beta_max and (lb >= a.beta_every or (change < 0.01 and lb >= 15)):
            beta *= 2; lb = 0
            if mma is not None:
                mma.reset()
        elif beta >= a.beta_max and lb >= 30 and feas and (change < 0.01 or abs(hist[-11]["ctot"] - ctot) / ctot < 1e-3):
            break
        if a.save_every and it % a.save_every == 0:          # 中途存盘（同一个文件，跑完再覆盖）
            xs = cp.asnumpy(pipe.forward(x, beta)[0]).astype(np.float32)
            np.savez_compressed(f"{a.out}/result.npz", x=cp.asnumpy(x).astype(np.float32), xi=xs, NX=NX, NY=NY, NZ=NZ, h=h, dz=M.dz, NT=M.NT,
                                comps_b=comps_b, names=np.array(M.names), target=target, it=it, **{"col_" + k: v for k, v in M.cols.items()})
            json.dump(dict(args=vars(a), names=M.names, hist=hist, base=dict(comp=comps_b.tolist()), target_mm3=target),
                      open(f"{a.out}/history.json", "w"), ensure_ascii=False, indent=1)
        if a.time_budget and time.time() - t0 > a.time_budget:
            print(f"到时间上限 {a.time_budget:.0f}s，第 {it} 轮停下（β={beta:g}）", flush=True)
            break
    xi, _ = pipe.forward(x, beta)
    xi = cp.asnumpy(xi)
    np.savez_compressed(f"{a.out}/result.npz", x=cp.asnumpy(x).astype(np.float32), xi=xi.astype(np.float32), NX=NX, NY=NY, NZ=NZ, h=h, dz=M.dz, NT=M.NT,
                        comps_b=comps_b, names=np.array(M.names), target=target, **{"col_" + k: v for k, v in M.cols.items()})
    json.dump(dict(args=vars(a), names=M.names, hist=hist, base=dict(comp=comps_b.tolist()), target_mm3=target),
              open(f"{a.out}/history.json", "w"), ensure_ascii=False, indent=1)
    fig_layers(M, xi, f"{a.out}/layers_final.png", f"最终 β={beta:g}")
    fig_sections(M, xi, f"{a.out}/sections_final.png", "剖面（x = 常数）")
    print(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
