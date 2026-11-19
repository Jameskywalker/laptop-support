"""
后处理：密度场 → 平滑轮廓多边形 → 合并精确非设计几何 → 尺寸检查 → 0.5 mm 网格 FE 复核
用法: STAND_VARIANT=desk40 python postprocess.py [拓扑优化结果目录]
"""
import sys, os, json, pickle
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import measure
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypardiso
import params as P
import topopt2d as TO

run = sys.argv[1] if len(sys.argv) > 1 else P.TOPO_DIR
MD = P.MODEL_DIR
r = np.load(f"{run}/result.npz")
nelx, nely, h = int(r["nelx"]), int(r["nely"]), float(r["h"])
G = r["xphys"].reshape(nelx, nely).T   # G[j, i]

# ---------- 1. 轮廓提取 ----------
d = 0.2
xs = np.arange(-2, nelx * h + 2, d); ys = np.arange(-2, nely * h + 2, d)
XX, YY = np.meshgrid(xs, ys)
Gs = gaussian_filter(G, 0.6)
Fd = map_coordinates(Gs, [YY.ravel() / h - 0.5, XX.ravel() / h - 0.5], order=1, mode="constant", cval=0).reshape(XX.shape)
rings = []
for c in measure.find_contours(Fd, 0.5):
    if len(c) < 8: continue
    pg = Polygon(np.c_[xs[0] + c[:, 1] * d, ys[0] + c[:, 0] * d]).buffer(0)
    if pg.area > 1.0: rings.append(pg)
region = rings[0]
for pg in rings[1:]:
    region = region.symmetric_difference(pg)

domain, solids = P.geometry(True)
for k in [k for k in solids if k.startswith("joint_")]:
    solids[k] = Polygon(solids[k].exterior).buffer(-3, join_style="mitre").buffer(3).difference(P.joint_slot())   # 外轮廓倒圆 R3（方槽两侧薄壁不能参与开运算）
shape = shapely.unary_union([region] + list(solids.values())).intersection(domain)
shape = shape.buffer(1.5).buffer(-1.5)          # 闭运算：内凹处倒圆 R1.5
shape = shape.buffer(-1.0).buffer(1.0)          # 开运算：外凸处倒圆 R1
shape = shape.intersection(domain)              # 保证顶面严格平整
if isinstance(shape, MultiPolygon):
    parts = sorted(shape.geoms, key=lambda g: -g.area)
    print("警告：存在多个分离块，面积", [round(g.area, 1) for g in parts])
    shape = parts[0]
holes = [h_ for h_ in shape.interiors if Polygon(h_).area >= 20.0]
shape = Polygon(shape.exterior, holes).simplify(0.02)
shape = shape.difference(P.joint_slot())        # 横梁卡扣穿过的贯通方槽
assert shape.is_valid and shape.geom_type == "Polygon", shape.geom_type

# ---------- 2. 最小尺寸检查（开/闭运算保持拓扑的最大半径） ----------
def topo(g):
    gs = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    return len(gs), sum(len(x.interiors) for x in gs)
base_topo = topo(shape)
min_member = None
for rr in np.arange(0.5, 6.01, 0.25):
    op = shape.buffer(-rr).buffer(rr)
    if topo(op) != base_topo or shape.difference(op).area > 0.15 * shape.area:
        break
    min_member = 2 * rr
hull = shape.convex_hull
min_hole = None
for rr in np.arange(0.5, 6.01, 0.25):
    cl = shape.buffer(rr).buffer(-rr)
    if topo(cl)[1] != base_topo[1]:
        break
    min_hole = 2 * rr

area = shape.area
mass_solid = area * P.T * 1.24e-3
info = dict(area_mm2=area, holes=len(shape.interiors), min_member_mm=min_member, min_hole_mm=min_hole,
            volume_cm3=area * P.T / 1000, mass_solid_g=mass_solid, n_vertices=len(shape.exterior.coords) + sum(len(i.coords) for i in shape.interiors),
            bbox=shape.bounds)
print(json.dumps(info, ensure_ascii=False, indent=1))
os.makedirs(MD, exist_ok=True)
with open(f"{MD}/profile.pkl", "wb") as fh:
    pickle.dump(shape, fh)
with open(f"{MD}/profile.json", "w") as fh:
    json.dump(dict(exterior=list(shape.exterior.coords), holes=[list(i.coords) for i in shape.interiors]), fh)

# ---------- 3. FE 复核（0.5 mm 网格，实际几何） ----------
hv = 0.5
m = TO.Mesh(hv, True)
inside = shapely.contains_xy(shape, m.xc, m.yc)
m.in_dom = inside
used = np.zeros(m.nn, bool)
el_nodes = np.stack([m.edof[:, 0] // 2, m.edof[:, 2] // 2, m.edof[:, 4] // 2, m.edof[:, 6] // 2], 1)
used[el_nodes[inside].ravel()] = True
m.node_in = used
F, names, imp = TO.build_loads(m)
# 额外的误用工况：手肘撑在前部 / 转轴处 150 N（全部压在一侧支撑上；沿斜面分量由前挡承担）
_s, _n, _top, _lip, _add, _press = TO.load_tools(m)
fa = np.zeros(m.ndof); _press(fa, 10, 40, 150.0)
fb = np.zeros(m.ndof); _press(fb, 175, P.S_END, 150.0)
F = np.column_stack([F, fa, fb])
names = list(names) + ["前部误用150N", "后部误用150N"]
free, fixed = TO.build_bc(m)
KE = TO.element_stiffness(P.NU)
Ee = np.where(inside, P.E_MAT * P.T, P.E_MAT * P.T * 1e-9)
K = sp.csr_matrix(((KE.ravel()[None, :] * Ee[:, None]).ravel(), (m.iK, m.jK)), shape=(m.ndof, m.ndof))
solver = pypardiso.PyPardisoSolver(mtype=2)
Kf = sp.triu(K[free][:, free], format="csr")
Ug = np.zeros((m.ndof, F.shape[1])); Ug[free] = solver.solve(Kf, F[free])
# 单元中心应力
nu = P.NU
D = P.E_MAT / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
xin = np.array([-1, 1, 1, -1]); etn = np.array([-1, -1, 1, 1])
B = np.zeros((3, 8)); dNx = 0.25 * xin * 2 / hv; dNy = 0.25 * etn * 2 / hv
B[0, 0::2] = dNx; B[1, 1::2] = dNy; B[2, 0::2] = dNy; B[2, 1::2] = dNx
res = []
vm_all = []
lip_zone = inside & (np.hypot(m.xc - P.P0[0], m.yc - P.P0[1]) < 25.0)   # 前挡及其根部
for k, nm in enumerate(names):
    ue = Ug[m.edof, k]
    sig = ue @ (D @ B).T
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2 + 3 * sig[:, 2] ** 2)
    vm[~inside] = 0
    vm_all.append(vm)
    disp = np.hypot(Ug[0::2, k], Ug[1::2, k])[used]
    res.append(dict(工况=nm, 最大位移_mm=float(disp.max()), 最大VonMises_MPa=float(vm.max()), VM_99_9pct_MPa=float(np.percentile(vm[inside], 99.9)),
                    前挡区最大VM_MPa=float(vm[lip_zone].max())))
for rr_ in res:
    print(rr_)
with open(f"{MD}/verify.json", "w") as fh:
    json.dump(dict(info=info, fe=res), fh, ensure_ascii=False, indent=1)

# ---------- 4. 图 ----------
from matplotlib import font_manager as fm
for f in ("/mnt/c/Windows/Fonts/msyh.ttc",):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axs[0]
ax.imshow(1 - G, cmap="gray", origin="lower", extent=[0, nelx * h, 0, nely * h], alpha=0.35)
ex = np.array(shape.exterior.coords); ax.fill(ex[:, 0], ex[:, 1], color="#4a78a8", alpha=0.85, lw=0)
for it in shape.interiors:
    c = np.array(it.coords); ax.fill(c[:, 0], c[:, 1], color="white", lw=0)
ax.plot(ex[:, 0], ex[:, 1], "k-", lw=0.6)
for _i, _j in enumerate(P.JOINTS):
    c = np.array(P.joint_slot(_j).exterior.coords)
    ax.plot(c[:, 0], c[:, 1], "r--", lw=1, label="横梁卡扣方槽（贯通）" if _i == 0 else None)
# 笔记本示意
lx, ly = P.sn2xy([0, P.LAPTOP_D, P.LAPTOP_D, 0, 0], [0, 0, 16, 16, 0])
ax.plot(lx, ly, color="#888", lw=1.2, label="笔记本（示意）")
ax.set_aspect("equal"); ax.set_xlim(-10, 265); ax.set_ylim(-8, 215)
ax.set_title(f"最终轮廓：面积 {area:.0f} mm²，最小杆宽≈{min_member} mm，最小孔宽≈{min_hole} mm")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.2)
ax = axs[1]
k = names.index("自重")
vmg = m.grid(np.where(inside, vm_all[k], np.nan))
im = ax.imshow(vmg, origin="lower", extent=[0, m.nelx * hv, 0, m.nely * hv], cmap="viridis")
ax.set_aspect("equal"); plt.colorbar(im, ax=ax, shrink=0.8, label="Von Mises (MPa)")
ax.set_title(f"复核：笔记本自重工况应力（最大 {res[k]['最大VonMises_MPa']:.2f} MPa，最大位移 {res[k]['最大位移_mm']*1000:.1f} µm）")
fig.savefig(f"{MD}/profile_and_stress.png", dpi=110, bbox_inches="tight")
print("saved")
