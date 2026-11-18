"""
校验 + 预览渲染：STL 水密性、卡扣臂插入量、估算打印重量、装配/爆炸/打印摆放图（支持多根横梁）
用法: python render.py
"""
import json, os
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon, box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import font_manager as fm
import params as P

for f in ("/mnt/c/Windows/Fonts/msyh.ttc",):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
OUT = P.MODEL_DIR

meshes = {nm: trimesh.load(f"{OUT}/{nm}.stl") for nm in ("support_left", "support_right", "beam")}
chk = {nm: dict(watertight=bool(m.is_watertight), faces=len(m.faces), volume_cm3=round(m.volume / 1000, 2),
                extents=[round(v, 1) for v in m.extents]) for nm, m in meshes.items()}

# --- 世界坐标装配（与 build3d.py 一致的变换） ---
def T_support_left():
    M = np.eye(4)
    M[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])   # local (x,y,z) -> (z, x, y)
    M[:3, 3] = [-P.SPACING / 2 - P.T / 2, 0, 0]
    return M
L = meshes["support_left"].copy(); L.apply_transform(T_support_left())
R = L.copy(); R.apply_transform(np.diag([-1, 1, 1, 1]))   # 镜像；trimesh 会自动翻转面朝向
def T_beam(j):
    """横梁打印坐标 → 世界坐标：X 沿横梁轴，截面长边 a → (0,cos φ,sin φ)，短边 b → (0,−sin φ,cos φ)"""
    ph = np.deg2rad(j["phi"]); c, s = np.cos(ph), np.sin(ph)
    M = np.eye(4)
    M[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    M[:3, 3] = [0, j["x"] + P.BEAM_W / 2 * s, j["y"] - P.BEAM_W / 2 * c]
    return M
Bs = [meshes["beam"].copy().apply_transform(T_beam(j)) for j in P.JOINTS]
NB = len(Bs)

# 卡扣臂插入量：横梁落在左支撑板厚范围内的体积（每根应 ≈ 2 × 2.5 × 16 × 15 = 1.2 cm³）
slab = trimesh.creation.box(extents=[P.T, 600, 600]); slab.apply_translation([-P.SPACING / 2, 150, 100])
try:
    ins = sum(trimesh.boolean.intersection([B, slab], engine="manifold").volume for B in Bs)
    clash = sum(trimesh.boolean.intersection([B, S], engine="manifold").volume for B in Bs for S in (L, R))
except Exception as e:
    ins = clash = float("nan"); print("boolean failed", e)
chk["beams"] = NB
chk["tenon_in_left_slab_cm3"] = round(ins / 1000, 2)
chk["beam_support_clash_mm3"] = round(clash, 3)

# --- 打印重量估算（3 圈墙 0.42 mm，顶底各 4 层 0.2 mm，内部 15% 填充，PLA 1.24 g/cm³） ---
prof = json.load(open(f"{OUT}/profile.json"))
pg = Polygon(prof["exterior"], prof["holes"])
wall = 3 * 0.42; skin = 2 * 4 * 0.2
shell_area = pg.area - pg.buffer(-wall).area
core_area = pg.buffer(-wall).area
v_sup = (shell_area * P.T + core_area * skin + core_area * (P.T - skin) * 0.15) / 1000
cs = box(0, 0, P.BEAM_H, P.BEAM_W); Lb = float(meshes["beam"].extents[0])
v_beam = ((cs.area - cs.buffer(-wall).area) * Lb + cs.length * Lb * 0 + 2 * cs.area * skin * 0 +
          (cs.buffer(-wall).area * (2 * skin / Lb) + cs.buffer(-wall).area * 0.15) * Lb) / 1000
# 横梁平躺打印：上下表面为 skin，四周为墙
core_h = P.BEAM_W - skin; core_w = P.BEAM_H - 2 * wall
v_beam = (Lb * (P.BEAM_H * P.BEAM_W - core_w * core_h) + Lb * core_w * core_h * 0.15) / 1000
chk["print_mass_g_est"] = dict(support_each=round(v_sup * 1.24), beam=round(v_beam * 1.24), beams=NB, total=round((2 * v_sup + NB * v_beam) * 1.24))

# --- 稳定性（倾覆）粗估：13-14 寸本 1.6 kg，机身 1.15 kg + 屏 0.45 kg，屏幕竖直 ---
mb, ms = 1.15, 0.45
xb, _ = P.sn2xy(110, 8)
xh, yh = P.sn2xy(215, 16)
x_cg = (mb * xb + ms * (xh + 3)) / (mb + ms)
stand_mass = (2 * v_sup + NB * v_beam) * 1.24 / 1000
x_stand = P.X_MAX / 2
h_push = yh + 200
W = 9.81 * (mb + ms); Ws = 9.81 * stand_mass
F_back = (W * (P.X_MAX - x_cg) + Ws * (P.X_MAX - x_stand)) / h_push
F_fwd = (W * x_cg + Ws * x_stand) / h_push
chk["tipping"] = dict(cg_x_mm=round(float(x_cg), 1), push_height_mm=round(float(h_push)), F_tip_backward_N=round(float(F_back), 1), F_tip_forward_N=round(float(F_fwd), 1))
json.dump(chk, open(f"{OUT}/check.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(chk, indent=1, ensure_ascii=False))

# --- 渲染 ---
def laptop_meshes():
    out = []
    # 机身：s∈[0,220], n∈[0,16], X∈±155
    def prism(s0, s1, n0, n1, X0, X1):
        corners = []
        for X in (X0, X1):
            for (s, n) in ((s0, n0), (s1, n0), (s1, n1), (s0, n1)):
                x, y = P.sn2xy(s, n); corners.append([X, x, y])
        return trimesh.convex.convex_hull(np.array(corners))
    out.append(prism(0, P.LAPTOP_D, 0, 16, -P.LAPTOP_W / 2, P.LAPTOP_W / 2))
    # 屏幕：竖直，从转轴向上 210，厚 6
    xh, yh = P.sn2xy(P.LAPTOP_D - 4, 16)
    lid = trimesh.creation.box(extents=[P.LAPTOP_W, 6, 210]); lid.apply_translation([0, xh + 3, yh + 105])
    out.append(lid)
    return out

LIGHT = np.array([0.35, -0.55, 0.75]); LIGHT /= np.linalg.norm(LIGHT)
def draw(ax, mesh, color, alpha=1.0, edge=False):
    tri = mesh.vertices[mesh.faces]
    nrm = mesh.face_normals
    shade = 0.35 + 0.65 * np.clip(nrm @ LIGHT, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(color))
    cols = np.clip(base[None, :] * shade[:, None], 0, 1)
    pc = Poly3DCollection(tri, facecolors=np.c_[cols, np.full(len(cols), alpha)], edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)

def setup(ax, lim, elev, azim):
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()

C_SUP, C_BEAM, C_LAP = "#4a78a8", "#e0913a", "#9a9a9a"
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for m_, c_ in [(L, C_SUP), (R, C_SUP)] + [(B, C_BEAM) for B in Bs]: draw(ax, m_, c_)
for lm in laptop_meshes(): draw(ax, lm, C_LAP, alpha=0.25)
setup(ax, ((-170, 170), (-10, 260), (0, 440)), 18, -125)
ax.set_title("装配 + 13/14 寸笔记本（半透明示意）")
ax = fig.add_subplot(1, 2, 2, projection="3d")
Le = L.copy(); Le.apply_translation([-60, 0, 0]); Re = R.copy(); Re.apply_translation([60, 0, 0])
for m_, c_ in [(Le, C_SUP), (Re, C_SUP)] + [(B, C_BEAM) for B in Bs]: draw(ax, m_, c_)
setup(ax, ((-200, 200), (-10, 260), (0, 200)), 22, -60)
ax.set_title("爆炸视图（从后方看：横梁两端卡扣穿过支撑方槽，卡钩扣住外侧面）")
fig.tight_layout(); fig.savefig(f"{OUT}/preview_assembly.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# 打印摆放（俯视）：H3C 热床；两片支撑放得下就同盘，横梁另一盘
from shapely import affinity
BW, BD = P.BED_W, P.BED_D
right_fl = affinity.scale(pg, 1, -1, origin=(0, 0))
_bo = json.load(open(f"{OUT}/beam_outline.json"))
beam_fl = Polygon(_bo["exterior"], _bo["holes"])
def place(g, cx, cy):
    mnx, mny, mxx, mxy = g.bounds
    return affinity.translate(g, cx - (mnx + mxx) / 2, cy - (mny + mxy) / 2)
sy = pg.bounds[3] - pg.bounds[1]; gap = 10
if 2 * sy + gap <= BD - 10 and (pg.bounds[2] - pg.bounds[0]) <= BW - 10:
    plates = [[(place(pg, BW / 2, BD / 2 + gap / 2 + sy / 2), C_SUP), (place(right_fl, BW / 2, BD / 2 - gap / 2 - sy / 2), C_SUP)],
              [(place(beam_fl, BW / 2, BD / 2 + (i - (NB - 1) / 2) * (P.BEAM_H + 15)), C_BEAM) for i in range(NB)]]
    titles = ["第 1 盘：支撑 ×2", f"第 2 盘：横梁 ×{NB}"]
else:
    plates = [[(place(pg, BW / 2, BD / 2), C_SUP)], [(place(right_fl, BW / 2, BD / 2), C_SUP)],
              [(place(beam_fl, BW / 2, BD / 2 + (i - (NB - 1) / 2) * (P.BEAM_H + 15)), C_BEAM) for i in range(NB)]]
    titles = ["支撑", "支撑", f"横梁 ×{NB}"]
fig, axs = plt.subplots(1, len(plates), figsize=(6 * len(plates), 5.6))
for ax, items, tt in zip(np.atleast_1d(axs), plates, titles):
    ax.add_patch(plt.Rectangle((0, 0), BW, BD, fill=False, ls="--", color="gray"))
    for g, c in items:
        xs, ys = g.exterior.xy; ax.fill(xs, ys, color=c, alpha=0.9)
        for it in g.interiors:
            xs, ys = it.xy; ax.fill(xs, ys, color="white")
    ax.set_aspect("equal"); ax.set_xlim(-5, BW + 5); ax.set_ylim(-5, BD + 5); ax.set_title(tt)
fig.suptitle(f"打印摆放（虚线 = {P.PRINTER} 热床 {BW:.0f}×{BD:.0f} mm；支撑任一侧面贴床（方槽贯通）；横梁 40 mm 面贴床；图为第一层截面）")
fig.savefig(f"{OUT}/preview_print_layout.png", dpi=100, bbox_inches="tight"); plt.close(fig)
print("rendered")
