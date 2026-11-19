"""
三维优化预览网页：把三维优化支撑、二维方案支撑、横梁网格和复核数据嵌入 web/template3d.html → web/support3d_preview.html
用法: STAND_VARIANT=desk40 python make_web3d.py   （需先运行 topopt3d.py 和 export3d.py）
"""
import base64, json
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon
import params as P
import export3d as E


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(v, f):
    v = np.asarray(v, float); f = np.asarray(f)
    mn = v.min(0); sz = v.max(0) - mn; sz[sz == 0] = 1
    q = np.round((v - mn) / sz * 65535).astype("<u2")
    u32 = len(v) >= 65536
    return dict(min=mn.round(5).tolist(), size=sz.round(5).tolist(), pos=b64(q), idx=b64(f.astype("<u4" if u32 else "<u2")), u32=u32)


r = np.load(f"{P.OUT_DIR}/topo3d/result3d.npz")
NX, NY, h, dz, NZ = int(r["NX"]), int(r["NY"]), float(r["h"]), float(r["dz"]), int(r["NZ"])
regs = E.level_regions(r["tq"], NX, NY, h, dz)
m3 = E.build_mesh({n: g.simplify(0.08) for n, g in regs.items()}, dz)          # 显示用，轮廓放粗一点
m2 = trimesh.load(f"{P.MODEL_DIR}/support_left.stl")
mb = trimesh.load(f"{P.MODEL_DIR}/beam.stl")
ver = json.load(open(f"{P.OUT_DIR}/model3d/verify3d.json"))
prof = json.load(open(f"{P.MODEL_DIR}/profile.json"))
pg2 = Polygon(prof["exterior"], prof["holes"])

# 每层截面积（cm²）：第 k 层（z 从 (k-1)·dz 到 k·dz）用列高 ≥ k·dz 的轮廓
keys = sorted(regs)
area3 = [round(regs[min([n for n in keys if n >= k], default=keys[-1])].area / 100, 3) if k <= keys[-1] else 0.0 for k in range(1, NZ + 1)]
nT = int(round(P.T / dz))
area2 = [round(pg2.area / 100, 3) if k <= nT else 0.0 for k in range(1, NZ + 1)]

# 装配矩阵（与 render.py 相同；行优先）
Ls = np.eye(4); Ls[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]; Ls[:3, 3] = [-P.SPACING / 2 - P.T / 2, 0, 0]
Rs = np.diag([-1, 1, 1, 1]) @ Ls
beams = []
for j in P.JOINTS:
    ph = np.deg2rad(j["phi"]); c, s = np.cos(ph), np.sin(ph)
    M = np.eye(4); M[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    M[:3, 3] = [0, j["x"] + P.BEAM_W / 2 * s, j["y"] - P.BEAM_W / 2 * c]
    beams.append(M.ravel().round(6).tolist())

names = ver["names"]
mis = names.index("后部误用150N")
st2, st3 = ver["raw"]["二维方案拉伸"]["stress"][mis], ver["raw"]["三维优化"]["stress"][mis]
hist = json.load(open(f"{P.OUT_DIR}/topo3d/history3d.json"))
data = dict(
    nz=NZ, dz=dz, tmax=NZ * dz, kz=hist["args"]["kz"], bed=[P.BED_W, P.BED_D],
    center=[float(m3.bounds[:, 0].mean()), float(m3.bounds[:, 1].mean())],
    vol3=float(E.trimesh.load(f"{P.OUT_DIR}/model3d/support3d_left.stl").volume / 1000), vol2=pg2.area * P.T / 1000,
    mass3=ver["mass"]["三维优化"]["petg_g"], mass2=ver["mass"]["二维方案拉伸"]["petg_g"],
    setting=ver["print_setting"], weighted=ver["weighted_ratio"], weighted_solid=ver["weighted_ratio_solid"],
    warp=[st2["z向位移_mm"], st3["z向位移_mm"]], inplane=[st2["面内位移_mm"], st3["面内位移_mm"]],
    table=[dict(name=row["工况"], ratio=row["柔度比"]) for row in ver["table"] if row["工况"] not in ("前部误用150N", "后部误用150N")],
    stress=[["Von Mises", st2["VM_99_9pct_MPa"], st3["VM_99_9pct_MPa"]],
            ["层间拉应力", st2["层间拉应力_99_9pct_MPa"], st3["层间拉应力_99_9pct_MPa"]],
            ["层间剪应力", st2["层间剪应力_99_9pct_MPa"], st3["层间剪应力_99_9pct_MPa"]]],
    area3=area3, area2=area2,
    supportMats=[Ls.ravel().round(6).tolist(), Rs.ravel().round(6).tolist()], beamMats=beams,
    mesh3=pack(m3.vertices, m3.faces), mesh2=pack(m2.vertices, m2.faces), beam=pack(mb.vertices, mb.faces),
)
html = open("web/template3d.html", encoding="utf-8").read().replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
open("web/support3d_preview.html", "w", encoding="utf-8").write(html)
print(f"三维支撑显示网格 {len(m3.vertices)} 顶点 / {len(m3.faces)} 面；页面 {len(html)/1e6:.2f} MB")
