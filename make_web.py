"""
生成网页预览：把各方案的网格 / 轮廓 / 密度场 / 复核数据嵌入 web/template.html → web/stand_preview.html
用法: python make_web.py [方案...]   默认 desk40 raised25（需先跑完对应方案的 run_all.sh）
"""
import os, sys, json, base64, importlib
import numpy as np
import trimesh
from shapely.geometry import Polygon, box
import params as P

LABELS = {"desk40": "40° · 选定", "raised25": "25° · 上一版"}
ORDER = sys.argv[1:] or ["desk40", "raised25"]
DEFAULT = "desk40" if "desk40" in ORDER else ORDER[0]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def panel_info():
    """面板说明和刚度：有 panel_contact（卡扣按真实接触重新优化）就用它，否则用 panel.py 的复核"""
    pc = f"{P.OUT_DIR}/panel_contact/verify.json"
    if os.path.exists(pc):
        v = json.load(open(pc)); r = {a["工况"]: (a, b) for a, b in zip(v["ref"], v["new"])}
        tw = [b["柔度"] / a["柔度"] for k, (a, b) in r.items() if "绕法线" in k]
        rk = r["侧向错动"]
        return dict(note="前、顶两个接口合成一块；卡扣按真实接触（台肩、槽壁、卡钩单向接触）拓扑优化，四分之一对称，外框加三角桁架；平放打印，四角卡扣与横梁端头相同",
                    spec=f"侧向错动的柔度是 X 斜撑版的 {rk[1]['柔度'] / rk[0]['柔度']:.2f}；单侧支撑绕面板法线扭 2 N·m 时是 {max(tw):.2f}")
    v = json.load(open(f"{P.OUT_DIR}/panel/verify.json"))
    return dict(note="前、顶两个接口合成一块，拓扑优化成 X 形斜撑；平放打印，四角卡扣与横梁端头相同",
                spec=f"上部受 {v['fe'][0]['力_N']:g} N 横向力，卡扣处错动 {v['fe'][0]['加载点最大位移_mm'] * 1000:.1f} µm")


def pack(fn):
    m = trimesh.load(fn)
    v = np.asarray(m.vertices); f = np.asarray(m.faces)
    assert len(v) < 65536
    mn = v.min(0); sz = v.max(0) - mn; sz[sz == 0] = 1
    q = np.round((v - mn) / sz * 65535).astype("<u2")
    return dict(min=mn.round(5).tolist(), size=sz.round(5).tolist(), pos=b64(q), idx=b64(f.astype("<u2")), nv=int(len(v)), nf=int(len(f)))


ring = lambda c: [[round(x, 1), round(y, 1)] for x, y in c]


def snap_section(md):
    """横梁右端卡扣剖面（u = 离支撑内侧面的距离），供网页画剖面图"""
    bo = json.load(open(f"{md}/beam_outline.json"))
    half = bo["half"]
    g = Polygon(bo["exterior"], bo["holes"]).intersection(box(half - 24, -30, half + 30, 30))
    polys = list(g.geoms) if hasattr(g, "geoms") else [g]
    r2 = lambda c: [[round(x - half, 2), round(y, 2)] for x, y in c]
    rep = json.load(open(f"{md}/build_report.json"))["snap"]
    return dict(beam=[dict(ext=r2(p.exterior.coords), holes=[r2(h.coords) for h in p.interiors]) for p in polys],
                T=P.T, slot_h=P.TENON_H / 2 + P.CLEAR, joint_h=P.BEAM_H / 2 + P.JOINT_MARGIN, **rep)
variants = {}
for v in ORDER:
    os.environ["STAND_VARIANT"] = v
    importlib.reload(P)
    md, td = P.MODEL_DIR, P.TOPO_DIR
    prof = json.load(open(f"{md}/profile.json"))
    pg = Polygon(prof["exterior"], prof["holes"]).simplify(0.08)
    domain, _ = P.geometry(True)
    dom = max(domain.geoms, key=lambda g: g.area) if hasattr(domain, "geoms") else domain
    r = np.load(f"{td}/result.npz")
    nelx, nely = int(r["nelx"]), int(r["nely"])
    G = r["xphys"].reshape(nelx, nely).T[::-1]
    verify = json.load(open(f"{md}/verify.json"))
    variants[v] = dict(
        label=LABELS.get(v, v),
        params=dict(mode=P.MODE, S=P.SPACING, T=P.T, joints=[dict(name=j["name"], cn=j["cn"], x=j["x"], y=j["y"], phi=j["phi"], part=j.get("part", "beam")) for j in P.JOINTS],
                    panel_ref=(P.panel_joints()[0]["name"] if P.panel_joints() else None),
                    slots=[ring(P.joint_slot(j).exterior.coords) for j in P.JOINTS], BW=P.BEAM_W, BH=P.BEAM_H, TW=P.TENON_W, TH=P.TENON_H,
                    CLEAR=P.CLEAR, theta=P.THETA_DEG, P0=[float(x) for x in P.P0], S_END=P.S_END, X_MAX=P.X_MAX,
                    LIP_T=P.LIP_T, LIP_H=P.LIP_H, LW=P.LAPTOP_W, LD=P.LAPTOP_D),
        snap=snap_section(md),
        meshes=dict(support=pack(f"{md}/support_left.stl"), beam=pack(f"{md}/beam.stl"),
                    **({"panel": pack(f"{md}/panel.stl")} if P.panel_joints() else {})),
        panel=panel_info() if P.panel_joints() else None,
        profile=dict(ext=ring(pg.exterior.coords), holes=[ring(h.coords) for h in pg.interiors]),
        domain=ring(dom.exterior.coords),
        density=dict(w=nelx, h=nely, hmm=float(r["h"]), data=b64(np.clip(np.round(G * 255), 0, 255).astype(np.uint8))),
        fe=verify["fe"], info=verify["info"], check=json.load(open(f"{md}/check.json")),
    )
data = dict(order=ORDER, default=DEFAULT, variants=variants, bed=dict(name=P.PRINTER, w=P.BED_W, d=P.BED_D))
html = open("web/template.html", encoding="utf-8").read()
assert "/*__DATA__*/null" in html
out = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
open("web/stand_preview.html", "w", encoding="utf-8").write(out)
print("web/stand_preview.html", len(out.encode()), "bytes;", ", ".join(f"{k}: {v['meshes']['support']['nf']} 面" for k, v in variants.items()))
