"""
无填充三维优化预览网页：把三维优化支撑、尖拱空腔对照、二维方案、面板、后横梁的网格和复核数据嵌入
web/template_noinfill.html → web/noinfill_preview.html
用法: STAND_VARIANT=desk40 python make_web_noinfill.py [--dir out/desk40/model_noinfill]   （需先运行 topopt3d_noinfill.py 和 export3d_noinfill.py）
"""
import argparse, base64, json, os
import numpy as np
import trimesh
import manifold3d as m3d
import params as P


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(m):
    v = np.asarray(m.vertices, float); f = np.asarray(m.faces)
    mn = v.min(0); sz = v.max(0) - mn; sz[sz == 0] = 1
    q = np.round((v - mn) / sz * 65535).astype("<u2")
    u32 = len(v) >= 65536
    return dict(min=mn.round(5).tolist(), size=sz.round(5).tolist(), pos=b64(q), idx=b64(f.astype("<u4" if u32 else "<u2")), u32=u32)


def display_mesh(fn, tol):
    """显示用网格：manifold 简化（tol mm 以内），面数降到原来的几分之一"""
    m = trimesh.load(fn)
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    mm = man.simplify(tol).to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def T_beam(j):
    """横梁 / 面板打印坐标 → 世界坐标（与 render.py 相同）"""
    ph = np.deg2rad(j["phi"]); c, s = np.cos(ph), np.sin(ph)
    M = np.eye(4); M[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    M[:3, 3] = [0, j["x"] + P.BEAM_W / 2 * s, j["y"] - P.BEAM_W / 2 * c]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{P.OUT_DIR}/model_noinfill")
    ap.add_argument("--tol", type=float, default=0.06)
    ap.add_argument("--cmp-dir", default=None, help="另一次导出的目录，取它的三维设计作对照（比值各按它自己的二维方案算），例如孔位改动前的那次")
    ap.add_argument("--names", default="新孔位三维,新孔位,旧孔位三维,旧孔位", help="--cmp-dir 时的标签：本次全称,本次简称,对照全称,对照简称")
    ap.add_argument("--lead", default="横梁孔上移 5 mm、接口区贴齐斜面后重新优化；", help="--cmp-dir 时导语前面加的话")
    ap.add_argument("--cmp-note", default="旧孔位三维：孔中心离斜面 18 mm 时的上一版（屈曲 ≥ 20、最薄 1.9 mm），比值按它自己那时的二维方案算。")
    a = ap.parse_args()
    ver = json.load(open(f"{a.dir}/verify.json"))
    res = {r["key"]: dict(r, label=nm) for nm, r in ver["results"].items()}
    if a.cmp_dir:
        v2 = json.load(open(f"{a.cmp_dir}/verify.json"))
        res["free_old"] = [dict(r, label=nm, key="free_old") for nm, r in v2["results"].items() if r["key"] == "free"][0]
    names = ver["names"]
    cases = [c for c in names if c not in ("侧推后段", "z向体力", "前部误用150N", "后部误用150N")]
    iF, iR = names.index("前部误用150N"), names.index("后部误用150N")
    cmp = "free_old" if "free_old" in res else "gable"
    keys = ("free", cmp, "flat")
    ha = ver["args"]
    cons = bool(ha.get("buck") or ha.get("tmin"))
    designs = []
    for k in keys:
        r = res[k]
        st = r["stress"]
        designs.append(dict(key=k, label=r["label"], mass=r["petg_g"], weighted=r["weighted_orig"],
                            rel={c: r["rel"][names.index(c)] for c in cases},
                            buck_f=r["buckling"]["前部误用150N"], buck_r=r["buckling"]["后部误用150N"],
                            vm_f=st[iF]["VM_99_9pct_MPa"], vm_r=st[iR]["VM_99_9pct_MPa"],
                            z_f=st[iF]["z向位移_mm"] * 1000, z_r=st[iR]["z向位移_mm"] * 1000))
    src = dict(free=f"{a.dir}/support_noinfill_left.stl", gable=f"{a.dir}/support_gable_left.stl",
               free_old=(f"{a.cmp_dir}/support_noinfill_left.stl" if a.cmp_dir else
                         f"{a.dir}/support_ref_left.stl" if os.path.exists(f"{a.dir}/support_ref_left.stl") else f"{ver.get('ref_model_dir')}/support_noinfill_left.stl"))
    meshes = {k: display_mesh(src[k], a.tol) for k in ("free", cmp)}
    meshes["flat"] = trimesh.load(f"{P.MODEL_DIR}/support_left.stl")
    nm4 = a.names.split(",")
    labels = dict(free=nm4[0] if a.cmp_dir else "新三维" if cons else "无填充三维", free_old=nm4[2] if a.cmp_dir else "旧三维（无约束）", gable="尖拱空腔", flat="二维方案")
    short = dict(free=nm4[1] if a.cmp_dir else "新三维" if cons else "三维", free_old=nm4[3] if a.cmp_dir else "旧三维", gable="尖拱", flat="二维")
    Ls = np.eye(4); Ls[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]; Ls[:3, 3] = [-P.SPACING / 2 - P.T / 2, 0, 0]
    Rs = np.diag([-1, 1, 1, 1]) @ Ls
    rear = [j for j in P.JOINTS if j.get("part", "beam") == "beam"][0]
    m0 = meshes["free"]
    gw = ver.get("gable_wall_mm")
    notes = [
        "打印：侧面贴床，不加支撑。切片时内部按实心打印（墙圈数加够，或稀疏填充设 100%）；三个横梁接口区加修改器，仍用 4 圈墙 + 15–25% 填充。",
        ("没有封闭空腔（体素模型里两个十几 mm³ 的小空腔，顶是平的，在补 45° 斜坡时被补实了）。" if not ver["mesh"]["cavities_mm3"] else
         "封闭空腔留在零件里面，FDM 不用清理。" + ("空腔顶部按 45° 逐层收成尖顶，不用搭桥。" if ver["overhang"]["短桥面积_mm2"] < 1 else "空腔顶部最后一层是跨度不到 2 mm 的短桥。")),
        "左右两片是镜像件，都是 z = 0 的那面贴床；装配时这一面朝外，卡钩扣在这一面。",
        ("尖拱空腔对照：同一个二维轮廓，内部挖成 45° 尖顶空腔，墙厚 " + (f"{gw:.2f}" if gw else "—") + " mm 时与二维方案同样用料。") if cmp == "gable" else
        (a.cmp_note if a.cmp_dir else
         "旧三维：上一版无填充三维优化（只有用料约束），同一个三维模型、同样的复核方法重新算的。"),
        "侧向位移：面内载荷引起的厚度方向（z）位移。误用工况按装配状态算：面板和横梁拉住三个接口的 z 向。",
    ]
    sm = ver.get("smooth")
    if sm:
        sv = sm["sigma_vox"]; sv = sv if isinstance(sv, list) else [sv, sv]
        sig = f"{sv[0]:g}" if sv[0] == sv[1] else f"平面内 {sv[0]:g}、厚度方向 {sv[1]:g}"
        notes.insert(1, f"表面：优化网格是 {P.T / int(ha['nz']):.3f} mm 的立方体素，直接导出的话缓坡上是这么高的台阶。STL 先把体素平滑（σ = {sig} 个体素），"
                        f"再按 {sm['layer_mm']:g} mm 打印层重建：伸出超过 45° 的地方在下面补斜坡，再削掉补不到的细边，用料调回原值；复核数据也是按平滑后的形状算的。")
    B = np.load(f"{a.dir}/binary.npz")["B"]
    t = B.sum(2); thin = float((t[t > 0] <= 1).mean())          # 只有一层体素厚的柱子占比
    th = res["free"].get("thin"); tho = res.get(cmp, {}).get("thin")
    thin_fact = None
    if th:
        thin_fact = (f"比 {th['tmin_mm']:g} mm 薄的部分占体积 {th['薄于tmin体积比例'] * 100:.1f}%" + (f"（{short[cmp]} {tho['薄于tmin体积比例'] * 100:.0f}%）" if tho else "")
                     + f"；只有一层体素厚的柱子 {th['单层柱子比例'] * 100:.1f}%" + (f"（{short[cmp]} {tho['单层柱子比例'] * 100:.0f}%）" if tho else ""))
    if cons:
        notes.insert(0, f"这一版在优化里加了两个约束：两个 150 N 误用工况的线性屈曲系数 ≥ {ha.get('buck', 0):g}（按腐蚀后的设计算，偏保守），"
                        f"最小厚度 {ha.get('tmin', 0):g} mm（比它薄的板和杆在腐蚀设计里消失，只吃料不出力）。")
    pads_stl = f"{P.OUT_DIR}/pads/pads_assembled_one_support.stl"
    pads_info = None
    if os.path.exists(pads_stl):          # 可选：TPU 卡套（pads.py）
        pj = json.load(open(f"{P.OUT_DIR}/pads/pads.json"))
        pads_info = (f"U 形卡套，每片支撑 4 件（顶面垫、前挡垫、前后脚套），两片共 {pj['合计_g（两片支撑）']:g} g TPU；"
                     f"顶面垫底厚 {pj['params']['base']:g} mm，脚套底厚 {pj['params']['foot']:g} mm")
    data = dict(T=P.T, thin=thin, thinFact=thin_fact, cmpKey=cmp, short=short,
                leadExtra=((a.lead if a.cmp_dir else "") + (f"这一版加了屈曲系数 ≥ {ha.get('buck', 0):g} 和最小厚度 {ha.get('tmin', 0):g} mm 两个约束。" if cons else "")), nz=(int(round(P.T / ha["dz"])) if ha.get("dz") else int(ha["nz"])), h=(ha["dz"] if ha.get("dz") else P.T / int(ha["nz"])),
                hxy=(ha.get("hxy") or None), layer=(sm["layer_mm"] if sm else ha.get("dz")), warp=float(ha.get("warp", 0.0)), bed=[P.BED_W, P.BED_D],
                center=[float(m0.bounds[:, 0].mean()), float(m0.bounds[:, 1].mean())], xmax=float(m0.bounds[1, 0]),
                cavities=ver["mesh"]["cavities_mm3"], overhang=dict(area=ver["overhang"]["悬空总面积_mm2"], bridge=ver["overhang"]["短桥面积_mm2"]),
                cases=cases, designs=designs, notes=notes,
                models=[dict(key=k, label=labels[k], mesh=pack(meshes[k])) for k in keys],
                supportMats=[Ls.ravel().round(6).tolist(), Rs.ravel().round(6).tolist()],
                beamMat=T_beam(rear).ravel().round(6).tolist(), panelMat=T_beam(P.panel_joints()[0]).ravel().round(6).tolist(),
                beam=pack(trimesh.load(f"{P.MODEL_DIR}/beam.stl")), panel=pack(trimesh.load(f"{P.MODEL_DIR}/panel.stl")),
                pads=pack(trimesh.load(pads_stl)) if pads_info else None, padsInfo=pads_info)
    html = open("web/template_noinfill.html", encoding="utf-8").read().replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open("web/noinfill_preview.html", "w", encoding="utf-8").write(html)
    print("网格面数", {k: len(m.faces) for k, m in meshes.items()}, f"；页面 {len(html)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
