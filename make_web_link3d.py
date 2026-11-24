"""
面板 / 后横梁三维优化预览网页：现在的插指版零件和 link3d.py 的三维结果放在一起（可旋转、剖切），附各工况柔度比
web/template_link3d.html → web/link3d_preview.html
用法: python make_web_link3d.py --beam 标签=目录 [...] --panel 标签=目录 [...]   （目录里要有 report.json 和 <part>_3d.stl）
"""
import argparse, base64, json, os
import numpy as np
import trimesh
import manifold3d as m3d
import params as P


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(fn, tol):
    m = trimesh.load(fn)
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    if tol > 0:
        man = man.simplify(tol)
    mm = man.to_mesh()
    v = np.asarray(mm.vert_properties, float)[:, :3]; f = np.asarray(mm.tri_verts)
    mn = v.min(0); sz = v.max(0) - mn; sz[sz == 0] = 1
    q = np.round((v - mn) / sz * 65535).astype("<u2")
    u32 = len(v) >= 65536
    return dict(min=mn.round(4).tolist(), size=sz.round(4).tolist(), pos=b64(q), idx=b64(f.astype("<u4" if u32 else "<u2")), u32=u32, ntri=int(len(f)))


def part_data(part, specs, tol):
    cur = f"{P.OUT_DIR}/clip_trial/{part}_clip.stl"
    reps = []
    for spec in specs:
        label, d = spec.split("=", 1)
        reps.append((label, d, json.load(open(f"{d}/report.json"))))
    ref = reps[0][2]
    names = [r["工况"] for r in ref["new"]]; groups = [r["组"] for r in ref["new"]]
    # 同一零件下的多个对比方案用同一套权重算加权柔度：优先用含 weights 字段的方案（新版 relaxed），否则回退旧权重
    w_src = next((rep["weights"] for _, _, rep in reps if "weights" in rep), None)
    w = np.array(w_src, float) if w_src is not None else np.array([1.0 if (part, n) in (("panel", "侧向错动"), ("beam", "前后错动")) else 0.5 for n in names])
    gi = np.array([g == "面内" for g in groups])
    designs, ratios = [], {}
    for label, d, rep in reps:
        key = os.path.basename(os.path.normpath(d))
        ratios[key] = [round(float(x), 3) for x in rep["ratio"]]
        r = np.array(rep["ratio"])
        designs.append(dict(key=key, label=label, mass_g=rep["mass_g"], proj=rep.get("proj_mm2"), h=rep["h"],
                            weighted=round(float(w @ r / w.sum()), 3), inplane=round(float(w[gi] @ r[gi] / w[gi].sum()), 3),
                            outplane=round(float(w[~gi] @ r[~gi] / w[~gi].sum()), 3), mesh=pack(f"{d}/{part}_3d.stl", tol)))
    cur_d = dict(key="current", label="现在（插指版，二维拉伸）", mass_g=ref["ref_mass_g"], proj=ref.get("ref_proj_mm2"), h=ref["h"],
                 weighted=1.0, inplane=1.0, outplane=1.0, mesh=pack(cur, 0))
    ratios["current"] = [1.0] * len(names)
    return dict(part=part, designs=[cur_d] + designs, ratios=ratios, cases=names, groups=groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", nargs="+", default=[])
    ap.add_argument("--panel", nargs="+", default=[])
    ap.add_argument("--tol", type=float, default=0.08, help="显示用网格简化容差 mm")
    ap.add_argument("--out", default="web/link3d_preview.html")
    a = ap.parse_args()
    data = dict(parts=[])
    if a.beam:
        data["parts"].append(dict(part_data("beam", a.beam, a.tol), name="后横梁"))
    if a.panel:
        data["parts"].append(dict(part_data("panel", a.panel, a.tol), name="面板"))
    html = open("web/template_link3d.html", encoding="utf-8").read().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open(a.out, "w", encoding="utf-8").write(html)
    print(a.out, f"{len(html) / 1e6:.2f} MB", [(p["part"], [(d["key"], d["mesh"]["ntri"]) for d in p["designs"]]) for p in data["parts"]])


if __name__ == "__main__":
    main()
