"""Slice STL via local Bambu Studio CLI using a reference .3mf template (bambu_slice.py).

By default applies Stage 1 pure path-planning settings (no speed/accel/flow/mode changes):
  - wall_generator = arachne
  - enable_arc_fitting = 1
  - resolution = 0.05
  - seam_position = nearest

Usage:
    python bambu_slice.py template.3mf model.stl outdir [--raw] [--rot DEG] [--set k=v ...]
"""
import argparse
import json
import os
import re
import subprocess
import zipfile
import numpy as np
import trimesh

STAGE1_OVERRIDES = {
    "wall_generator": "arachne",
    "enable_arc_fitting": "1",
    "resolution": "0.05",
    "seam_position": "nearest",
}


def slice_with_bambu(
    template_3mf,
    stl_path,
    outdir,
    bambu_dir=None,
    overrides=None,
    raw=False,
    rot=None,
    bed="165,160",
):
    """Replace mesh in template 3MF, apply config overrides, and slice with local Bambu Studio."""
    if bambu_dir is None:
        bambu_dir = os.path.expanduser("~/.local/opt/bambustudio")
    os.makedirs(outdir, exist_ok=True)

    m = trimesh.load(stl_path, force="mesh")
    lo, hi = m.bounds
    c = (lo + hi) / 2.0

    with zipfile.ZipFile(template_3mf) as zin:
        tm = zin.read("3D/3dmodel.model").decode("utf-8")
        ms = zin.read("Metadata/model_settings.config").decode("utf-8")
        obj = zin.read("3D/Objects/object_1.model").decode("utf-8")

        off = re.search(
            r'(<component [^>]*transform="1 0 0 0 1 0 0 0 1 )([-\d.]+) ([-\d.]+) ([-\d.]+)(")',
            tm,
        )
        V = m.vertices - c
        verts = "\n".join('     <vertex x="%.6f" y="%.6f" z="%.6f"/>' % tuple(v) for v in V)
        tris = "\n".join('     <triangle v1="%d" v2="%d" v3="%d"/>' % tuple(f) for f in m.faces)
        head = obj[: obj.index("<vertices>") + len("<vertices>")]
        tail = obj[obj.index("</triangles>") :]
        newobj = head + "\n" + verts + "\n    </vertices>\n    <triangles>\n" + tris + "\n    " + tail

        tm2 = tm
        if off is not None and any(abs(float(g)) > 1e-6 for g in off.groups()[1:4]):
            tm2 = tm.replace(
                off.group(0),
                "%s%.6f %.6f %.6f%s" % (off.group(1), c[0], c[1], c[2], off.group(5)),
            )

        if rot is not None:
            th = np.radians(rot)
            C, S = np.cos(th), np.sin(th)
            Pxy = m.vertices[:, :2] @ np.array([[C, S], [-S, C]])
            mid = (Pxy.min(0) + Pxy.max(0)) / 2.0
            bx, by = map(float, bed.split(","))
            item = re.search(r'(<item [^>]*transform=")([^"]+)(")', tm2)
            new_t = "%.9f %.9f 0 %.9f %.9f 0 0 0 1 %.5f %.5f 0" % (
                C, S, -S, C, bx - mid[0], by - mid[1]
            )
            tm2 = tm2.replace(item.group(0), item.group(1) + new_t + item.group(3))

        ms2 = re.sub(r'face_count="\d+"', 'face_count="%d"' % len(m.faces), ms)
        ms2 = re.sub(
            r'(key="matrix" value="1 0 0 )[-\d.]+( 0 1 0 )[-\d.]+( 0 0 1 )[-\d.]+',
            lambda mm: "%s%.6f%s%.6f%s%.6f" % (mm.group(1), c[0], mm.group(2), c[1], mm.group(3), c[2]),
            ms2,
        )
        for k, v in zip("xyz", c):
            ms2 = re.sub(
                r'(key="source_offset_%s" value=")[-\d.]+' % k,
                lambda mm: mm.group(1) + "%.6f" % v,
                ms2,
            )
        ms2 = re.sub(r'value="[^"]*\.stl"', 'value="%s"' % os.path.basename(stl_path), ms2)

        cfg_overrides = {} if raw else dict(STAGE1_OVERRIDES)
        if overrides:
            cfg_overrides.update(overrides)

        src = os.path.join(outdir, "in.3mf")
        with zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED) as zout:
            for it in zin.infolist():
                if it.filename.startswith("Metadata/plate_") or it.filename == "Metadata/slice_info.config":
                    continue
                data = zin.read(it.filename)
                if it.filename == "3D/Objects/object_1.model":
                    data = newobj.encode("utf-8")
                elif it.filename == "3D/3dmodel.model":
                    data = tm2.encode("utf-8")
                elif it.filename == "Metadata/model_settings.config":
                    data = ms2.encode("utf-8")
                elif it.filename == "Metadata/project_settings.config" and cfg_overrides:
                    cfg = json.loads(data)
                    for k, v in cfg_overrides.items():
                        if k not in cfg:
                            raise SystemExit(f"Unknown config key: {k}")
                        cfg[k] = [v] * len(cfg[k]) if isinstance(cfg[k], list) else v
                    data = json.dumps(cfg, indent=4).encode("utf-8")
                zout.writestr(it, data)

    env = dict(
        os.environ,
        LC_ALL="C",
        LD_LIBRARY_PATH=bambu_dir + "/squashfs-root/bin:" + bambu_dir + "/libs/usr/lib/x86_64-linux-gnu",
    )
    proc = subprocess.run(
        [
            bambu_dir + "/squashfs-root/bin/bambu-studio",
            "--slice", "1",
            "--arrange", "0",
            "--outputdir", outdir,
            "--export-3mf", "sliced.3mf",
            src,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    res_path = os.path.join(outdir, "result.json")
    if not os.path.exists(res_path):
        raise RuntimeError(f"Bambu Studio slice failed: {proc.stderr[-400:]}")
    res = json.load(open(res_path))
    plate = res["sliced_plates"][0] if res.get("sliced_plates") else {}
    pred_s = int(round(plate.get("main_predication", 0)))

    sliced_3mf = os.path.join(outdir, "sliced.3mf")
    wt = "?"
    retrs = 0
    g23 = 0
    with zipfile.ZipFile(sliced_3mf) as zs:
        gc = zs.read("Metadata/plate_1.gcode").decode("utf-8", errors="ignore")
        for ln in gc.splitlines()[:25]:
            if "total filament weight" in ln:
                wt = ln.split(":")[-1].strip()
        retrs = gc.count("; WIPE_START")
        g23 = sum(1 for ln in gc.splitlines() if (ln.startswith("G2 ") or ln.startswith("G3 ")) and " E" in ln)

    return dict(
        status=res.get("error_string", ""),
        pred_s=pred_s,
        weight_g=wt,
        retracts=retrs,
        arcs=g23,
        feature_times=plate.get("feature_type_times", {}),
        sliced_3mf=sliced_3mf,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="Reference .3mf / .gcode.3mf template")
    ap.add_argument("stl", help="Model STL to slice")
    ap.add_argument("outdir", help="Output directory")
    ap.add_argument("--bambu", default=os.path.expanduser("~/.local/opt/bambustudio"))
    ap.add_argument("--raw", action="store_true", help="Do not apply Stage 1 path-planning overrides")
    ap.add_argument("--rot", type=float, default=None)
    ap.add_argument("--bed", default="165,160")
    ap.add_argument("--set", action="append", default=[], help="Override key=value")
    a = ap.parse_args()

    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = v

    info = slice_with_bambu(
        a.template, a.stl, a.outdir, bambu_dir=a.bambu, overrides=overrides, raw=a.raw, rot=a.rot, bed=a.bed
    )
    t = info["pred_s"]
    print(
        f"{info['status']}  Bambu {t//3600}h {t%3600//60:02d}m {t%60:02d}s ({t}s) | "
        f"wt={info['weight_g']}g | retr={info['retracts']} | G2/G3={info['arcs']}"
    )
    for k, v in sorted(info["feature_times"].items(), key=lambda kv: -kv[1]):
        if v >= 1.0:
            print(f"  {k:<26} {v:7.1f}s")


if __name__ == "__main__":
    main()
