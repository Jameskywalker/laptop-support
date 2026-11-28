"""Comprehensive quality, physics, and geometry verification gate for planned .gcode.3mf (verify_toolpath.py).

Usage:
    python verify_toolpath.py stage1.gcode.3mf planned.gcode.3mf [--layers 5,10,20,30,40,50,60,65,70]
"""
import argparse
import collections
import hashlib
import math
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

import gcode_time as GT
import infill_spiral as IS

NUM_RE = re.compile(r"([XYZEFIJPS])(-?\d*\.?\d+)")


def _extrusion_stats_by_layer(lines):
    """Return (layer_vol_mm3_dict, feature_len_dict, ow_speed_hist, layer_lines_dict)."""
    chg = [i for i, l in enumerate(lines) if l.startswith("; CHANGE_LAYER")]
    end_idx = next((i for i, l in enumerate(lines) if "; MACHINE_END_GCODE_START" in l), len(lines))
    n_layers = len(chg)
    layer_chunks = {
        k + 1: lines[chg[k] : (chg[k + 1] if k + 1 < n_layers else end_idx)]
        for k in range(n_layers)
    }

    x = y = 0.0
    F = 600.0
    feat = ""
    inwipe = False
    layer = 0
    layer_e = collections.Counter()
    feat_len = collections.Counter()
    ow_speeds = collections.Counter()

    for ln in lines:
        st = ln.strip()
        if st.startswith("; CHANGE_LAYER"):
            layer += 1
        elif st.startswith("; FEATURE:"):
            feat = st[10:].strip()
        elif st.startswith("; WIPE_START"):
            inwipe = True
        elif st.startswith("; WIPE_END"):
            inwipe = False
        code = st.split(";", 1)[0].strip()
        if not code:
            continue
        w = code.split()[0]
        if w not in ("G0", "G1", "G2", "G3"):
            continue
        pr = {k: float(v) for k, v in NUM_RE.findall(code)}
        if "F" in pr:
            F = pr["F"]
        nx, ny = pr.get("X", x), pr.get("Y", y)
        de = pr.get("E", 0.0)
        has_arc = "I" in pr or "J" in pr
        if de > 1e-9 and not inwipe and (nx != x or ny != y or has_arc):
            if w in ("G2", "G3") and has_arc:
                dl, *_ = GT._planar_arc_geom(w, x, y, nx, ny, pr.get("I", 0.0), pr.get("J", 0.0))
            else:
                dl = math.hypot(nx - x, ny - y)
            layer_e[layer] += de
            feat_len[feat] += dl
            if feat == "Outer wall":
                ow_speeds[round(F / 60.0, 1)] += round(dl, 3)
        x, y = nx, ny

    layer_vol = {L: e * IS.FILA_AREA for L, e in layer_e.items()}
    return layer_vol, dict(feat_len), dict(ow_speeds), layer_chunks


def _layer_extrusion_footprint(lines, up_to_layer):
    """Build physical extruded footprint Polygon for each requested layer directly from G-code."""
    x = y = 0.0
    lw = 0.42
    inwipe = False
    layer = 0
    targets = set(up_to_layer)
    by_layer_r = collections.defaultdict(lambda: collections.defaultdict(list))
    cur_pts = []
    cur_r = None
    cur_L = None

    def flush():
        nonlocal cur_pts, cur_r, cur_L
        if cur_L in targets and len(cur_pts) >= 2 and cur_r is not None:
            by_layer_r[cur_L][cur_r].append(LineString(cur_pts))
        cur_pts = []
        cur_r = None

    for raw in lines:
        st = raw.strip()
        if st.startswith("; CHANGE_LAYER"):
            flush()
            layer += 1
            cur_L = layer
            if layer > max(targets):
                break
            continue
        if st.startswith("; LINE_WIDTH:"):
            flush()
            lw = float(st[13:].strip())
            continue
        if st.startswith("; WIPE_START"):
            flush()
            inwipe = True
            continue
        if st.startswith("; WIPE_END"):
            flush()
            inwipe = False
            continue
        code = st.split(";", 1)[0].strip()
        if not code:
            continue
        w = code.split()[0]
        if w not in ("G0", "G1", "G2", "G3"):
            continue
        pr = {k: float(v) for k, v in NUM_RE.findall(code)}
        nx, ny = pr.get("X", x), pr.get("Y", y)
        de = pr.get("E", 0.0)
        has_arc = "I" in pr or "J" in pr
        if layer in targets and de > 1e-9 and not inwipe and (nx != x or ny != y or has_arc):
            r = round(0.5 * lw, 3)
            pts = (
                IS._sample_arc(w, x, y, nx, ny, pr.get("I", 0.0), pr.get("J", 0.0), step=0.25)
                if (w in ("G2", "G3") and has_arc)
                else [(x, y), (nx, ny)]
            )
            if cur_r == r and cur_pts and math.hypot(cur_pts[-1][0] - pts[0][0], cur_pts[-1][1] - pts[0][1]) < 1e-3:
                cur_pts.extend(pts[1:])
            else:
                flush()
                cur_r = r
                cur_pts = list(pts)
        else:
            flush()
        x, y = nx, ny
    flush()

    out = {}
    for L in targets:
        polys = [
            shapely.MultiLineString(ls_list).buffer(r, cap_style=1, join_style=1, quad_segs=2)
            for r, ls_list in by_layer_r[L].items()
            if ls_list
        ]
        out[L] = unary_union(polys) if polys else Polygon()
    return out


def verify(stage1_3mf, planned_3mf, check_layers):
    ok = True
    # 1. 3MF structural integrity
    with zipfile.ZipFile(planned_3mf) as z:
        names = set(z.namelist())
        gname = [n for n in names if re.match(r"Metadata/plate_\d+\.gcode$", n)][0]
        gbytes = z.read(gname)
        gtext = gbytes.decode("utf-8")
        md5_actual = hashlib.md5(gbytes).hexdigest().upper()
        md5_file = z.read(gname + ".md5").decode("utf-8").strip().upper()
        if md5_actual != md5_file:
            print(f"FAIL: MD5 mismatch {md5_actual} != {md5_file}")
            ok = False
        ET.fromstring(z.read("Metadata/slice_info.config").decode("utf-8"))
        m73_lines = [l for l in gtext.splitlines() if l.startswith("M73 L") or l.startswith("M73 P")]
        n_layers_pl = sum(1 for l in gtext.splitlines() if l.startswith("; CHANGE_LAYER"))
        if len(m73_lines) < max(20, 2 * n_layers_pl - 6):
            print(f"FAIL: Insufficient M73 lines ({len(m73_lines)} for {n_layers_pl} layers)")
            ok = False

    lines_s1 = GT.load_gcode_lines(stage1_3mf)
    lines_pl = gtext.split("\n")

    # 2. Kinematic & print-mode audit
    t_s1, _, _, _, aud_s1 = GT.simulate(lines_s1, check_limits=True)
    t_pl, _, cat_pl, _, aud_pl = GT.simulate(lines_pl, check_limits=True)
    if aud_pl["n_q_violations"] or aud_pl["n_arc_violations"] or aud_pl["n_v_violations"]:
        print(f"FAIL: Kinematic violation: {aud_pl}")
        ok = False

    # 3. Extrusion conservation, Layer 1 & Layer N verbatim, Outer wall speed preservation
    vol_s1, flen_s1, ows_s1, ch_s1 = _extrusion_stats_by_layer(lines_s1)
    vol_pl, flen_pl, ows_pl, ch_pl = _extrusion_stats_by_layer(lines_pl)

    tot_v_s1 = sum(vol_s1.values())
    tot_v_pl = sum(vol_pl.values())
    wt_diff_pct = 100.0 * (tot_v_pl - tot_v_s1) / tot_v_s1
    if not (-3.5 <= wt_diff_pct <= 5.5):
        print(f"FAIL: Total volume diff {wt_diff_pct:+.2f}% outside [-3.5%, +5.5%] vs Stage 1")
        ok = False

    # Layer 1 and Layer N extrusion exact check
    for L in (1, max(ch_s1)):
        if abs(vol_s1[L] - vol_pl[L]) > 0.05:
            print(f"FAIL: Layer {L} volume changed ({vol_s1[L]:.3f} -> {vol_pl[L]:.3f})")
            ok = False

    # Outer wall length & speed preservation
    ow_len_diff = abs(flen_pl.get("Outer wall", 0) - flen_s1.get("Outer wall", 0))
    if ow_len_diff > 1.0:
        print(f"FAIL: Outer wall length changed by {ow_len_diff:.2f} mm")
        ok = False
    for spd, l1 in ows_s1.items():
        l2 = ows_pl.get(spd, 0.0)
        if abs(l1 - l2) > 1.0:
            print(f"FAIL: Outer wall speed {spd} mm/s length mismatch ({l1:.2f} vs {l2:.2f})")
            ok = False

    # 4. Solid slice volume & coverage audit
    man = IS.load_bed_manifold_from_3mf(planned_3mf)
    n_layers = max(ch_pl)
    solids = IS.slice_all_solids(man, [(L - 0.5) * 0.2 for L in range(1, n_layers + 1)])

    if check_layers is None:
        lo_l = max(2, round(0.05 * n_layers))
        hi_l = max(lo_l, n_layers - 2)
        base_ls = [max(2, min(n_layers - 1, round(lo_l + i * (hi_l - lo_l) / 8.0))) for i in range(9)]
        trans_ls = [k + 1 for k in range(1, n_layers - 1) if solids[k].area - solids[k + 1].area > 200.0]
        check_layers = sorted(set(base_ls + trans_ls))
        print(f"Auto-selected check layers ({len(check_layers)}): {check_layers}")

    vol_ratios = [vol_pl[L] / max(solids[L - 1].area * 0.2, 1e-6) for L in range(2, n_layers)]
    mean_vr = sum(vol_ratios) / len(vol_ratios)
    min_vr, max_vr = min(vol_ratios), max(vol_ratios)
    if not (0.96 <= mean_vr <= 1.04) or min_vr < 0.94 or max_vr > 1.07:
        print(f"FAIL: Layer volume ratios out of range: mean={mean_vr:.3f}, min={min_vr:.3f}, max={max_vr:.3f}")
        ok = False

    foots = _layer_extrusion_footprint(lines_pl, check_layers)
    worst_oob = 0.0
    min_cov = 100.0
    max_void = 0.0
    for L in check_layers:
        sol = solids[L - 1]
        ft = foots[L]
        cov = 100.0 * sol.intersection(ft).area / max(sol.area, 1e-6)
        oob = 100.0 * ft.difference(sol.buffer(0.15)).area / max(sol.area, 1e-6)
        missed = sol.difference(ft)
        mg = list(missed.geoms) if missed.geom_type == "MultiPolygon" else ([missed] if not missed.is_empty else [])
        big_sliver = max([g.area for g in mg], default=0.0)
        cav = missed.buffer(-0.15, join_style=2).buffer(0.15, join_style=2) if not missed.is_empty else Polygon()
        cg = list(cav.geoms) if cav.geom_type == "MultiPolygon" else ([cav] if not cav.is_empty else [])
        big_void = max([g.area for g in cg], default=0.0)
        min_cov = min(min_cov, cov)
        worst_oob = max(worst_oob, oob)
        max_void = max(max_void, big_void)
        print(f"  L{L:2d}: phys_cov={cov:.2f}%, oob={oob:.3f}%, max_cav={big_void:.2f}mm2 (sliver {big_sliver:.2f}), vol_ratio={vol_ratios[L-2]:.3f}")
        if cov < 95.0 or oob > 0.05 or big_void > 1.0:
            print(f"FAIL: Layer {L} failed coverage/oob/void gate")
            ok = False

    cad_tot_vol = sum(s.area * 0.2 for s in solids)
    cad_ratio = tot_v_pl / max(cad_tot_vol, 1e-6)
    if not (0.95 <= cad_ratio <= 1.02):
        print(f"FAIL: Total volume vs CAD solid {100*cad_ratio:.1f}% outside [95%, 102%]")
        ok = False

    print(
        f"\nSUMMARY ({planned_3mf}): {'PASS' if ok else 'FAIL'}\n"
        f"  3MF valid & MD5 OK ({md5_actual[:12]}...), M73 count={len(m73_lines)}\n"
        f"  Sim time: {GT.fmt_hms(t_s1)} -> {GT.fmt_hms(t_pl)} ({100*(t_pl-t_s1)/t_s1:+.1f}%)\n"
        f"  Total vol: {tot_v_s1/1000:.2f} -> {tot_v_pl/1000:.2f} cm3 ({wt_diff_pct:+.2f}% vs Stage1; {100*cad_ratio:.1f}% of CAD {cad_tot_vol/1000:.2f} cm3)\n"
        f"  Layer vol/solid*0.2: mean={mean_vr:.3f}, range=[{min_vr:.3f}, {max_vr:.3f}]\n"
        f"  Coverage min={min_cov:.2f}%, worst OOB={worst_oob:.3f}%, max void={max_void:.2f} mm2\n"
        f"  Limits: maxQ={aud_pl['max_q_seen']:.2f}, q_viol={aud_pl['n_q_violations']}, arc_viol={aud_pl['n_arc_violations']}, v_viol={aud_pl['n_v_violations']}"
    )
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1_3mf")
    ap.add_argument("planned_3mf")
    ap.add_argument("--layers", default="5,10,20,30,40,50,60,65,70")
    a = ap.parse_args()
    check_layers = None if a.layers == "auto" else [int(v) for v in a.layers.split(",")]
    sys.exit(0 if verify(a.stage1_3mf, a.planned_3mf, check_layers) else 1)
