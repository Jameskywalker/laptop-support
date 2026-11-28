"""Full Stage 2A + Stage 3 toolpath planner & .gcode.3mf packager (toolpath_plan.py).

Takes a Stage-1 sliced .gcode.3mf (Arachne + G2/G3 arcs + res=0.05 + nearest seam):
  - Preserves Layer 1, Layer N, all kept walls (with exact overhang speeds),
    Bottom surface, Bridge, and true flat horizontal Top surface verbatim.
  - Replaces fragmented interior features (Sparse infill, Internal solid infill,
    Floating vertical shell, Gap infill, and sloped-step Top surface slivers) with
    infill_spiral.py's unified Connected Fermat Spirals + G2/G3 circular arcs.
  - Optimizes atom sequence per layer (greedy + Or-opt with precedence constraints,
    inter-layer start continuity, open-spiral reversal, and closed-spiral rotation).
  - Applies intra-solid combing for short interior-to-interior jumps inside solid(L).
  - Rewrites M73, header predictions, MD5, and slice_info.config into output .gcode.3mf.

Usage:
    python toolpath_plan.py stage1.gcode.3mf -o out_planned.gcode.3mf [--report r.json]
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import re
import time
import zipfile
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

import gcode_time as GT
import infill_spiral as IS

NUM_RE = re.compile(r"([XYZEFIJPS])(-?\d*\.?\d+)")
WALL_FEATS = {"Outer wall", "Inner wall", "Overhang wall"}
OUTER_FEATS = {"Outer wall", "Overhang wall"}
KEPT_ALWAYS = {"Outer wall", "Inner wall", "Overhang wall", "Bottom surface", "Bridge"}

RETRACT_MIN = 1.0
ZHOP_DZ = 0.4
SPIRAL_R = 1.217
WIPE_DIST = 2.0
WIPE_E = 0.38
RET_E = 0.02
DER_E = 0.4
TRAVEL_F = 84000
TRAVEL_ACC = 25000
SHORT_OUTER_ACC = 250
T_ZHOP_PENALTY = 0.14
T_COMB_PENALTY = 0.005


def fmt_num(v, nd=3):
    s = ("%." + str(nd) + "f") % v
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    if s in ("", "-", "-0"):
        s = "0"
    return s


def _params(line):
    return {k: float(v) for k, v in NUM_RE.findall(line.split(";", 1)[0])}


def _fan_key(cmd):
    m = re.search(r"\bP(\d+)", cmd)
    return "P" + m.group(1) if m else "P0"


def _arc_center_and_pts(cmd, x0, y0, x1, y1, I, J, step=0.25):
    return IS._sample_arc(cmd, x0, y0, x1, y1, I, J, step=step)


def _wipe_trail_from_moves(extr_moves, dist=WIPE_DIST):
    """Build backward 2 mm wipe trail [(x,y), ...] from a list of (cmd, x0, y0, x1, y1, I, J)."""
    if not extr_moves:
        return []
    pts = []
    acc = 0.0
    for cmd, x0, y0, x1, y1, I, J in reversed(extr_moves):
        seg_pts = (
            _arc_center_and_pts(cmd, x0, y0, x1, y1, I, J, step=0.35)
            if cmd in ("G2", "G3")
            else [(x0, y0), (x1, y1)]
        )
        for k in range(len(seg_pts) - 1, 0, -1):
            ax, ay = seg_pts[k]
            bx, by = seg_pts[k - 1]
            if not pts:
                pts.append((ax, ay))
            dl = math.hypot(ax - bx, ay - by)
            if dl < 1e-6:
                continue
            if acc + dl >= dist:
                u = (dist - acc) / dl
                pts.append((ax + u * (bx - ax), ay + u * (by - ay)))
                return pts
            pts.append((bx, by))
            acc += dl
    return pts


def _split_layer_sections(chunk):
    """Split a layer chunk into header, init_wipe, keep_hdr, first_travel, body, tail_meta."""
    n = len(chunk)
    # head: ; CHANGE_LAYER, ; Z_HEIGHT, ; LAYER_HEIGHT
    i = 0
    while i < n and i < 6 and chunk[i].strip().startswith("; ") and not chunk[i].strip().startswith("; WIPE_START"):
        if chunk[i].strip().startswith("; layer num"):
            break
        i += 1
    head = chunk[:i]

    # init_wipe: up through ; WIPE_END and optional G1 E-.02...
    j = i
    saw_wipe = False
    while j < n:
        st = chunk[j].strip()
        if st.startswith("; layer num") or st.startswith("M73 "):
            break
        if st.startswith("; WIPE_END"):
            saw_wipe = True
            j += 1
            if j < n and "E-" in chunk[j].split(";", 1)[0]:
                j += 1
            break
        j += 1
    if not saw_wipe:
        # Check if there is a pure retract before layer num
        init_wipe = chunk[i:j]
    else:
        init_wipe = chunk[i:j]

    # keep_hdr + first_travel up to first ; FEATURE:
    k = j
    while k < n and not chunk[k].strip().startswith("; FEATURE:"):
        k += 1
    pre_feat = chunk[j:k]
    # Split pre_feat into keep_hdr (up to ; OBJECT_ID or M104/M106 before M204 S25000 / G3 Z) and travel
    split_idx = len(pre_feat)
    for idx, ln in enumerate(pre_feat):
        st = ln.strip()
        if st.startswith("M204 S25000") or (st.startswith("G3 ") and "P1" in st):
            split_idx = idx
            break
    keep_hdr = pre_feat[:split_idx]
    first_travel = pre_feat[split_idx:]

    # tail_meta: trailing M622.1 / M1002 / M622 / M971 / M623 timelapse lines after last extrusion/wipe
    end_idx = n
    while end_idx > k:
        st = chunk[end_idx - 1].strip()
        if not st or st.startswith(("M622", "M1002", "M971", "M623", "; close powerlost", "M1003 S0")):
            end_idx -= 1
        else:
            break
    body = chunk[k:end_idx]
    tail_meta = chunk[end_idx:]
    return head, init_wipe, keep_hdr, first_travel, body, tail_meta


def _extract_kept_atoms(body, x0, y0, F0, acc0, fan0, flat_mask):
    """Extract kept wall/bottom/bridge/flat-top atoms verbatim from Stage-1 layer body.

    Returns (atoms, end_xy, end_F, end_acc, end_fan, base_sparse_fan).
    """
    x, y = x0, y0
    F = F0
    acc = acc0
    fan = dict(fan0)
    feat = ""
    lw = "0.42"
    inwipe = False
    base_sparse_fan = dict(fan0)

    atoms = []
    cur = None

    def start_atom(ax, ay, a_feat, a_lw, a_F, a_acc, a_fan):
        return dict(
            kind="K",
            is_wall=a_feat in WALL_FEATS,
            start_feat=a_feat,
            feats={a_feat},
            raw_prefix=[],
            lines=[],
            start=(ax, ay),
            end=(ax, ay),
            ext_moves=[],
            ext_moves_full=[],
            is_closed=False,
            in_F=a_F,
            in_acc=a_acc,
            in_fan=dict(a_fan),
            out_F=a_F,
            out_acc=a_acc,
            out_fan=dict(a_fan),
            cur_feat=a_feat,
            cur_lw=a_lw,
            feed_F=a_F,
            width=float(a_lw),
        )

    def close_atom():
        nonlocal cur
        if cur is None:
            return
        if cur["ext_moves"]:
            cur["wipe_trail"] = _wipe_trail_from_moves(cur["ext_moves"])
            cur["is_closed"] = (
                math.hypot(cur["end"][0] - cur["start"][0], cur["end"][1] - cur["start"][1]) < 0.6
            )
            cur["is_wall"] = any(f in WALL_FEATS for f in cur["feats"])
            atoms.append(cur)
        cur = None

    pending_lines = []
    for raw in body:
        st = raw.strip()
        if not st:
            continue
        if st.startswith("; FEATURE:"):
            feat = st[10:].strip()
            pending_lines.append(raw)
            continue
        if st.startswith("; LINE_WIDTH:"):
            lw = st[13:].strip()
            pending_lines.append(raw)
            continue
        if st.startswith("; WIPE_START"):
            close_atom()
            inwipe = True
            pending_lines.clear()
            continue
        if st.startswith("; WIPE_END"):
            inwipe = False
            pending_lines.clear()
            continue
        if st.startswith(";"):
            if cur is not None:
                cur["lines"].append(raw)
            else:
                pending_lines.append(raw)
            continue

        code = st.split(";", 1)[0].strip()
        if not code:
            continue
        w = code.split()[0]
        if w == "M204":
            m = re.search(r"S(\d*\.?\d+)", code)
            if m:
                acc = float(m.group(1))
            if cur is not None:
                cur["lines"].append(raw)
                cur["out_acc"] = acc
            else:
                pending_lines.append(raw)
            continue
        if w == "M106":
            fk = _fan_key(code)
            fan[fk] = code
            if feat == "Sparse infill":
                base_sparse_fan[fk] = code
            if cur is not None:
                cur["lines"].append(raw)
                cur["out_fan"] = dict(fan)
            else:
                pending_lines.append(raw)
            continue
        if w == "G17":
            continue
        if w not in ("G0", "G1", "G2", "G3"):
            if cur is not None:
                cur["lines"].append(raw)
            else:
                pending_lines.append(raw)
            continue

        pr = _params(code)
        if "F" in pr:
            F = pr["F"]
        nx, ny = pr.get("X", x), pr.get("Y", y)
        de = pr.get("E", 0.0)
        has_arc = "I" in pr or "J" in pr
        has_z = "Z" in pr

        if inwipe or de < -1e-9:
            close_atom()
            pending_lines.clear()
            x, y = nx, ny
            continue

        # Pure feedrate-only line (e.g. "G1 F13587.299" from Arachne variable width): keep inside active atom!
        if "F" in pr and abs(de) < 1e-9 and not has_z and not has_arc and "X" not in pr and "Y" not in pr:
            if cur is not None:
                cur["lines"].append(raw)
                cur["out_F"] = F
            else:
                pending_lines.append(raw)
            continue

        if de > 1e-9 and not has_z and (nx != x or ny != y or has_arc):
            keep = feat in KEPT_ALWAYS
            if not keep and feat == "Top surface" and not flat_mask.is_empty:
                mx, my = 0.5 * (x + nx), 0.5 * (y + ny)
                if flat_mask.distance(Point(mx, my)) < 0.35:
                    keep = True
            if keep:
                if cur is not None and (feat == "Inner wall") != (cur["cur_feat"] == "Inner wall"):
                    close_atom()
                if cur is None:
                    cur = start_atom(x, y, feat, lw, F, acc, fan)
                    cur["raw_prefix"] = [
                        f"; FEATURE: {feat}",
                        f"; LINE_WIDTH: {lw}",
                    ]
                    cur["lines"].append(f"G1 F{fmt_num(F, 3)}")
                    cur["lines"].append(f"M204 S{fmt_num(acc, 0)}")
                    pending_lines.clear()
                else:
                    if feat != cur["cur_feat"]:
                        cur["lines"].append(f"; FEATURE: {feat}")
                        cur["cur_feat"] = feat
                        cur["feats"].add(feat)
                    if lw != cur["cur_lw"]:
                        cur["lines"].append(f"; LINE_WIDTH: {lw}")
                        cur["cur_lw"] = lw
                cur["lines"].append(raw)
                cur["ext_moves"].append((w, x, y, nx, ny, pr.get("I", 0.0), pr.get("J", 0.0)))
                cur["ext_moves_full"].append((w, x, y, nx, ny, pr.get("I", 0.0), pr.get("J", 0.0), de, F, float(lw)))
                cur["end"] = (nx, ny)
                cur["out_F"] = F
                cur["out_acc"] = acc
                cur["out_fan"] = dict(fan)
            else:
                close_atom()
                pending_lines.clear()
                if feat == "Sparse infill":
                    base_sparse_fan = dict(fan)
        else:
            # Pure travel or Z-move or pure-E deretract
            if abs(de) < 1e-9 and not has_z and (nx != x or ny != y):
                # Short in-run non-retract travel (e.g. Inner -> Inner < 1mm)
                if (
                    cur is not None
                    and math.hypot(nx - x, ny - y) < RETRACT_MIN
                    and feat in KEPT_ALWAYS
                    and (feat == "Inner wall") == (cur["cur_feat"] == "Inner wall")
                ):
                    for pl in pending_lines:
                        if pl.strip().startswith(("; FEATURE:", "; LINE_WIDTH:")):
                            cur["lines"].append(pl)
                            if pl.strip().startswith("; FEATURE:"):
                                cf = pl.strip()[10:].strip()
                                cur["cur_feat"] = cf
                                cur["feats"].add(cf)
                            else:
                                cur["cur_lw"] = pl.strip()[13:].strip()
                    pending_lines.clear()
                    cur["lines"].append(raw)
                    cur["end"] = (nx, ny)
                else:
                    close_atom()
                    pending_lines.clear()
            else:
                close_atom()
                pending_lines.clear()
        x, y = nx, ny

    close_atom()
    return atoms, (x, y), F, acc, fan, base_sparse_fan


def _wrap_spiral_atom(sp_atom, fan_dict):
    """Wrap an infill_spiral atom dict into the common planner atom format."""
    mvs = sp_atom["moves"]
    trail_in = [(m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for m in mvs]
    return dict(
        kind="S",
        is_wall=False,
        start_feat=sp_atom["feature"],
        feats={sp_atom["feature"]},
        moves=mvs,
        start=sp_atom["start"],
        end=sp_atom["end"],
        is_closed=sp_atom["is_closed"],
        width=sp_atom["width"],
        feed_F=sp_atom["feed_F"],
        in_F=float(sp_atom["feed_F"]),
        in_acc=20000.0,
        in_fan=dict(fan_dict),
        out_F=float(sp_atom["feed_F"]),
        out_acc=20000.0,
        out_fan=dict(fan_dict),
        wipe_trail=_wipe_trail_from_moves(trail_in),
    )


def _reverse_moves(moves):
    """Reverse an open sequence of 11-tuple moves (cmd, x0, y0, x1, y1, I, J, de, dl, w_loc, feed_F)."""
    out = []
    for mv in reversed(moves):
        cmd, x0, y0, x1, y1, I, J, de, dl, w_loc, f_loc = mv
        if cmd == "G1":
            out.append(("G1", x1, y1, x0, y0, 0.0, 0.0, de, dl, w_loc, f_loc))
        else:
            ncmd = "G3" if cmd == "G2" else "G2"
            cx = x0 + I
            cy = y0 + J
            nI = round(cx - x1, 3)
            nJ = round(cy - y1, 3)
            r0 = math.hypot(nI, nJ)
            r1 = math.hypot(x0 - (x1 + nI), y0 - (y1 + nJ))
            if abs(r1 - r0) >= 0.0018:
                best = None
                best_e = abs(r1 - r0)
                for di in (-0.001, 0.0, 0.001):
                    for dj in (-0.001, 0.0, 0.001):
                        Ii, Ji = round(nI + di, 3), round(nJ + dj, 3)
                        e = abs(math.hypot(x0 - (x1 + Ii), y0 - (y1 + Ji)) - math.hypot(Ii, Ji))
                        if e < best_e:
                            best_e = e
                            best = (Ii, Ji)
                if best is not None and best_e < 0.0018:
                    nI, nJ = best
                else:
                    out.append(("G1", x1, y1, x0, y0, 0.0, 0.0, de, math.hypot(x0 - x1, y0 - y1), w_loc, f_loc))
                    continue
            out.append((ncmd, x1, y1, x0, y0, nI, nJ, de, dl, w_loc, f_loc))
    return out


def _rotate_closed_moves(moves, target_xy):
    """Rotate a closed loop of moves to start at the vertex closest to target_xy."""
    pts = np.array([(m[1], m[2]) for m in moves], dtype=np.float64)
    dists = np.hypot(pts[:, 0] - target_xy[0], pts[:, 1] - target_xy[1])
    k = int(np.argmin(dists))
    if k == 0:
        return moves
    return moves[k:] + moves[:k]


def _optimize_inner_wall_atom(atom, solid_poly, fair=True):
    """Merge micro-F jitter, apply constrained fairing inside solid, and refit G2/G3 arcs on pure Inner wall atoms."""
    mvs_full = atom.get("ext_moves_full", [])
    if atom["kind"] != "K" or atom["feats"] != {"Inner wall"} or len(mvs_full) < 3:
        return atom

    # Partition into contiguous sub-runs of consistent speed (+-15%) and line width
    runs = []
    cur_run = [mvs_full[0]]
    ref_F = mvs_full[0][8]
    ref_lw = mvs_full[0][9]
    for mv in mvs_full[1:]:
        prev = cur_run[-1]
        gap = math.hypot(mv[1] - prev[3], mv[2] - prev[4])
        if gap >= 0.05:
            return atom  # keep atoms with internal travel jumps verbatim
        if abs(mv[8] - ref_F) / max(ref_F, 1.0) >= 0.15 or abs(mv[9] - ref_lw) >= 0.02:
            runs.append(cur_run)
            cur_run = [mv]
            ref_F = mv[8]
            ref_lw = mv[9]
        else:
            cur_run.append(mv)
    runs.append(cur_run)

    safe_zone = solid_poly.buffer(-0.42) if (fair and solid_poly is not None and not solid_poly.is_empty) else Polygon()
    all_moves = []
    for r_idx, run in enumerate(runs):
        pts = []
        tot_e = 0.0
        tot_l = 0.0
        for w, x0, y0, x1, y1, I, J, de, f_mv, lw_mv in run:
            tot_e += de
            if w in ("G2", "G3") and (abs(I) > 1e-9 or abs(J) > 1e-9):
                dl, *_ = GT._planar_arc_geom(w, x0, y0, x1, y1, I, J)
                spts = IS._sample_arc(w, x0, y0, x1, y1, I, J, step=0.25)
            else:
                dl = math.hypot(x1 - x0, y1 - y0)
                spts = [(x0, y0), (x1, y1)]
            tot_l += dl
            if not pts:
                pts.extend(spts)
            else:
                pts.extend(spts[1:])
        if len(pts) < 2 or tot_l < 1e-4:
            return atom

        med_F = float(np.median([m[8] for m in run]))
        med_lw = round(float(np.median([m[9] for m in run])), 3)
        e_per_mm = tot_e / tot_l
        v_cap = min(med_F / 60.0, 350.0, 17.95 / max(e_per_mm * IS.FILA_AREA, 1e-9))
        f_run = round(v_cap * 60.0)

        pts_arr = np.asarray(pts, dtype=np.float64)
        if fair and len(pts_arr) >= 5 and not safe_zone.is_empty:
            p0 = pts_arr.copy()
            p = p0.copy()
            is_cl_run = math.hypot(p0[-1, 0] - p0[0, 0], p0[-1, 1] - p0[0, 1]) < 0.05
            for _ in range(4):
                if is_cl_run and len(p) >= 6:
                    q = p[:-1]
                    q = q + 0.40 * (0.5 * (np.roll(q, 1, 0) + np.roll(q, -1, 0)) - q)
                    q = q - 0.42 * (0.5 * (np.roll(q, 1, 0) + np.roll(q, -1, 0)) - q)
                    p = np.vstack([q, q[0:1]])
                else:
                    p[1:-1] = p[1:-1] + 0.40 * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
                    p[1:-1] = p[1:-1] - 0.42 * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
                d = p - p0
                dn = np.hypot(d[:, 0], d[:, 1])
                m = dn > 0.06
                if m.any():
                    p[m] = p0[m] + d[m] * (0.06 / dn[m, None])
            # Clamp any vertex that stepped outside safe_zone back to p0
            inside = shapely.contains_xy(safe_zone, p[:, 0], p[:, 1])
            if not np.all(inside):
                p[~inside] = p0[~inside]
            pts_arr = p

        if len(pts_arr) >= 3:
            ls_s = LineString(pts_arr).simplify(0.05, preserve_topology=False)
            if len(ls_s.coords) >= 2:
                pts_arr = np.asarray(ls_s.coords, dtype=np.float64)

        arc_mvs = IS.fit_arcs_to_polyline(pts_arr, tol=0.07)
        if not arc_mvs:
            return atom
        for mv in arc_mvs:
            if mv[0] == "G1":
                _, x0, y0, x1, y1, dl = mv
                all_moves.append(("G1", x0, y0, x1, y1, 0.0, 0.0, dl * e_per_mm, dl, med_lw, f_run))
            else:
                cmd, x0, y0, x1, y1, I, J, dl = mv
                all_moves.append((cmd, x0, y0, x1, y1, I, J, dl * e_per_mm, dl, med_lw, f_run))

    if not all_moves:
        return atom
    ext_mvs = [(m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for m in all_moves]
    st = (all_moves[0][1], all_moves[0][2])
    en = (all_moves[-1][3], all_moves[-1][4])
    is_cl = math.hypot(en[0] - st[0], en[1] - st[1]) < 0.05
    return dict(
        kind="IW",
        is_wall=True,
        start_feat="Inner wall",
        feats={"Inner wall"},
        moves=all_moves,
        ext_moves=ext_mvs,
        start=st,
        end=en,
        is_closed=is_cl,
        width=float(all_moves[0][9]),
        feed_F=float(all_moves[0][10]),
        in_F=float(all_moves[0][10]),
        in_acc=atom["in_acc"],
        in_fan=dict(atom["in_fan"]),
        out_F=float(all_moves[-1][10]),
        out_acc=atom["in_acc"],
        out_fan=dict(atom["out_fan"]),
        wipe_trail=_wipe_trail_from_moves(ext_mvs),
    )


def _kept_footprint_from_atoms(kept_atoms, overlap=0.063):
    """Build exact kept polygon footprint from finalized kept atoms (K and IW)."""
    by_r = collections.defaultdict(list)
    for a in kept_atoms:
        mvs = a.get("moves") if a["kind"] == "IW" else a.get("ext_moves_full", [])
        cur_pts = []
        cur_r = None
        for mv in mvs:
            cmd, x0, y0, x1, y1, I, J = mv[:7]
            lw = mv[9]
            r = round(max(0.5 * lw - overlap, 0.05), 3)
            pts = (
                IS._sample_arc(cmd, x0, y0, x1, y1, I, J)
                if (cmd in ("G2", "G3") and (abs(I) > 1e-9 or abs(J) > 1e-9))
                else [(x0, y0), (x1, y1)]
            )
            if cur_r == r and cur_pts and math.hypot(cur_pts[-1][0] - pts[0][0], cur_pts[-1][1] - pts[0][1]) < 0.05:
                cur_pts.extend(pts[1:])
            else:
                if len(cur_pts) >= 2 and cur_r is not None:
                    by_r[cur_r].append(LineString(cur_pts))
                cur_r = r
                cur_pts = list(pts)
        if len(cur_pts) >= 2 and cur_r is not None:
            by_r[cur_r].append(LineString(cur_pts))
    polys = [
        shapely.MultiLineString(llist).buffer(r, cap_style=1, join_style=1, quad_segs=2)
        for r, llist in by_r.items()
        if llist
    ]
    return shapely.ops.unary_union(polys) if polys else Polygon()


# ---------------- Worker for Parallel Layer Geometry Generation ----------------
def _process_layer_geom(task):
    L, seg, x_st, y_st, F_st, acc_st, fan_st, solid_wkb, flat_wkb, stagger_layers = task
    solid_poly = shapely.from_wkb(solid_wkb)
    flat_mask = shapely.from_wkb(flat_wkb)
    kept_atoms, end_xy, end_F, end_acc, end_fan, base_fan = _extract_kept_atoms(
        seg, x_st, y_st, F_st, acc_st, fan_st, flat_mask
    )
    kept_atoms = [
        _optimize_inner_wall_atom(a, solid_poly, fair=False)
        if (a["kind"] == "K" and a["feats"] == {"Inner wall"} and len(a.get("ext_moves_full", [])) >= 3)
        else a
        for a in kept_atoms
    ]
    kept_foot = _kept_footprint_from_atoms(kept_atoms, overlap=0.063)
    res = IS.generate_layer_interior(solid_poly, kept_foot, L, stagger_layers=stagger_layers)
    spiral_atoms = [_wrap_spiral_atom(sa, base_fan) for sa in res["atoms"]]
    return L, kept_atoms, spiral_atoms, end_xy, end_F, end_acc, end_fan


# ---------------- Islands, Precedence & Sequencing ----------------
def _assign_islands_and_precedence(atoms, solid_poly):
    """Assign solid island IDs and precedence (walls -> spirals, inner -> outer wall)."""
    comps = (
        list(solid_poly.geoms)
        if solid_poly.geom_type == "MultiPolygon"
        else ([solid_poly] if not solid_poly.is_empty else [])
    )
    comp_tree = shapely.STRtree(comps) if comps else None

    for a in atoms:
        pt = Point(a["start"])
        if comp_tree is not None:
            hits = comp_tree.query(pt, predicate="intersects")
            a["island"] = int(hits[0]) if len(hits) else int(comp_tree.nearest(pt))
        else:
            a["island"] = 0
        a["preds"] = set()

    # Within each island: all walls precede all non-walls (spirals/cores/top/bottom)
    by_isl = collections.defaultdict(list)
    for idx, a in enumerate(atoms):
        by_isl[a["island"]].append(idx)

    for isl, idxs in by_isl.items():
        walls = [i for i in idxs if atoms[i]["is_wall"]]
        nonwalls = [i for i in idxs if not atoms[i]["is_wall"]]
        for j in nonwalls:
            atoms[j]["preds"].update(walls)

    # Outer wall / Overhang wall follows adjacent Inner wall
    iw = []
    iw_geos = []
    for i, a in enumerate(atoms):
        if a["kind"] in ("K", "IW") and "Inner wall" in a["feats"] and not (OUTER_FEATS & a["feats"]):
            segs = [
                ((m[1], m[2]), (m[3], m[4]))
                for m in a["ext_moves"]
                if math.hypot(m[3] - m[1], m[4] - m[2]) > 1e-4
            ]
            if segs:
                iw.append(i)
                iw_geos.append(shapely.MultiLineString(segs))
    if iw:
        tree = shapely.STRtree(iw_geos)
        for j, a in enumerate(atoms):
            if a["kind"] == "K" and (OUTER_FEATS & a["feats"]):
                pts = [(m[3], m[4]) for m in a["ext_moves"]]
                if not pts:
                    continue
                step = max(1, len(pts) // 20)
                sample = pts[::step]
                cnt = collections.Counter()
                for p in sample:
                    for h in tree.query(Point(p), predicate="dwithin", distance=0.75):
                        cnt[int(h)] += 1
                for h, c in cnt.items():
                    i = iw[h]
                    if i != j and c >= max(2, 0.25 * len(sample)):
                        a["preds"].add(i)


def _t_travel(d, acc):
    if d <= 0:
        return 0.0
    a = min(acc, 9000.0)
    v = 1400.0
    if d >= v * v / a:
        return d / v + v / a
    return 2.0 * math.sqrt(d / a)


def _can_comb(p0, p1, prev_is_wall, next_is_wall, shrunk_solid):
    if prev_is_wall or next_is_wall or shrunk_solid is None or shrunk_solid.is_empty:
        return False
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if not (RETRACT_MIN <= d <= 15.0):
        return False
    return bool(shrunk_solid.contains(LineString([p0, p1])))


def _trans_cost(p0, p1, prev_is_wall, next_atom, shrunk_solid, use_combing):
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if d < RETRACT_MIN:
        acc = SHORT_OUTER_ACC if (next_atom["start_feat"] in OUTER_FEATS) else TRAVEL_ACC
        return _t_travel(d, acc)
    if use_combing and _can_comb(p0, p1, prev_is_wall, next_atom["is_wall"], shrunk_solid):
        return T_COMB_PENALTY + _t_travel(d, TRAVEL_ACC)
    return T_ZHOP_PENALTY + _t_travel(d, TRAVEL_ACC)


def _order_layer_atoms(atoms, p0, shrunk_solid, use_combing):
    """Order atoms respecting precedence; returns list of (idx, rev, rot_start_xy)."""
    n = len(atoms)
    if n == 0:
        return []
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for j, a in enumerate(atoms):
        for i in a["preds"]:
            succs[i].append(j)
            indeg[j] += 1

    avail = {i for i in range(n) if indeg[i] == 0}
    order = []
    cur_xy = p0
    prev_is_wall = True

    while avail:
        best = None
        best_c = 1e18
        for i in avail:
            a = atoms[i]
            if a["kind"] in ("S", "IW") and a["is_closed"]:
                # Closest vertex on closed loop
                mvs = a["moves"]
                pts = np.array([(m[1], m[2]) for m in mvs], dtype=np.float64)
                d_arr = np.hypot(pts[:, 0] - cur_xy[0], pts[:, 1] - cur_xy[1])
                k = int(np.argmin(d_arr))
                st = (float(pts[k, 0]), float(pts[k, 1]))
                c = _trans_cost(cur_xy, st, prev_is_wall, a, shrunk_solid, use_combing)
                if c < best_c:
                    best_c = c
                    best = (i, False, st, st)
            elif a["kind"] in ("S", "IW") and not a["is_closed"]:
                for rev in (False, True):
                    st = a["end"] if rev else a["start"]
                    en = a["start"] if rev else a["end"]
                    c = _trans_cost(cur_xy, st, prev_is_wall, a, shrunk_solid, use_combing)
                    if c < best_c:
                        best_c = c
                        best = (i, rev, st, en)
            else:
                st = a["start"]
                c = _trans_cost(cur_xy, st, prev_is_wall, a, shrunk_solid, use_combing)
                if c < best_c:
                    best_c = c
                    best = (i, False, st, a["end"])

        i, rev, st, en = best
        order.append([i, rev, st])
        avail.remove(i)
        cur_xy = en
        prev_is_wall = atoms[i]["is_wall"]
        for s in succs[i]:
            indeg[s] -= 1
            if indeg[s] == 0:
                avail.add(s)

    # Or-opt single-atom relocation improvement respecting precedence
    def seq_endpoints(seq_list):
        pts_list = []
        for idx, rev, st in seq_list:
            a = atoms[idx]
            if a["kind"] in ("S", "IW") and a["is_closed"]:
                pts_list.append((st, st))
            elif rev:
                pts_list.append((a["end"], a["start"]))
            else:
                pts_list.append((a["start"], a["end"]))
        return pts_list

    def total_cost(seq_list):
        pts_list = seq_endpoints(seq_list)
        c = _trans_cost(p0, pts_list[0][0], True, atoms[seq_list[0][0]], shrunk_solid, use_combing)
        for q in range(1, len(seq_list)):
            prev_a = atoms[seq_list[q - 1][0]]
            cur_a = atoms[seq_list[q][0]]
            c += _trans_cost(pts_list[q - 1][1], pts_list[q][0], prev_a["is_wall"], cur_a, shrunk_solid, use_combing)
        return c

    if 3 <= n <= 160:
        cur_cost = total_cost(order)
        for _ in range(2):
            improved = False
            pos_of = {item[0]: idx for idx, item in enumerate(order)}
            for pos in range(n):
                item = order[pos]
                ai = item[0]
                lo_pos = max([pos_of[p] + 1 for p in atoms[ai]["preds"]] + [0])
                hi_pos = min([pos_of[s] for s in succs[ai]] + [n])
                if hi_pos - lo_pos <= 1:
                    continue
                rem = order[:pos] + order[pos + 1 :]
                # Compute valid insertion range in `rem`
                lo_r = lo_pos - (1 if lo_pos > pos else 0)
                hi_r = hi_pos - (1 if hi_pos > pos else 0)
                best_ins = None
                best_new_c = cur_cost - 1e-4
                for ins in range(lo_r, hi_r):
                    if ins == pos:
                        continue
                    cand = rem[:ins] + [item] + rem[ins:]
                    nc = total_cost(cand)
                    if nc < best_new_c:
                        best_new_c = nc
                        best_ins = cand
                if best_ins is not None:
                    order = best_ins
                    cur_cost = best_new_c
                    pos_of = {it[0]: idx for idx, it in enumerate(order)}
                    improved = True
                    break
            if not improved:
                break
    return order


# ---------------- G-code Emitter ----------------
class _Emitter:
    def __init__(self):
        self.out = []
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.f = 600.0
        self.acc = 20000.0
        self.fan = {}
        self.g17_active = False

    def w(self, line):
        self.out.append(line)

    def set_acc(self, acc):
        if abs(acc - self.acc) > 0.5:
            self.w("M204 S%s" % fmt_num(acc, 0))
            self.acc = acc

    def set_f(self, f):
        if abs(f - self.f) > 0.5:
            self.w("G1 F%s" % fmt_num(f, 3))
            self.f = f

    def set_fan(self, target_fan):
        for k, cmd in target_fan.items():
            if self.fan.get(k) != cmd:
                self.w(cmd)
                self.fan[k] = cmd

    def ensure_g17(self):
        if not self.g17_active:
            self.w("G17")
            self.g17_active = True

    def wipe_retract(self, wipe_trail, wipe_f):
        if wipe_trail and len(wipe_trail) >= 2:
            self.w("; WIPE_START")
            wf = max(min(wipe_f * 0.8, 15000.0), 3000.0)
            self.set_f(wf)
            pts = wipe_trail
            lens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
            tot = max(sum(lens), 1e-6)
            for (nx, ny), dl in zip(pts[1:], lens):
                de = -WIPE_E * (dl / tot)
                self.w(f"G1 X{fmt_num(nx)} Y{fmt_num(ny)} E{fmt_num(de, 5)}")
                self.x, self.y = nx, ny
            self.w("; WIPE_END")
        self.w(f"G1 E-{fmt_num(RET_E, 5)} F1800")
        self.f = 1800.0

    def lift_travel(self, z_cur, dst, z_dst):
        self.set_acc(TRAVEL_ACC)
        self.w("G17")
        self.g17_active = True
        zh = max(z_cur, z_dst) + ZHOP_DZ
        self.w(f"G3 Z{fmt_num(zh)} I{fmt_num(SPIRAL_R)} J0 P1 F{TRAVEL_F}")
        self.w(f"G1 X{fmt_num(dst[0])} Y{fmt_num(dst[1])} Z{fmt_num(zh)}")
        self.w(f"G1 Z{fmt_num(z_dst)}")
        self.w(f"G1 E{fmt_num(DER_E, 5)} F1800")
        self.x, self.y, self.z = dst[0], dst[1], z_dst
        self.f = 1800.0

    def comb_travel(self, dst):
        self.set_acc(TRAVEL_ACC)
        self.w(f"G1 X{fmt_num(dst[0])} Y{fmt_num(dst[1])} F{TRAVEL_F}")
        self.x, self.y = dst[0], dst[1]
        self.f = float(TRAVEL_F)

    def short_travel(self, dst, to_outer):
        self.set_acc(SHORT_OUTER_ACC if to_outer else TRAVEL_ACC)
        self.w(f"G1 X{fmt_num(dst[0])} Y{fmt_num(dst[1])} F{TRAVEL_F}")
        self.x, self.y = dst[0], dst[1]
        self.f = float(TRAVEL_F)

    def emit_atom(self, atom, rev, rot_xy):
        if atom["kind"] == "K":
            self.set_fan(atom["in_fan"])
            self.ensure_g17()
            for ln in atom["raw_prefix"]:
                self.w(ln)
            for ln in atom["lines"]:
                self.w(ln)
            self.x, self.y = atom["end"]
            self.f = atom["out_F"]
            self.acc = atom["out_acc"]
            self.fan = dict(atom["out_fan"])
            return

        # Spiral / core / optimized Inner wall atom
        self.set_fan(atom["in_fan"])
        self.ensure_g17()
        self.w(f"; FEATURE: {atom['start_feat']}")
        mvs = atom["moves"]
        if atom["is_closed"]:
            mvs = _rotate_closed_moves(mvs, rot_xy)
        elif rev:
            mvs = _reverse_moves(mvs)

        cur_w = None
        self.set_acc(atom["in_acc"] if atom["kind"] == "IW" else 20000.0)
        for cmd, x0, y0, x1, y1, I, J, de, _, w_mv, f_mv in mvs:
            if cur_w is None or abs(w_mv - cur_w) > 1e-4:
                self.w(f"; LINE_WIDTH: {fmt_num(w_mv, 3)}")
                cur_w = w_mv
            self.set_f(f_mv)
            if cmd == "G1":
                self.w(f"G1 X{fmt_num(x1)} Y{fmt_num(y1)} E{fmt_num(de, 5)}")
            else:
                self.w(f"{cmd} X{fmt_num(x1)} Y{fmt_num(y1)} I{fmt_num(I)} J{fmt_num(J)} E{fmt_num(de, 5)}")
            self.x, self.y = x1, y1


def _atom_wipe_trail(atom, rev, rot_xy):
    """Compute backward wipe trail from the actual end of an emitted (possibly rotated/reversed) atom."""
    if atom["kind"] in ("S", "IW"):
        mvs = atom["moves"]
        if atom["is_closed"] and rot_xy is not None:
            mvs = _rotate_closed_moves(mvs, rot_xy)
        elif rev:
            mvs = _reverse_moves(mvs)
        else:
            return atom["wipe_trail"]
        return _wipe_trail_from_moves([(m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for m in mvs])
    return atom["wipe_trail"]


def _rewrite_metadata(lines, sim_scale, total_e_mm):
    """Rewrite header predictions, filament usage, and per-layer M73 progress."""
    tot, cum, _, _ = GT.simulate(lines)
    tot_s = tot * sim_scale
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("M73 P") and "R" in ln and i > 0 and not ln.startswith("M73 P100"):
            continue
        out.append(ln)
        if ln.startswith("M73 L"):
            t = cum[i] * sim_scale
            pct = min(99, int(100.0 * t / max(tot_s, 1e-6)))
            rem_min = int(max(tot_s - t, 0.0) // 60)
            out.append(f"M73 P{pct} R{rem_min}")

    vol_mm3 = total_e_mm * IS.FILA_AREA
    wt_g = (vol_mm3 / 1000.0) * 1.25  # Bambu PETG Basic density = 1.25
    hms = GT.fmt_hms(tot_s)
    for i, ln in enumerate(out[:25]):
        if ln.startswith("; model printing time:"):
            out[i] = f"; model printing time: {hms}; total estimated time: {hms}"
        elif ln.startswith("; total filament length [mm] :"):
            out[i] = f"; total filament length [mm] : {total_e_mm:.2f}"
        elif ln.startswith("; total filament volume [cm^3] :"):
            out[i] = f"; total filament volume [cm^3] : {vol_mm3:.2f}"
        elif ln.startswith("; total filament weight [g] :"):
            out[i] = f"; total filament weight [g] : {wt_g:.2f}"
    return out, tot_s, wt_g


def plan_3mf(src_3mf, out_3mf, report_path=None, use_combing=True, stagger_layers=True, workers=4, layers_only=None):
    t_start = time.perf_counter()
    with zipfile.ZipFile(src_3mf) as z:
        gname = [n for n in z.namelist() if re.match(r"Metadata/plate_\d+\.gcode$", n)][0]
        raw_text = z.read(gname).decode("utf-8", errors="ignore")
    lines = raw_text.split("\n")

    # Baseline Stage-1 timing and Bambu prediction
    t_stage1_sim, _, cat_before, _ = GT.simulate(lines)
    m_bambu = re.search(r"model printing time: (?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+)s", raw_text[:3000])
    bambu_s = (
        int(m_bambu.group(1) or 0) * 3600 + int(m_bambu.group(2) or 0) * 60 + int(m_bambu.group(3))
        if m_bambu
        else t_stage1_sim
    )
    sim_scale = bambu_s / t_stage1_sim

    # Partition preamble, layers, epilogue
    chg_idxs = [i for i, l in enumerate(lines) if l.startswith("; CHANGE_LAYER")]
    end_idx = next((i for i, l in enumerate(lines) if "; MACHINE_END_GCODE_START" in l), len(lines))
    n_layers = len(chg_idxs)
    preamble = lines[: chg_idxs[0]]
    layer_chunks = [
        lines[chg_idxs[k] : (chg_idxs[k + 1] if k + 1 < n_layers else end_idx)]
        for k in range(n_layers)
    ]
    epilogue = lines[end_idx:]

    # Parse layer Z heights
    layer_zs = []
    for ch in layer_chunks:
        zh = None
        for ln in ch[:8]:
            if ln.startswith("; Z_HEIGHT:"):
                zh = float(ln.split(":", 1)[1].strip())
                break
        layer_zs.append(zh if zh is not None else 0.2 * (len(layer_zs) + 1))

    # Slice exact bed-space mesh via Manifold
    man = IS.load_bed_manifold_from_3mf(src_3mf)
    mid_zs = [z - 0.1 for z in layer_zs]
    solids = IS.slice_all_solids(man, mid_zs)
    print(f"Sliced {n_layers} layers from bed mesh in {time.perf_counter() - t_start:.2f}s")

    # Forward modal state scan over original file to seed each layer's initial state
    x = y = z_pos = 0.0
    F = 600.0
    acc = 20000.0
    fan = {}
    for ln in preamble:
        c = ln.split(";", 1)[0].strip()
        if not c:
            continue
        w = c.split()[0]
        if w == "M204":
            m = re.search(r"S(\d*\.?\d+)", c)
            if m:
                acc = float(m.group(1))
        elif w == "M106":
            fan[_fan_key(c)] = c
        elif w in ("G0", "G1", "G2", "G3"):
            pr = _params(c)
            if "F" in pr:
                F = pr["F"]
            x, y, z_pos = pr.get("X", x), pr.get("Y", y), pr.get("Z", z_pos)

    layer_init_state = []
    layer_sections = []
    for k, ch in enumerate(layer_chunks):
        head, init_wipe, keep_hdr, first_travel, body, tail_meta = _split_layer_sections(ch)
        layer_sections.append((head, init_wipe, keep_hdr, first_travel, body, tail_meta))
        # Advance state across header + init_wipe + keep_hdr + first_travel to body start
        for ln in head + init_wipe + keep_hdr + first_travel:
            c = ln.split(";", 1)[0].strip()
            if not c:
                continue
            w = c.split()[0]
            if w == "M204":
                m = re.search(r"S(\d*\.?\d+)", c)
                if m:
                    acc = float(m.group(1))
            elif w == "M106":
                fan[_fan_key(c)] = c
            elif w in ("G0", "G1", "G2", "G3"):
                pr = _params(c)
                if "F" in pr:
                    F = pr["F"]
                x, y, z_pos = pr.get("X", x), pr.get("Y", y), pr.get("Z", z_pos)
        layer_init_state.append((x, y, F, acc, dict(fan)))
        for ln in body + tail_meta:
            c = ln.split(";", 1)[0].strip()
            if not c:
                continue
            w = c.split()[0]
            if w == "M204":
                m = re.search(r"S(\d*\.?\d+)", c)
                if m:
                    acc = float(m.group(1))
            elif w == "M106":
                fan[_fan_key(c)] = c
            elif w in ("G0", "G1", "G2", "G3"):
                pr = _params(c)
                if "F" in pr:
                    F = pr["F"]
                x, y, z_pos = pr.get("X", x), pr.get("Y", y), pr.get("Z", z_pos)

    # Plan Layers 2 .. N-1 (0-based indices 1 .. n_layers-2)
    target_indices = [
        k for k in range(1, n_layers - 1) if (layers_only is None or (k + 1) in layers_only)
    ]
    tasks = []
    for k in target_indices:
        L = k + 1
        x_st, y_st, F_st, acc_st, fan_st = layer_init_state[k]
        _, _, _, _, body, _ = layer_sections[k]
        flat_mask = IS.flat_top_mask_for_layer(solids, k)
        tasks.append(
            (L, body, x_st, y_st, F_st, acc_st, fan_st, shapely.to_wkb(solids[k]), shapely.to_wkb(flat_mask), stagger_layers)
        )

    t_geom0 = time.perf_counter()
    layer_plans = {}
    if workers > 1 and len(tasks) > 4:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_process_layer_geom, tasks):
                L, kept_atoms, spiral_atoms, end_xy, end_F, end_acc, end_fan = res
                layer_plans[L] = (kept_atoms, spiral_atoms)
    else:
        for tsk in tasks:
            L, kept_atoms, spiral_atoms, end_xy, end_F, end_acc, end_fan = _process_layer_geom(tsk)
            layer_plans[L] = (kept_atoms, spiral_atoms)
    print(f"Generated interior spirals for {len(tasks)} layers in {time.perf_counter() - t_geom0:.2f}s")

    # Assemble full G-code sequentially with inter-layer continuity
    em = _Emitter()
    for ln in preamble:
        em.w(ln)
        c = ln.split(";", 1)[0].strip()
        if c:
            w = c.split()[0]
            if w == "M204":
                m = re.search(r"S(\d*\.?\d+)", c)
                if m:
                    em.acc = float(m.group(1))
            elif w == "M106":
                em.fan[_fan_key(c)] = c
            elif w in ("G0", "G1", "G2", "G3"):
                pr = _params(c)
                if "F" in pr:
                    em.f = pr["F"]
                em.x, em.y, em.z = pr.get("X", em.x), pr.get("Y", em.y), pr.get("Z", em.z)

    prev_planned = False
    prev_last_atom = None
    prev_last_rev = False
    prev_last_rot = None
    stats = []
    tot_zhop = 0
    tot_comb = 0
    tot_short = 0

    for k in range(n_layers):
        L = k + 1
        head, init_wipe, keep_hdr, first_travel, body, tail_meta = layer_sections[k]
        z_cur = layer_zs[k]

        order = None
        if L in layer_plans:
            kept_atoms, spiral_atoms = layer_plans[L]
            all_atoms = kept_atoms + spiral_atoms
            if all_atoms:
                _assign_islands_and_precedence(all_atoms, solids[k])
                shrunk = solids[k].buffer(-0.55) if not solids[k].is_empty else Polygon()
                order = _order_layer_atoms(all_atoms, (em.x, em.y), shrunk, use_combing)

        if not order:
            # Unmodified layer (Layer 1, Layer N, or empty degenerate layer)
            for ln in head:
                em.w(ln)
            if prev_planned and prev_last_atom is not None:
                # Replace original init_wipe with planned Layer N-1 wipe+retract, then lift to Stage1's first XY!
                trail = _atom_wipe_trail(prev_last_atom, prev_last_rev, prev_last_rot)
                em.wipe_retract(trail, prev_last_atom["out_F"])
                for ln in keep_hdr:
                    em.w(ln)
                    c = ln.split(";", 1)[0].strip()
                    if c.startswith("M106"):
                        em.fan[_fan_key(c)] = c
                # Parse target XY from first_travel
                dst_x, dst_y = em.x, em.y
                for ln in first_travel:
                    c = ln.split(";", 1)[0].strip()
                    if c.startswith(("G0 ", "G1 ")) and "Z" not in c and "E" not in c:
                        pr = _params(c)
                        dst_x, dst_y = pr.get("X", dst_x), pr.get("Y", dst_y)
                        break
                    elif c.startswith(("G0 ", "G1 ")) and ("X" in c or "Y" in c):
                        pr = _params(c)
                        dst_x, dst_y = pr.get("X", dst_x), pr.get("Y", dst_y)
                em.lift_travel(layer_zs[k - 1], (dst_x, dst_y), z_cur)
            else:
                for ln in init_wipe + keep_hdr + first_travel:
                    em.w(ln)
                    c = ln.split(";", 1)[0].strip()
                    if c:
                        w = c.split()[0]
                        if w == "M204":
                            m = re.search(r"S(\d*\.?\d+)", c)
                            if m:
                                em.acc = float(m.group(1))
                        elif w == "M106":
                            em.fan[_fan_key(c)] = c
                        elif w in ("G0", "G1", "G2", "G3"):
                            pr = _params(c)
                            if "F" in pr:
                                em.f = pr["F"]
                            em.x, em.y, em.z = pr.get("X", em.x), pr.get("Y", em.y), pr.get("Z", em.z)
            for ln in body + tail_meta:
                em.w(ln)
                c = ln.split(";", 1)[0].strip()
                if c:
                    w = c.split()[0]
                    if w == "M204":
                        m = re.search(r"S(\d*\.?\d+)", c)
                        if m:
                            em.acc = float(m.group(1))
                    elif w == "M106":
                        em.fan[_fan_key(c)] = c
                    elif w in ("G0", "G1", "G2", "G3"):
                        pr = _params(c)
                        if "F" in pr:
                            em.f = pr["F"]
                        em.x, em.y, em.z = pr.get("X", em.x), pr.get("Y", em.y), pr.get("Z", em.z)
            em.g17_active = False
            prev_planned = False
            continue

        for ln in head:
            em.w(ln)
        if prev_planned and prev_last_atom is not None:
            trail = _atom_wipe_trail(prev_last_atom, prev_last_rev, prev_last_rot)
            em.wipe_retract(trail, prev_last_atom["out_F"])
        else:
            for ln in init_wipe:
                em.w(ln)
            em.f = 1800.0
        for ln in keep_hdr:
            em.w(ln)
            c = ln.split(";", 1)[0].strip()
            if c.startswith("M106"):
                em.fan[_fan_key(c)] = c

        first_idx, first_rev, first_st = order[0]
        em.lift_travel(layer_zs[k - 1] if k > 0 else 0.0, first_st, z_cur)
        em.z = z_cur
        l_zhop = 1
        l_comb = 0
        l_short = 0
        l_trav = math.hypot(first_st[0] - em.x, first_st[1] - em.y)

        for q, (idx, rev, rot_xy) in enumerate(order):
            a = all_atoms[idx]
            if q > 0:
                prev_a = all_atoms[order[q - 1][0]]
                st_xy = (
                    rot_xy
                    if (a["kind"] in ("S", "IW") and a["is_closed"])
                    else (a["end"] if rev else a["start"])
                )
                d = math.hypot(st_xy[0] - em.x, st_xy[1] - em.y)
                l_trav += d
                if d < RETRACT_MIN:
                    em.short_travel(st_xy, a["start_feat"] in OUTER_FEATS)
                    l_short += 1
                elif use_combing and _can_comb((em.x, em.y), st_xy, prev_a["is_wall"], a["is_wall"], shrunk):
                    em.comb_travel(st_xy)
                    l_comb += 1
                else:
                    trail = _atom_wipe_trail(prev_a, order[q - 1][1], order[q - 1][2])
                    em.wipe_retract(trail, prev_a["out_F"])
                    em.lift_travel(z_cur, st_xy, z_cur)
                    l_zhop += 1
            em.emit_atom(a, rev, rot_xy)

        for ln in tail_meta:
            em.w(ln)

        prev_planned = True
        prev_last_atom = all_atoms[order[-1][0]]
        prev_last_rev = order[-1][1]
        prev_last_rot = order[-1][2]
        tot_zhop += l_zhop
        tot_comb += l_comb
        tot_short += l_short
        stats.append(
            dict(
                layer=L,
                kept=len(kept_atoms),
                spirals=len(spiral_atoms),
                zhops=l_zhop,
                combs=l_comb,
                shorts=l_short,
            )
        )

    for ln in epilogue:
        em.w(ln)

    # Compute total physical extruded filament length E (spatial extrusion moves outside wipe)
    total_e = 0.0
    inw = False
    for ln in em.out:
        st = ln.strip()
        if st.startswith("; WIPE_START"):
            inw = True
        elif st.startswith("; WIPE_END"):
            inw = False
        elif st.startswith(("G1 ", "G2 ", "G3 ")) and not inw:
            pr = _params(st)
            if pr.get("E", 0.0) > 0 and ("X" in pr or "Y" in pr):
                total_e += pr["E"]

    out_lines, tot_s_scaled, wt_g = _rewrite_metadata(em.out, sim_scale, total_e)
    t_after_sim, _, cat_after, _, aud = GT.simulate(out_lines, check_limits=True)

    gtext = "\n".join(out_lines)
    with zipfile.ZipFile(src_3mf) as zin, zipfile.ZipFile(out_3mf, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename == gname:
                data = gtext.encode("utf-8")
            elif it.filename == gname + ".md5":
                data = hashlib.md5(gtext.encode("utf-8")).hexdigest().upper().encode("utf-8")
            elif it.filename == "Metadata/slice_info.config":
                data = re.sub(
                    rb'key="prediction" value="\d+"',
                    b'key="prediction" value="%d"' % int(round(tot_s_scaled)),
                    data,
                )
                data = re.sub(
                    rb'key="weight" value="[0-9.]+"',
                    b'key="weight" value="%.2f"' % wt_g,
                    data,
                )
                data = re.sub(
                    rb'used_m="[0-9.]+"',
                    b'used_m="%.2f"' % (total_e / 1000.0),
                    data,
                )
                data = re.sub(
                    rb'used_g="[0-9.]+"',
                    b'used_g="%.2f"' % wt_g,
                    data,
                )
            zout.writestr(it, data)

    elapsed = time.perf_counter() - t_start
    print(
        f"Done in {elapsed:.1f}s -> {out_3mf}\n"
        f"  Time model: {GT.fmt_hms(t_stage1_sim)} -> {GT.fmt_hms(t_after_sim)} ({100*(t_after_sim-t_stage1_sim)/t_stage1_sim:+.1f}% vs Stage 1)\n"
        f"  Bambu scaled: {GT.fmt_hms(bambu_s)} -> {GT.fmt_hms(tot_s_scaled)} | wt={wt_g:.2f}g (E={total_e/1000:.2f}m)\n"
        f"  Transitions: zhops={tot_zhop}, combs={tot_comb}, short={tot_short}\n"
        f"  Audit: maxQ={aud['max_q_seen']:.2f}, qv={aud['n_q_violations']}, arcv={aud['n_arc_violations']}, vv={aud['n_v_violations']}"
    )
    for k in sorted(set(cat_before) | set(cat_after), key=lambda x: -cat_before.get(x, 0)):
        print(f"    {k:<26} {cat_before.get(k, 0):7.1f} -> {cat_after.get(k, 0):7.1f} s")

    if report_path:
        rep = dict(
            src=src_3mf,
            out=out_3mf,
            elapsed_s=elapsed,
            stage1_bambu_s=bambu_s,
            stage1_sim_s=t_stage1_sim,
            planned_sim_s=t_after_sim,
            planned_bambu_scaled_s=tot_s_scaled,
            weight_g=wt_g,
            filament_mm=total_e,
            zhops=tot_zhop,
            combs=tot_comb,
            shorts=tot_short,
            audit=aud,
            cat_before=cat_before,
            cat_after=cat_after,
            layers=stats,
        )
        json.dump(rep, open(report_path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Stage 1 .gcode.3mf")
    ap.add_argument("-o", "--out", required=True, help="Output planned .gcode.3mf")
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-combing", action="store_true")
    ap.add_argument("--no-stagger", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--layers", default=None, help="Comma-separated 1-based layer numbers")
    a = ap.parse_args()
    only = None if a.layers is None else set(int(v) for v in a.layers.split(","))
    plan_3mf(
        a.src,
        a.out,
        report_path=a.report,
        use_combing=not a.no_combing,
        stagger_layers=not a.no_stagger,
        workers=a.workers,
        layers_only=only,
    )


if __name__ == "__main__":
    main()
