"""Unified interior Connected Fermat Spiral + G2/G3 arc generator (infill_spiral.py).

For each internal layer (2 .. N-1) of a sliced .gcode.3mf:
  1. Slices the bed-space 3D mesh with manifold3d for exact solid cross-sections solid(L).
  2. Identifies true horizontal exposed Top surface ledges (width >= 1.5 mm, area >= 12 mm^2)
     to preserve verbatim; tiny sloped-step Top surface slivers are merged into the interior.
  3. Reconstructs the footprint kept_foot(L) of all kept features (walls, bottom, bridge,
     kept flat top surface) using exact active LINE_WIDTH and G2/G3 arc sampling.
  4. Computes unified interior P_int(L) = solid(L) - kept_foot(L), peels non-overlapping
     thick offset rings (W = 0.45 mm), splices nested rings into continuous non-crossing
     Connected Fermat Spirals with layer-staggered splice sites, and recovers narrow cores
     via raster skeletonization with adaptive line width.
  5. Fits firmware-exact G2/G3 circular arcs (perp-bisector center projection, |R1-R0| < 0.002 mm)
     and computes exact Slic3r/Bambu extrusion E and volumetric-flow-capped feedrates.
"""
import math
import re
import zipfile
from collections import defaultdict
import manifold3d as m3d
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

NUM_RE = re.compile(r"([XYZEFIJPS])(-?\d*\.?\d+)")
FILA_AREA = math.pi * (1.75 / 2.0) ** 2
KEPT_ALWAYS = {"Outer wall", "Inner wall", "Overhang wall", "Bottom surface", "Bridge"}
INTERIOR_DROP = {
    "Sparse infill",
    "Internal solid infill",
    "Floating vertical shell",
    "Gap infill",
}


def mm3_per_mm(w, h=0.2):
    """Exact Bambu / Slic3r pill-shaped extrusion cross-section area (mm^2)."""
    return w * h - (h * h) * (1.0 - math.pi / 4.0)


def load_bed_manifold_from_3mf(path):
    """Extract bed-space triangle mesh from a .3mf file and build a Manifold."""
    with zipfile.ZipFile(path) as z:
        tm = z.read("3D/3dmodel.model").decode("utf-8")
        obj = z.read("3D/Objects/object_1.model").decode("utf-8")
    comp_m = re.search(r'<component [^>]*transform="([^"]+)"', tm)
    item_m = re.search(r'<item [^>]*transform="([^"]+)"', tm)
    Mc = (
        np.array(list(map(float, comp_m.group(1).split()))).reshape(4, 3)
        if comp_m
        else np.vstack([np.eye(3), np.zeros(3)])
    )
    Mi = np.array(list(map(float, item_m.group(1).split()))).reshape(4, 3)
    verts = np.array(
        [
            (float(x), float(y), float(z))
            for x, y, z in re.findall(
                r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"/>', obj
            )
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            (int(a), int(b), int(c))
            for a, b, c in re.findall(
                r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"/>', obj
            )
        ],
        dtype=np.int64,
    )
    verts = verts @ Mc[:3, :] + Mc[3, :]
    verts = verts @ Mi[:3, :] + Mi[3, :]
    mesh = m3d.Mesh(
        vert_properties=np.asarray(verts, np.float32),
        tri_verts=np.asarray(faces, np.uint32),
    )
    mesh.merge()
    return m3d.Manifold(mesh)


def _cs_to_shapely(cs):
    rings = cs.to_polygons()
    if not rings:
        return Polygon()
    pgs = []
    for r in rings:
        if len(r) < 3:
            continue
        p = Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            pgs.append(p)
    if not pgs:
        return Polygon()
    out = shapely.symmetric_difference_all(pgs)
    return out if out.is_valid else out.buffer(0)


def slice_all_solids(manifold, layer_zs):
    """Slice Manifold at each layer mid-plane z and return list of Shapely geometries."""
    return [_cs_to_shapely(manifold.slice(float(z))) for z in layer_zs]


def flat_top_mask_for_layer(solids, layer_idx_0):
    """Return Shapely mask of true flat horizontal Top surface ledges on 0-based layer_idx_0."""
    if layer_idx_0 + 1 >= len(solids):
        return solids[layer_idx_0]
    exp = solids[layer_idx_0].difference(solids[layer_idx_0 + 1])
    if exp.is_empty or exp.area < 12.0:
        return Polygon()
    core = exp.buffer(-0.75, join_style=2).buffer(+0.75, join_style=2)
    if core.is_empty:
        return Polygon()
    comps = list(core.geoms) if core.geom_type == "MultiPolygon" else [core]
    big = [c for c in comps if c.area >= 12.0]
    if not big:
        return Polygon()
    return unary_union(big).buffer(0.9, join_style=2).intersection(exp)


def _sample_arc(w, x, y, nx, ny, I, J, step=0.25):
    cx, cy = x + I, y + J
    R = math.hypot(I, J)
    if R < 1e-6:
        return [(x, y), (nx, ny)]
    a0 = math.atan2(-J, -I)
    a1 = math.atan2(ny - cy, nx - cx)
    ccw = w == "G3"
    if ccw:
        dth = (a1 - a0) % (2.0 * math.pi)
        if dth < 1e-9 and (nx - x) ** 2 + (ny - y) ** 2 < 1e-6:
            dth = 2.0 * math.pi
    else:
        dth = -((a0 - a1) % (2.0 * math.pi))
        if abs(dth) < 1e-9 and (nx - x) ** 2 + (ny - y) ** 2 < 1e-6:
            dth = -2.0 * math.pi
    nseg = max(2, int(math.ceil(abs(dth) * R / step)))
    th = np.linspace(a0, a0 + dth, nseg + 1)
    return [(cx + R * math.cos(t), cy + R * math.sin(t)) for t in th]


def compute_kept_footprint(layer_lines, x0, y0, flat_mask, overlap=0.063):
    """Build kept extruded polygon footprint from contiguous kept runs on a layer."""
    x, y = x0, y0
    feat = ""
    lw = 0.42
    inwipe = False
    # Group consecutive kept moves of same rounded buffer radius into polylines for fast buffering
    by_r = defaultdict(list)
    cur_pts = []
    cur_r = None

    def flush():
        nonlocal cur_pts, cur_r
        if len(cur_pts) >= 2 and cur_r is not None:
            by_r[cur_r].append(LineString(cur_pts))
        cur_pts = []
        cur_r = None

    for raw in layer_lines:
        st = raw.strip()
        if st.startswith("; FEATURE:"):
            flush()
            feat = st[10:].strip()
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
        is_ext = de > 1e-9 and not inwipe and (nx != x or ny != y or has_arc)
        if is_ext:
            keep = feat in KEPT_ALWAYS
            if not keep and feat == "Top surface" and not flat_mask.is_empty:
                mx, my = 0.5 * (x + nx), 0.5 * (y + ny)
                if flat_mask.distance(Point(mx, my)) < 0.35:
                    keep = True
            if keep:
                r = round(max(0.5 * lw - overlap, 0.05), 3)
                pts = (
                    _sample_arc(w, x, y, nx, ny, pr.get("I", 0.0), pr.get("J", 0.0))
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
        else:
            flush()
        x, y = nx, ny
    flush()

    polys = []
    for r, lines_list in by_r.items():
        if not lines_list:
            continue
        mls = shapely.MultiLineString(lines_list)
        polys.append(mls.buffer(r, cap_style=1, join_style=1, quad_segs=2))
    if not polys:
        return Polygon()
    return unary_union(polys)


# ---------------- Offset Ring Peeling & Non-Crossing Fermat Splice ----------------
def _fair_closed_pts(pts, iters=6, lam=0.45, mu=-0.47, max_disp=0.12):
    """Constrained non-shrinking Taubin smoothing on closed ring vertices."""
    p0 = np.asarray(pts, dtype=np.float64)
    if len(p0) < 5:
        return p0
    p = p0.copy()
    for _ in range(iters):
        p = p + lam * (0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p)
        p = p + mu * (0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p)
        d = p - p0
        dn = np.hypot(d[:, 0], d[:, 1])
        m = dn > max_disp
        if m.any():
            p[m] = p0[m] + d[m] * (max_disp / dn[m, None])
    return p


def _peel_thick_rings(P_int, S=0.4071, W=0.45, layer_h=0.2, stagger_odd=False, W_odd0=0.62):
    """Peel non-overlapping closed offset rings from thick interior regions using bead spacing S.

    When stagger_odd=True, depth 0 uses W_odd0 (0.62 mm) so all interior bead seams shift inward
    relative to even layers (brick-wall staggering) with zero wall gap.
    Returns (rings, residual_poly) where rings is list of (depth, coords_Nx2, LineString, w_ring).
    """
    uncov = P_int
    rings = []
    depth = 0
    S_odd0 = W_odd0 - layer_h * (1.0 - math.pi / 4.0)
    while not uncov.is_empty and uncov.area > 0.12:
        if depth == 0 and stagger_odd:
            r_open = 0.48  # require >=0.96 mm width so 0.62 mm odd-layer rings do not overlap inside thin walls
            s_cur = S_odd0
            w_cur = W_odd0
        else:
            r_open = 0.24
            s_cur = S
            w_cur = W
        thick = uncov.buffer(-r_open, join_style=1, quad_segs=3).buffer(r_open, join_style=1, quad_segs=3)
        if thick.is_empty or thick.area < 0.12:
            if depth == 0 and stagger_odd:
                depth += 1
                continue
            break
        inset = thick.buffer(-0.5 * s_cur, join_style=1, quad_segs=3)
        if inset.is_empty:
            if depth == 0 and stagger_odd:
                depth += 1
                continue
            break
        geoms = list(inset.geoms) if inset.geom_type == "MultiPolygon" else [inset]
        layer_lines = []
        for g in geoms:
            for ring in [g.exterior] + list(g.interiors):
                ls = LineString(ring.coords).simplify(0.03, preserve_topology=False)
                if ls.length >= 0.8:
                    pts = np.asarray(ls.coords)[:-1]
                    if len(pts) >= 3:
                        rings.append((depth, pts, ls, w_cur))
                        layer_lines.append(ls)
        if not layer_lines:
            if depth == 0 and stagger_odd:
                depth += 1
                continue
            break
        covered = shapely.MultiLineString(layer_lines).buffer(0.5 * s_cur, cap_style=1, join_style=1, quad_segs=2)
        uncov = uncov.difference(covered)
        if not uncov.is_empty:
            uncov = uncov.buffer(-0.035, join_style=2).buffer(0.035, join_style=2)
            if not uncov.is_empty:
                uncov = uncov.simplify(0.02, preserve_topology=True)
        depth += 1
    return rings, uncov


def _resample_ring(pts, max_step=0.45):
    """Ensure closed ring vertex spacing allows finding clean gap-cut splice points."""
    out = []
    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        out.append(p0)
        if d > max_step:
            k = int(math.ceil(d / max_step))
            for s in range(1, k):
                t = s / k
                out.append(p0 * (1.0 - t) + p1 * t)
    return np.asarray(out, dtype=np.float64)


def _advance_along_ring(pts, idx, dist):
    """Return (point, next_vert_idx, prev_vert_idx) after advancing `dist` forward along closed ring `pts`."""
    n = len(pts)
    rem = dist
    cur = idx
    for _ in range(n):
        nxt = (cur + 1) % n
        seg = math.hypot(pts[nxt, 0] - pts[cur, 0], pts[nxt, 1] - pts[cur, 1])
        if seg >= rem and seg > 1e-9:
            u = rem / seg
            pt = pts[cur] * (1.0 - u) + pts[nxt] * u
            return pt, nxt
        rem -= seg
        cur = nxt
    return pts[(idx + 1) % n].copy(), (idx + 2) % n


def _segs_intersect(a0, a1, b0, b1):
    """Return True if segment (a0, a1) properly or nearly intersects (b0, b1)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(b0, b1, a0)
    d2 = cross(b0, b1, a1)
    d3 = cross(a0, a1, b0)
    d4 = cross(a0, a1, b1)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _choose_splice_pair(pts_p, pts_c, target_dir, forbid_mask=None, max_bridge=0.75, gap=0.4071):
    """Find best non-crossing gap-cut splice between parent ring `pts_p` and child ring `pts_c`.

    forbid_mask: boolean array of length len(pts_p) marking vertices already reserved by cuts.
    Returns (pi, p1, p_nxt, c_ordered_rot, c_end, c_stop_idx, cost) or None.
    """
    diff = pts_p[:, None, :] - pts_c[None, :, :]
    dmat = np.hypot(diff[:, :, 0], diff[:, :, 1])
    if forbid_mask is not None and forbid_mask.any():
        dmat[forbid_mask, :] = 1e9
    min_d = float(dmat.min())
    if min_d > max_bridge:
        return None
    thresh = min(max_bridge, max(min_d + 0.12, 0.58))
    pi_cand, ci_cand = np.where(dmat <= thresh)
    if len(pi_cand) == 0:
        return None

    ctr = pts_c.mean(axis=0)
    best_cand = None
    best_score = 1e18
    step = max(1, len(pi_cand) // 24)
    for k in range(0, len(pi_cand), step):
        pi = int(pi_cand[k])
        ci = int(ci_cand[k])
        d = float(dmat[pi, ci])
        vec = pts_p[pi] - ctr
        vn = math.hypot(vec[0], vec[1]) + 1e-9
        align = (vec[0] * target_dir[0] + vec[1] * target_dir[1]) / vn
        score = d - 0.08 * align
        if score < best_score:
            best_score = score
            best_cand = (pi, ci)

    pi, ci = best_cand
    p0 = pts_p[pi]
    p1, p_nxt = _advance_along_ring(pts_p, pi, gap)
    # Require p_nxt != pi and forward span stays within small step (no wrap collision)
    if forbid_mask is not None and forbid_mask[p_nxt % len(pts_p)]:
        return None

    n_c = len(pts_c)
    best_orient = None
    best_ocost = 1e18
    for rev in (False, True):
        c_ord = pts_c[::-1] if rev else pts_c
        ci_ord = (n_c - 1 - ci) if rev else ci
        c_rot = np.roll(c_ord, -ci_ord, axis=0)
        c0 = c_rot[0]
        # Walk backwards `gap` mm from c0 (= c_rot[0]) to find exit point c_end
        segs = np.hypot(
            np.diff(np.vstack([c_rot, c_rot[0:1]]), axis=0)[:, 0],
            np.diff(np.vstack([c_rot, c_rot[0:1]]), axis=0)[:, 1],
        )
        tot_c = float(segs.sum())
        if tot_c <= 2.5 * gap:
            continue
        target_l = tot_c - gap
        acc = 0.0
        c_end = c_rot[-1]
        c_last_v = n_c - 1
        for m in range(n_c):
            sl = float(segs[m])
            if acc + sl >= target_l - 1e-6:
                u = max(0.0, min(1.0, (target_l - acc) / max(sl, 1e-9)))
                p_nxt_v = c_rot[(m + 1) % n_c]
                c_end = c_rot[m] * (1.0 - u) + p_nxt_v * u
                c_last_v = m  # last full vertex before c_end on c_rot
                break
            acc += sl

        if _segs_intersect(p0, c0, c_end, p1):
            continue
        cost = math.hypot(c0[0] - p0[0], c0[1] - p0[1]) + math.hypot(p1[0] - c_end[0], p1[1] - c_end[1])
        if cost < best_ocost:
            best_ocost = cost
            best_orient = (pi, p1, p_nxt, c_rot, c_end, c_last_v, cost)
    return best_orient


def _build_spliced_spirals(rings, layer_num, S=0.4071, post_simp=0.09):
    """Splice connected offset rings into continuous non-crossing spiral polylines.

    Returns list of (runs_list, is_closed) where runs_list is [(pts_Nx2, width_mm), ...]
    with consecutive sub-runs sharing exact endpoints.
    """
    n = len(rings)
    if n == 0:
        return []
    resamp = [_resample_ring(r[1], max_step=0.45) for r in rings]
    geoms = [r[2] for r in rings]
    depths = [r[0] for r in rings]
    widths = [r[3] for r in rings]

    edges = []
    tree = shapely.STRtree(geoms)
    for i in range(n):
        for j in tree.query(geoms[i], predicate="dwithin", distance=0.85):
            j = int(j)
            if j > i and abs(depths[i] - depths[j]) <= 1:
                d = float(geoms[i].distance(geoms[j]))
                if d <= 0.85:
                    edges.append((d, i, j))
    edges.sort()

    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    adj = [[] for _ in range(n)]
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            adj[i].append(j)
            adj[j].append(i)

    az = math.radians((layer_num * 137.508) % 360.0)
    target_dir = (math.cos(az), math.sin(az))

    visited = [False] * n
    spirals = []

    def mark_forbid(fmask, pts_arr, idx, radius_mm=0.95):
        p = pts_arr[idx]
        d = np.hypot(pts_arr[:, 0] - p[0], pts_arr[:, 1] - p[1])
        fmask |= d <= radius_mm

    def trace_ring_with_ports(u, pts_u, entry_cut, child_ports):
        """Return list of (pt_xy, seg_width_from_prev) or (pts, seg_widths) for node u."""
        m = len(pts_u)
        wu = widths[u]
        if entry_cut is None:
            if not child_ports:
                pts_o = list(pts_u) + [pts_u[0]]
                return pts_o, [wu] * len(pts_u), True
            ports_sorted = sorted(child_ports, key=lambda x: x[0])
            s0 = ports_sorted[0]
            s0_pi, s0_p1, s0_pnxt, s0_crot, s0_cend, s0_clast, s0_v, s0_sub = s0
            out_pts = [s0_p1]
            out_w = []
            rem = []
            for p in ports_sorted[1:]:
                off = (p[0] - s0_pnxt) % m
                rem.append((off, p))
            rem.sort(key=lambda x: x[0])
            cur = s0_pnxt
            cur_off = 0
            end_off = (s0_pi - s0_pnxt) % m
            for off, p in rem:
                pi, p1, p_nxt, c_rot, c_end, c_last_v, v_idx, v_sub = p
                while cur_off <= off and cur_off <= end_off:
                    out_pts.append(pts_u[cur])
                    out_w.append(wu)
                    cur = (cur + 1) % m
                    cur_off += 1
                c_pts, c_ws = trace_ring_with_ports(v_idx, c_rot, (c_end, c_last_v), v_sub)
                wb = min(wu, widths[v_idx])
                out_pts.extend(c_pts)
                out_w.append(wb)
                out_w.extend(c_ws)
                out_pts.append(p1)
                out_w.append(wb)
                step_skip = (p_nxt - cur) % m
                cur = p_nxt
                cur_off += step_skip
            while cur_off <= end_off:
                out_pts.append(pts_u[cur])
                out_w.append(wu)
                cur = (cur + 1) % m
                cur_off += 1
            c0_pts, c0_ws = trace_ring_with_ports(s0_v, s0_crot, (s0_cend, s0_clast), s0_sub)
            wb0 = min(wu, widths[s0_v])
            out_pts.extend(c0_pts)
            out_w.append(wb0)
            out_w.extend(c0_ws)
            out_pts.append(s0_p1)
            out_w.append(wb0)
            return out_pts, out_w, True

        c_end, c_last_v = entry_cut
        out_pts = [pts_u[0]]
        out_w = []
        ports_sorted = sorted([p for p in child_ports if 0 < p[0] < c_last_v], key=lambda x: x[0])
        cur = 1
        for p in ports_sorted:
            pi, p1, p_nxt, c_rot, ch_end, ch_last, v_idx, v_sub = p
            if pi < cur or p_nxt > c_last_v:
                continue
            while cur <= pi:
                out_pts.append(pts_u[cur])
                out_w.append(wu)
                cur += 1
            c_pts, c_ws = trace_ring_with_ports(v_idx, c_rot, (ch_end, ch_last), v_sub)
            wb = min(wu, widths[v_idx])
            out_pts.extend(c_pts)
            out_w.append(wb)
            out_w.extend(c_ws)
            out_pts.append(p1)
            out_w.append(wb)
            cur = p_nxt
        while cur <= c_last_v:
            out_pts.append(pts_u[cur])
            out_w.append(wu)
            cur += 1
        out_pts.append(c_end)
        out_w.append(wu)
        return out_pts, out_w

    def plan_tree(u, pts_u, entry_last_v=None):
        """Plan valid non-overlapping child splices for node `u`, splitting fallbacks as separate roots."""
        visited[u] = True
        fmask = np.zeros(len(pts_u), dtype=bool)
        if entry_last_v is not None:
            mark_forbid(fmask, pts_u, 0, radius_mm=0.95)
            mark_forbid(fmask, pts_u, entry_last_v, radius_mm=0.95)
            fmask[entry_last_v:] = True

        ports = []
        fallbacks = []
        for v in adj[u]:
            if visited[v]:
                continue
            gap_use = 0.43 if (widths[u] > 0.5 or widths[v] > 0.5) else S
            sp = _choose_splice_pair(pts_u, resamp[v], target_dir, forbid_mask=fmask, max_bridge=0.85, gap=gap_use)
            if sp is None:
                fallbacks.append(v)
                continue
            pi, p1, p_nxt, c_rot, c_end, c_last_v, _ = sp
            if entry_last_v is not None and not (0 < pi < p_nxt <= entry_last_v):
                fallbacks.append(v)
                continue
            mark_forbid(fmask, pts_u, pi, radius_mm=0.95)
            mark_forbid(fmask, pts_u, p_nxt % len(pts_u), radius_mm=0.70)
            v_sub, v_fb = plan_tree(v, c_rot, entry_last_v=c_last_v)
            fallbacks.extend(v_fb)
            ports.append((pi, p1, p_nxt, c_rot, c_end, c_last_v, v, v_sub))
        return ports, fallbacks

    order = sorted(range(n), key=lambda i: (depths[i], -geoms[i].length))
    queue = list(order)
    while queue:
        i = queue.pop(0)
        if visited[i]:
            continue
        pts_u = resamp[i]
        ports, fbs = plan_tree(i, pts_u, entry_last_v=None)
        pts_list, seg_ws, is_cl = trace_ring_with_ports(i, pts_u, None, ports)
        if len(pts_list) >= 2 and len(seg_ws) == len(pts_list) - 1:
            # Group into constant-width contiguous sub-runs; simplify each keeping exact endpoints
            runs = []
            r_pts = [pts_list[0], pts_list[1]]
            r_w = seg_ws[0]
            for k in range(1, len(seg_ws)):
                if abs(seg_ws[k] - r_w) < 1e-6:
                    r_pts.append(pts_list[k + 1])
                else:
                    ls_r = LineString(r_pts).simplify(post_simp, preserve_topology=False)
                    runs.append((np.asarray(ls_r.coords, dtype=np.float64), r_w))
                    r_pts = [pts_list[k], pts_list[k + 1]]
                    r_w = seg_ws[k]
            ls_r = LineString(r_pts).simplify(post_simp, preserve_topology=False)
            runs.append((np.asarray(ls_r.coords, dtype=np.float64), r_w))
            spirals.append((runs, is_cl))
        for fb in fbs:
            if not visited[fb]:
                queue.insert(0, fb)
    return spirals


# ---------------- Narrow Core Skeleton Recovery ----------------
def _recover_narrow_cores(uncov, px=0.05, min_area=0.80, min_len=2.5, cav_area_thresh=0.70, cav_min_len=1.0):
    """Extract medial centerlines with adaptive local width from thin residual polygons."""
    if uncov.is_empty or uncov.area < min_area:
        return []
    geoms = list(uncov.geoms) if uncov.geom_type == "MultiPolygon" else [uncov]
    chains = []
    for pg in geoms:
        if pg.area < min_area:
            continue
        cav = pg.buffer(-0.15, join_style=2).buffer(0.15, join_style=2)
        if cav.is_empty:
            max_cav_pg = 0.0
        elif cav.geom_type == "MultiPolygon":
            max_cav_pg = max((g.area for g in cav.geoms), default=0.0)
        else:
            max_cav_pg = cav.area
        eff_min_len = cav_min_len if max_cav_pg >= cav_area_thresh else min_len
        x0, y0, x1, y1 = pg.bounds
        nx = max(3, int(math.ceil((x1 - x0) / px)) + 3)
        ny = max(3, int(math.ceil((y1 - y0) / px)) + 3)
        xs = x0 - px + np.arange(nx) * px
        ys = y0 - px + np.arange(ny) * px
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        mask = shapely.contains_xy(pg, gx.ravel(), gy.ravel()).reshape(nx, ny)
        if not mask.any():
            continue
        edt = distance_transform_edt(mask) * px
        if edt.max() < 0.13:
            continue
        skel = skeletonize(mask)
        pts_idx = np.argwhere(skel)
        if len(pts_idx) < 2:
            continue

        idx_map = {(int(i), int(j)): k for k, (i, j) in enumerate(pts_idx)}
        m = len(pts_idx)
        nbrs = [[] for _ in range(m)]
        for k, (i, j) in enumerate(pts_idx):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    nb = idx_map.get((int(i + di), int(j + dj)))
                    if nb is not None:
                        nbrs[k].append(nb)

        visited_edge = set()
        raw_chains = []
        starts = [k for k in range(m) if len(nbrs[k]) != 2] + [k for k in range(m) if len(nbrs[k]) == 2]
        for s in starts:
            for nb in nbrs[s]:
                e = (min(s, nb), max(s, nb))
                if e in visited_edge:
                    continue
                path_idx = [s, nb]
                visited_edge.add(e)
                cur = nb
                prev = s
                while len(nbrs[cur]) == 2:
                    n0, n1 = nbrs[cur]
                    nxt = n1 if n0 == prev else n0
                    e2 = (min(cur, nxt), max(cur, nxt))
                    if e2 in visited_edge:
                        break
                    visited_edge.add(e2)
                    path_idx.append(nxt)
                    prev, cur = cur, nxt
                deg_end = (len(nbrs[path_idx[0]]), len(nbrs[path_idx[-1]]))
                coords = np.array([(xs[pts_idx[k][0]], ys[pts_idx[k][1]]) for k in path_idx], dtype=np.float64)
                widths = np.array([np.clip(2.0 * edt[pts_idx[k][0], pts_idx[k][1]], 0.22, 0.62) for k in path_idx])
                raw_chains.append((coords, float(np.median(widths)), deg_end))

        # Keep main chains; drop tiny dead-end branch spurs (< 0.9 mm attached to a >=3 branch node)
        for coords, w_mean, (d0, d1) in raw_chains:
            ls = LineString(coords)
            is_spur = (d0 == 1 and d1 >= 3) or (d1 == 1 and d0 >= 3)
            if is_spur and ls.length < 1.0:
                continue
            if ls.length < eff_min_len:
                continue
            ls_s = ls.simplify(0.05, preserve_topology=False)
            sc = np.asarray(ls_s.coords, dtype=np.float64)
            if len(sc) >= 2:
                chains.append((sc, w_mean))
    return chains


# ---------------- Firmware-Exact G2/G3 Circular Arc Fitting ----------------
def _fit_circle_bisector(pts, p0_r, p1_r):
    """Fit circle to pts and project center onto exact perpendicular bisector of (p0_r, p1_r)."""
    dx = p1_r[0] - p0_r[0]
    dy = p1_r[1] - p0_r[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return None
    mx = 0.5 * (p0_r[0] + p1_r[0])
    my = 0.5 * (p0_r[1] + p1_r[1])
    nx_u, ny_u = -dy / L, dx / L

    if len(pts) == 3:
        # Exact circumcircle through 3 points
        ax, ay = pts[0]
        bx, by = pts[1]
        cx_, cy_ = pts[2]
        d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-8:
            return None
        a2 = ax * ax + ay * ay
        b2 = bx * bx + by * by
        c2 = cx_ * cx_ + cy_ * cy_
        ux = (a2 * (by - cy_) + b2 * (cy_ - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx_ - bx) + b2 * (ax - cx_) + c2 * (bx - ax)) / d
    else:
        # Kasa algebraic circle fit shifted to midpoint
        P = pts - np.array([mx, my])
        px, py = P[:, 0], P[:, 1]
        z = px * px + py * py
        A = np.column_stack([2.0 * px, 2.0 * py, np.ones(len(P))])
        try:
            sol, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            return None
        ux = mx + float(sol[0])
        uy = my + float(sol[1])

    # Project (ux, uy) onto perpendicular bisector of rounded endpoints
    h = (ux - mx) * nx_u + (uy - my) * ny_u
    cx = mx + h * nx_u
    cy = my + h * ny_u
    return cx, cy


def fit_arcs_to_polyline(
    pts,
    tol=0.05,
    r_min=0.45,
    r_max=500.0,
    min_sweep_deg=3.0,
    max_sweep_deg=140.0,
    corner_deg=36.0,
):
    """Convert a 2D polyline (Nx2) into exact rounded G1/G2/G3 segments.

    Returns list of tuples:
      ("G1", x0, y0, x1, y1, length)
      ("G2"|"G3", x0, y0, x1, y1, I, J, arc_length)
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return []

    # Round all vertices to 3 decimals up-front so every endpoint matches emitted G-code!
    rpts = np.round(pts, 3)
    # Remove zero-length duplicates
    clean = [rpts[0]]
    for p in rpts[1:]:
        if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) >= 0.002:
            clean.append(p)
    rpts = np.asarray(clean, dtype=np.float64)
    if len(rpts) < 2:
        return []

    # Split at sharp corners (> corner_deg)
    cos_thresh = math.cos(math.radians(corner_deg))
    splits = [0]
    for k in range(1, len(rpts) - 1):
        v1 = rpts[k] - rpts[k - 1]
        v2 = rpts[k + 1] - rpts[k]
        l1 = math.hypot(v1[0], v1[1])
        l2 = math.hypot(v2[0], v2[1])
        if l1 > 1e-6 and l2 > 1e-6:
            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            if dot < cos_thresh:
                splits.append(k)
    splits.append(len(rpts) - 1)

    moves = []
    for s_idx in range(len(splits) - 1):
        a, b = splits[s_idx], splits[s_idx + 1]
        sub = rpts[a : b + 1]
        m = len(sub)
        i = 0
        while i < m - 1:
            p0 = sub[i]
            best = None
            max_j = min(m - 1, i + 40)
            for j in range(i + 2, max_j + 1):
                p1 = sub[j]
                if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.1:
                    continue
                fit = _fit_circle_bisector(sub[i : j + 1], p0, p1)
                if fit is None:
                    continue
                cx, cy = fit
                I = round(cx - p0[0], 3)
                J = round(cy - p0[1], 3)
                R0 = math.hypot(I, J)
                if not (r_min <= R0 <= r_max):
                    continue
                cx_r, cy_r = p0[0] + I, p0[1] + J
                R1 = math.hypot(p1[0] - cx_r, p1[1] - cy_r)
                if abs(R1 - R0) >= 0.0018:
                    # Try 1-unit ULP adjustment on I/J to minimize radius mismatch
                    best_ij = None
                    best_err = abs(R1 - R0)
                    for di in (-0.001, 0.0, 0.001):
                        for dj in (-0.001, 0.0, 0.001):
                            Ii = round(I + di, 3)
                            Ji = round(J + dj, 3)
                            r0i = math.hypot(Ii, Ji)
                            r1i = math.hypot(p1[0] - (p0[0] + Ii), p1[1] - (p0[1] + Ji))
                            err = abs(r1i - r0i)
                            if err < best_err:
                                best_err = err
                                best_ij = (Ii, Ji, r0i)
                    if best_ij is None or best_err >= 0.0018:
                        continue
                    I, J, R0 = best_ij
                    cx_r, cy_r = p0[0] + I, p0[1] + J

                # Check sweep direction and intermediate vertex deviations
                a0 = math.atan2(-J, -I)
                a1 = math.atan2(p1[1] - cy_r, p1[0] - cx_r)
                # Orientation from cross product of (p0->mid) and (mid->p1)
                mid_p = sub[(i + j) // 2]
                cr = (mid_p[0] - p0[0]) * (p1[1] - p0[1]) - (mid_p[1] - p0[1]) * (p1[0] - p0[0])
                if abs(cr) < 1e-7:
                    continue
                ccw = cr > 0  # if mid is to the right of p0->p1, center is to the left -> CCW (G3)!
                if ccw:
                    dth = (a1 - a0) % (2.0 * math.pi)
                else:
                    dth = -((a0 - a1) % (2.0 * math.pi))
                sweep_deg = math.degrees(abs(dth))
                if not (min_sweep_deg <= sweep_deg <= max_sweep_deg):
                    continue

                win = sub[i + 1 : j]
                dists = np.hypot(win[:, 0] - cx_r, win[:, 1] - cy_r)
                if np.max(np.abs(dists - R0)) > tol:
                    if np.max(np.abs(dists - R0)) > 2.5 * tol:
                        break
                    continue

                # Also check chord midpoints so long straight segments never get replaced by bulging arcs!
                mids = 0.5 * (sub[i:j] + sub[i + 1 : j + 1])
                mdists = np.hypot(mids[:, 0] - cx_r, mids[:, 1] - cy_r)
                if np.max(np.abs(mdists - R0)) > tol:
                    if np.max(np.abs(mdists - R0)) > 2.5 * tol:
                        break
                    continue

                # Verify angular monotonicity of interior points within [0, dth]
                angs = np.arctan2(win[:, 1] - cy_r, win[:, 0] - cx_r)
                if ccw:
                    rel = (angs - a0) % (2.0 * math.pi)
                    ok_ang = np.all(rel <= dth + 0.05) and np.all(np.diff(rel) >= -0.05)
                else:
                    rel = (a0 - angs) % (2.0 * math.pi)
                    ok_ang = np.all(rel <= abs(dth) + 0.05) and np.all(np.diff(rel) >= -0.05)
                if not ok_ang:
                    continue

                arc_len = R0 * abs(dth)
                cmd = "G3" if ccw else "G2"
                best = (j, (cmd, float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), float(I), float(J), float(arc_len)))

            if best is not None:
                j_best, mv = best
                moves.append(mv)
                i = j_best
            else:
                p1 = sub[i + 1]
                dl = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                moves.append(("G1", float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), float(dl)))
                i += 1
    return moves


def generate_layer_interior(
    solid_poly,
    kept_foot,
    layer_num,
    layer_h=0.2,
    W=0.45,
    q_max=18.0,
    arc_tol=0.07,
    stagger_layers=True,
    W_odd0=0.62,
    fair_rings=True,
    ring_simp=0.06,
    max_disp=0.06,
    absorb_cores=True,
):
    """Generate unified interior spiral + core atoms for one layer.

    Each move in atom["moves"] is an 11-tuple:
      (cmd, x0, y0, x1, y1, I, J, de, dl, w_loc, feed_F)
    """
    if solid_poly.is_empty:
        return dict(atoms=[], P_int=Polygon(), resid_area=0.0)
    P_int = solid_poly.difference(kept_foot)
    if not P_int.is_empty:
        P_int = P_int.buffer(-0.04, join_style=2).buffer(0.04, join_style=2)
    if P_int.is_empty or P_int.area < 0.15:
        return dict(atoms=[], P_int=Polygon(), resid_area=0.0)

    S = W - layer_h * (1.0 - math.pi / 4.0)
    stagger_odd = stagger_layers and (layer_num % 2 == 1)
    rings, uncov = _peel_thick_rings(
        P_int, S=S, W=W, layer_h=layer_h, stagger_odd=stagger_odd, W_odd0=W_odd0
    )
    if fair_rings and rings:
        faired = []
        for depth, pts, ls, w_cur in rings:
            pf = _fair_closed_pts(pts, iters=6, max_disp=max_disp)
            lsf = LineString(list(pf) + [tuple(pf[0])]).simplify(ring_simp, preserve_topology=False)
            if lsf.length >= 0.8 and len(lsf.coords) >= 4:
                faired.append((depth, np.asarray(lsf.coords)[:-1], lsf, w_cur))
            else:
                faired.append((depth, pts, ls, w_cur))
        rings = faired
        cov_polys = [
            ls.buffer(0.5 * (w - layer_h * (1.0 - math.pi / 4.0)), cap_style=1, join_style=1, quad_segs=2)
            for _, _, ls, w in rings
        ]
        uncov = P_int.difference(unary_union(cov_polys))
        if not uncov.is_empty:
            uncov = uncov.buffer(-0.035, join_style=2).buffer(0.035, join_style=2)
    if absorb_cores and not uncov.is_empty and rings:
        uc_geoms = list(uncov.geoms) if uncov.geom_type == "MultiPolygon" else [uncov]
        ring_geoms = [r[2] for r in rings]
        rtree = shapely.STRtree(ring_geoms)
        keep_uc = []
        widened = set()
        for pg in uc_geoms:
            if pg.area < 0.80:
                continue
            if pg.area < 3.0 and pg.buffer(-0.22).is_empty:
                hits = rtree.query(pg, predicate="dwithin", distance=0.35)
                cands = [
                    int(h)
                    for h in hits
                    if rings[int(h)][3] < 0.50 and rings[int(h)][2].length * 0.11 <= 2.5 * pg.area
                ]
                if cands:
                    best_h = max(cands, key=lambda h: rings[h][0])
                    new_cov = rings[best_h][2].buffer(0.5 * 0.56, cap_style=1, join_style=1, quad_segs=2)
                    rem = pg.difference(new_cov)
                    rem_cav = rem.buffer(-0.14, join_style=2).buffer(0.14, join_style=2) if not rem.is_empty else Polygon()
                    if rem_cav.is_empty or rem_cav.area < 0.25:
                        widened.add(best_h)
                        continue
            keep_uc.append(pg)
        for idx_w in widened:
            d_, p_, l_, _ = rings[idx_w]
            rings[idx_w] = (d_, p_, l_, 0.56)
        uncov = unary_union(keep_uc) if keep_uc else Polygon()
    spirals = _build_spliced_spirals(rings, layer_num, S=S, post_simp=0.06)
    cores = _recover_narrow_cores(uncov)

    def _flow_params(w_val):
        a_val = mm3_per_mm(w_val, layer_h)
        v_cap = 250.0 if w_val < 0.36 else 350.0
        v_val = min(v_cap, q_max / a_val)
        return a_val / FILA_AREA, v_val, round(v_val * 60.0)

    atoms = []
    for runs, is_cl in spirals:
        emoves = []
        w_first = None
        v_first = None
        f_first = None
        for pts_run, w_run in runs:
            arc_moves = fit_arcs_to_polyline(pts_run, tol=arc_tol)
            if not arc_moves:
                continue
            e_per_mm, v_run, f_run = _flow_params(w_run)
            if w_first is None:
                w_first, v_first, f_first = round(w_run, 3), v_run, f_run
            for mv in arc_moves:
                if mv[0] == "G1":
                    _, x0, y0, x1, y1, dl = mv
                    emoves.append(("G1", x0, y0, x1, y1, 0.0, 0.0, dl * e_per_mm, dl, round(w_run, 3), f_run))
                else:
                    cmd, x0, y0, x1, y1, I, J, dl = mv
                    emoves.append((cmd, x0, y0, x1, y1, I, J, dl * e_per_mm, dl, round(w_run, 3), f_run))
        if not emoves:
            continue
        # Ensure exact continuity between sub-runs in case endpoint rounding shifted by <= 0.001 mm
        is_cl_real = is_cl and math.hypot(emoves[-1][3] - emoves[0][1], emoves[-1][4] - emoves[0][2]) < 0.05
        atoms.append(
            dict(
                moves=emoves,
                width=w_first,
                speed_mm_s=v_first,
                feed_F=f_first,
                is_closed=is_cl_real,
                start=(emoves[0][1], emoves[0][2]),
                end=(emoves[-1][3], emoves[-1][4]),
                feature="Sparse infill",
            )
        )

    for pts_c, w_loc in cores:
        if len(pts_c) >= 3:
            ls_c = LineString(pts_c).simplify(0.08, preserve_topology=False)
            if len(ls_c.coords) >= 2:
                pts_c = np.asarray(ls_c.coords, dtype=np.float64)
        arc_moves = fit_arcs_to_polyline(pts_c, tol=arc_tol)
        if not arc_moves:
            continue
        e_per_mm_loc, v_loc, f_loc = _flow_params(w_loc)
        w_r = round(w_loc, 3)
        emoves = []
        for mv in arc_moves:
            if mv[0] == "G1":
                _, x0, y0, x1, y1, dl = mv
                emoves.append(("G1", x0, y0, x1, y1, 0.0, 0.0, dl * e_per_mm_loc, dl, w_r, f_loc))
            else:
                cmd, x0, y0, x1, y1, I, J, dl = mv
                emoves.append((cmd, x0, y0, x1, y1, I, J, dl * e_per_mm_loc, dl, w_r, f_loc))
        atoms.append(
            dict(
                moves=emoves,
                width=w_r,
                speed_mm_s=v_loc,
                feed_F=f_loc,
                is_closed=False,
                start=(emoves[0][1], emoves[0][2]),
                end=(emoves[-1][3], emoves[-1][4]),
                feature="Sparse infill",
            )
        )

    return dict(atoms=atoms, P_int=P_int, resid_area=uncov.area if not uncov.is_empty else 0.0)


def raster_audit_layer(solid_poly, kept_foot, atoms, true_wall_foot=None, px=0.12):
    """Audit coverage, volume conservation, boundary safety, and arc precision on one layer."""
    if solid_poly.is_empty:
        return dict(cov_pct=100.0, phys_cov_pct=100.0, vol_ratio=1.0, oob_pct=0.0, worst_arc_err=0.0)
    spiral_polys = []
    worst_arc_err = 0.0
    for a in atoms:
        by_r = defaultdict(list)
        cur_pts = []
        cur_r = None
        for mv in a["moves"]:
            cmd, x0, y0, x1, y1, I, J, _, _ = mv[:9]
            w_mv = mv[9] if len(mv) > 9 else a["width"]
            r = round(0.5 * w_mv, 3)
            if cmd in ("G2", "G3"):
                r0 = math.hypot(I, J)
                r1 = math.hypot(x1 - (x0 + I), y1 - (y0 + J))
                worst_arc_err = max(worst_arc_err, abs(r1 - r0))
                pts = _sample_arc(cmd, x0, y0, x1, y1, I, J, step=0.25)
            else:
                pts = [(x0, y0), (x1, y1)]
            if cur_r == r and cur_pts and math.hypot(cur_pts[-1][0] - pts[0][0], cur_pts[-1][1] - pts[0][1]) < 0.05:
                cur_pts.extend(pts[1:])
            else:
                if len(cur_pts) >= 2 and cur_r is not None:
                    by_r[cur_r].append(LineString(cur_pts))
                cur_r = r
                cur_pts = list(pts)
        if len(cur_pts) >= 2 and cur_r is not None:
            by_r[cur_r].append(LineString(cur_pts))
        for r, llist in by_r.items():
            spiral_polys.append(shapely.MultiLineString(llist).buffer(r, cap_style=1, join_style=1, quad_segs=2))

    spiral_foot = unary_union(spiral_polys) if spiral_polys else Polygon()
    P_int = solid_poly.difference(kept_foot)
    cov_area = P_int.intersection(spiral_foot).area
    cov_pct = 100.0 * cov_area / max(P_int.area, 1e-6)

    if true_wall_foot is not None:
        tot_foot = unary_union([true_wall_foot, spiral_foot])
        phys_cov_pct = 100.0 * solid_poly.intersection(tot_foot).area / max(solid_poly.area, 1e-6)
    else:
        phys_cov_pct = cov_pct

    extr_mm3 = sum(sum(mv[7] for mv in a["moves"]) for a in atoms) * FILA_AREA
    vol_ratio = extr_mm3 / max(P_int.area * 0.2, 1e-6)

    oob = spiral_foot.difference(solid_poly.buffer(0.15))
    oob_pct = 100.0 * oob.area / max(solid_poly.area, 1e-6)

    return dict(
        P_int_area=P_int.area,
        cov_pct=cov_pct,
        phys_cov_pct=phys_cov_pct,
        vol_ratio=vol_ratio,
        oob_pct=oob_pct,
        worst_arc_err=worst_arc_err,
    )


if __name__ == "__main__":
    import argparse
    import time
    import gcode_time as GT

    ap = argparse.ArgumentParser()
    ap.add_argument("src_3mf", help="Stage 1 .gcode.3mf")
    ap.add_argument("--layers", default="10,30,50,65")
    a = ap.parse_args()

    t0 = time.perf_counter()
    man = load_bed_manifold_from_3mf(a.src_3mf)
    lines = GT.load_gcode_lines(a.src_3mf)
    layer_bounds = {}
    cur_l = 0
    s_i = 0
    for idx, s in enumerate(lines):
        if s.startswith("; CHANGE_LAYER"):
            if cur_l > 0:
                layer_bounds[cur_l] = (s_i, idx)
            cur_l += 1
            s_i = idx
    layer_bounds[cur_l] = (s_i, len(lines))
    zs = [(L - 0.5) * 0.2 for L in range(1, cur_l + 1)]
    solids = slice_all_solids(man, zs)
    header = [s for s in lines[: layer_bounds[1][0]] if s.startswith("; machine_max") or s.startswith("; default_jerk")]
    print(f"Loaded & sliced {cur_l} layers in {time.perf_counter() - t0:.2f}s")

    for L in [int(v) for v in a.layers.split(",")]:
        tl = time.perf_counter()
        s0, s1 = layer_bounds[L]
        seg = lines[s0:s1]
        # Track starting XY from previous layer end
        x_st = y_st = 0.0
        for raw in lines[:s0]:
            c = raw.split(";", 1)[0].strip()
            if c.startswith(("G0 ", "G1 ", "G2 ", "G3 ")):
                pr = {k: float(v) for k, v in NUM_RE.findall(c)}
                x_st = pr.get("X", x_st)
                y_st = pr.get("Y", y_st)
        flat_mask = flat_top_mask_for_layer(solids, L - 1)
        kept_foot = compute_kept_footprint(seg, x_st, y_st, flat_mask)
        true_wall = compute_kept_footprint(seg, x_st, y_st, flat_mask, overlap=0.0)
        res = generate_layer_interior(solids[L - 1], kept_foot, L)
        aud = raster_audit_layer(solids[L - 1], kept_foot, res["atoms"], true_wall_foot=true_wall)

        # Kinematic simulation of generated interior atoms
        gc_sim = ["M204 S20000"]
        n_g1 = n_g23 = 0
        for at in res["atoms"]:
            sx, sy = at["start"]
            gc_sim.append(f"G1 X{sx:.3f} Y{sy:.3f} F84000")
            cur_f = at["feed_F"]
            cur_w = at["width"]
            gc_sim.append(f"; LINE_WIDTH: {cur_w}")
            gc_sim.append(f"G1 F{cur_f}")
            for mv in at["moves"]:
                cmd, x0, y0, x1, y1, I, J, de, _, w_mv, f_mv = mv
                if w_mv != cur_w:
                    gc_sim.append(f"; LINE_WIDTH: {w_mv}")
                    cur_w = w_mv
                if f_mv != cur_f:
                    gc_sim.append(f"G1 F{f_mv}")
                    cur_f = f_mv
                if cmd == "G1":
                    gc_sim.append(f"G1 X{x1:.3f} Y{y1:.3f} E{de:.5f}")
                    n_g1 += 1
                else:
                    gc_sim.append(f"{cmd} X{x1:.3f} Y{y1:.3f} I{I:.3f} J{J:.3f} E{de:.5f}")
                    n_g23 += 1
        t_int, _, _, _ = GT.simulate(header + ["; FEATURE: Sparse infill"] + gc_sim)
        t_orig, _, cat_orig, _ = GT.simulate(header + seg)
        t_walls = cat_orig.get("Outer wall", 0) + cat_orig.get("Inner wall", 0) + cat_orig.get("Overhang wall", 0)
        t_nonwall_orig = t_orig - t_walls
        print(
            f"L{L:2d} ({time.perf_counter()-tl:.2f}s): atoms={len(res['atoms']):2d}, G1={n_g1:4d}, G2/3={n_g23:4d} | "
            f"phys_cov={aud['phys_cov_pct']:.1f}% (P_int {aud['cov_pct']:.1f}%), vol_ratio={aud['vol_ratio']:.3f}, "
            f"oob={aud['oob_pct']:.2f}%, arc_err={aud['worst_arc_err']:.4f} | "
            f"int_t={t_int:.1f}s vs orig_nonwall={t_nonwall_orig:.1f}s (layer_save={t_nonwall_orig - t_int - 2.0:+.1f}s)"
        )
