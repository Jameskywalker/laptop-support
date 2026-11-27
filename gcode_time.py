"""G-code kinematic print time estimator & physical limit verifier (gcode_time.py).

Models Bambu Studio / PrusaSlicer trapezoidal velocity profiles with classic jerk
cornering and true planar G2/G3 circular arc kinematics (entry/exit tangents and
centripetal acceleration cap). Reads machine limits from G-code CONFIG_BLOCK.

Usage:
    python gcode_time.py plate_1.gcode [--by-layer] [--check-limits] [--k-cent 0.7]
"""
import argparse
import collections
import math
import re
import zipfile

NUM = re.compile(r"([XYZEFIJRPS])(-?\d*\.?\d+)")
FILA_AREA = math.pi * (1.75 / 2.0) ** 2


def machine_from_config(lines):
    """Parse machine kinematic limits from Bambu G-code CONFIG_BLOCK."""
    cfg = {}
    on = False
    for line in lines:
        if line.startswith("; CONFIG_BLOCK_START"):
            on = True
            continue
        if line.startswith("; CONFIG_BLOCK_END"):
            break
        if on and " = " in line:
            k, v = line[2:].split(" = ", 1)
            cfg[k.strip()] = v.strip()

    def first(k, d):
        try:
            return float(cfg[k].split(",")[0])
        except (KeyError, ValueError):
            return d

    return dict(
        maxv=dict(
            x=first("machine_max_speed_x", 1400.0),
            y=first("machine_max_speed_y", 1400.0),
            z=first("machine_max_speed_z", 35.0),
            e=first("machine_max_speed_e", 30.0),
        ),
        maxa=dict(
            x=first("machine_max_acceleration_x", 25000.0),
            y=first("machine_max_acceleration_y", 25000.0),
            z=first("machine_max_acceleration_z", 1000.0),
            e=first("machine_max_acceleration_e", 5000.0),
        ),
        jerk=dict(
            x=first("machine_max_jerk_x", 9.0),
            y=first("machine_max_jerk_y", 9.0),
            z=first("machine_max_jerk_z", 3.0),
            e=first("machine_max_jerk_e", 2.5),
        ),
        a_extr=first("machine_max_acceleration_extruding", 25000.0),
        a_trav=first("machine_max_acceleration_travel", 9000.0),
        a_retr=first("machine_max_acceleration_retracting", 5000.0),
        q_max=first("filament_max_volumetric_speed", 18.0),
    )


class _Move:
    __slots__ = (
        "d", "v", "a", "safe", "axv_in", "axv_out",
        "vj", "v0", "v1", "t", "cat", "layer", "line",
    )


def _planar_arc_geom(w, x, y, nx, ny, I, J):
    """Return (arc_xy_len, R, tx0, ty0, tx1, ty1, radius_err) for planar G2/G3."""
    cx = x + I
    cy = y + J
    r0 = math.hypot(I, J)
    r1 = math.hypot(nx - cx, ny - cy)
    radius_err = abs(r1 - r0)
    if r0 < 1e-9:
        dx, dy = nx - x, ny - y
        dl = math.hypot(dx, dy)
        if dl < 1e-9:
            return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, radius_err
        ux, uy = dx / dl, dy / dl
        return dl, 0.0, ux, uy, ux, uy, radius_err

    a0 = math.atan2(-J, -I)
    a1 = math.atan2(ny - cy, nx - cx)
    ccw = (w == "G3")
    if ccw:
        dtheta = (a1 - a0) % (2.0 * math.pi)
        if dtheta < 1e-9 and (nx - x) ** 2 + (ny - y) ** 2 < 1e-6:
            dtheta = 2.0 * math.pi
        tx0, ty0 = -math.sin(a0), math.cos(a0)
        tx1, ty1 = -math.sin(a1), math.cos(a1)
    else:
        dtheta = (a0 - a1) % (2.0 * math.pi)
        if dtheta < 1e-9 and (nx - x) ** 2 + (ny - y) ** 2 < 1e-6:
            dtheta = 2.0 * math.pi
        tx0, ty0 = math.sin(a0), -math.cos(a0)
        tx1, ty1 = math.sin(a1), -math.cos(a1)
    return r0 * dtheta, r0, tx0, ty0, tx1, ty1, radius_err


def _jerk_junction(axv_prev, axv_cur, jk, prev_v, cur_v, prev_safe, cur_safe):
    """Compute classic-jerk junction speed between two consecutive moves."""
    psl = prev_v > cur_v
    ssf = cur_v / prev_v if psl else prev_v / cur_v
    vj = cur_v if psl else prev_v
    vf = 1.0
    lim = False
    for i in range(4):
        ve, vn = axv_prev[i], axv_cur[i]
        if psl:
            ve *= ssf
        if lim:
            ve *= vf
            vn *= vf
        if ve > vn:
            j = (ve - vn) if (vn > 0 or ve < 0) else max(ve, -vn)
        else:
            j = (vn - ve) if (vn < 0 or ve > 0) else max(-ve, vn)
        if j > jk[i]:
            vf *= jk[i] / j
            lim = True
    if lim:
        vj *= vf
    th = vj * 0.99
    if prev_safe > th and cur_safe > th:
        vj = cur_safe
    return vj


def simulate(lines, mach=None, k_cent=1.6, check_limits=False):
    """Simulate G-code timing and optionally audit physical/kinematic limits.

    Returns (total_s, cum_line_s, cat_dict, layer_dict) if check_limits is False,
    or (total_s, cum_line_s, cat_dict, layer_dict, audit_dict) if check_limits is True.
    """
    if mach is None:
        mach = machine_from_config(lines)
    MAXV, MAXA, JERK = mach["maxv"], mach["maxa"], mach["jerk"]
    q_cap = mach.get("q_max", 18.0)

    x = y = z = 0.0
    F = 600.0
    acc = 20000.0
    feat = "Start"
    layer = 0
    inwipe = False
    bl = []

    max_q_seen = 0.0
    q_violations = []
    arc_violations = []
    v_violations = []

    for li, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        if s[0] == ";":
            if s.startswith("; FEATURE:"):
                feat = s[10:].strip()
            elif s.startswith("; CHANGE_LAYER"):
                layer += 1
            elif s.startswith("; WIPE_START"):
                inwipe = True
            elif s.startswith("; WIPE_END"):
                inwipe = False
            continue
        code = s.split(";", 1)[0].strip()
        if not code:
            continue
        w = code.split()[0]
        if w == "M204":
            m = re.search(r"S(\d*\.?\d+)", code)
            if m:
                acc = float(m.group(1))
            continue
        if w == "G4":
            m = re.search(r"([PS])(\d*\.?\d+)", code)
            if m:
                b = _Move()
                b.d = 0.0
                b.t = float(m.group(2)) / (1000.0 if m.group(1) == "P" else 1.0)
                b.cat, b.layer, b.line = "dwell", layer, li
                bl.append(b)
            continue
        if w not in ("G0", "G1", "G2", "G3"):
            continue

        p = {k: float(v) for k, v in NUM.findall(code)}
        if "F" in p:
            F = p["F"]
        nx, ny, nz = p.get("X", x), p.get("Y", y), p.get("Z", z)
        de = p.get("E", 0.0)
        dx, dy, dz = nx - x, ny - y, nz - z

        is_spiral_zhop = w in ("G2", "G3") and (
            "P" in p or (abs(dz) > 1e-9 and abs(de) < 1e-9 and ("I" in p or "J" in p))
        )
        is_planar_arc = w in ("G2", "G3") and not is_spiral_zhop and ("I" in p or "J" in p)

        if is_planar_arc:
            I = p.get("I", 0.0)
            J = p.get("J", 0.0)
            d_xy, R, tx0, ty0, tx1, ty1, r_err = _planar_arc_geom(w, x, y, nx, ny, I, J)
            dxyz = math.hypot(d_xy, dz)
            if check_limits and r_err > 0.003:
                arc_violations.append((li + 1, layer, r_err, R))
            dx_in, dy_in = d_xy * tx0, d_xy * ty0
            dx_out, dy_out = d_xy * tx1, d_xy * ty1
        else:
            dxyz = math.sqrt(dx * dx + dy * dy + dz * dz)
            R = 0.0
            dx_in, dy_in = dx, dy
            dx_out, dy_out = dx, dy

        x, y, z = nx, ny, nz
        if dxyz < 1e-9 and abs(de) < 1e-9:
            continue

        b = _Move()
        b.d = dxyz if dxyz >= 1e-9 else abs(de)
        b.layer, b.line = layer, li
        if dxyz < 1e-9:
            b.cat = "retract" if de < 0 else "deretract"
        elif inwipe:
            b.cat = "wipe"
        elif de > 1e-9:
            b.cat = feat
        elif abs(dz) > 1e-9:
            b.cat = "zhop"
        else:
            b.cat = "travel"

        if dxyz < 1e-9:
            a = mach["a_retr"]
        elif de > 1e-9:
            a = min(acc, mach["a_extr"])
        else:
            a = min(acc, mach["a_trav"])

        v = F / 60.0
        if check_limits:
            if de > 1e-9 and dxyz > 0.1 and not inwipe:
                q_val = (de * FILA_AREA) / (b.d / max(v, 1e-9))
                max_q_seen = max(max_q_seen, q_val)
                if q_val > q_cap + 0.5:
                    q_violations.append((li + 1, layer, feat, q_val))
            if de > 1e-9 and v > 350.5:
                v_violations.append((li + 1, layer, feat, v))

        if is_planar_arc and R > 1e-6:
            a_xy = min(a, MAXA["x"], MAXA["y"])
            v = min(v, math.sqrt(k_cent * a_xy * R))

        # Worst-case axis ratios along the move for v and a bounds
        if is_planar_arc:
            # An arc sweeps across X and Y, so cap v and a conservatively by min(MAXV x,y), min(MAXA x,y)
            v = min(v, MAXV["x"], MAXV["y"])
            a = min(a, MAXA["x"], MAXA["y"])
            if dz:
                r_z = b.d / abs(dz)
                v = min(v, MAXV["z"] * r_z)
                a = min(a, MAXA["z"] * r_z)
            if de:
                r_e = b.d / abs(de)
                v = min(v, MAXV["e"] * r_e)
                a = min(a, MAXA["e"] * r_e)
        else:
            for k, c in (("x", dx), ("y", dy), ("z", dz), ("e", de)):
                if c:
                    r = b.d / abs(c)
                    v = min(v, MAXV[k] * r)
                    a = min(a, MAXA[k] * r)

        safe = v
        for k, c in (("x", dx_in), ("y", dy_in), ("z", dz), ("e", de)):
            if c:
                safe = min(safe, JERK[k] * b.d / abs(c))
        if is_planar_arc:
            for k, c in (("x", dx_out), ("y", dy_out)):
                if c:
                    safe = min(safe, JERK[k] * b.d / abs(c))

        b.v, b.a, b.safe = v, a, safe
        b.axv_in = tuple(v * c / b.d for c in (dx_in, dy_in, dz, de))
        b.axv_out = tuple(v * c / b.d for c in (dx_out, dy_out, dz, de))
        bl.append(b)

    # Classic jerk junction cornering
    prev = None
    jk = (JERK["x"], JERK["y"], JERK["z"], JERK["e"])
    for b in bl:
        if b.d == 0:
            prev = None
            b.vj = 0.0
            continue
        if prev is not None:
            vj = _jerk_junction(prev.axv_out, b.axv_in, jk, prev.v, b.v, prev.safe, b.safe)
        else:
            vj = b.safe
        b.vj = min(vj, b.v)
        prev = b

    # Backward & forward trapezoidal lookahead passes
    n = len(bl)
    nxt = 0.0
    for i in range(n - 1, -1, -1):
        b = bl[i]
        if b.d == 0:
            nxt = 0.0
            continue
        b.v1 = nxt
        b.v0 = min(b.vj, math.sqrt(b.v1 * b.v1 + 2.0 * b.a * b.d))
        nxt = b.v0

    pv = 0.0
    for i in range(n):
        b = bl[i]
        if b.d == 0:
            pv = 0.0
            continue
        b.v0 = min(b.v0, pv)
        nx_ = bl[i + 1].v0 if i + 1 < n and bl[i + 1].d > 0 else 0.0
        b.v1 = min(math.sqrt(b.v0 * b.v0 + 2.0 * b.a * b.d), nx_)
        pv = b.v1
        v0, v1, a, d, vc = b.v0, b.v1, b.a, b.d, b.v
        da = (vc * vc - v0 * v0) / (2.0 * a)
        dd = (vc * vc - v1 * v1) / (2.0 * a)
        if da + dd <= d:
            b.t = (vc - v0) / a + (vc - v1) / a + (d - da - dd) / vc
        else:
            vp = max(math.sqrt(max((2.0 * a * d + v0 * v0 + v1 * v1) / 2.0, 0.0)), v0, v1)
            b.t = (vp - v0) / a + (vp - v1) / a

    cat = collections.Counter()
    lay = collections.Counter()
    cum = [0.0] * len(lines)
    t = 0.0
    bi = 0
    for li in range(len(lines)):
        while bi < n and bl[bi].line == li:
            t += bl[bi].t
            cat[bl[bi].cat] += bl[bi].t
            lay[bl[bi].layer] += bl[bi].t
            bi += 1
        cum[li] = t

    if check_limits:
        audit = dict(
            max_q_seen=max_q_seen,
            q_violations=q_violations[:20],
            n_q_violations=len(q_violations),
            arc_violations=arc_violations[:20],
            n_arc_violations=len(arc_violations),
            v_violations=v_violations[:20],
            n_v_violations=len(v_violations),
        )
        return t, cum, dict(cat), dict(lay), audit
    return t, cum, dict(cat), dict(lay)


def fmt_hms(t):
    t = int(round(t))
    return "%dh %02dm %02ds" % (t // 3600, (t % 3600) // 60, t % 60)


def load_gcode_lines(path):
    if path.endswith(".3mf"):
        with zipfile.ZipFile(path) as z:
            gname = [n for n in z.namelist() if re.match(r"Metadata/plate_\d+\.gcode$", n)][0]
            return z.read(gname).decode("utf-8", errors="replace").split("\n")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().split("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gcode", help=".gcode or .gcode.3mf file")
    ap.add_argument("--by-layer", action="store_true")
    ap.add_argument("--check-limits", action="store_true")
    ap.add_argument("--k-cent", type=float, default=1.6)
    a = ap.parse_args()
    lines = load_gcode_lines(a.gcode)
    res = simulate(lines, k_cent=a.k_cent, check_limits=a.check_limits)
    if a.check_limits:
        tot, _, cat, lay, audit = res
    else:
        tot, _, cat, lay = res
    print("Total time %s (%.1f s)" % (fmt_hms(tot), tot))
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print("  %-26s %7.1f s  %5.1f%%" % (k, v, 100.0 * v / tot))
    if a.by_layer:
        for k in sorted(lay):
            print("  Layer %3d  %6.1f s" % (k, lay[k]))
    if a.check_limits:
        print(
            "Audit: max_Q=%.2f mm3/s, q_viol=%d, arc_viol=%d, v_viol=%d"
            % (audit["max_q_seen"], audit["n_q_violations"], audit["n_arc_violations"], audit["n_v_violations"])
        )
