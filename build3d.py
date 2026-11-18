"""
三维建模：支撑、横梁（均为打印姿态）、装配体；导出 STEP + STL
用法: STAND_VARIANT=desk40 python build3d.py   （需先运行 postprocess.py 生成 profile.json）
连接方式：卡扣，无螺钉。
打印姿态说明：
  支撑：侧面轮廓贴床，沿 z 拉伸 T；横梁方槽是轮廓里的贯通孔 → 全部为竖直壁，无悬空。
        支撑是纯拉伸体、正反对称，所以左右两片是同一个零件（打印两份即可）。
  横梁：40 mm 宽面贴床；卡扣臂、卡钩、导入斜面、让位槽都是平面图形沿 z 拉伸 → 无悬空；
        卡扣臂在打印平面内弯曲，弯曲应力沿层内方向，不会把层间撕开。
"""
import json
import math
import shapely
from shapely.geometry import Polygon, LineString, box
from shapely.geometry.polygon import orient
from build123d import (Wire, Face, Vector, Location, Plane, Axis, extrude, chamfer, mirror,
                       export_step, export_stl, Compound)
import params as P

OUT = P.MODEL_DIR
prof = json.load(open(f"{OUT}/profile.json"))


def face_from_poly(poly):
    poly = orient(poly, 1.0)                     # 外轮廓逆时针 → 面法向 +Z

    def wire(coords):
        pts = [tuple(p) for p in coords][:-1]
        return Wire.make_polygon([Vector(x, y, 0) for x, y in pts], close=True)
    return Face(wire(poly.exterior.coords), [wire(i.coords) for i in poly.interiors])


def extrude_poly(poly, h):
    s = extrude(face_from_poly(poly), amount=h)
    bb = s.bounding_box()
    assert abs(bb.min.Z) < 1e-6 and abs(bb.max.Z - h) < 1e-6, f"拉伸方向错误: {bb}"
    return s


# ---------------- 支撑 ----------------
sup_poly = Polygon(prof["exterior"], prof["holes"])
assert sup_poly.intersection(P.joint_slot()).area < 1e-6, "profile.json 里没有横梁方槽，请先重新运行 postprocess.py"
support_left = extrude_poly(sup_poly, P.T)
support_right = mirror(support_left, about=Plane.XZ)   # 与左支撑全等（纯拉伸体），仅摆放方向不同

# ---------------- 横梁（打印姿态：长度沿 X，BEAM_H 沿 Y，BEAM_W 沿 Z 竖直） ----------------
L_body = P.SPACING - P.T                 # 两支撑内侧面间距
half = L_body / 2
yA = P.TENON_H / 2                       # 卡扣臂外表面
yI = yA - P.ARM_T                        # 卡扣臂内表面
uc = P.T + P.SNAP_PLAY                   # 卡钩扣合面（从支撑内侧面量起）
uL = uc + P.HOOK_FLAT + P.HOOK_RAMP      # 卡扣臂端部


def beam_outline():
    parts = [box(-half, -P.BEAM_H / 2, half, P.BEAM_H / 2)]
    cuts = []
    for sx in (1, -1):
        def T_(u, y):
            return (sx * (half + u), y)
        for sy in (1, -1):
            parts.append(Polygon([T_(-1, sy * yI), T_(uL, sy * yI), T_(uL, sy * yA), T_(-1, sy * yA)]))          # 卡扣臂
            parts.append(Polygon([T_(uc, sy * (yA - 0.01)), T_(uc, sy * (yA + P.HOOK_H)),
                                  T_(uc + P.HOOK_FLAT, sy * (yA + P.HOOK_H)), T_(uL, sy * (yA - 0.01))]))         # 卡钩 + 导入斜面
            w = P.RELIEF_W
            cuts.append(LineString([T_(-P.ARM_ROOT + w / 2, sy * (yA + w / 2)), T_(2, sy * (yA + w / 2))]).buffer(w / 2))  # 外侧让位槽
        cuts.append(Polygon([T_(-P.ARM_ROOT, -yI), T_(3, -yI), T_(3, yI), T_(-P.ARM_ROOT, yI)]).buffer(-2).buffer(2))     # 两臂之间让位
    g = shapely.unary_union(parts).difference(shapely.unary_union(cuts))
    assert g.geom_type == "Polygon", g.geom_type
    return orient(g, 1.0)


beam_poly = beam_outline()
json.dump(dict(exterior=list(beam_poly.exterior.coords), holes=[list(i.coords) for i in beam_poly.interiors], half=half),
          open(f"{OUT}/beam_outline.json", "w"))
beam = extrude_poly(beam_poly, P.BEAM_W)
try:   # 梁身四条长边倒角 1 mm（只选 |Y| = BEAM_H/2 的整长边，不碰卡扣）
    long_edges = beam.edges().filter_by(Axis.X).filter_by(
        lambda e: abs(abs(e.center().Y) - P.BEAM_H / 2) < 1e-6 and abs(e.center().X) < 1e-6)
    beam = chamfer(long_edges, length=1.0)
except Exception as e:
    print("横梁长边倒角跳过:", e)

# ---------------- 装配（世界坐标：X 左右，Y 前→后，Z 向上） ----------------
left_w = support_left.moved(Location(Plane(origin=(-P.SPACING / 2 - P.T / 2, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))))
right_w = mirror(left_w, about=Plane.YZ)
beams_w = []
for j in P.JOINTS:   # 横梁截面长边 a → (0,cos φ,sin φ)，短边 b → (0,−sin φ,cos φ)
    c, s = math.cos(math.radians(j["phi"])), math.sin(math.radians(j["phi"]))
    beams_w.append(beam.moved(Location(Plane(origin=(0, j["x"] + P.BEAM_W / 2 * s, j["y"] - P.BEAM_W / 2 * c),
                                             x_dir=(1, 0, 0), z_dir=(0, -s, c)))))

parts = {"support_left": support_left, "support_right": support_right, "beam": beam}
report = {}
for nm, pt in parts.items():
    export_step(pt, f"{OUT}/{nm}.step")
    export_stl(pt, f"{OUT}/{nm}.stl", tolerance=0.01, angular_tolerance=0.1)
    bb = pt.bounding_box()
    report[nm] = dict(volume_cm3=round(pt.volume / 1000, 2), bbox_mm=[round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)], valid=pt.is_valid)
asm = Compound(label="laptop_stand", children=[left_w, right_w] + beams_w)
export_step(asm, f"{OUT}/assembly.step")
export_stl(asm, f"{OUT}/assembly.stl", tolerance=0.01, angular_tolerance=0.1)
bb = asm.bounding_box()
report["assembly"] = dict(bbox_mm=[round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)])
report["beams"] = [dict(name=j["name"], x=j["x"], y=j["y"], phi=j["phi"]) for j in P.JOINTS]
report["interference_beam_left_mm3"] = round(sum((b & left_w).volume for b in beams_w), 3)
report["interference_beam_right_mm3"] = round(sum((b & right_w).volume for b in beams_w), 3)
report["interference_beam_beam_mm3"] = round(sum((a & b).volume for i, a in enumerate(beams_w) for b in beams_w[i + 1:]), 3)
report["snap"] = dict(arm_t=P.ARM_T, flex_len_mm=round(P.ARM_ROOT + uc, 2), deflection_mm=round(P.HOOK_H - P.CLEAR, 2),
                      strain_pct=round(100 * P.snap_strain(), 2), hook_overlap_mm=round(P.HOOK_H - P.CLEAR, 2),
                      hook_protrude_mm=round(uL - P.T, 2))
json.dump(report, open(f"{OUT}/build_report.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
