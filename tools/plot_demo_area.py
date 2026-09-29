# -*- coding: utf-8 -*-
"""시연 지역 후보를 그림으로 남긴다. find_demo_area.py 와 같은 기준으로 계산한다.

    python plot_demo_area.py 폴리곤파일 [격자크기m]
"""
import sys, json
from collections import defaultdict
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SRC = sys.argv[1] if len(sys.argv) > 1 else "zoning.geojson"
CELL = int(sys.argv[2]) if len(sys.argv) > 2 else 500
TOUCH, MIN_FOREST, TOP = 5, 1.0, 3
FOREST = ("보전녹지지역", "자연녹지지역", "생산녹지지역")
RESID = ("제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역",
         "준주거지역", "제1종전용주거지역", "제2종전용주거지역")

to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform
to_deg = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform

items = []
for f in json.load(open(SRC, encoding="utf-8"))["features"]:
    p = f["properties"]
    try:
        g = shp_transform(to_m, shape(f["geometry"]))
    except Exception:
        continue
    if g.is_empty:
        continue
    items.append((int(p.get("dyear") or 0), p.get("uname", ""), g if g.is_valid else g.buffer(0)))
items.sort(key=lambda t: -t[0])

groups, taken = defaultdict(list), None
for _, name, g in items:
    g2 = g.difference(taken) if taken is not None else g
    if g2.is_empty:
        continue
    taken = g2 if taken is None else unary_union([taken, g2])
    if name in FOREST:
        groups["산림"].append(g2)
    elif name in RESID:
        groups["주거"].append(g2)

forest, resid = unary_union(groups["산림"]), unary_union(groups["주거"])
parts = list(forest.geoms) if forest.geom_type == "MultiPolygon" else [forest]
forest = unary_union([g for g in parts if g.area >= MIN_FOREST * 1e6])
edge = forest.boundary.intersection(resid.buffer(TOUCH))

minx, miny, maxx, maxy = forest.union(resid).bounds
cells, y = [], miny
while y < maxy:
    x = minx
    while x < maxx:
        c = box(x, y, x + CELL, y + CELL)
        part = edge.intersection(c)
        if not part.is_empty and part.length > 0:
            fa, ra = forest.intersection(c).area, resid.intersection(c).area
            cells.append((part.length * (0.3 + min(fa, ra) / max(fa, ra, 1)), c))
        x += CELL
    y += CELL
cells.sort(key=lambda t: -t[0])

fig, ax = plt.subplots(figsize=(10, 8))
def draw(geom, **kw):
    gs = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for g in gs:
        ax.add_patch(MplPoly(list(g.exterior.coords), **kw))

draw(forest, facecolor="#5f8a4e", edgecolor="none", alpha=0.75)
draw(resid, facecolor="#d98032", edgecolor="none", alpha=0.75)
for g in ([edge] if edge.geom_type == "LineString" else list(edge.geoms)):
    if g.geom_type == "LineString":
        xs, ys = g.xy
        ax.plot(xs, ys, color="#b32020", linewidth=1.4)

for i, (_, c) in enumerate(cells[:TOP], 1):
    bx, by, bx2, by2 = c.bounds
    ax.add_patch(Rectangle((bx, by), bx2-bx, by2-by, fill=False,
                           edgecolor="#1f3864", linewidth=2.2))
    lon, lat = to_deg(c.centroid.x, c.centroid.y)
    ax.annotate(f"{i}", (c.centroid.x, by2), color="#1f3864", fontsize=15,
                fontweight="bold", ha="center", va="bottom")
    print(f"{i}순위  {lat:.5f}, {lon:.5f}")

# 멀리 떨어진 작은 폴리곤 때문에 여백이 생기므로 산덩어리 범위에 맞춘다.
fx, fy, fx2, fy2 = forest.bounds
m = 600
ax.set_xlim(fx - m, fx2 + m); ax.set_ylim(fy - m, fy2 + m); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor("#cccccc")
ax.set_title("시연 지역 후보  강릉시 서부\n녹색 산림 계열, 주황 주거 계열, 붉은 선 맞닿은 경계",
             fontsize=13, pad=14)
handles = [MplPoly([(0,0)], facecolor="#5f8a4e", alpha=.75, label="산림 계열"),
           MplPoly([(0,0)], facecolor="#d98032", alpha=.75, label="주거 계열")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig("results/demo_area_candidates.png", dpi=150)
print("\n저장: results/demo_area_candidates.png")
