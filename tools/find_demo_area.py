# -*- coding: utf-8 -*-
"""시연 지역 후보를 용도지역 데이터로 고른다.

산림과 주거지가 맞닿은 곳을 눈으로 찍지 않고, 두 용도가 실제로 접하는 경계의
길이를 재서 순위를 매긴다. 격자로 잘라 칸별로 재고 위쪽 몇 칸을 내놓는다.

    python find_demo_area.py 폴리곤파일 [격자크기m]
"""
import sys, json, math
from collections import defaultdict
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer

SRC = sys.argv[1] if len(sys.argv) > 1 else "zoning.geojson"
CELL = int(sys.argv[2]) if len(sys.argv) > 2 else 500     # 격자 한 변(m)
TOUCH = 5                                                  # 맞닿음으로 보는 거리(m)
MIN_FOREST = 1.0                                           # 산덩어리 최소 넓이(km²)

# 용도지역 이름으로 나눈다. 녹지지역이 곧 산림은 아니므로 뒤에서 임상도로 확인해야 한다.
FOREST = ("보전녹지지역", "자연녹지지역", "생산녹지지역")
RESID = ("제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역",
         "준주거지역", "제1종전용주거지역", "제2종전용주거지역")

to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform
to_deg = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform

gj = json.load(open(SRC, encoding="utf-8"))
print(f"폴리곤 {len(gj['features'])}건")

# 지정 연도가 다른 이력이 겹쳐 있으므로 최신을 우선해 중복을 걷어낸다.
items = []
for f in gj["features"]:
    p = f["properties"]
    try:
        g = shp_transform(to_m, shape(f["geometry"]))
    except Exception:
        continue
    if g.is_empty:
        continue
    if not g.is_valid:
        g = g.buffer(0)
    items.append((int(p.get("dyear") or 0), p.get("uname", ""), g))
items.sort(key=lambda t: -t[0])

groups = defaultdict(list)
taken = None
for _, name, g in items:
    g2 = g.difference(taken) if taken is not None else g
    if g2.is_empty:
        continue
    taken = g2 if taken is None else unary_union([taken, g2])
    if name in FOREST:
        groups["산림"].append(g2)
    elif name in RESID:
        groups["주거"].append(g2)

forest = unary_union(groups["산림"]) if groups["산림"] else None
resid = unary_union(groups["주거"]) if groups["주거"] else None
if forest is None or resid is None:
    raise SystemExit("산림이나 주거 용도가 없습니다. 범위를 넓혀 받으세요.")

print(f"산림 계열 {forest.area/1e6:.1f}km²   주거 계열 {resid.area/1e6:.1f}km²")

# 두 용도가 맞닿은 선
edge = forest.boundary.intersection(resid.buffer(TOUCH))
print(f"맞닿은 경계 길이 {edge.length/1000:.1f}km")
if edge.is_empty:
    raise SystemExit("맞닿은 곳이 없습니다.")

# 녹지지역이라도 도심 공원이나 작은 언덕이면 시연에 맞지 않는다. 이어진 덩어리가
# 충분히 큰 것만 산림으로 본다.
parts = list(forest.geoms) if forest.geom_type == "MultiPolygon" else [forest]
big = [g for g in parts if g.area >= MIN_FOREST * 1e6]
print(f"녹지 덩어리 {len(parts)}개 중 {MIN_FOREST}km² 이상 {len(big)}개")
if not big:
    raise SystemExit("큰 산덩어리가 없습니다. MIN_FOREST 를 낮추거나 범위를 넓히세요.")
forest_big = unary_union(big)
edge = forest_big.boundary.intersection(resid.buffer(TOUCH))
print(f"큰 덩어리만 본 경계 길이 {edge.length/1000:.1f}km")

minx, miny, maxx, maxy = forest_big.union(resid).bounds
cells = []
y = miny
while y < maxy:
    x = minx
    while x < maxx:
        cell = box(x, y, x + CELL, y + CELL)
        part = edge.intersection(cell)
        if not part.is_empty and part.length > 0:
            cells.append((part.length,
                          forest_big.intersection(cell).area,
                          resid.intersection(cell).area,
                          cell))
        x += CELL
    y += CELL

# 경계가 길고 두 용도가 모두 충분히 들어 있는 칸이 좋다.
def score(c):
    length, fa, ra, _ = c
    balance = min(fa, ra) / max(fa, ra, 1)
    return length * (0.3 + balance)

cells.sort(key=score, reverse=True)
print(f"\n격자 {CELL}m, 경계가 지나는 칸 {len(cells)}개\n")
print(f"{'순위':<4}{'중심 위도, 경도':<26}{'경계(m)':>9}{'산림':>8}{'주거':>8}")
for i, (length, fa, ra, cell) in enumerate(cells[:8], 1):
    cx, cy = cell.centroid.x, cell.centroid.y
    lon, lat = to_deg(cx, cy)
    area = CELL * CELL
    print(f"{i:<4}{lat:.5f}, {lon:.5f}{'':<8}{length:>9.0f}"
          f"{fa/area*100:>7.0f}%{ra/area*100:>7.0f}%")

best = cells[0][3]
bminx, bminy, bmaxx, bmaxy = best.bounds
lo1, la1 = to_deg(bminx, bminy)
lo2, la2 = to_deg(bmaxx, bmaxy)
print(f"\n1순위 칸 범위: {lo1:.5f} {la1:.5f} {lo2:.5f} {la2:.5f}")
print(f"지도에서 보기: https://map.kakao.com/link/map/시연후보,"
      f"{to_deg(best.centroid.x, best.centroid.y)[1]:.5f},"
      f"{to_deg(best.centroid.x, best.centroid.y)[0]:.5f}")
