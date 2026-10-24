# -*- coding: utf-8 -*-
"""오차 반경 안의 건물 수를 센다 (스키마의 evidence.buildings_in_radius).

용도지역만으로는 거를 수 없는 오탐이 있다. 굴뚝연기는 실제로 연기라서 사진만으로는
산불과 구분되지 않는다. 구분하려면 "연기가 건물 위에서 났는가"를 봐야 한다.

건물 하나를 대표점 하나로 보고 센다. 폴리곤을 다 열어 반경과 겹치는지 보는 것이
정확하지만, 건물 바닥면적이 중앙값 84m²(한 변 9m 남짓)이라 반경 10m 이상에서는
대표점으로 세도 차이가 거의 없다. 10MB 짜리 폴리곤을 알람마다 여는 것보다 낫다.

    python count_buildings.py 건물요약csv 지점csv [반경...]

지점 csv 는 lat, lon 칸이 있으면 된다. 없으면 x_5179, y_5179 를 쓴다.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_landuse import to_5179

RADII = (10, 30, 50, 100)


def load_buildings(path):
    pts = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                pts.append((float(r["x_5179"]), float(r["y_5179"]),
                            float(r.get("area_m2") or 0)))
            except (TypeError, ValueError):
                continue
    return pts


def grid_of(pts, cell):
    """격자에 넣어 가까운 건물만 보게 한다. 1만 5천 채를 지점마다 전부 보면 느리다."""
    g = {}
    for x, y, a in pts:
        g.setdefault((int(x // cell), int(y // cell)), []).append((x, y, a))
    return g


def count_within(g, cell, x, y, r):
    n = 0
    span = int(r // cell) + 1
    gx, gy = int(x // cell), int(y // cell)
    rr = r * r
    for i in range(gx - span, gx + span + 1):
        for j in range(gy - span, gy + span + 1):
            for bx, by, _ in g.get((i, j), ()):
                if (bx - x) ** 2 + (by - y) ** 2 <= rr:
                    n += 1
    return n


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    bpath, ppath = sys.argv[1], sys.argv[2]
    radii = [int(v) for v in sys.argv[3:]] or list(RADII)

    pts = load_buildings(bpath)
    print(f"건물 {len(pts)}채")
    cell = max(radii) * 2
    g = grid_of(pts, cell)

    with open(ppath, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"지점 {len(rows)}곳\n")

    head = f"  {'지점':6}{'구간':6}"
    for r in radii:
        head += f"{str(r) + 'm':>8}"
    print(head)
    out = []
    for row in rows:
        if row.get("lat") and row.get("lon"):
            x, y = to_5179(float(row["lon"]), float(row["lat"]))
        else:
            x, y = float(row["x_5179"]), float(row["y_5179"])
        counts = [count_within(g, cell, x, y, r) for r in radii]
        out.append((row, counts))
        line = f"  {row.get('pos_id', ''):6}{row.get('band', ''):6}"
        for c in counts:
            line += f"{c:>8}"
        print(line)

    # 구간별 평균. 산림이 많은 곳일수록 건물이 적어야 말이 된다
    print()
    bands = {}
    for row, counts in out:
        bands.setdefault(row.get("band", ""), []).append(counts)
    print(f"  {'구간':8}{'지점':>5}" + "".join(f"{str(r) + 'm 평균':>12}" for r in radii))
    for b, cs in sorted(bands.items()):
        line = f"  {b:8}{len(cs):>5}"
        for i in range(len(radii)):
            line += f"{sum(c[i] for c in cs) / len(cs):>12.1f}"
        print(line)


if __name__ == "__main__":
    main()
