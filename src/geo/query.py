# -*- coding: utf-8 -*-
"""오차 반경 안의 용도지역 비율, 건물, 지형 경사를 조회한다 (WBS 3.4).

3.3 transform.py가 낸 position(lat, lon, error_radius_m)을 받아 알람의
land_use[]와 evidence를 채울 값을 돌려준다. 외부 패키지 없이 동작한다.

    python3 src/geo/query.py --lat 37.73679 --lon 128.906239 --radius 74 \
        --landuse 용도지역폴더/landuse_5179.geojson \
        --buildings 건물폴더/buildings_5179.geojson \
        --dem 37808.img 37804.img

계산 방법
    - 용도지역 비율: 오차 원 안에 지점을 고르게 깔고(기본 약 2,000개) 각 지점이
      어느 용도지역 폴리곤에 드는지 센다. 조회 지점 수는 verdict_rule에 남긴다.
    - 건물 수: 오차 원과 겹치는 건물 폴리곤 수 (LT_C_SPBD에는 용도 칸이 없어 전체 건물).
    - 건물 근접 점수: 오차 원 안 후보 지점 중 건물에서 15m 안에 있는 비율
      (7주차 굴뚝 판정 방법. 후보 지점은 위와 같은 격자를 쓴다).
    - 지형 경사: 추정 좌표가 속한 DEM 칸을 가운데로 한 3x3칸(270m) 경사.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dem import Dem  # noqa: E402

N_POINTS = 2000          # 오차 원 안 조회 지점 수(대략)
NEAR_BUILDING_M = 15.0   # 건물 근접 판단 거리 (7주차 굴뚝 판정과 같음)
NEAREST_SEARCH_M = 300.0  # 가장 가까운 건물을 찾는 최소 거리. 반경이 더 크면 반경까지 찾는다


# ---------- 좌표 변환: 위도·경도 → EPSG:5179 ----------
def to_5179(lon, lat):
    a = 6378137.0
    f = 1 / 298.257222101
    e2 = f * (2 - f)
    k0, lon0, lat0, fe, fn = 0.9996, math.radians(127.5), math.radians(38), 1e6, 2e6

    def m(p):
        return a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * p
                    - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * p)
                    + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * p)
                    - (35 * e2**3 / 3072) * math.sin(6 * p))

    p, l = math.radians(lat), math.radians(lon)
    ep2 = e2 / (1 - e2)
    n = a / math.sqrt(1 - e2 * math.sin(p) ** 2)
    t = math.tan(p) ** 2
    c = ep2 * math.cos(p) ** 2
    aa = (l - lon0) * math.cos(p)
    x = fe + k0 * n * (aa + (1 - t + c) * aa**3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * aa**5 / 120)
    y = fn + k0 * (m(p) - m(lat0) + n * math.tan(p) * (aa * aa / 2 + (5 - t + 9 * c + 4 * c * c) * aa**4 / 24
                                                      + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * aa**6 / 720))
    return x, y


# ---------- 기하 계산 ----------
def _polys(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def _bbox(rings):
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return min(xs), min(ys), max(xs), max(ys)


def _inside(x, y, rings):
    """짝홀 규칙. 구멍(안쪽 고리)도 같이 처리된다."""
    inside = False
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _dist_to_poly(x, y, rings):
    if _inside(x, y, rings):
        return 0.0
    best = float("inf")
    for ring in rings:
        for i in range(len(ring) - 1):
            best = min(best, _seg_dist(x, y, ring[i][0], ring[i][1], ring[i + 1][0], ring[i + 1][1]))
    return best


def circle_points(cx, cy, r, n=N_POINTS):
    """원 안에 고른 격자 지점. 반경이 0이면 중심 한 점."""
    if r <= 0:
        return [(cx, cy)]
    step = r * math.sqrt(math.pi / n)
    k = int(r // step) + 1
    pts = []
    for i in range(-k - 1, k + 1):
        for j in range(-k - 1, k + 1):
            # 반 칸 비켜서 깔아 중심선 위 지점이 한쪽으로 쏠리지 않게 한다
            x, y = cx + (i + 0.5) * step, cy + (j + 0.5) * step
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                pts.append((x, y))
    return pts


# ---------- 데이터 ----------
GRID_M = 100.0  # 건물·용도지역을 미리 나눠 둘 칸 크기 (속도용, 결과는 같음)


def _grid(bboxes):
    """다각형 bbox 목록 → {(칸 x, 칸 y): [다각형 번호, ...]} (번호 순)."""
    g = {}
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        for gx in range(int(x0 // GRID_M), int(x1 // GRID_M) + 1):
            for gy in range(int(y0 // GRID_M), int(y1 // GRID_M) + 1):
                g.setdefault((gx, gy), []).append(i)
    return g


class GeoData:
    def __init__(self, landuse_path, buildings_path=None, dem_paths=None):
        self.landuse = []      # (name, rings, bbox)
        for f in json.load(open(landuse_path, encoding="utf-8"))["features"]:
            name = (f.get("properties") or {}).get("uname") or "(이름 없음)"
            for rings in _polys(f["geometry"]):
                self.landuse.append((name, rings, _bbox(rings)))
        self.buildings = []    # (rings, bbox)
        if buildings_path:
            for f in json.load(open(buildings_path, encoding="utf-8"))["features"]:
                for rings in _polys(f["geometry"]):
                    self.buildings.append((rings, _bbox(rings)))
        self.dem = Dem(dem_paths) if dem_paths else None
        # 속도: 칸마다 걸치는 다각형 번호를 미리 적어 두고, 근처 칸만 본다
        self._lu_grid = _grid([b for _, _, b in self.landuse])
        self._bd_grid = _grid([b for _, b in self.buildings])

    def landuse_at(self, x, y):
        for i in self._lu_grid.get((int(x // GRID_M), int(y // GRID_M)), ()):
            name, rings, (x0, y0, x1, y1) = self.landuse[i]
            if x0 <= x <= x1 and y0 <= y <= y1 and _inside(x, y, rings):
                return name
        return None

    def landuse_lookup(self, cx, cy, r):
        """원 하나를 조회할 때 쓰는 빠른 landuse_at (결과는 같음).
        원의 위아래 범위에 걸치는 변만 남겨, 지점마다 다각형 전체 변을 보지 않게 한다."""
        ymin, ymax = cy - r, cy + r
        ids = set()
        for gx in range(int((cx - r) // GRID_M), int((cx + r) // GRID_M) + 1):
            for gy in range(int(ymin // GRID_M), int(ymax // GRID_M) + 1):
                ids.update(self._lu_grid.get((gx, gy), ()))
        cands = []
        for i in sorted(ids):
            name, rings, bbox = self.landuse[i]
            edges = []
            for ring in rings:
                n = len(ring)
                for k in range(n):
                    x1, y1 = ring[k][0], ring[k][1]
                    x2, y2 = ring[(k + 1) % n][0], ring[(k + 1) % n][1]
                    if max(y1, y2) > ymin and min(y1, y2) <= ymax:
                        edges.append((x1, y1, x2, y2))
            cands.append((name, bbox, edges))

        def at(x, y):
            for name, (x0, y0, x1, y1), edges in cands:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    inside = False
                    for ax, ay, bx, by in edges:
                        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                            inside = not inside
                    if inside:
                        return name
            return None
        return at

    def _near_ids(self, x, y, d):
        """(x, y)에서 d 안에 걸칠 수 있는 건물 번호 (근처 칸 후보, 번호 순)."""
        ids = set()
        for gx in range(int((x - d) // GRID_M), int((x + d) // GRID_M) + 1):
            for gy in range(int((y - d) // GRID_M), int((y + d) // GRID_M) + 1):
                ids.update(self._bd_grid.get((gx, gy), ()))
        return sorted(ids)

    def buildings_near(self, x, y, d):
        """(x, y)에서 d m 안에 있는 건물 목록."""
        out = []
        for i in self._near_ids(x, y, d):
            rings, (x0, y0, x1, y1) = self.buildings[i]
            if x0 - d <= x <= x1 + d and y0 - d <= y <= y1 + d and _dist_to_poly(x, y, rings) <= d:
                out.append(rings)
        return out


def query(geo, lat, lon, radius_m, n_points=N_POINTS):
    """position 하나를 조회해 (land_use, evidence, info)를 돌려준다."""
    cx, cy = to_5179(lon, lat)
    pts = circle_points(cx, cy, radius_m, n_points)

    counts = {}
    landuse_at = geo.landuse_lookup(cx, cy, radius_m)
    for x, y in pts:
        name = landuse_at(x, y)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    hit = sum(counts.values())
    land_use = [{"name": k, "ratio": round(v / hit, 4)}
                for k, v in sorted(counts.items(), key=lambda kv: -kv[1])] if hit else []

    evidence = {"buildings_in_radius": None, "nearest_building_m": None,
                "terrain_slope_deg": None, "dem_resolution_m": None}
    info = {"n_points": len(pts), "landuse_coverage": round(hit / len(pts), 4)}

    if geo.buildings:
        evidence["buildings_in_radius"] = len(geo.buildings_near(cx, cy, radius_m))
        # 추정 위치(원 중심)에서 가장 가까운 건물 외곽까지 거리. 찾는 거리 안에 없으면 null (11/19 회의)
        far = max(radius_m, NEAREST_SEARCH_M)
        dists = [_dist_to_poly(cx, cy, r) for r in geo.buildings_near(cx, cy, far)]
        evidence["nearest_building_m"] = round(min(dists), 1) if dists else None
        near_cands = [(r, _bbox(r)) for r in geo.buildings_near(cx, cy, radius_m + NEAR_BUILDING_M)]
        d = NEAR_BUILDING_M
        near = sum(1 for x, y in pts
                   if any(x0 - d <= x <= x1 + d and y0 - d <= y <= y1 + d and _dist_to_poly(x, y, r) <= d
                          for r, (x0, y0, x1, y1) in near_cands))
        info["near_building_ratio"] = round(near / len(pts), 4)

    if geo.dem:
        s = geo.dem.slope_deg(cx, cy)
        evidence["terrain_slope_deg"] = None if s is None else round(s, 1)
        evidence["dem_resolution_m"] = geo.dem.cell
        h = geo.dem.height(cx, cy)
        info["ground_height_m"] = None if h is None else round(h, 1)

    return land_use, evidence, info


def main():
    ap = argparse.ArgumentParser(description="오차 반경 안 용도지역, 건물, 경사 조회 (WBS 3.4)")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius", type=float, required=True, help="오차 반경(m)")
    ap.add_argument("--landuse", required=True, help="landuse_5179.geojson")
    ap.add_argument("--buildings", help="buildings_5179.geojson")
    ap.add_argument("--dem", nargs="*", help="DEM IMG 파일들 (예: 37808.img 37804.img)")
    ap.add_argument("--points", type=int, default=N_POINTS, help="조회 지점 수")
    a = ap.parse_args()

    geo = GeoData(a.landuse, a.buildings, a.dem)
    land_use, evidence, info = query(geo, a.lat, a.lon, a.radius, a.points)
    print(json.dumps({"land_use": land_use, "evidence": evidence, "info": info},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
