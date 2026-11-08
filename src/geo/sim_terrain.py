# -*- coding: utf-8 -*-
"""시뮬레이터 지형 높이 격자를 DEM 처럼 읽는다 (WBS 4.7 측정용).

시뮬레이터 프레임으로 위치 오차를 잴 때 V-World 90m DEM 을 넣으면, 시뮬레이터가
그린 Cesium 지형과 높이 차이가 자리마다 달라서(발화점 +13m, 드론 자리 +25~31m)
탐지가 밑동을 정확히 짚어도 600m 에서 94m, 800m 에서 123m 가 틀렸다. DEM 을 빼면
드론과 발화점의 높이 차(2~8m)가 남아 10~22m 가 틀렸다.

시뮬레이터에서 뽑은 지형을 그대로 쓰면 이 두 오차가 모두 없어지고, 남는 오차는
탐지가 어느 픽셀을 짚었는가에서만 나온다.

    격자 뽑기: sim/index.html 콘솔에서  await terrainGrid()  ->  sim/terrain.json

    from sim_terrain import SimTerrain
    t = SimTerrain("sim/terrain.json")
    t.height(x, y)         # EPSG:5179 좌표의 지면 높이(m), 격자 밖이면 None

Dem 과 같은 height(x, y) 를 가지므로 transform.to_position(..., dem=t) 로 넣는다.
실제 드론 사진에는 쓰지 않는다. 그때는 DEM 이 맞다.
"""
import json
import math

from query import to_5179


class SimTerrain:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        self.lat0, self.lon0 = g["lat0"], g["lon0"]
        self.half, self.step, self.n = g["half_m"], g["step_m"], g["n"]
        self.m_lat, self.m_lon = g["m_per_deg_lat"], g["m_per_deg_lon"]
        self.h = g["heights"]            # 남 → 북 줄마다 서 → 동
        # 5179 -> 경위도를 뒤집으려고 격자 가운데에서 기울기를 미리 잰다
        self.x0, self.y0 = to_5179(self.lon0, self.lat0)
        d = 1e-3
        xe, ye = to_5179(self.lon0 + d, self.lat0)
        xn, yn = to_5179(self.lon0, self.lat0 + d)
        self.jac = ((xe - self.x0) / d, (xn - self.x0) / d,
                    (ye - self.y0) / d, (yn - self.y0) / d)

    def _lonlat(self, x, y):
        """EPSG:5179 -> 경위도. 가운데 기울기로 시작해 몇 번 고친다."""
        a, b, c, d = self.jac
        det = a * d - b * c
        lon, lat = self.lon0, self.lat0
        for _ in range(4):
            fx, fy = to_5179(lon, lat)
            rx, ry = x - fx, y - fy
            lon += (d * rx - b * ry) / det
            lat += (-c * rx + a * ry) / det
            if math.hypot(rx, ry) < 0.01:
                break
        return lon, lat

    def _at(self, i, j):
        return self.h[i * self.n + j]

    cell = None          # query.py 가 dem_resolution_m 에 쓴다

    def slope_deg(self, x, y, d=30.0):
        """지면 경사(도). 둘레 d m 높이 차로 어림한다."""
        hs = [self.height(x + dx, y + dy) for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d))]
        if any(h is None for h in hs):
            return None
        gx, gy = (hs[0] - hs[1]) / (2 * d), (hs[2] - hs[3]) / (2 * d)
        return math.degrees(math.atan(math.hypot(gx, gy)))

    def height(self, x, y):
        lon, lat = self._lonlat(x, y)
        north = (lat - self.lat0) * self.m_lat + self.half
        east = (lon - self.lon0) * self.m_lon + self.half
        fi, fj = north / self.step, east / self.step
        i, j = int(math.floor(fi)), int(math.floor(fj))
        if i < 0 or j < 0 or i + 1 >= self.n or j + 1 >= self.n:
            return None
        di, dj = fi - i, fj - j
        return ((1 - di) * (1 - dj) * self._at(i, j) + (1 - di) * dj * self._at(i, j + 1)
                + di * (1 - dj) * self._at(i + 1, j) + di * dj * self._at(i + 1, j + 1))
