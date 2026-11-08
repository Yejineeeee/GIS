# -*- coding: utf-8 -*-
"""자곡동 지형에서 연기 밑동이 능선에 가려지는 촬영 자리를 찾는다 (4.7 가림 판정 평가용).

산림(자연녹지) 안에 발화점을 깔고, 드론을 발화점에서 D m, 방위 30도 간격, 지면 위 h m 에
둔다. 드론에서 발화점 밑동(지면 + 3m)까지의 시선이 지형에 막히고, 연기 기둥 위쪽
(지면 + 40m)은 보이면 '가린 장면'이다. 같은 발화점에서 밑동이 보이는 자리를 하나 골라
'보이는 장면'으로 짝지어, 가림 판정이 보이는 장면을 가렸다고 잘못 보는지도 잰다.

    python tools/find_occluded.py sim/terrain_jagok3d.json sim/occl_shots.json

정답 판단은 지형 자료로만 한다(탐지나 occlusion.py 를 쓰지 않는다).
"""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/geo"))

from query import GeoData, to_5179  # noqa: E402
from sim_terrain import SimTerrain  # noqa: E402

R = 6378137.0
H = 120              # 드론 지면 위 높이 (시뮬레이터 기본값)
DISTS = (400, 600)
BASE_M, TOP_M = 3.0, 40.0
N_PAIRS = 20


def move(lat, lon, e, n):
    return lat + math.degrees(n / R), lon + math.degrees(e / (R * math.cos(math.radians(lat))))


def ground(dem, lat, lon):
    return dem.height(*to_5179(lon, lat))


def blocked(dem, cam, tgt, z_cam, z_tgt):
    """cam 에서 tgt 까지 시선이 지형에 막히나. 끝 20m 는 보지 않는다(목표 자리 지면)."""
    (la0, lo0), (la1, lo1) = cam, tgt
    x0, y0 = to_5179(lo0, la0)
    x1, y1 = to_5179(lo1, la1)
    L = math.hypot(x1 - x0, y1 - y0)
    d = 10.0
    while d < L - 20:
        t = d / L
        g = ground(dem, la0 + (la1 - la0) * t, lo0 + (lo1 - lo0) * t)
        if g is not None and g > z_cam + (z_tgt - z_cam) * t:
            return True
        d += 10.0
    return False


def main():
    dem = SimTerrain(sys.argv[1])
    out = Path(sys.argv[2])
    geo = GeoData(str(ROOT / "data/geo_jagok/landuse_5179.geojson"))
    rng = random.Random(0)
    fires = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            la, lo = move(dem.lat0, dem.lon0, j * 150, i * 150)
            if geo.landuse_at(*to_5179(lo, la)) == "자연녹지지역" and ground(dem, la, lo) is not None:
                fires.append((round(la, 6), round(lo, 6)))
    rng.shuffle(fires)
    pairs = []
    for la, lo in fires:
        gf = ground(dem, la, lo)
        occl, vis = [], []
        for D in DISTS:
            for b in range(0, 360, 30):
                br = math.radians(b)
                cla, clo = move(la, lo, D * math.sin(br), D * math.cos(br))
                gc = ground(dem, cla, clo)
                if gc is None:
                    continue
                zc = gc + H
                base = blocked(dem, (cla, clo), (la, lo), zc, gf + BASE_M)
                top = blocked(dem, (cla, clo), (la, lo), zc, gf + TOP_M)
                shot = {"D": D, "bearing": b}
                if base and not top:
                    occl.append(shot)
                elif not base:
                    vis.append(shot)
        if occl and vis:
            o = rng.choice(occl)
            v = rng.choice([s for s in vis if s["D"] == o["D"]] or vis)
            pairs.append({"fire": {"lat": la, "lon": lo}, "occluded": o, "visible": v})
        if len(pairs) >= N_PAIRS:
            break
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"산림 발화점 {len(fires)}곳 중 가린 자리가 있는 곳으로 {len(pairs)}쌍 -> {out}")


if __name__ == "__main__":
    main()
