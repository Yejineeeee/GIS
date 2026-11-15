# -*- coding: utf-8 -*-
"""시뮬레이터에 세울 건물만 골라 작은 geojson 으로 뽑는다.

`fetch_buildings.py` 가 받아 둔 15,876채는 10MB 가 넘어 브라우저에서 그리면
느리다. 시뮬레이터 카메라가 도는 범위만 남기고 쓰지 않는 칸을 버린다.

층수(`gro_flo_co`)만 남기고 건물관리번호나 도로명은 버린다. 화면에 세우는 데는
높이만 쓰기 때문이다. 좌표는 소수점 7자리로 끊는다. 7자리면 1cm 쯤이라
건물 모서리에는 넉넉하다.

    python tools/make_sim_buildings.py data/geo_moraegogae/buildings_4326.geojson sim/buildings.geojson

나온 파일은 `.gitignore` 의 `*.geojson` 에 걸려 저장소에 올라가지 않는다.
받아 쓰는 사람은 이 스크립트를 다시 돌리면 된다.
"""
import argparse
import json
import math
import os

# 연기를 피우는 자리. sim/index.html 의 FIRE 와 같은 값이다
FIRE_LON, FIRE_LAT = 128.9089, 37.7331

# 자동 비행 반지름이 0.0045도(약 500m)고 sweep 으로 가장 멀리 간 것이 612m 다.
# 그 바깥도 화면에 들어오므로 넉넉히 1.5km 로 자른다.
RADIUS_M = 1500.0

# 층당 높이. 건물 데이터에 실제 높이 칸이 없어 층수로 어림한다.
FLOOR_M = 3.0


def meters(lon, lat):
    """발화점에서 떨어진 거리를 미터로 돌려준다."""
    dy = (lat - FIRE_LAT) * 111320.0
    dx = (lon - FIRE_LON) * 111320.0 * math.cos(math.radians(FIRE_LAT))
    return math.hypot(dx, dy)


def centroid(geom):
    """폴리곤을 감싸는 네모의 가운데. 건물은 작아서 이 어림으로 충분하다."""
    xs, ys = [], []

    def walk(node):
        if isinstance(node, list) and node and isinstance(node[0], (int, float)):
            xs.append(node[0])
            ys.append(node[1])
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(geom.get("coordinates"))
    if not xs:
        return None
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def floors(props):
    """지상 층수. 빈 칸이나 숫자가 아닌 값은 1층으로 본다."""
    raw = str(props.get("gro_flo_co", "")).strip()
    try:
        n = int(float(raw))
    except ValueError:
        return 1
    return n if n >= 1 else 1


def round_coords(node, nd=7):
    if isinstance(node, list) and node and isinstance(node[0], (int, float)):
        return [round(node[0], nd), round(node[1], nd)]
    return [round_coords(child, nd) for child in node]


def main():
    ap = argparse.ArgumentParser(description="시뮬레이터용 건물 geojson 만들기")
    ap.add_argument("src", help="fetch_buildings.py 가 낸 buildings_4326.geojson")
    ap.add_argument("out", help="나올 파일 (보통 sim/buildings.geojson)")
    ap.add_argument("--radius", type=float, default=RADIUS_M,
                    help="발화점에서 자를 거리, 미터 (기본 %(default)s)")
    ap.add_argument("--fire", nargs=2, type=float, metavar=("LON", "LAT"),
                    help="발화점 (기본은 모래고개입구. 자곡동은 127.101707 37.470338)")
    args = ap.parse_args()
    if args.fire:
        global FIRE_LON, FIRE_LAT
        FIRE_LON, FIRE_LAT = args.fire

    with open(args.src, encoding="utf-8") as fh:
        src = json.load(fh)

    kept, tall = [], 0
    for feat in src.get("features", []):
        geom = feat.get("geometry") or {}
        c = centroid(geom)
        if c is None or meters(c[0], c[1]) > args.radius:
            continue
        n = floors(feat.get("properties") or {})
        if n >= 3:
            tall += 1
        kept.append({
            "type": "Feature",
            # 층수와 높이만 남긴다. 높이는 Cesium 이 그대로 쓴다.
            "properties": {"floors": n, "height_m": round(n * FLOOR_M, 1)},
            "geometry": {"type": geom.get("type"),
                         "coordinates": round_coords(geom.get("coordinates"))},
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": kept}, fh,
                  ensure_ascii=False, separators=(",", ":"))

    before = len(src.get("features", []))
    mb = os.path.getsize(args.out) / 1048576.0
    src_mb = os.path.getsize(args.src) / 1048576.0
    print(f"원본        {before:,}채  {src_mb:.1f}MB")
    print(f"반경 {args.radius:.0f}m 안  {len(kept):,}채  {mb:.1f}MB")
    print(f"3층 이상    {tall:,}채")
    print(f"층당 높이   {FLOOR_M}m")
    print(f"저장 -> {args.out}")


if __name__ == "__main__":
    main()
