"""시연 지역 주변 건물 폴리곤을 V-World에서 받아 저장한다 (WBS 3.2).

용도지역만으로는 구름, 안개, 굴뚝 같은 오탐을 거를 수 없다. 특히 굴뚝연기는
실제로 연기라서 사진만으로 산불과 구분되지 않는다. 구분하려면 "연기가 건물 위에서
났는가"를 봐야 하고, 그래서 건물 위치가 필요하다. 스키마의
evidence.buildings_in_radius 가 이 값을 쓴다.

받는 레이어
    LT_C_SPBD  건물 (MultiPolygon). 속성에 건물관리번호, 지상 층수, 주소가 있다

주의. 이 레이어에는 건물 용도(주거·상업·공업) 속성이 없다. 주거 건물만 세려면
용도지역 폴리곤과 겹쳐 봐야 한다. 굴뚝 판단에는 용도와 무관하게 건물 유무가 쓰인다.

사용법
    export VWORLD_KEY=발급받은키        # 키는 코드나 문서에 적지 않는다
    export VWORLD_DOMAIN=localhost     # 키 발급 때 등록한 도메인. 없으면 거부당한다
    python3 tools/fetch_buildings.py <출력 폴더> [최소경도 최소위도 최대경도 최대위도]

    기본 범위: 128.880 37.712 128.932 37.762 (fetch_landuse.py 와 같은 팀 공통 범위)

출력 (<출력 폴더> 안)
    buildings_4326.geojson   받은 폴리곤 원본 (위도·경도)
    buildings_5179.geojson   범위로 자른 폴리곤 (EPSG:5179, DEM·용도지역과 같은 좌표계)
    buildings_summary.csv    건물 한 채마다 대표점과 면적, 층수

좌표 변환과 다각형 자르기는 fetch_landuse.py 의 것을 그대로 쓴다.
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_landuse import (DEFAULT_BBOX, clip_ring, fetch_layer, polygons_of,
                           ring_area, to_5179)

LAYER = "LT_C_SPBD"
# V-World 는 geomFilter 한 번에 10km² 까지만 받는다. 시연 범위가 26km² 라 격자로
# 쪼개 여러 번 부른다. 경계에 걸친 건물이 두 타일에 모두 나오므로 건물관리번호로
# 중복을 지운다.
TILE_LIMIT_KM2 = 8.0


def centroid(ring):
    """고리의 무게중심. 반경 안 건물 수를 셀 때 건물 하나를 점 하나로 보기 위한 값."""
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-9:                       # 면적이 0에 가까우면 평균으로 대신한다
        n = len(ring)
        return sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n
    return cx / (3 * a), cy / (3 * a)


def main():
    if len(sys.argv) not in (2, 6):
        print(__doc__)
        sys.exit(1)
    out = Path(sys.argv[1])
    bbox = tuple(float(v) for v in sys.argv[2:6]) if len(sys.argv) == 6 else DEFAULT_BBOX
    key = (os.environ.get("VWORLD_KEY") or "").strip()
    if not key:
        sys.exit("VWORLD_KEY가 없습니다. 터미널에서 export VWORLD_KEY=발급받은키 를 먼저 입력하세요.")
    domain = os.environ.get("VWORLD_DOMAIN", "").strip()
    if not domain:
        sys.exit("VWORLD_DOMAIN이 없습니다. 키를 발급받을 때 등록한 도메인을 넣으세요 (예: localhost).")

    lo0, la0, lo1, la1 = bbox
    total = ring_area([to_5179(lo0, la0), to_5179(lo1, la0),
                       to_5179(lo1, la1), to_5179(lo0, la1)])

    out.mkdir(parents=True, exist_ok=True)

    # 범위를 10km² 아래로 쪼갠다
    n = 1
    while (total / 1e6) / (n * n) > TILE_LIMIT_KM2:
        n += 1
    dlon, dlat = (lo1 - lo0) / n, (la1 - la0) / n
    print(f"범위 {total / 1e6:.1f}km² 를 {n}x{n} 타일로 나눠 받는다 "
          f"(타일당 약 {total / 1e6 / n / n:.1f}km²)")

    seen, feats = set(), []
    for i in range(n):
        for j in range(n):
            tile = (lo0 + i * dlon, la0 + j * dlat,
                    lo0 + (i + 1) * dlon, la0 + (j + 1) * dlat)
            got = fetch_layer(LAYER, key, domain, tile)
            new = 0
            for f in got:
                sn = (f.get("properties") or {}).get("bd_mgt_sn")
                if sn and sn in seen:
                    continue            # 타일 경계에 걸쳐 두 번 나온 건물
                if sn:
                    seen.add(sn)
                feats.append(f)
                new += 1
            print(f"  타일 {i + 1},{j + 1}: 받은 {len(got)}개, 새로 {new}개")
    print(f"건물({LAYER}): 중복 뺀 {len(feats)}개")

    raw, clipped, rows = [], [], []
    for f in feats:
        props = dict(f.get("properties") or {})
        raw.append({"type": "Feature", "properties": props, "geometry": f["geometry"]})
        polys, area = [], 0.0
        big = None                           # 가장 넓은 고리를 대표점 계산에 쓴다
        for poly in polygons_of(f["geometry"]):
            rings = [clip_ring(ring, bbox) for ring in poly]
            if not rings or not rings[0]:
                continue
            rings = [[to_5179(x, y) for x, y in r] for r in rings if r]
            outer = ring_area(rings[0])
            area += outer - sum(ring_area(r) for r in rings[1:])
            if big is None or outer > big[0]:
                big = (outer, rings[0])
            polys.append([[list(p) for p in r + r[:1]] for r in rings])
        if not polys or big is None:
            continue
        clipped.append({"type": "Feature", "properties": props,
                        "geometry": {"type": "MultiPolygon", "coordinates": polys}})
        cx, cy = centroid(big[1])
        try:
            floors = int(props.get("gro_flo_co") or 0)
        except ValueError:
            floors = 0
        rows.append([props.get("bd_mgt_sn", ""), round(cx, 2), round(cy, 2),
                     round(area, 1), floors, props.get("buld_nm", ""),
                     props.get("sigungu", ""), props.get("gu", "")])

    with open(out / "buildings_4326.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": raw}, f, ensure_ascii=False)
    with open(out / "buildings_5179.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name",
                           "properties": {"name": "urn:ogc:def:crs:EPSG::5179"}},
                   "features": clipped}, f, ensure_ascii=False)
    rows.sort(key=lambda r: -r[3])          # 넓은 것부터. 아래 요약도 이 순서를 쓴다
    with open(out / "buildings_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bd_mgt_sn", "x_5179", "y_5179", "area_m2", "floors",
                    "buld_nm", "sigungu", "gu"])
        w.writerows(rows)

    built = sum(r[3] for r in rows)
    print(f"\n범위: 경도 {lo0}~{lo1}, 위도 {la0}~{la1} (면적 {total / 1e6:.2f}km²)")
    print(f"  범위 안 건물 {len(rows)}채, 바닥면적 합 {built / 1e4:.1f}ha "
          f"(범위의 {built / total * 100:.1f}%)")
    if rows:
        fl = sorted(r[4] for r in rows if r[4] > 0)
        areas = sorted(r[3] for r in rows)
        print(f"  바닥면적 중앙값 {areas[len(areas) // 2]:.0f}m², "
              f"가장 넓은 건물 {rows[0][3]:.0f}m² ({rows[0][5] or '이름 없음'})")
        print(f"  층수가 적힌 건물 {len(fl)}채, 중앙값 {fl[len(fl) // 2]}층, "
              f"최고 {fl[-1]}층")
    print("완료")


if __name__ == "__main__":
    main()
