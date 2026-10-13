# -*- coding: utf-8 -*-
"""시연 지역 안에서 평가셋 사진을 가상 배치할 위치를 고른다.

71330 드론 사진의 GPS 는 장소 대표값이고 절반은 기본값이라 위치로 쓸 수 없다
(docs/data_select.md). 그래서 평가셋 사진은 시연 지역 안의 정해진 위치에 가상으로
배치한다. 그 위치를 여기서 고른다.

촬영지 좌표 자체는 산 한가운데라 어느 반경에서도 1순위 비율이 1에 가깝다. 세 판정을
모두 보려면 산림과 주거가 섞이는 지점을 함께 넣어야 한다.

판정은 넣지 않는다. 판정 임계값은 3.5 모의 실험에서 정해지므로, 여기서는 1순위
비율만 재서 구간별로 고르고 임계값이 정해진 뒤 비율로 판정을 계산한다.

    python make_demo_positions.py 폴리곤파일 [출력csv]

폴리곤파일은 tools/fetch_zoning.py 로 받은 용도지역 GeoJSON 이다.
"""
import sys, csv, json
from collections import defaultdict
from shapely.geometry import shape, Point
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer

SRC = sys.argv[1] if len(sys.argv) > 1 else "zoning.geojson"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/demo_positions.csv"

SITE = (37.736790, 128.906239)     # 모래고개입구 라벨 GPS
# 촬영지를 중심으로 한 변 3.6km 네모를 훑는다. 1.2km 로 좁혔을 때는 1순위 비율이
# 낮은 칸이 14개뿐이라 미확정 쪽 위치를 채울 수 없었다.
SPAN, STEP = 1800, 40              # 훑는 범위(m)와 격자 간격(m)
RADII = [10, 30, 50, 100]
PICK_R = 50                        # 구간을 나눌 때 기준이 되는 반경
MIN_GAP = 120                      # 뽑은 위치 사이 최소 거리(m)
PER_BAND = 12                      # 구간마다 뽑을 개수
MIN_COV = 0.90                     # 반경 안에 용도지역이 이만큼은 있어야 한다

# 1순위 비율 구간. 임계값과 무관하게 나눌 수 있도록 넉넉히 띄웠다.
BANDS = [("높음", 0.90, 1.01), ("중간", 0.55, 0.80), ("낮음", 0.0, 0.50)]

FOREST = ("보전녹지지역", "자연녹지지역", "생산녹지지역")
RESID = ("제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역",
         "준주거지역", "제1종전용주거지역", "제2종전용주거지역")

fwd = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
inv = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)


def load_zones(path):
    """용도지역 폴리곤을 읽는다. 지정 연도가 다른 이력이 겹쳐 있으므로
    최신을 우선해 중복을 걷어낸다."""
    gj = json.load(open(path, encoding="utf-8"))
    items = []
    for f in gj["features"]:
        p = f["properties"]
        try:
            g = shp_transform(fwd.transform, shape(f["geometry"]))
        except Exception:
            continue
        if g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
        items.append((int(p.get("dyear") or 0), p.get("uname", ""), g))
    items.sort(key=lambda t: -t[0])

    taken, zones = None, defaultdict(list)
    for _, name, g in items:
        piece = g if taken is None else g.difference(taken)
        if piece.is_empty:
            continue
        zones[name].append(piece)
        taken = piece if taken is None else unary_union([taken, piece])
    return {k: unary_union(v) for k, v in zones.items()}, len(gj["features"])


def measure(zones, pt, r):
    """반경 안 용도별 넓이를 비율로 돌려준다.

    비율은 원 넓이가 아니라 용도지역이 있는 넓이로 나눈다. 용도지역 레이어가
    산간 지역을 덮지 않는 곳이 있어, 원 넓이로 나누면 데이터가 없는 것과
    여러 용도가 섞인 것이 구분되지 않는다. 덮인 비율은 따로 돌려준다.
    """
    circle = pt.buffer(r)
    areas = [(k, circle.intersection(g).area) for k, g in zones.items()]
    areas = sorted([(k, a) for k, a in areas if a > 0], key=lambda t: -t[1])
    covered = sum(a for _, a in areas)
    if covered <= 0:
        return [], 0.0
    return [(k, a / covered) for k, a in areas], covered / circle.area


def mixes_forest_resid(parts):
    """산림 계열과 주거 계열이 함께 들어 있는지. 녹지끼리 섞인 것은 이 과제가
    보려는 모호함이 아니므로 가려낸다."""
    f = sum(r for k, r in parts if k in FOREST)
    d = sum(r for k, r in parts if k in RESID)
    return f >= 0.05 and d >= 0.05


def main():
    zones, n_poly = load_zones(SRC)
    print(f"폴리곤 {n_poly}건, 용도 {len(zones)}종")

    cx, cy = fwd.transform(SITE[1], SITE[0])
    site = Point(cx, cy)

    found = []
    for dx in range(-SPAN, SPAN + 1, STEP):
        for dy in range(-SPAN, SPAN + 1, STEP):
            pt = Point(cx + dx, cy + dy)
            parts, cov = measure(zones, pt, PICK_R)
            if cov < MIN_COV or not parts:
                continue
            top = parts[0][1]
            band = next((b for b, lo, hi in BANDS if lo <= top < hi), None)
            if band is None:
                continue
            found.append({"pt": pt, "band": band, "top": top,
                          "mix": mixes_forest_resid(parts),
                          "dist": site.distance(pt)})

    print(f"\n훑은 칸 가운데 쓸 수 있는 칸 {len(found)}개")
    for b, _, _ in BANDS:
        c = [f for f in found if f["band"] == b]
        print(f"   1순위 {b:3} {len(c):5}칸   그중 산림과 주거가 섞인 칸 "
              f"{sum(1 for f in c if f['mix'])}개")

    # 구간마다 산림과 주거가 섞인 곳을 먼저, 촬영지에 가까운 곳을 먼저 고른다.
    # 한 곳에 몰리지 않게 뽑은 위치끼리 최소 거리를 둔다.
    picked = []
    for b, _, _ in BANDS:
        pool = sorted((f for f in found if f["band"] == b),
                      key=lambda f: (not f["mix"], f["dist"]))
        chosen = []
        for f in pool:
            if any(f["pt"].distance(c["pt"]) < MIN_GAP for c in chosen):
                continue
            chosen.append(f)
            if len(chosen) == PER_BAND:
                break
        if len(chosen) < PER_BAND:
            print(f"   주의: 1순위 {b} 구간에서 {PER_BAND}개가 필요한데 "
                  f"{len(chosen)}개만 뽑혔다")
        picked.extend(chosen)

    rows = []
    for i, f in enumerate(sorted(picked, key=lambda f: (f["band"], f["dist"])), 1):
        lon, lat = inv.transform(f["pt"].x, f["pt"].y)
        row = {"pos_id": f"P{i:03d}", "lat": round(lat, 6), "lon": round(lon, 6),
               "band": f["band"], "forest_resid_mix": int(f["mix"]),
               "dist_from_site_m": round(f["dist"])}
        for r in RADII:
            parts, cov = measure(zones, f["pt"], r)
            row[f"top_{r}"] = parts[0][0] if parts else ""
            row[f"ratio_{r}"] = round(parts[0][1], 3) if parts else 0
            row[f"second_{r}"] = parts[1][0] if len(parts) > 1 else ""
            row[f"cov_{r}"] = round(cov, 3)
        rows.append(row)

    cols = (["pos_id", "lat", "lon", "band", "forest_resid_mix",
             "dist_from_site_m"] +
            [f"{p}_{r}" for r in RADII for p in ("top", "ratio", "second", "cov")])
    with open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n위치 {len(rows)}개 저장 -> {OUT}")
    print(f"\n{'위치':<7}{'구간':<5}{'섞임':<5}{'촬영지에서':>9}" +
          "".join(f"{str(r) + 'm 1순위':>12}" for r in RADII))
    for row in rows:
        print(f"{row['pos_id']:<7}{row['band']:<5}"
              f"{'예' if row['forest_resid_mix'] else '아니오':<5}"
              f"{row['dist_from_site_m']:>8}m" +
              "".join(f"{row[f'ratio_{r}']:>12.2f}" for r in RADII))


if __name__ == "__main__":
    main()
