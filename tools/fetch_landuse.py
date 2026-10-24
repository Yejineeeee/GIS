"""시연 지역 주변 용도지역 폴리곤을 V-World에서 받아 저장한다 (WBS 3.2).

받는 레이어 (용도지역 4종을 모두 받아야 산간 지역이 비지 않는다)
    LT_C_UQ111 도시지역 / LT_C_UQ112 관리지역 / LT_C_UQ113 농림지역 / LT_C_UQ114 자연환경보전지역

사용법
    export VWORLD_KEY=발급받은키        # 키는 코드나 문서에 적지 않는다
    python3 tools/fetch_landuse.py <출력 폴더> [최소경도 최소위도 최대경도 최대위도]

    기본 범위: 128.880 37.712 128.932 37.762 (모래고개입구 주변, 팀 공통 범위)

출력 (<출력 폴더> 안)
    landuse_4326.geojson   받은 폴리곤 원본 (위도·경도)
    landuse_5179.geojson   범위로 자른 폴리곤 (EPSG:5179, DEM과 같은 좌표계)
    landuse_summary.csv    용도지역별 면적과 비율

외부 패키지 없이 파이썬 기본 기능만 쓴다.
"""
import csv
import json
import math
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LAYERS = {
    "LT_C_UQ111": "도시지역",
    "LT_C_UQ112": "관리지역",
    "LT_C_UQ113": "농림지역",
    "LT_C_UQ114": "자연환경보전지역",
}
API = "https://api.vworld.kr/req/data"
DEFAULT_BBOX = (128.880, 37.712, 128.932, 37.762)  # 경도·위도 (예진 지정 범위)
NAME_KEYS = ("uname", "UNAME", "name", "NAME")  # 용도지역 이름이 들어 있는 속성 후보


# ---------- 좌표 변환: 위도·경도(EPSG:4326) → EPSG:5179 (GRS80 TM, DEM과 같은 좌표계) ----------
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


# ---------- 다각형 계산 ----------
def ring_area(ring):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def clip_ring(ring, box):
    """사각형 box(xmin, ymin, xmax, ymax)로 고리 하나를 자른다 (Sutherland-Hodgman)."""
    xmin, ymin, xmax, ymax = box
    edges = [
        (lambda p: p[0] >= xmin, lambda p, q: (xmin, p[1] + (q[1] - p[1]) * (xmin - p[0]) / (q[0] - p[0]))),
        (lambda p: p[0] <= xmax, lambda p, q: (xmax, p[1] + (q[1] - p[1]) * (xmax - p[0]) / (q[0] - p[0]))),
        (lambda p: p[1] >= ymin, lambda p, q: (p[0] + (q[0] - p[0]) * (ymin - p[1]) / (q[1] - p[1]), ymin)),
        (lambda p: p[1] <= ymax, lambda p, q: (p[0] + (q[0] - p[0]) * (ymax - p[1]) / (q[1] - p[1]), ymax)),
    ]
    pts = [tuple(p) for p in ring]
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    for inside, cross in edges:
        if not pts:
            break
        out = []
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
        pts = out
    return pts if len(pts) >= 3 else []


def polygons_of(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


# ---------- V-World 요청 ----------
def request(params):
    url = API + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLError):
            sys.exit("SSL 인증서 오류: 파이썬 인증서 설정 문제. 화면을 캡처해서 공유해주세요.\n" + str(e))
        raise


def fetch_layer(layer, key, domain, bbox4326):
    feats, page = [], 1
    while True:
        params = {
            "service": "data", "request": "GetFeature", "data": layer, "key": key,
            "format": "json", "crs": "EPSG:4326", "size": 1000, "page": page,
            "geometry": "true", "attribute": "true",
            "geomFilter": "BOX(%f,%f,%f,%f)" % bbox4326,
        }
        if domain:
            params["domain"] = domain
        res = request(params)["response"]
        status = res.get("status")
        if status == "NOT_FOUND":
            return feats
        if status != "OK":
            sys.exit(f"{layer} 요청 실패: {json.dumps(res.get('error', res), ensure_ascii=False)}")
        feats += res["result"]["featureCollection"]["features"]
        if page >= int(res.get("page", {}).get("total", 1)):
            return feats
        page += 1


def name_of(props):
    for k in NAME_KEYS:
        if props.get(k):
            return props[k]
    return "(이름 없음)"


def main():
    if len(sys.argv) not in (2, 6):
        print(__doc__)
        sys.exit(1)
    out = Path(sys.argv[1])
    bbox = tuple(float(v) for v in sys.argv[2:6]) if len(sys.argv) == 6 else DEFAULT_BBOX
    key = os.environ.get("VWORLD_KEY")
    if not key:
        sys.exit("VWORLD_KEY가 없습니다. 터미널에서 export VWORLD_KEY=발급받은키 를 먼저 입력하세요.")
    domain = os.environ.get("VWORLD_DOMAIN", "")

    # 범위 사각형의 실제 면적 (EPSG:5179에서 계산)
    lo0, la0, lo1, la1 = bbox
    total = ring_area([to_5179(lo0, la0), to_5179(lo1, la0), to_5179(lo1, la1), to_5179(lo0, la1)])

    out.mkdir(parents=True, exist_ok=True)
    raw, clipped, areas = [], [], {}
    for layer, kind in LAYERS.items():
        feats = fetch_layer(layer, key, domain, bbox)
        print(f"{kind}({layer}): {len(feats)}개")
        for f in feats:
            props = dict(f.get("properties") or {})
            props["layer"], props["kind"] = layer, kind
            raw.append({"type": "Feature", "properties": props, "geometry": f["geometry"]})
            name = name_of(props)
            polys, area = [], 0.0
            for poly in polygons_of(f["geometry"]):
                # 위도·경도 상태에서 범위로 자른 뒤 EPSG:5179로 바꾼다
                rings = [clip_ring(ring, bbox) for ring in poly]
                if not rings[0]:
                    continue
                rings = [[to_5179(x, y) for x, y in r] for r in rings if r]
                area += ring_area(rings[0]) - sum(ring_area(r) for r in rings[1:])
                polys.append([[list(p) for p in r + r[:1]] for r in rings])
            if polys:
                clipped.append({"type": "Feature", "properties": props,
                                "geometry": {"type": "MultiPolygon", "coordinates": polys}})
                areas[(kind, name)] = areas.get((kind, name), 0.0) + area

    with open(out / "landuse_4326.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": raw}, f, ensure_ascii=False)
    with open(out / "landuse_5179.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::5179"}},
                   "features": clipped}, f, ensure_ascii=False)

    covered = sum(areas.values())
    with open(out / "landuse_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "uname", "area_m2", "share_pct"])
        for (kind, name), a in sorted(areas.items(), key=lambda kv: -kv[1]):
            w.writerow([kind, name, round(a), round(a / total * 100, 1)])
        rest = max(total - covered, 0)
        w.writerow(["(없음)", "용도지역 없음", round(rest), round(rest / total * 100, 1)])

    print(f"\n범위: 경도 {lo0}~{lo1}, 위도 {la0}~{la1} (면적 {total / 1e6:.2f}km²)")
    for (kind, name), a in sorted(areas.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:8s} {name:20s} {a / total * 100:5.1f}%")
    print(f"  용도지역이 덮은 비율 {covered / total * 100:.1f}% (100%보다 크면 레이어끼리 겹친 것)")
    if raw:
        print("속성 예시:", json.dumps(raw[0]["properties"], ensure_ascii=False)[:300])
    print("완료")


if __name__ == "__main__":
    main()
