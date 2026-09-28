# -*- coding: utf-8 -*-
"""용도지역 폴리곤을 V-World 데이터 API로 내려받아 GeoJSON으로 저장한다.

인증키는 환경변수 VWORLD_KEY 에서 읽는다. 데이터 API는 등록한 도메인을
Referer로 보내야 하므로 REFERER 값을 인증키에 등록한 주소와 맞춘다.

    python fetch_zoning.py 최소경도 최소위도 최대경도 최대위도 [출력파일]
"""
import os, sys, json, time, urllib.parse, urllib.request

KEY = (os.environ.get("VWORLD_KEY") or "").strip().strip('"').strip("'")
REFERER = "http://localhost"
LAYER = "LT_C_UQ111"          # 용도지역
PAGE_SIZE = 1000              # 데이터 API 최대치

if not KEY:
    raise SystemExit("환경변수 VWORLD_KEY 가 없습니다.")

args = sys.argv[1:]
if len(args) >= 4:
    minx, miny, maxx, maxy = map(float, args[:4])
    out = args[4] if len(args) > 4 else "zoning.geojson"
else:
    minx, miny, maxx, maxy = 127.00, 37.48, 127.06, 37.52   # 예시 영역
    out = "zoning.geojson"


def fetch(page):
    params = {
        "service": "data", "request": "GetFeature", "data": LAYER,
        "key": KEY, "format": "json", "size": str(PAGE_SIZE), "page": str(page),
        "geometry": "true", "attribute": "true", "crs": "EPSG:4326",
        "geomFilter": f"BOX({minx},{miny},{maxx},{maxy})",
    }
    url = "https://api.vworld.kr/req/data?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Referer": REFERER})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


feats, page = [], 1
while True:
    res = fetch(page)
    resp = res.get("response", {})
    if resp.get("status") != "OK":
        print("응답 상태:", resp.get("status"), resp.get("error", {}))
        break
    fc = resp.get("result", {}).get("featureCollection", {})
    got = fc.get("features", [])
    feats.extend(got)
    total = int(resp.get("record", {}).get("total", 0) or 0)
    print(f"  page {page}: {len(got)}건 (누적 {len(feats)} / 전체 {total})")
    if page == 1 and got:
        print("  속성 필드:", list(got[0].get("properties", {}).keys()))
        print("  첫 건 속성:", got[0].get("properties"))
    if not got or len(feats) >= total or len(got) < PAGE_SIZE:
        break
    page += 1
    time.sleep(0.3)

with open(out, "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
print(f"저장 완료: {out} ({len(feats)}건)")
