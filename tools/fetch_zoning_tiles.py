# -*- coding: utf-8 -*-
"""넓은 범위의 용도지역을 V-World 에서 작은 타일로 나눠 받는다 (docs/demo_area.md).

데이터 API 는 한 번에 10km² 를 넘는 상자를 받지 않아, 약 6km² 타일로 나눠 받고
겹쳐 나온 폴리곤은 지운다. 도시지역(UQ111), 관리지역(UQ112), 농림지역(UQ113),
자연환경보전지역(UQ114)을 레이어마다 파일 하나로 저장한다.

    setx VWORLD_KEY "발급받은키"      # 키는 코드나 문서에 적지 않는다
    python tools/fetch_zoning_tiles.py 126.76 37.42 127.19 37.71 data/zoning_seoul
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

LAYERS = ["LT_C_UQ111", "LT_C_UQ112", "LT_C_UQ113", "LT_C_UQ114"]
STEP = 0.025          # 경위도 0.025도 = 약 2.2km x 2.8km


def get(key, layer, b, page):
    p = {"service": "data", "request": "GetFeature", "data": layer, "key": key, "domain": "localhost",
         "format": "json", "size": "1000", "page": str(page), "geometry": "true", "attribute": "true",
         "crs": "EPSG:4326", "geomFilter": "BOX(%f,%f,%f,%f)" % b}
    req = urllib.request.Request("https://api.vworld.kr/req/data?" + urllib.parse.urlencode(p),
                                 headers={"Referer": "http://localhost"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read().decode("utf-8"))["response"]


def main():
    key = (os.environ.get("VWORLD_KEY") or "").strip().strip('"')
    if not key:
        sys.exit("VWORLD_KEY 가 없습니다.")
    minx, miny, maxx, maxy = map(float, sys.argv[1:5])
    outdir = sys.argv[5]
    os.makedirs(outdir, exist_ok=True)
    tiles, x = [], minx
    while x < maxx - 1e-9:
        y = miny
        while y < maxy - 1e-9:
            tiles.append((x, y, min(x + STEP, maxx), min(y + STEP, maxy)))
            y += STEP
        x += STEP
    for layer in LAYERS:
        seen, feats, err = set(), [], 0
        for b in tiles:
            page = 1
            while True:
                r = get(key, layer, b, page)
                if r.get("status") == "NOT_FOUND":
                    break
                if r.get("status") != "OK":
                    err += 1
                    print(layer, b, r.get("status"), (r.get("error") or {}).get("text"))
                    break
                got = r["result"]["featureCollection"]["features"]
                for f in got:            # 타일 경계에 걸친 폴리곤은 두 번 나온다
                    k = json.dumps(f["geometry"]["coordinates"][:1])[:200] + str(f["properties"])
                    if k not in seen:
                        seen.add(k)
                        feats.append(f)
                if len(got) < 1000:
                    break
                page += 1
            time.sleep(0.15)
        with open(f"{outdir}/{layer}.geojson", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
        print(layer, len(feats), "오류", err, flush=True)


if __name__ == "__main__":
    main()
