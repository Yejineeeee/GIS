# -*- coding: utf-8 -*-
"""용도지역이 실제 땅과 맞는지 환경부 세분류 토지피복지도(2021)로 확인한다 (docs/demo_area.md).

환경공간정보서비스 지도 서버(WMS)에서 1km 칸을 5m 로 받아 색을 분류 코드로 바꾼다.
인증키는 필요 없다. 경계에서 섞인 색은 빼므로 칸의 약 80% 만 분류된다.

    python tools/landcover_check.py 자곡동,37.4716,127.1017,data/zoning_seoul \
        노암동,37.74349,128.9045,data/zoning_gn

분류 코드: 1xx 시가화, 2xx 농업, 3xx 산림, 4xx 초지, 5xx 습지, 6xx 나지, 7xx 수역
"""
import io
import json
import sys
import urllib.request

import numpy as np
import shapely
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape

WMS = "https://api.mcee.go.kr/geoserver/wms"
LAYER = "EGIS:lv3_2021_g"
N, PX = 200, 5.0
FOREST_Z = ("녹지", "관리", "농림", "자연환경")
T = Transformer.from_crs(4326, 5179, always_xy=True)


def palette():
    url = f"{WMS}?service=WMS&version=1.1.1&request=GetLegendGraphic&layer={LAYER}&format=application/json"
    leg = json.loads(urllib.request.urlopen(url, timeout=60).read().decode("utf-8"))
    ent = [e for e in leg["Legend"][0]["rules"][0]["symbolizers"][0]["Raster"]["colormap"]["entries"]
           if e["quantity"] not in ("0", "65536")]
    pal = np.array([[int(e["color"][i:i + 2], 16) for i in (1, 3, 5)] for e in ent])
    return pal, np.array([int(e["quantity"]) for e in ent])


def landcover(cx, cy, pal, codes):
    url = (f"{WMS}?service=WMS&version=1.1.1&request=GetMap&layers={LAYER}&styles=&srs=EPSG:5179"
           "&bbox=%f,%f,%f,%f&width=%d&height=%d&format=image/png" % (cx - 500, cy - 500, cx + 500, cy + 500, N, N))
    im = np.asarray(Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=150).read())).convert("RGBA")).astype(int)
    d = ((im[..., :3].reshape(-1, 3)[:, None, :] - pal[None]) ** 2).sum(-1)
    code = codes[d.argmin(1)].reshape(N, N)
    code[(im[..., 3] < 128) | (d.min(1).reshape(N, N) > 300)] = 0       # 빈칸, 경계에서 섞인 색
    return code


def zoning(zdir, cx, cy):
    """0 없음, 1 녹지 계열, 2 주거, 3 그 밖."""
    lab = np.zeros((N, N), np.int8)
    gx, gy = np.meshgrid(cx - 500 + (np.arange(N) + 0.5) * PX, cy + 500 - (np.arange(N) + 0.5) * PX)
    view = shapely.box(cx - 500, cy - 500, cx + 500, cy + 500)
    items = []
    for l in ["LT_C_UQ111", "LT_C_UQ112", "LT_C_UQ113"]:
        for f in json.load(open(f"{zdir}/{l}.geojson", encoding="utf-8"))["features"]:
            u = f["properties"].get("uname") or ""
            if u not in ("", "도시지역미지정"):
                items.append((int(f["properties"].get("dyear") or 0), u, f["geometry"]))
    for _, u, g in sorted(items, key=lambda t: t[0]):
        g = shapely.transform(shape(g), lambda c: np.column_stack(T.transform(c[:, 0], c[:, 1])))
        if g.intersects(view):
            lab[shapely.contains_xy(g, gx, gy)] = 1 if any(s in u for s in FOREST_Z) else 2 if "주거" in u else 3
    return lab


def main():
    pal, codes = palette()
    print("이름 | 분류된 비율 | 녹지계열: 면적, 실제 숲, 시가화, 농지 | 주거: 면적, 실제 숲, 시가화, 농지")
    for arg in sys.argv[1:]:
        name, lat, lon, zdir = arg.split(",")
        cx, cy = T.transform(float(lon), float(lat))
        lc, z = landcover(cx, cy, pal, codes), zoning(zdir, cx, cy)
        ok = lc > 0
        groups = {"숲": (lc >= 300) & (lc < 400), "시가화": (lc >= 100) & (lc < 200), "농지": (lc >= 200) & (lc < 300)}
        out = [name, f"{ok.mean():.2f}"]
        for k in (1, 2):
            m = (z == k) & ok
            out.append(f"{(z == k).mean():.2f}, " + ", ".join(
                f"{g[m].mean():.2f}" if m.sum() else "-" for g in groups.values()))
        print(" | ".join(out), flush=True)


if __name__ == "__main__":
    main()
