# -*- coding: utf-8 -*-
"""넓은 범위에서 시연 지역 후보를 찾는다 (docs/demo_area.md).

용도지역을 10m 격자로 그리고, 40m 마다 반경 안 용도 비율로 판정(src/pipeline/verdict.py
규칙)을 낸다. 1km 칸(500m 씩 밀며)마다 확정, 모호, 미확정 비율과 산림 계열과 주거가
함께 걸리는 점의 비율("섞임")을 센다. 반경은 자세 오차 항을 넣은 반경으로 드론 거리
200, 300, 400m 일 때 값(36, 56, 83m)이다 (docs/verdict_compare.md).

    python tools/scan_demo_area.py data/zoning_seoul 126.76,37.42,127.19,37.71 out.json --sido 서울
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from verdict import AMBIGUOUS_SUM, MIN_COVERAGE, threshold_for  # noqa: E402

PX = 10.0          # 격자 10m
STEP = 4           # 40m 마다 한 점
RADII = (36, 56, 83)
FOREST = {"보전녹지지역", "자연녹지지역", "생산녹지지역", "농림지역", "보전관리지역",
          "생산관리지역", "자연환경보전지역"}
RES = {"제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역", "준주거지역",
       "제1종전용주거지역", "제2종전용주거지역"}
LAYERS = ["LT_C_UQ111", "LT_C_UQ112", "LT_C_UQ113", "LT_C_UQ114"]


def load(zdir, sido):
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    items = []
    for l in LAYERS:
        p = Path(zdir) / f"{l}.geojson"
        if not p.exists():
            continue
        for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
            a = f["properties"]
            if a.get("uname") in (None, "도시지역미지정"):
                continue
            if sido and sido not in (a.get("sido_name") or ""):
                continue
            g = shapely.transform(shape(f["geometry"]), lambda c: np.column_stack(
                to5179.transform(c[:, 0], c[:, 1])))
            items.append((int(a.get("dyear") or 0), a["uname"], g))
    return sorted(items, key=lambda t: t[0])      # 오래된 것부터 그려 최신이 덮는다


def rasterize(items, box):
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    x0, y0 = to5179.transform(box[0], box[1])
    x1, y1 = to5179.transform(box[2], box[3])
    X0, Y0, X1, Y1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    W, H = int((X1 - X0) / PX), int((Y1 - Y0) / PX)
    names = sorted({u for _, u, _ in items})
    idx = {u: i + 1 for i, u in enumerate(names)}
    lab = np.zeros((H, W), np.int16)
    xs = X0 + (np.arange(W) + 0.5) * PX
    ys = Y1 - (np.arange(H) + 0.5) * PX
    for _, u, g in items:
        a, b, c, d = g.bounds
        i0, i1 = max(0, int((a - X0) / PX)), min(W, int((c - X0) / PX) + 1)
        j0, j1 = max(0, int((Y1 - d) / PX)), min(H, int((Y1 - b) / PX) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        lab[j0:j1, i0:i1][shapely.contains_xy(g, gx, gy)] = idx[u]
    return lab, names, (X0, Y1)


def disk_frac(onehot, r):
    """반경 r(m) 원 안에서 차지하는 비율. FFT 합성곱."""
    k = int(np.ceil(r / PX))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    ker = ((xx * PX) ** 2 + (yy * PX) ** 2 <= r * r).astype(np.float32)
    sh = (onehot.shape[0] + 2 * k, onehot.shape[1] + 2 * k)
    out = np.fft.irfft2(np.fft.rfft2(onehot, sh) * np.fft.rfft2(ker, sh), sh)
    return np.clip(out[k:k + onehot.shape[0], k:k + onehot.shape[1]] / ker.sum(), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zoning_dir")
    ap.add_argument("box", help="최소경도,최소위도,최대경도,최대위도")
    ap.add_argument("out")
    ap.add_argument("--sido", default="", help="이 시도 폴리곤만 쓴다 (예: 서울)")
    a = ap.parse_args()

    lab, names, (X0, Y1) = rasterize(load(a.zoning_dir, a.sido), [float(v) for v in a.box.split(",")])
    print(f"격자 {lab.shape[1]}x{lab.shape[0]}, 용도지역이 덮은 비율 {(lab > 0).mean():.3f}")
    verdicts, forest, resid, cov56 = {}, None, None, None
    for r in RADII:
        fr = np.stack([disk_frac((lab == i).astype(np.float32), r)[::STEP, ::STEP]
                       for i in range(1, len(names) + 1)])
        cov = fr.sum(0)
        srt = np.sort(fr, 0)[::-1]
        p1, p2 = srt[0], srt[1]
        t = threshold_for(r)
        v = np.full(p1.shape, 2, np.int8)                     # 0 확정, 1 모호, 2 미확정
        ok = cov >= MIN_COVERAGE
        v[ok & (p1 >= t)] = 0
        v[ok & (p1 < t) & (p1 + p2 >= AMBIGUOUS_SUM)] = 1
        verdicts[r] = v
        if r == 56:
            forest = sum(fr[i] for i, u in enumerate(names) if u in FOREST)
            resid = sum(fr[i] for i, u in enumerate(names) if u in RES)
            cov56 = cov
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    C = int(1000 / (PX * STEP))
    h, w = cov56.shape
    cells = []
    for j in range(0, h - C + 1, C // 2):
        for i in range(0, w - C + 1, C // 2):
            sl = (slice(j, j + C), slice(i, i + C))
            if (cov56[sl] >= MIN_COVERAGE).mean() < 0.9:       # 용도지역이 덮지 않는 칸은 뺀다
                continue
            cx, cy = X0 + (i + C / 2) * PX * STEP, Y1 - (j + C / 2) * PX * STEP
            lon, lat = to4326.transform(cx, cy)
            row = {"lat": round(lat, 5), "lon": round(lon, 5), "x": cx, "y": cy,
                   "mixed": round(float(((forest[sl] >= 0.1) & (resid[sl] >= 0.1)).mean()), 3),
                   "forest": round(float(forest[sl].mean()), 3), "resid": round(float(resid[sl].mean()), 3)}
            for r in RADII:
                row[f"v{r}"] = [round(float((verdicts[r][sl] == k).mean()), 3) for k in range(3)]
            cells.append(row)
    Path(a.out).write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")
    print(f"1km 칸 {len(cells)}개 -> {a.out}")

    # 후보: 주거 15% 이상, 섞임이 큰 순, 서로 2.5km 이상
    pick = []
    for c in sorted((c for c in cells if c["resid"] >= 0.15), key=lambda c: -c["mixed"]):
        if all(np.hypot(c["x"] - p["x"], c["y"] - p["y"]) >= 2500 for p in pick):
            pick.append(c)
        if len(pick) == 8:
            break
    print("\n위도, 경도, 섞임, 산림:주거, 확정/모호/미확정 (반경 36, 56, 83m)")
    for c in pick:
        print(c["lat"], c["lon"], c["mixed"], f"{c['forest']}:{c['resid']}", c["v36"], c["v56"], c["v83"])


if __name__ == "__main__":
    main()
