# -*- coding: utf-8 -*-
"""VLM 에게 보여 준 잘린 사진이 땅에서 어디인지 GIS 로 재고, VLM 점수와 비교한다.

tools/vlm_landcover.py 결과(landcover*.jsonl)를 읽는다. 잘린 영역의 픽셀 12x12 개를 시뮬레이터
지형까지 광선으로 따라가 용도지역과 건물을 센다. 모호 판정에서는 잘린 영역을 완벽하게
읽었을 때(녹지 비율과 시가지 비율 중 큰 쪽)와 밑동 한 점의 용도를 상한으로 같이 센다.

    python tools/vlm_landcover_gis.py 0.08 landcover_0.08.jsonl
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/geo", "src/pipeline"):
    sys.path.insert(0, str(ROOT / p))
from query import GeoData, to_5179, _inside  # noqa: E402
from transform import target_pixel, ray, move  # noqa: E402
from sim_terrain import SimTerrain  # noqa: E402

geo = GeoData(str(ROOT / "data/geo_jagok/landuse_5179.geojson"),
              str(ROOT / "data/geo_jagok/buildings_5179.geojson"))
ter = SimTerrain(str(ROOT / "sim/terrain_jagok.json"))
CELL = 50.0
bucket = defaultdict(list)
for rings, (x0, y0, x1, y1) in geo.buildings:
    for i in range(int(x0 // CELL), int(x1 // CELL) + 1):
        for j in range(int(y0 // CELL), int(y1 // CELL) + 1):
            bucket[(i, j)].append((rings, (x0, y0, x1, y1)))


def in_bldg(x, y):
    for rings, (x0, y0, x1, y1) in bucket.get((int(x // CELL), int(y // CELL)), ()):
        if x0 <= x <= x1 and y0 <= y <= y1 and _inside(x, y, rings):
            return True
    return False


def kind(name):
    if not name:
        return "기타"
    if "녹지" in name or "보전" in name or "농림" in name:
        return "녹지"
    if "주거" in name or "상업" in name or "공업" in name:
        return "시가지"
    return "기타"


def hit(cam, sweep, px, py, W, H):
    """픽셀 광선을 지형과 만날 때까지 5m 씩 따라간다. (x, y) 5179."""
    e, n, u = ray(px, py, W, H, cam["heading_deg"], cam["pitch_deg"], cam["hfov_deg"])
    L = math.sqrt(e * e + n * n + u * u)
    e, n, u = e / L, n / L, u / L
    z0 = sweep["ground_m"] + sweep["h"]
    t = 5.0
    while t < 4000:
        lat, lon = move(cam["lat"], cam["lon"], e * t, n * t)
        x, y = to_5179(lon, lat)
        g = ter.height(x, y)
        if g is None:
            g = sweep["fire_ground_m"]
        if z0 + u * t <= g:
            return x, y
        t += 5.0
    return None


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


SIDE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
JS = sys.argv[2] if len(sys.argv) > 2 else "landcover.jsonl"
rows = []
for name in ("pipeline_boundary", "pipeline_boundary2"):
    src = ROOT / "results" / name
    dets = {}
    for f in (src / "detections").glob("*.json"):
        fr = json.load(open(f, encoding="utf-8"))
        for d in fr.get("detections") or []:
            dets[d["id"]] = (fr, d)
    al_by = {}
    for f in (src / "alerts").glob("*.json"):
        al = json.load(open(f, encoding="utf-8"))
        al_by[al["alert_id"]] = al
    for line in open(src / JS, encoding="utf-8"):
        r = json.loads(line)
        fr, d = dets[al_by[r["alert"]]["detection_id"]]
        meta = json.load(open(ROOT / fr["image"].replace("\\", "/").replace(".png", ".json"), encoding="utf-8"))
        cam, sw = meta["camera"], meta["sweep"]
        W, H = fr["image_size"]["width"], fr["image_size"]["height"]
        px, py = target_pixel(d)
        side = W * SIDE
        N = 12
        cnt = defaultdict(int)
        bl = tot = 0
        far = []
        for i in range(N):
            for j in range(N):
                qx = px - side / 2 + side * (i + 0.5) / N
                qy = py - side / 2 + side * (j + 0.5) / N
                if not (0 <= qx < W and 0 <= qy < H):
                    continue
                h = hit(cam, sw, qx, qy, W, H)
                if h is None:
                    continue
                tot += 1
                cnt[kind(geo.landuse_at(*h))] += 1
                bl += in_bldg(*h)
                far.append(h)
        # 가운데 3x3 (밑동 바로 둘레)
        c = hit(cam, sw, px, py, W, H)
        ckind = kind(geo.landuse_at(*c)) if c else "기타"
        xs = [p[0] for p in far]
        ys = [p[1] for p in far]
        span = max(max(xs) - min(xs), max(ys) - min(ys)) if far else 0
        r.update(set=name, green=cnt["녹지"] / tot if tot else 0, urban=cnt["시가지"] / tot if tot else 0,
                 bldg=bl / tot if tot else 0, center_kind=ckind, span_m=span)
        rows.append(r)

json.dump(rows, open(ROOT / "results" / f"landcover_gis_{SIDE}.json", "w", encoding="utf-8"), ensure_ascii=False)

for setname in ("pipeline_boundary", "pipeline_boundary2", None):
    R = [r for r in rows if setname is None or r["set"] == setname]
    print(f"\n== {setname or '합계'} ({len(R)}개) ==")
    spans = sorted(r["span_m"] for r in R)
    print(f"잘린 영역 땅 위 크기(가로세로 중 긴 쪽) 중앙값 {spans[len(spans)//2]:.0f}m")
    mixed = sum(0.2 <= r["green"] <= 0.8 for r in R)
    print(f"숲·시가지 섞인 영역(녹지 20~80%) {mixed}/{len(R)}")
    print(f"상관(순위) VLM점수 vs 녹지비율-시가지비율 {spearman([r['vlm'] for r in R], [r['green'] - r['urban'] for r in R]):.2f}")
    print(f"상관(순위) VLM점수 vs 건물 비율         {spearman([r['vlm'] for r in R], [r['bldg'] for r in R]):.2f}")
    print(f"상관(순위) P(숲) vs 녹지 비율          {spearman([r['p_forest'] for r in R], [r['green'] for r in R]):.2f}")
    print(f"상관(순위) P(건물) vs 건물 비율        {spearman([r['p_built'] for r in R], [r['bldg'] for r in R]):.2f}")
    print(f"상관(순위) 녹지비율 vs 불위치 정답(녹지=1) {spearman([r['green'] for r in R], [1.0 if r['truth']=='녹지' else 0.0 for r in R]):.2f}")
    # 기준별 AUC
    for lab, key in (("불 위치 정답", "truth"), ("밑동 픽셀 땅", "center_kind")):
        g = [r["vlm"] for r in R if r[key] == "녹지"]
        b = [r["vlm"] for r in R if r[key] == "시가지"]
        print(f"AUC VLM vs {lab}: {auc(g, b):.2f}  (녹지 {len(g)}, 시가지 {len(b)})")
    g = [r["vlm"] for r in R if r["green"] >= 0.8]
    b = [r["vlm"] for r in R if r["urban"] >= 0.8]
    print(f"AUC VLM, 잘린 영역이 한쪽으로 뚜렷한 것만 (녹지>=80% {len(g)} vs 시가지>=80% {len(b)}): {auc(g, b):.2f}")
    g = [r["vlm"] for r in R if r["bldg"] <= 0.02]
    b = [r["vlm"] for r in R if r["bldg"] >= 0.15]
    print(f"AUC VLM, 건물 거의 없음(<=2% {len(g)}) vs 건물 많음(>=15% {len(b)}): {auc(g, b):.2f}")
    agree = sum(r["center_kind"] == r["truth"] for r in R)
    print(f"밑동 픽셀 땅 종류 = 불 위치 정답: {agree}/{len(R)}")

    # 모호 판정 풀기: 녹지/시가지가 1, 2순위인 모호 알람
    M = [r for r in R if r["verdict"] == "모호" and {r["k1"], r["k2"]} == {"녹지", "시가지"}]
    top = sum(r["k1"] == r["truth"] for r in M)
    print(f"모호(녹지/시가지) {len(M)}개, GIS 1순위 맞음 {top}")
    for tau in (0.2, 0.4):
        P = [(r, "녹지" if r["vlm"] >= tau else "시가지") for r in M if abs(r["vlm"]) >= tau]
        ok = sum(p == r["truth"] for r, p in P)
        ag = [(r, p) for r, p in P if p == r["k1"]]
        dis = [(r, p) for r, p in P if p != r["k1"]]
        print(f"  문턱 {tau}: VLM 이 고름 {len(P)}, 맞음 {ok} | GIS와 같음 {len(ag)} (맞음 {sum(p == r['truth'] for r, p in ag)})"
              f" | 반대 {len(dis)} (VLM 맞음 {sum(p == r['truth'] for r, p in dis)})")
    o = sum(("녹지" if r["green"] >= r["urban"] else "시가지") == r["truth"] for r in M)
    c = sum(r["center_kind"] == r["truth"] for r in M)
    print(f"  상한: 잘린 영역을 완벽하게 읽음 {o}, 밑동 한 점의 용도 {c}")
