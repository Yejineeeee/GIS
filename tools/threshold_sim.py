# -*- coding: utf-8 -*-
"""판정 임계값을 정하기 위한 모의 실험. 폴리곤 파일만 있으면 돌아간다.

정답 위치를 뽑아 실제 용도를 기록하고, 오차 반경만큼 위치를 흔든 뒤 주변
용도 비율로 판정한다. 임계값을 훑으며 확정 비율과 확정 정확도를 기록한다.

    python threshold_sim.py [폴리곤파일] [정답지점수] [반경안조회지점수]
"""
import json, sys, random, csv, math
from collections import defaultdict

try:
    from shapely.geometry import shape, Point
    from shapely.ops import transform as shp_transform, unary_union
    from shapely import STRtree
    from pyproj import Transformer
except ImportError:
    raise SystemExit("먼저 설치하세요:  pip install shapely pyproj")

SRC = sys.argv[1] if len(sys.argv) > 1 else "zoning.geojson"
N_POINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000     # 정답 지점 수
K_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 30      # 지점 샘플 방식의 조회 지점 수
RADII = [10, 30, 50, 100]                                      # 오차 반경(m)
LABEL_FIELD = "uname"
SEED = 42

random.seed(SEED)
to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform

print(f"[1] 폴리곤 읽기: {SRC}")
gj = json.load(open(SRC, encoding="utf-8"))
items = []
for f in gj["features"]:
    try:
        g = shp_transform(to_m, shape(f["geometry"]))
    except Exception:
        continue
    if g.is_empty:
        continue
    if not g.is_valid:
        g = g.buffer(0)
    props = f["properties"]
    lab = (props.get(LABEL_FIELD) or "").strip() or "미지정"
    try:
        yr = int(props.get("dyear") or 0)
    except ValueError:
        yr = 0
    items.append((yr, g, lab))
# 같은 땅에 여러 이력이 겹쳐 있어 최신 지정 연도를 우선한다
items.sort(key=lambda x: -x[0])
polys = [g for _, g, _ in items]
labels = [l for _, _, l in items]
years = [y for y, _, _ in items]
print(f"    폴리곤 {len(polys)}개, 용도 종류 {len(set(labels))}개")
for lab, cnt in sorted(((l, labels.count(l)) for l in set(labels)), key=lambda x: -x[1])[:8]:
    print(f"      {lab}: {cnt}")

tree = STRtree(polys)
area_union = unary_union(polys)
minx, miny, maxx, maxy = area_union.bounds


def label_at(pt):
    cands = [i for i in tree.query(pt) if polys[i].contains(pt)]
    if not cands:
        return None
    return labels[min(cands)]   # polys는 최신 연도 순으로 정렬되어 있음


print(f"[2] 정답 지점 {N_POINTS}곳 추출")
truth = []
guard = 0
while len(truth) < N_POINTS and guard < N_POINTS * 200:
    guard += 1
    p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
    lab = label_at(p)
    if lab:
        truth.append((p, lab))
print(f"    확보: {len(truth)}곳")


def ratios_by_samples(center, r, k=K_SAMPLES):
    cnt = defaultdict(int)
    hit = 0
    for _ in range(k):
        a = random.uniform(0, 2 * math.pi)
        d = r * math.sqrt(random.random())
        p = Point(center.x + d * math.cos(a), center.y + d * math.sin(a))
        lab = label_at(p)
        if lab:
            cnt[lab] += 1
            hit += 1
    if hit == 0:
        return {}, k
    return {k2: v / hit for k2, v in cnt.items()}, k - hit


def ratios_by_area(center, r):
    circle = center.buffer(r, quad_segs=32)
    tot = circle.area
    out = defaultdict(float)
    taken = None
    for i in sorted(tree.query(circle)):      # 최신 연도 우선
        inter = polys[i].intersection(circle)
        if inter.is_empty:
            continue
        if taken is not None:
            inter = inter.difference(taken)
            if inter.is_empty:
                continue
        out[labels[i]] += inter.area
        taken = inter if taken is None else taken.union(inter)
    covered = sum(out.values())
    if covered <= 0:
        return {}, 1.0
    miss = max(0.0, 1 - covered / tot)
    return {k2: v / covered for k2, v in out.items()}, miss


def judge_top(ratios, t_conf, t_amb, miss_limit=0.4, miss=0.0):
    """규칙 1: 최다 비율 기준"""
    if not ratios or miss > miss_limit:
        return "미확정", None, None
    s = sorted(ratios.items(), key=lambda x: -x[1])
    p1 = s[0][1]
    second = s[1][0] if len(s) > 1 else None
    if p1 >= t_conf:
        return "확정", s[0][0], second
    if p1 >= t_amb:
        return "모호", s[0][0], second
    return "미확정", s[0][0], second


def judge_gap(ratios, g_conf, g_amb, miss_limit=0.4, miss=0.0):
    """규칙 2: 1순위와 2순위 격차 기준"""
    if not ratios or miss > miss_limit:
        return "미확정", None, None
    s = sorted(ratios.items(), key=lambda x: -x[1])
    p1 = s[0][1]
    p2 = s[1][1] if len(s) > 1 else 0.0
    second = s[1][0] if len(s) > 1 else None
    gap = p1 - p2
    if gap >= g_conf:
        return "확정", s[0][0], second
    if gap >= g_amb:
        return "모호", s[0][0], second
    return "미확정", s[0][0], second


print(f"[3] 반경별 조회 결과 계산 (지점 샘플 {K_SAMPLES}곳 / 면적 비율)")
cases = []   # (radius, method, truth_label, ratios, miss)
for r in RADII:
    for pt, lab in truth:
        a = random.uniform(0, 2 * math.pi)
        d = r * math.sqrt(random.random())
        est = Point(pt.x + d * math.cos(a), pt.y + d * math.sin(a))
        rs, miss_s = ratios_by_samples(est, r)
        cases.append((r, "지점샘플", lab, rs, miss_s / K_SAMPLES))
        ra, miss_a = ratios_by_area(est, r)
        cases.append((r, "면적비율", lab, ra, miss_a))
    print(f"    반경 {r}m 완료")

print("[4] 임계값 스윕")
rows = []
grid = [round(x, 2) for x in [0.40 + 0.05 * i for i in range(12)]]   # 0.40~0.95
for rule_name, fn, thr_list in [
    ("최다비율", judge_top, grid),
    ("1-2순위격차", judge_gap, [round(x, 2) for x in [0.05 + 0.05 * i for i in range(12)]]),
]:
    for r in RADII:
        for method in ["지점샘플", "면적비율"]:
            subset = [c for c in cases if c[0] == r and c[1] == method]
            for t in thr_list:
                t_amb = round(t * 0.6, 3)
                n_conf = n_conf_ok = n_amb = n_amb_ok = 0
                for _, _, lab, ratios, miss in subset:
                    verdict, top1, top2 = fn(ratios, t, t_amb, miss=miss)
                    if verdict == "확정":
                        n_conf += 1
                        n_conf_ok += (top1 == lab)
                    elif verdict == "모호":
                        n_amb += 1
                        n_amb_ok += (lab in (top1, top2))
                rows.append({
                    "규칙": rule_name, "반경m": r, "방식": method, "임계값": t,
                    "확정비율": round(n_conf / len(subset), 3),
                    "확정정확도": round(n_conf_ok / n_conf, 3) if n_conf else "",
                    "모호비율": round(n_amb / len(subset), 3),
                    "모호_2순위내포함": round(n_amb_ok / n_amb, 3) if n_amb else "",
                })

out_csv = "threshold_sim_result.csv"
with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"    저장: {out_csv} ({len(rows)}행)")

print("\n[5] 목표 정확도를 만족하는 가장 낮은 임계값")
print(f"{'규칙':<12}{'반경':>6}{'방식':>10}{'목표':>7}{'임계값':>8}{'확정비율':>10}{'확정정확도':>10}")
for rule_name in ["최다비율", "1-2순위격차"]:
    for r in RADII:
        for method in ["지점샘플", "면적비율"]:
            for target in [0.90, 0.95, 0.99]:
                ok = [x for x in rows
                      if x["규칙"] == rule_name and x["반경m"] == r and x["방식"] == method
                      and x["확정정확도"] != "" and x["확정정확도"] >= target and x["확정비율"] > 0]
                if ok:
                    best = min(ok, key=lambda x: x["임계값"])
                    print(f"{rule_name:<12}{r:>5}m{method:>10}{int(target*100):>6}%{best['임계값']:>8}"
                          f"{best['확정비율']:>10}{best['확정정확도']:>10}")
                else:
                    print(f"{rule_name:<12}{r:>5}m{method:>10}{int(target*100):>6}%{'없음':>8}")
