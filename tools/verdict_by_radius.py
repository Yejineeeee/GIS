# -*- coding: utf-8 -*-
"""오차 반경별 판정 결과를 정리한다 (WBS 6.5).

3.6 정식 판정(src/geo/judge.py)이 오차 반경에 따라 어떻게 나오는지, 그리고 확정했을 때
얼마나 맞는지를 표로 낸다. 선행 작업: 3.5 임계값(docs/threshold.md), 3.6 판정(docs/verdict.md).

    python3 tools/verdict_by_radius.py <landuse_5179.geojson> [--n 500]

1. 평가셋(6.2 확정본, results/eval_facts.jsonl) 100장의 반경별 판정
   - 사실 목록의 용도 비율에 3.6 판정을 다시 적용해, 사실 목록에 적힌 판정과 같은지 확인한다
2. 모의 실험: 반경별 판정 정확도
   - 시연 지역(모래고개입구 범위, 가장자리 200m 제외)에서 정답 지점 n곳을 고르게 뽑는다
     (3.5와 다른 시드를 써서, 임계값을 고를 때 쓰지 않은 지점으로 확인한다)
   - 반경 R마다 정답 지점을 원 안 아무 곳으로 옮긴 위치를 '추정 위치'로 보고,
     추정 위치를 중심으로 반경 R 안 용도 비율을 조회해 3.6으로 판정한다
   - 확정이면 1순위 용도가 정답 지점의 용도와 같은지, 모호면 1·2순위 안에 있는지 본다
   - 반경 10, 30, 50, 100m에 더해 3.5 범위 밖인 150, 200m도 넣어 100m 초과 규칙을 확인한다

출력
    results/verdict_by_radius.csv   반경별 결과표
    docs/verdict_by_radius.md       설명 문서 (결과 숫자가 들어감)

외부 패키지 없이 동작한다.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "geo"))
from query import GeoData, query, to_5179  # noqa: E402
from judge import judge, THRESHOLDS, TARGET, AMBIGUOUS_SUM  # noqa: E402

SEED = 65                                   # 3.5(시드 42)와 다른 지점을 쓴다
BBOX = (128.880, 37.712, 128.932, 37.762)   # 모래고개입구 범위 (docs/landuse.md)
MARGIN_M = 200.0
RADII_EVAL = [10, 30, 50, 100]
RADII_SIM = [10, 30, 50, 100, 150, 200]
VERDICTS = ("확정", "모호", "미확정")


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "-"


# ---------- 1. 평가셋 ----------
def eval_table(facts_path):
    facts = [json.loads(line) for line in open(facts_path, encoding="utf-8") if line.strip()]
    rows, mismatch = [], 0
    for r in RADII_EVAL:
        xs = [x for x in facts if x["error_radius_m"] == r]
        c = {v: 0 for v in VERDICTS}
        for x in xs:
            v, _ = judge(x["land_use"], r)
            c[v] += 1
            mismatch += v != x["verdict"]
        rows.append({"반경_m": r, "장수": len(xs), "확정": c["확정"], "모호": c["모호"], "미확정": c["미확정"]})
    reviewed = all(x.get("review") not in (None, "-", "검토 전") for x in facts)
    return rows, len(facts), mismatch, reviewed


# ---------- 2. 모의 실험 ----------
def random_points(geo, n, rng):
    """범위 안쪽(가장자리 MARGIN_M 제외)에서 용도지역이 있는 지점을 고르게 뽑는다."""
    x0, y0 = to_5179(BBOX[0], BBOX[1])
    x1, y1 = to_5179(BBOX[2], BBOX[3])
    x0, y0, x1, y1 = x0 + MARGIN_M, y0 + MARGIN_M, x1 - MARGIN_M, y1 - MARGIN_M
    pts, tries = [], 0
    while len(pts) < n and tries < n * 50:
        tries += 1
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        name = geo.landuse_at(x, y)
        if name is not None:
            pts.append((x, y, name))
    return pts


def inverse_5179(x, y):
    """EPSG:5179 → 위도·경도 (query()가 위도·경도를 받으므로). 뉴턴법으로 to_5179를 거꾸로 푼다."""
    lon, lat = 127.5 + (x - 1e6) / 88000.0, 38.0 + (y - 2e6) / 111000.0
    for _ in range(8):
        fx, fy = to_5179(lon, lat)
        dx, dy = x - fx, y - fy
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            break
        e = 1e-6
        ax, ay = to_5179(lon + e, lat)
        bx, by = to_5179(lon, lat + e)
        j11, j21 = (ax - fx) / e, (ay - fy) / e
        j12, j22 = (bx - fx) / e, (by - fy) / e
        det = j11 * j22 - j12 * j21
        lon += (dx * j22 - dy * j12) / det
        lat += (j11 * dy - j21 * dx) / det
    return lat, lon


def simulate(geo, n, rng):
    truth = random_points(geo, n, rng)
    rows = []
    for r in RADII_SIM:
        c = {v: 0 for v in VERDICTS}
        conf_ok = amb_ok = point_ok = 0
        for x, y, name in truth:
            ang, dist = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random())   # 원 안에 고르게
            ex, ey = x + dist * math.cos(ang), y + dist * math.sin(ang)
            lat, lon = inverse_5179(ex, ey)
            land_use, _, info = query(geo, lat, lon, r)
            v, _ = judge(land_use, r, info.get("landuse_coverage", 1.0))
            c[v] += 1
            if v == "확정":
                conf_ok += land_use[0]["name"] == name
            elif v == "모호":
                amb_ok += name in (land_use[0]["name"], land_use[1]["name"])
            point_ok += geo.landuse_at(ex, ey) == name                     # 한 점 조회와 비교용
        m = len(truth)
        rows.append({"반경_m": r, "지점": m,
                     "확정": c["확정"], "모호": c["모호"], "미확정": c["미확정"],
                     "확정_정확도": pct(conf_ok, c["확정"]),
                     "모호_정답포함": pct(amb_ok, c["모호"]),
                     "틀린_단정": pct(c["확정"] - conf_ok, m),
                     "한점조회_틀린_단정": pct(m - point_ok, m)})
        print(f"  반경 {r:>3}m: 확정 {c['확정']} 모호 {c['모호']} 미확정 {c['미확정']} "
              f"/ 확정 정확도 {rows[-1]['확정_정확도']} / 틀린 단정 {rows[-1]['틀린_단정']} "
              f"(한 점 조회 {rows[-1]['한점조회_틀린_단정']})")
    return rows, len(truth)


# ---------- 출력 ----------
def write_doc(path, ev_rows, n_eval, mismatch, reviewed, sim_rows, n_sim):
    t = THRESHOLDS[TARGET]
    L = ["# 오차 반경별 판정 결과 (WBS 6.5)\n",
         "3.6 정식 판정이 오차 반경에 따라 어떻게 나오는지, 확정했을 때 얼마나 맞는지 정리했다.\n",
         "담당: 이수진 / 상태: 결과표 작성 완료 (모래고개입구 기준)\n",
         "선행 작업: 3.5 임계값 모의 실험(`docs/threshold.md`), 3.6 판정 구현(`docs/verdict.md`)\n",
         "```\npython3 tools/verdict_by_radius.py <landuse_5179.geojson>\n```\n",
         "## 적용한 판정 규칙 (3.6)\n",
         f"- 최다비율 규칙, 목표 정확도 {TARGET}% 임계값: "
         + " / ".join(f"{r}m {v:.2f}" for r, v in t) + " (사이 반경은 직선으로 잇는다)",
         f"- 모호: 1·2순위 합 {AMBIGUOUS_SUM} 이상",
         "- 미확정: 조회 결과 없음, 용도지역이 원의 60% 미만, 내려다보는 각이 얕음, 반경 100m 초과\n",
         f"## 1. 평가셋 {n_eval}장의 반경별 판정\n",
         "6.2 확정본" + ("" if reviewed else " (주의: 검토 전 사진 포함)")
         + "의 사실 목록에 3.6 판정을 다시 적용했다. 평가셋은 반경 4단계를 25장씩 돌아가며 붙였다.\n",
         "| 반경 | 장수 | 확정 | 모호 | 미확정 |", "|---|---|---|---|---|"]
    for r in ev_rows:
        L.append(f"| {r['반경_m']}m | {r['장수']} | {r['확정']} | {r['모호']} | {r['미확정']} |")
    L += ["",
          f"- 사실 목록에 적힌 판정과 3.6으로 다시 낸 판정이 다른 사진: {mismatch}장"
          + (" (같은 규칙이라 모두 일치)" if mismatch == 0 else ""),
          "- 반경이 커질수록 확정이 줄고 모호·미확정이 늘어난다. 위치가 불확실할수록 용도를 단정하지 않는다\n",
          f"## 2. 모의 실험: 반경별 판정 정확도 (정답 지점 {n_sim}곳)\n",
          "정답 지점을 반경 안 아무 곳으로 옮긴 위치에서 조회·판정하고, 정답 지점의 용도와 비교했다. "
          "3.5와 다른 시드로 뽑은 지점이라, 임계값을 고를 때 쓰지 않은 지점에서 확인한 결과다.\n",
          "| 반경 | 확정 | 모호 | 미확정 | 확정 정확도 | 모호 정답 포함 | 틀린 단정 | 한 점 조회 틀린 단정 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sim_rows:
        m = r["지점"]
        L.append(f"| {r['반경_m']}m | {pct(r['확정'], m)} | {pct(r['모호'], m)} | {pct(r['미확정'], m)} | "
                 f"{r['확정_정확도']} | {r['모호_정답포함']} | {r['틀린_단정']} | {r['한점조회_틀린_단정']} |")
    L += ["",
          "- **확정 정확도:** 확정한 것 중 1순위 용도가 정답과 같은 비율. 목표는 95%",
          "- **모호 정답 포함:** 모호한 것 중 정답이 1·2순위 안에 있는 비율",
          "- **틀린 단정:** 전체 지점 중 확정했는데 틀린 비율",
          "- **한 점 조회 틀린 단정:** 반경을 쓰지 않고 추정 위치 한 점의 용도로 단정했을 때 틀린 비율 (비교용)",
          "- 150m, 200m는 3.5에서 재지 않은 반경이라 3.6 규칙에 따라 모두 미확정이다\n",
          "## 한계\n",
          "- 위치 오차를 반경 안에 고르게 주었다. 실제 오차가 가운데에 몰리는지는 반영하지 않았다",
          "- 모래고개입구(강릉) 범위의 결과다. 시연 지역(서울 자곡동)에서는 용도지역 자료를 새로 받아 다시 재야 한다",
          "- 용도지역은 법적 구역이라 실제 토지 이용과 다를 수 있다",
          "- 평가셋 위치는 가상 배치라, 1번 표는 판정 분포만 보여 주고 정확도는 2번 모의 실험으로 본다"]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="오차 반경별 판정 결과 정리 (WBS 6.5)")
    ap.add_argument("landuse")
    ap.add_argument("--n", type=int, default=500, help="모의 실험 정답 지점 수")
    ap.add_argument("--facts", default=str(ROOT / "results" / "eval_facts.jsonl"))
    a = ap.parse_args()

    print("1. 평가셋 반경별 판정")
    ev_rows, n_eval, mismatch, reviewed = eval_table(a.facts)
    for r in ev_rows:
        print(f"  반경 {r['반경_m']:>3}m: 확정 {r['확정']} 모호 {r['모호']} 미확정 {r['미확정']}")
    print(f"  사실 목록과 다른 판정: {mismatch}장")

    print(f"\n2. 모의 실험 (정답 지점 {a.n}곳, 몇 분 걸림)")
    t0 = time.time()
    geo = GeoData(a.landuse)
    sim_rows, n_sim = simulate(geo, a.n, random.Random(SEED))
    print(f"  {time.time() - t0:.0f}초 걸림")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "verdict_by_radius.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sim_rows[0].keys()))
        w.writeheader()
        w.writerows(sim_rows)
    write_doc(ROOT / "docs" / "verdict_by_radius.md", ev_rows, n_eval, mismatch, reviewed, sim_rows, n_sim)
    print("\n저장: results/verdict_by_radius.csv, docs/verdict_by_radius.md")


if __name__ == "__main__":
    main()
