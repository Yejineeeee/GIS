# -*- coding: utf-8 -*-
"""지리 조회(3.4 query.py) 속도를 재고, 고치기 전과 결과가 같은지 확인한다.

시연 위치 36곳(docs/demo_positions.csv) x 반경 4단계(10, 30, 50, 100m) = 144번 조회한다.

    # 고치기 전: 결과를 저장
    python3 tools/bench_query.py <landuse_5179.geojson> <buildings_5179.geojson> <DEM.img ...> --save before.json
    # 고친 뒤: 저장한 결과와 비교
    python3 tools/bench_query.py <landuse_5179.geojson> <buildings_5179.geojson> <DEM.img ...> --compare before.json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "geo"))
from query import GeoData, query  # noqa: E402

RADII = [10, 30, 50, 100]


def main():
    ap = argparse.ArgumentParser(description="지리 조회 속도 재기")
    ap.add_argument("landuse")
    ap.add_argument("buildings")
    ap.add_argument("dem", nargs="*")
    ap.add_argument("--positions", default=str(ROOT / "docs" / "demo_positions.csv"))
    ap.add_argument("--save", help="결과를 저장할 JSON 파일")
    ap.add_argument("--compare", help="전에 저장한 JSON 파일과 결과 비교")
    a = ap.parse_args()

    t = time.perf_counter()
    geo = GeoData(a.landuse, a.buildings, a.dem or None)
    t_load = time.perf_counter() - t

    with open(a.positions, encoding="utf-8-sig") as f:
        pos = list(csv.DictReader(f))
    results, times = {}, []
    for p in pos:
        for r in RADII:
            t = time.perf_counter()
            land_use, ev, info = query(geo, float(p["lat"]), float(p["lon"]), r)
            times.append(time.perf_counter() - t)
            results[f"{p['pos_id']}_{r}"] = {"land_use": land_use, "evidence": ev, "info": info}

    times.sort()
    n = len(times)
    print(f"자료 읽기 {t_load:.1f}초 (처음 한 번)")
    print(f"조회 {n}번: 평균 {sum(times) / n:.3f}초, 중앙값 {times[n // 2]:.3f}초, 가장 느림 {times[-1]:.3f}초")
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump({"mean_s": sum(times) / n, "results": results}, f, ensure_ascii=False)
        print(f"저장: {a.save}")
    if a.compare:
        old = json.load(open(a.compare, encoding="utf-8"))
        diff = [k for k in results if results[k] != old["results"].get(k)]
        print(f"전과 같은 결과: {n - len(diff)}/{n}")
        for k in diff[:5]:
            print(f"  다름 {k}: 전 {old['results'].get(k)}\n         후 {results[k]}")
        print(f"평균 {old['mean_s']:.3f}초 → {sum(times) / n:.3f}초 ({old['mean_s'] / (sum(times) / n):.1f}배 빨라짐)")


if __name__ == "__main__":
    main()
