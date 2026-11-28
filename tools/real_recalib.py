# -*- coding: utf-8 -*-
"""실측 오차로 반경을 다시 맞추면 어떻게 되나 (가정 실험).

11/24 실측 29건에서 반경은 실제 오차의 5.1배였다(docs/radius_tightness.md). 실측 오차로 등각 보정
배율 q_real 을 구해 반경에 곱하면 반경이 줄고 확정이 늘 것이다. 다만 실측은 사람이 표적 픽셀을 찍어
연기 밑동을 찾는 탐지 오차가 없다. 그래서 시뮬레이션 사례를 두 가지로 만들어 같은 q_real 을 적용한다.

    밑동 정확   탐지 픽셀 = 정답 픽셀 (실측처럼 표적 위치를 아는 경우)
    연기 탐지   시뮬레이터 프레임에서 잰 V2 탐지 밑동 오차를 그대로 (지금 파이프라인)

두 경우 모두 사례 만드는 법, 시드(42), 자세 잡음(1배)은 tools/conformal_radius.py make 와 같다.
연기 탐지 쪽은 runs/conformal/test_n1.jsonl 과 같은 사례가 나온다.

    python tools/real_recalib.py <landuse_5179.geojson> --real results/drone_error_1124_all.csv --out runs/recalib
"""
import argparse
import copy
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import conformal_radius as cr  # noqa: E402
import verdict_coverage as vc  # noqa: E402

ALPHA = 0.1


def q_real(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["height_mode"] != "보정 없음"]
    s = [float(r["err_m"]) / float(r["radius_m"]) for r in rows]
    return cr.quantile(s, ALPHA), len(s)


def make(a, exact, out):
    if exact:
        orig = vc.frame_offsets

        def exact_frames():
            fr = copy.deepcopy(orig())
            for f in fr:
                f["V2"] = (0.0, 0.0, 0.0)
            return fr
        vc.frame_offsets = exact_frames
    try:
        cr.make(SimpleNamespace(landuse=a.landuse, dem=a.dem, n=a.n, points=a.points, seed=42,
                                noise_scale=1.0, out=str(out)))
    finally:
        if exact:
            vc.frame_offsets = orig
    return [json.loads(l) for l in open(out, encoding="utf-8")]


def score(geo, recs, k, points):
    ans = [vc.verdict(geo, r["lat"], r["lon"], r["radius"] * k, r["shallow"], r["truth"], points) for r in recs]
    n = len(recs)
    ok = sum(x == "맞는 확정" for x in ans)
    wrong = sum(x == "틀린 단정" for x in ans)
    return {"반경 배율": round(k, 3), "반경 중앙값 m": round(st.median(r["radius"] * k for r in recs), 1),
            "포함률": round(sum(r["err_m"] <= r["radius"] * k for r in recs) / n, 3),
            "확정 비율": round((ok + wrong) / n, 3),
            "확정 정확도": round(ok / (ok + wrong), 3) if ok + wrong else None,
            "틀린 단정": round(wrong / n, 3)}


def main():
    ap = argparse.ArgumentParser(description="실측 오차로 반경을 다시 맞추는 가정 실험")
    ap.add_argument("landuse")
    ap.add_argument("--real", default=str(ROOT / "results/drone_error_1124_all.csv"))
    ap.add_argument("--dem", nargs="+", default=[str(ROOT / "data/geo_moraegogae/37808.img"),
                                                 str(ROOT / "data/geo_moraegogae/37804.img")])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--out", default="runs/recalib")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q, n_real = q_real(a.real)
    print(f"실측 {n_real}건, 오차/반경 90% 분위 q_real = {q:.3f}", flush=True)
    geo = vc.GeoData(a.landuse)
    report = {"q_real": q, "실측 건수": n_real}
    for name, exact in (("밑동 정확", True), ("연기 탐지", False)):
        recs = make(a, exact, out / f"cases_{'exact' if exact else 'v2'}.jsonl")
        ratio = st.median(r["radius"] / max(r["err_m"], 0.1) for r in recs)
        res = {"반경/오차 중앙값": round(ratio, 2), "지금 반경": score(geo, recs, 1.0, a.points),
               "실측으로 다시 맞춘 반경": score(geo, recs, q, a.points)}
        report[name] = res
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
