# -*- coding: utf-8 -*-
"""오차 반경을 등각 예측(split conformal)으로 보정한다.

반경 공식(src/geo/transform.py)은 자세 잡음을 가정한 값으로 계산한다. 실제 잡음이 가정과 다르면
정답 위치가 반경 안에 들어오는 비율(포함률)이 떨어진다(잡음 2배에서 84% -> 71%,
docs/verdict_sensitivity.md). 등각 예측은 보정용 사례에서 "실제 오차 / 계산한 반경" 비율을 모아
그 분위수 q 를 반경에 곱한다. 보정 사례와 새 사례가 같은 분포에서 나오면, 분포를 가정하지 않고도
포함률이 1 - alpha 이상이 된다.

    s_i = err_i / r_i            (보정 사례 n 개)
    q   = s 의 ceil((n + 1)(1 - alpha)) 번째로 작은 값
    새 반경 = q x r

거리마다 따로 q 를 구하는 방식(Mondrian)도 같이 본다. 거리마다 포함률을 맞추기 위해서다.

사례는 tools/verdict_coverage.py 의 make_cases 를 그대로 불러 만든다(난수 순서가 같다).
보정 사례는 시드를 바꿔 따로 만들고, 시험 사례는 기존 실험과 같은 시드 42 를 쓴다.
verdict_coverage.py 와 verdict_sensitivity.py 는 고치지 않는다.

    python tools/conformal_radius.py make <landuse_5179.geojson> --seed 7  --out runs/conformal/calib_n1.jsonl
    python tools/conformal_radius.py make <landuse_5179.geojson> --seed 42 --out runs/conformal/test_n1.jsonl
    python tools/conformal_radius.py make <landuse_5179.geojson> --seed 42 --noise-scale 2 --out runs/conformal/test_n2.jsonl
    python tools/conformal_radius.py report <landuse_5179.geojson> --calib runs/conformal/calib_n1.jsonl \\
        --test runs/conformal/test_n1.jsonl runs/conformal/test_n2.jsonl
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verdict_coverage as vc  # noqa: E402
import verdict_sensitivity as vs  # noqa: E402

ALPHA = 0.1
FIXED_R = 50


def make(a):
    for k in vs.NOISE:
        vs.NOISE[k] *= a.noise_scale
    vc.K_LIST = [1.0]          # 판정은 report 에서 다시 한다. 난수를 쓰지 않는 부분만 줄인다
    vc.FIXED = []
    seen = []
    orig = vc.verdict

    def spy(geo, lat, lon, r, shallow, truth, n_points):
        seen.append((lat, lon))
        return orig(geo, lat, lon, r, shallow, truth, n_points)

    vc.verdict = spy
    recs = vc.make_cases(a)
    vc.verdict = orig
    assert len(seen) == len(recs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fo:
        for r, (lat, lon) in zip(recs, seen):
            r.update(lat=lat, lon=lon, noise_scale=a.noise_scale, seed=a.seed)
            r.pop("fixed", None)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)}개 -> {out}")


def quantile(scores, alpha):
    s = sorted(scores)
    n = len(s)
    i = math.ceil((n + 1) * (1 - alpha)) - 1
    return s[min(i, n - 1)]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stats(ans):
    n = len(ans)
    ok = sum(x == "맞는 확정" for x in ans)
    wrong = sum(x == "틀린 단정" for x in ans)
    conf = ok + wrong
    return conf / n, (ok / conf if conf else float("nan")), wrong / n


def cover(recs, radii):
    return sum(r["err_m"] <= R for r, R in zip(recs, radii)) / len(recs)


def report(a):
    geo = vc.GeoData(a.landuse)
    calib = load(a.calib)
    q = quantile([r["err_m"] / r["radius"] for r in calib], a.alpha)
    byD = defaultdict(list)
    for r in calib:
        byD[r["D"]].append(r["err_m"] / r["radius"])
    qD = {D: quantile(v, a.alpha) for D, v in byD.items()}
    print(f"보정 사례 {len(calib)}개 ({a.calib}), alpha={a.alpha}")
    print(f"  전체 q = {q:.3f}")
    print("  거리별 q: " + ", ".join(f"{D}m {qD[D]:.2f} (n={len(byD[D])})" for D in sorted(qD)))

    summary = {"calib": a.calib, "alpha": a.alpha, "q": q, "qD": qD, "tests": []}
    for tp in a.test:
        test = load(tp)
        ways = {
            "계산한 반경 x1 (지금)": [r["radius"] for r in test],
            f"보정 반경 x{q:.2f}": [r["radius"] * q for r in test],
            "거리별 보정 반경": [r["radius"] * qD.get(r["D"], q) for r in test],
            f"고정 {FIXED_R}m": [FIXED_R] * len(test),
        }
        print(f"\n== 시험 사례 {len(test)}개 ({tp}, 잡음 x{test[0]['noise_scale']})")
        print("  방식                     포함률   확정 비율  확정 정확도  틀린 단정   반경 중앙값")
        t = {"test": tp, "noise_scale": test[0]["noise_scale"], "rows": {}}
        for name, radii in ways.items():
            ans = [vc.verdict(geo, r["lat"], r["lon"], R, r["shallow"], r["truth"], a.points)
                   for r, R in zip(test, radii)]
            cv = cover(test, radii)
            s = stats(ans)
            med = sorted(radii)[len(radii) // 2]
            byd = {}
            for D in sorted({r["D"] for r in test}):
                idx = [i for i, r in enumerate(test) if r["D"] == D]
                byd[D] = sum(test[i]["err_m"] <= radii[i] for i in idx) / len(idx)
            t["rows"][name] = {"coverage": cv, "confirm": s[0], "acc": s[1], "wrong": s[2],
                               "median_r": med, "coverage_by_D": byd}
            print(f"  {name:<22}{cv:>8.0%}{s[0]:>10.0%}{s[1]:>12.1%}{s[2]:>10.1%}{med:>12.0f}m")
        print("  거리별 포함률:")
        for name, row in t["rows"].items():
            print(f"    {name:<22}" + "  ".join(f"{D}m {v:.0%}" for D, v in row["coverage_by_D"].items()))
        summary["tests"].append(t)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n요약: {a.json}")


def main():
    ap = argparse.ArgumentParser(description="오차 반경 등각 보정")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("landuse")
    m.add_argument("--dem", nargs="+", default=[str(ROOT / "data/geo_moraegogae/37808.img"),
                                                 str(ROOT / "data/geo_moraegogae/37804.img")])
    m.add_argument("--n", type=int, default=600)
    m.add_argument("--points", type=int, default=600)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--noise-scale", type=float, default=1.0)
    m.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("landuse")
    r.add_argument("--calib", required=True)
    r.add_argument("--test", nargs="+", required=True)
    r.add_argument("--alpha", type=float, default=ALPHA)
    r.add_argument("--points", type=int, default=600)
    r.add_argument("--json")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else report(a)


if __name__ == "__main__":
    main()
