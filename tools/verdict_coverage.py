# -*- coding: utf-8 -*-
"""같은 확정 비율에서 반경 판정과 한 점 조회를 견준다 (핵심 주장 보강).

반경 판정(D+)은 틀린 단정을 18.3% 에서 3.7% 로 줄이지만, 확정도 100% 에서 48% 로 준다.
확정을 줄이기만 해도 틀린 단정은 준다. 그래서 같은 비율만 확정할 때 어느 쪽이 더 정확한지 본다.

사례(발화점 600곳, 시뮬레이터 탐지 오차, 자세 잡음, 시드 42)를 만드는 법은
tools/verdict_sensitivity.py(= verdict_compare.py)와 한 줄도 다르지 않게 따라 한다. 검증기를
거친 탐지(V2)만 본다. 사례마다 아래를 기록한다.

    한 점 조회      추정 좌표 한 점의 용도
    반경 판정       D+ 반경에 k 배(0.5 ~ 2)를 곱한 반경으로 조회하고 verdict.judge 로 판정
    고정 반경 판정  반경을 10 ~ 150m 로 고정하고 같은 판정 규칙

비교하는 기준선 (한 점 조회 답은 그대로 두고, 무엇을 확정할지만 고른다)
    무작위로 고름      어느 비율에서나 정확도가 한 점 조회 전체와 같다
    가까운 것부터      드론에서 발화점까지 거리가 짧은 사례부터 확정
    반경이 작은 것부터  D+ 반경이 작은 사례부터 확정 (반경 판정과 같은 정보로 기권만 하는 것)

    python tools/verdict_coverage.py <landuse_5179.geojson> --out runs/verdict_coverage
    python tools/verdict_coverage.py --report runs/verdict_coverage/cases.jsonl
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verdict_sensitivity as vs  # noqa: E402
from verdict_sensitivity import (to_position, move, GeoData, query, to_5179, Dem, SimTerrain,  # noqa: E402
                                 sim_paths, judge, frame_offsets, dep_of, meters, BBOX, MARGIN_M,
                                 ALT, FIRE_BASE_M, NOISE)

K_LIST = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
FIXED = [10, 25, 50, 75, 100, 150]


def verdict(geo, lat, lon, r, shallow, truth, n_points):
    lu, _, qi = query(geo, lat, lon, r, n_points=n_points)
    kind, _ = judge(lu, r, qi["landuse_coverage"], shallow)
    if kind == "확정":
        return "맞는 확정" if lu[0]["name"] == truth else "틀린 단정"
    if kind == "모호":
        return "맞는 모호" if truth in [u["name"] for u in lu[:2]] else "틀린 모호"
    return "미확정"


def make_cases(a):
    """verdict_sensitivity.main 의 사례 만들기를 그대로 따른다. 난수를 부르는 순서가 같아야 한다."""
    rng = random.Random(a.seed)
    geo = GeoData(a.landuse)
    dem = Dem(a.dem)
    frames = frame_offsets()
    box = BBOX
    lat0, lat1, lon0, lon1 = box[1], box[3], box[0], box[2]
    mlat = MARGIN_M / 111320
    mlon = MARGIN_M / (111320 * math.cos(math.radians((lat0 + lat1) / 2)))
    recs, tries = [], 0
    while len(recs) < a.n and tries < a.n * 20:
        tries += 1
        flat = rng.uniform(lat0 + mlat, lat1 - mlat)
        flon = rng.uniform(lon0 + mlon, lon1 - mlon)
        fx, fy = to_5179(flon, flat)
        truth = geo.landuse_at(fx, fy)
        fg = dem.height(fx, fy)
        if truth is None or fg is None or fg < 1.0:
            continue
        fr = rng.choice(frames)
        D = fr["D"]
        bearing = rng.uniform(0, 360)
        dlat, dlon = move(flat, flon, D * math.sin(math.radians(bearing)), D * math.cos(math.radians(bearing)))
        dg = dem.height(*to_5179(dlon, dlat))
        if dg is None:
            continue
        heading = (bearing + 180) % 360
        dep = dep_of(ALT + dg - (fg + FIRE_BASE_M), D)
        pitch = fr["aim"] - dep
        W, H = fr["W"], fr["H"]
        f = (W / 2) / math.tan(math.radians(fr["hfov"]) / 2)
        tx, ty = W / 2, H / 2 + f * math.tan(math.radians(dep + pitch))
        cam = {"lat": dlat, "lon": dlon, "alt_m": ALT, "alt_ref": "지면", "heading_deg": heading,
               "pitch_deg": pitch, "hfov_deg": fr["hfov"], "source": "가상"}
        noisy = dict(cam)
        nlat, nlon = move(dlat, dlon, rng.gauss(0, NOISE["h"]), rng.gauss(0, NOISE["h"]))
        noisy.update(lat=nlat, lon=nlon, alt_m=ALT + rng.gauss(0, NOISE["alt"]),
                     heading_deg=(heading + rng.gauss(0, NOISE["heading"])) % 360,
                     pitch_deg=pitch + rng.gauss(0, NOISE["pitch"]))
        frame = {"image_size": {"width": W, "height": H}, "camera": noisy}
        res, bad = None, False
        for src in ("정답픽셀", "V0", "V2"):           # 원래 도구처럼 셋 다 계산해 같은 사례가 빠지게 한다
            dx, dy, _ = (0.0, 0.0, 0.0) if src == "정답픽셀" else fr[src]
            det = {"polygon": None, "bbox": [tx + dx - 0.5, ty + dy - 1, 1, 1]}
            try:
                pos, info = to_position(frame, det, dem=dem)
                to_position(frame, det, dem=dem, attitude_terms=False)
            except ValueError:
                bad = True
                break
            if src == "V2":
                res = (pos, info)
        if bad:
            continue
        pos, info = res
        r0 = pos["error_radius_m"]
        one = geo.landuse_at(*to_5179(pos["lon"], pos["lat"]))
        rec = {"truth": truth, "D": D, "radius": round(r0, 1), "shallow": info["shallow"],
               "err_m": round(meters(pos["lat"], pos["lon"], flat, flon), 1),
               "C": "미확정" if one is None else ("맞는 확정" if one == truth else "틀린 단정"),
               "k": {}, "fixed": {}}
        for k in K_LIST:
            rec["k"][str(k)] = verdict(geo, pos["lat"], pos["lon"], r0 * k, info["shallow"], truth, a.points)
        for R in FIXED:
            rec["fixed"][str(R)] = verdict(geo, pos["lat"], pos["lon"], R, info["shallow"], truth, a.points)
        recs.append(rec)
        if len(recs) % 50 == 0:
            print(f"  {len(recs)}/{a.n}", flush=True)
    return recs


def stats(answers):
    n = len(answers)
    ok = sum(x == "맞는 확정" for x in answers)
    wrong = sum(x == "틀린 단정" for x in answers)
    conf = ok + wrong
    return conf / n, (ok / conf if conf else float("nan")), wrong / n


def ranked(recs, key, frac):
    """key 가 작은 사례부터 frac 만큼 한 점 조회 답으로 확정하고 나머지는 기권한다."""
    order = sorted(range(len(recs)), key=lambda i: key(recs[i]))
    take = set(order[:round(frac * len(recs))])
    return [recs[i]["C"] if i in take else "미확정" for i in range(len(recs))]


def report(recs):
    n = len(recs)
    print(f"사례 {n}개 (V2)")
    c = stats([r["C"] for r in recs])
    print(f"\n한 점 조회 전체: 확정 {c[0]:.0%}, 확정 정확도 {c[1]:.1%}, 틀린 단정 {c[2]:.1%}")

    print("\n[반경 판정, 반경 배율 k]   확정 비율  확정 정확도  틀린 단정")
    curve = []
    for k in K_LIST:
        s = stats([r["k"][str(k)] for r in recs])
        curve.append(("반경 x" + str(k), s))
        print(f"  k={k:<5}{s[0]:>14.0%}{s[1]:>12.1%}{s[2]:>10.1%}")
    print("\n[고정 반경 + 판정]          확정 비율  확정 정확도  틀린 단정")
    for R in FIXED:
        s = stats([r["fixed"][str(R)] for r in recs])
        print(f"  {R:>4}m{s[0]:>15.0%}{s[1]:>12.1%}{s[2]:>10.1%}")

    print("\n[같은 확정 비율에서] 반경 판정의 확정 비율마다 기준선의 정확도")
    print("  반경 판정           확정 비율  반경 판정  가까운 것부터  반경이 작은 것부터  무작위")
    for name, s in curve:
        f = s[0]
        near = stats(ranked(recs, lambda r: (r["D"], r["radius"]), f))
        small = stats(ranked(recs, lambda r: r["radius"], f))
        print(f"  {name:<18}{f:>9.0%}{s[1]:>11.1%}{near[1]:>14.1%}{small[1]:>18.1%}{c[1]:>9.1%}")
    print("  (정확도는 확정한 것 중 맞은 비율. 기준선은 한 점 조회 답을 그대로 쓰고 무엇을 확정할지만 고른다)")


def main():
    ap = argparse.ArgumentParser(description="같은 확정 비율에서 반경 판정과 한 점 조회 비교")
    ap.add_argument("landuse", nargs="?")
    ap.add_argument("--dem", nargs="+", default=[str(ROOT / "data/geo_moraegogae/37808.img"),
                                                  str(ROOT / "data/geo_moraegogae/37804.img")])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/verdict_coverage")
    ap.add_argument("--report", help="저장된 cases.jsonl 로 표만 다시 낸다")
    a = ap.parse_args()
    if a.report:
        report([json.loads(l) for l in open(a.report, encoding="utf-8")])
        return
    recs = make_cases(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cases.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(recs)
    print(f"\n결과: {out / 'cases.jsonl'}")


if __name__ == "__main__":
    main()
