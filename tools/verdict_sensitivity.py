# -*- coding: utf-8 -*-
"""verdict_compare.py 에 자세 잡음 배율을 더한 민감도 실험이다.

핵심 수치(틀린 단정 18.3% -> 3.7%)는 평가에 넣은 자세 잡음과 반경 공식의 시그마가 같은 값이다.
실제 잡음이 가정보다 크면 어떻게 되는지 본다. 사례를 만드는 법과 채점은 verdict_compare.py 와
같고, 두 옵션만 더했다. verdict_compare.py 는 고치지 않았다.

    --noise-scale  평가에 섞는 자세 잡음(수평, 고도, 기수, 카메라 각도)에 곱한다. 반경은 그대로
                   두므로 "실제 잡음이 가정보다 컸을 때"가 된다
    --radius-scale 반경 공식의 자세 시그마(src/geo/transform.py 의 기수, 카메라 각도, 고도)에 곱한다.
                   "실제 잡음을 알고 반경을 맞춰 고쳤을 때"가 된다

    python tools/verdict_sensitivity.py <landuse_5179.geojson> --noise-scale 2 --out runs/verdict_sens/n2

아래는 verdict_compare.py 의 설명이다.

한 점 조회와 반경 판정이 틀린 용도를 단정하는 비율을 비교한다 (핵심 주장 검증).

"위치 오차를 고려하면 틀린 단정이 준다"를 정답 좌표가 있는 조건에서 잰다. 시뮬레이터로
발화점마다 프레임을 새로 찍으면 연기가 자라는 데만 한 장에 1분이 넘게 걸리므로,
시뮬레이터에서 이미 잰 탐지 오차를 지도 위 여러 발화점으로 옮겨 쓴다.

    python tools/verdict_compare.py <landuse_5179.geojson> --n 600

한 사례를 만드는 법
    1. 시연 지역 안에서 용도지역이 있는 발화점을 고르게 하나 뽑는다
    2. 시뮬레이터 프레임 31장(runs/verify/verify.jsonl) 중 하나를 뽑아 그 거리와 카메라
       각도를 쓰고, 드론은 발화점에서 아무 방향으로 그 거리만큼 떨어진 고도 120m 에 둔다
    3. 정답 픽셀을 기하로 구하고, 그 프레임에서 탐지 밑동이 정답 픽셀에서 벗어난 만큼
       (dx, dy) 옮겨 탐지 픽셀로 쓴다. 탐지는 V2(검증기 0.7 통과 중 conf 1등)와
       V0(conf 1등) 둘 다 본다
    4. 카메라 값에 자세 잡음(보통: 수평 2m, 고도 2m, 기수 2도, 카메라 각도 1도)을 섞어
       src/geo/transform.py 로 좌표와 반경을 낸다. 지형은 V-World DEM 하나로 정답과
       계산을 모두 하므로 지형 차이는 들어가지 않는다

비교하는 방식
    C   한 점 조회: 추정 좌표 한 점의 용도를 그대로 단정한다 (제안서 입력 조건 C)
    D   자세 오차 항 없는 반경 + 판정: transform.to_position(attitude_terms=False) 반경으로
        조회하고 src/pipeline/verdict.py 로 판정 (11주차 전 반경)
    D+  자세 오차 항을 넣은 반경 + 판정: transform.to_position 기본값. 반경에
        2 x sqrt((d sψ)^2 + (h sθ / sin^2 dep)^2 + (sh / tan dep)^2) 가 더해진다

채점 (정답 = 실제 발화점의 용도)
    확정  1순위 용도가 정답이면 맞는 확정, 아니면 틀린 단정
    모호  1, 2순위 안에 정답이 있으면 맞는 모호, 없으면 틀린 모호
    미확정 용도를 말하지 않는다
"""
import argparse
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

import transform  # noqa: E402
from transform import to_position, move  # noqa: E402
from query import GeoData, query, to_5179  # noqa: E402
from dem import Dem  # noqa: E402
from sim_terrain import SimTerrain, sim_paths  # noqa: E402
from verdict import judge  # noqa: E402

BBOX = (128.880, 37.712, 128.932, 37.762)   # 시연 지역 (docs/threshold.md)
MARGIN_M = 200
ALT = 120.0
FIRE_BASE_M = 2.0
TAU = 0.7
NOISE = dict(h=2.0, alt=2.0, heading=2.0, pitch=1.0)


def meters(la1, lo1, la2, lo2):
    R = 6378137.0
    dy = math.radians(la2 - la1) * R
    dx = math.radians(lo2 - lo1) * R * math.cos(math.radians(la1))
    return math.hypot(dx, dy)


def dep_of(dz, D):
    return math.degrees(math.atan2(dz, D))


def frame_offsets():
    """시뮬레이터 프레임마다 (거리, 조준 각 차이, V0 와 V2 탐지 픽셀의 정답 픽셀 대비 dx, dy, 오차)."""
    frames_dir, terrain, verify_dir = sim_paths()
    ter = SimTerrain(str(terrain))
    out = []
    for l in open(verify_dir / "verify.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r["dets"]:
            continue
        meta = json.load(open(frames_dir / r["image"].replace(".png", ".json"), encoding="utf-8"))
        c, W, H = meta["camera"], meta["image_size"]["width"], meta["image_size"]["height"]
        t = meta["truth"]
        fg = ter.height(*to_5179(t["fire_lon"], t["fire_lat"]))
        D = meta["sweep"]["D"]
        dep = dep_of(c["alt_m"] + meta["sweep"]["ground_m"] - (fg + FIRE_BASE_M), D)
        f = (W / 2) / math.tan(math.radians(c["hfov_deg"]) / 2)
        tx, ty = W / 2, H / 2 + f * math.tan(math.radians(dep + c["pitch_deg"]))
        v0 = max(r["dets"], key=lambda d: d["score"])
        ok = [d for d in r["dets"] if d["p_yes"] >= TAU]
        v2 = max(ok, key=lambda d: d["score"]) if ok else v0
        out.append({"image": r["image"], "D": D, "aim": c["pitch_deg"] + dep, "W": W, "H": H,
                    "hfov": c["hfov_deg"],
                    "V0": (v0["x"] - tx, v0["y"] - ty, v0["err_m"]),
                    "V2": (v2["x"] - tx, v2["y"] - ty, v2["err_m"])})
    return out


def score(kind, land_use, truth):
    if kind == "확정":
        return "맞는 확정" if land_use[0]["name"] == truth else "틀린 단정"
    if kind == "모호":
        return "맞는 모호" if truth in [u["name"] for u in land_use[:2]] else "틀린 모호"
    return "미확정"


def main():
    ap = argparse.ArgumentParser(description="한 점 조회 대 반경 판정 (틀린 단정 비율)")
    ap.add_argument("landuse")
    ap.add_argument("--dem", nargs="+", default=[str(ROOT / "data/geo_moraegogae/37808.img"),
                                                  str(ROOT / "data/geo_moraegogae/37804.img")])
    ap.add_argument("--sim-dem", action="store_true",
                    help="DEM 대신 시뮬레이터 지형 격자를 쓴다 (SIM_SITE 의 terrain). DEM 이 없는 지역용")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LON0", "LAT0", "LON1", "LAT1"),
                    help="발화점을 뽑을 범위 (기본: 모래고개입구 시연 지역)")
    ap.add_argument("--n", type=int, default=600, help="사례 수")
    ap.add_argument("--points", type=int, default=600, help="반경 조회 지점 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/verdict_sens/n1")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="평가에 섞는 자세 잡음 배율")
    ap.add_argument("--radius-scale", type=float, default=1.0, help="반경 공식의 자세 시그마 배율")
    a = ap.parse_args()
    for k in NOISE:
        NOISE[k] *= a.noise_scale
    transform.HEADING_SIGMA_DEG *= a.radius_scale
    transform.PITCH_SIGMA_DEG *= a.radius_scale
    transform.ALT_SIGMA_M *= a.radius_scale
    print(f"잡음 {NOISE}, 반경 시그마 기수 {transform.HEADING_SIGMA_DEG} 각도 {transform.PITCH_SIGMA_DEG} 고도 {transform.ALT_SIGMA_M}")

    rng = random.Random(a.seed)
    geo = GeoData(a.landuse)
    dem = SimTerrain(str(sim_paths()[1])) if a.sim_dem else Dem(a.dem)
    frames = frame_offsets()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    box = a.bbox or BBOX
    lat0, lat1 = box[1], box[3]
    lon0, lon1 = box[0], box[2]
    mlat = MARGIN_M / 111320
    mlon = MARGIN_M / (111320 * math.cos(math.radians((lat0 + lat1) / 2)))

    recs = []
    tries = 0
    while len(recs) < a.n and tries < a.n * 20:
        tries += 1
        flat = rng.uniform(lat0 + mlat, lat1 - mlat)
        flon = rng.uniform(lon0 + mlon, lon1 - mlon)
        fx, fy = to_5179(flon, flat)
        truth = geo.landuse_at(fx, fy)
        fg = dem.height(fx, fy)
        if truth is None or fg is None or fg < 1.0:     # 바다나 범위 밖
            continue
        fr = rng.choice(frames)
        D = fr["D"]
        bearing = rng.uniform(0, 360)                   # 발화점에서 본 드론 방향
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
        rec = {"fire": [round(flat, 6), round(flon, 6)], "truth": truth, "D": D, "frame": fr["image"],
               "bearing": round(bearing, 1)}
        noisy = dict(cam)
        nlat, nlon = move(dlat, dlon, rng.gauss(0, NOISE["h"]), rng.gauss(0, NOISE["h"]))
        noisy.update(lat=nlat, lon=nlon, alt_m=ALT + rng.gauss(0, NOISE["alt"]),
                     heading_deg=(heading + rng.gauss(0, NOISE["heading"])) % 360,
                     pitch_deg=pitch + rng.gauss(0, NOISE["pitch"]))
        bad = False
        for src in ("정답픽셀", "V0", "V2"):
            dx, dy, sim_err = (0.0, 0.0, 0.0) if src == "정답픽셀" else fr[src]
            px, py = tx + dx, ty + dy
            det = {"polygon": None, "bbox": [px - 0.5, py - 1, 1, 1]}
            frame = {"image_size": {"width": W, "height": H}, "camera": noisy}
            try:
                pos, info = to_position(frame, det, dem=dem)
                old, _ = to_position(frame, det, dem=dem, attitude_terms=False)
            except ValueError:
                bad = True
                break
            err = meters(pos["lat"], pos["lon"], flat, flon)
            ex, ey = to_5179(pos["lon"], pos["lat"])
            one = geo.landuse_at(ex, ey)
            res = {"err_m": round(err, 1), "C": "틀린 단정" if one != truth else "맞는 확정"}
            if one is None:
                res["C"] = "미확정"
            for name, r in (("D", old["error_radius_m"]), ("D+", pos["error_radius_m"])):
                lu, _, qi = query(geo, pos["lat"], pos["lon"], r, n_points=a.points)
                kind, _ = judge(lu, r, qi["landuse_coverage"], info["shallow"])
                res[name] = score(kind, lu, truth)
                res[name + "_radius"] = round(r, 1)
                res[name + "_inside"] = err <= r
            res["shallow"] = info["shallow"]
            rec[src] = res
        if bad:
            continue
        recs.append(rec)
        if len(recs) % 50 == 0:
            print(f"  {len(recs)}/{a.n}", flush=True)

    with open(out / "verdict_compare.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    report(recs)
    summary(recs)
    print(f"\n결과: {out / 'verdict_compare.jsonl'}")


KINDS = ["맞는 확정", "틀린 단정", "맞는 모호", "틀린 모호", "미확정"]


def summary(recs, src="V2"):
    """발표에 쓰는 요약: 방식마다 틀린 단정, 확정 비율, 확정했을 때 정확도, 반경 포함률."""
    n = len(recs)
    near = [r for r in recs if 200 <= r["D"] <= 400]
    print(f"\n[요약 {src}] 사례 {n}개, 그중 200~400m {len(near)}개")
    print("  방식   틀린 단정  확정 비율  확정 정확도  포함률  200~400m 틀린 단정")
    for m in ("C", "D", "D+"):
        c = Counter(r[src][m] for r in recs)
        conf = c["맞는 확정"] + c["틀린 단정"]
        acc = c["맞는 확정"] / conf if conf else float("nan")
        inc = "-" if m == "C" else f"{sum(r[src][m + '_inside'] for r in recs) / n:.0%}"
        wn = sum(r[src][m] == "틀린 단정" for r in near) / len(near) if near else float("nan")
        print(f"  {m:<5}{c['틀린 단정'] / n:>9.1%}{conf / n:>10.0%}{acc:>12.1%}{inc:>8}{wn:>16.1%}")


def report(recs):
    print(f"\n사례 {len(recs)}개. 정답 용도 분포: " +
          ", ".join(f"{k} {v}" for k, v in Counter(r["truth"] for r in recs).most_common(5)))
    for src in ("정답픽셀", "V0", "V2"):
        print(f"\n[{src}] 탐지 픽셀 {'= 정답 픽셀 (자세 잡음만)' if src == '정답픽셀' else '= 시뮬레이터 탐지 오차 + 자세 잡음'}")
        print(f"  {'방식':<6}" + "".join(f"{k:>9}" for k in KINDS) + f"{'반경 포함률':>10}")
        for m in ("C", "D", "D+"):
            c = Counter(r[src][m] for r in recs)
            inc = "" if m == "C" else f"{sum(r[src][m + '_inside'] for r in recs) / len(recs):>9.0%}"
            print(f"  {m:<6}" + "".join(f"{c[k] / len(recs):>9.1%}" for k in KINDS) + inc)
        by = defaultdict(list)
        for r in recs:
            by[r["D"]].append(r)
        print("  거리별 틀린 단정 (C / D / D+), 미확정 (D / D+)")
        for D in sorted(by):
            rs = by[D]
            w = [sum(r[src][m] == "틀린 단정" for r in rs) / len(rs) for m in ("C", "D", "D+")]
            u = [sum(r[src][m] == "미확정" for r in rs) / len(rs) for m in ("D", "D+")]
            print(f"    {D:>4}m {len(rs):>4}개  틀린 단정 {w[0]:.0%} / {w[1]:.0%} / {w[2]:.0%}"
                  f"   미확정 {u[0]:.0%} / {u[1]:.0%}   오차 중앙 {st.median(r[src]['err_m'] for r in rs):.0f}m")


if __name__ == "__main__":
    main()
