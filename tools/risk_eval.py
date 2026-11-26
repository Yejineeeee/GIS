# -*- coding: utf-8 -*-
"""위험도 평가셋: 시뮬레이터 발화점의 실제 위치로 매긴 위험도를 정답으로 두고 규칙을 채점한다.

위험도 규칙(src/vlm/risk.py)은 위치 판정과 "반경 안 건물·주거지"로 높음/보통/확인 필요를 낸다.
평가셋 사진에는 위험도 정답이 없어서, 정답 위치를 아는 시뮬레이션으로 평가셋을 새로 만든다.

정답 (실제 발화점 기준, 위치 오차 없음)
    높음  실제 발화점에서 BUFFER m 안에 건물이 있거나 주거지역이 걸린다
    보통  그 밖
    BUFFER 기본 50m: 국립산림과학원 공동 연구의 산림 인접 시설 관리 거리 30~50m 중 큰 값
    (Downloads 발표준비 위험도_규칙_근거_v2.md 와 같은 근거). 30m, 100m 도 같이 본다

예측 (파이프라인이 아는 값만 쓴다)
    반경 규칙   추정 위치 + 오차 반경으로 조회하고 판정한 뒤 risk.assess (지금 파이프라인)
    한 점 조회  추정 위치에서 정답과 같은 BUFFER m 로 조회 (오차를 무시하는 기준선, 확인 필요 없음)
    불꽃, 굴뚝, 검증기 결과는 넣지 않는다. 위치와 공간정보 부분만 채점한다

사례 (tools/verdict_sensitivity.py 의 사례 만들기를 따른다)
    자곡동 건물 자료 범위 안, 녹지지역(자연녹지, 생산녹지)에서 발화점을 고르게 뽑는다(산불 상황).
    자곡동 시뮬레이터 프레임의 거리와 탐지 픽셀 오차, 자세 잡음(1배, 2배)을 섞어 transform 으로
    좌표와 반경을 낸다. 지형은 시뮬레이터 지형 격자 하나로 정답과 계산을 모두 한다.

    python tools/risk_eval.py --data-root ../GIS --out runs/risk_eval
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src" / "vlm"))
import verdict_sensitivity as vs  # noqa: E402
import sim_terrain  # noqa: E402
from risk import assess  # noqa: E402

BBOX = (127.075, 37.446, 127.128, 37.497)   # 자곡동 건물 자료 범위 (data/geo_jagok/buildings_4326.geojson)
GREEN = ("자연녹지지역", "생산녹지지역", "보전녹지지역")
LEVELS_T = ("높음", "보통")
LEVELS_P = ("높음", "보통", "확인 필요")


def near_people(geo, lat, lon, r, points):
    """반경 r 안에 건물이 있거나 주거지역이 걸리는지. risk._near_people 과 같은 조건."""
    lu, ev, _ = vs.query(geo, lat, lon, r, n_points=points)
    return (ev["buildings_in_radius"] or 0) >= 1 or any("주거" in u["name"] for u in lu)


def make_cases(a, geo, dem, frames, noise_scale):
    noise = {k: v * noise_scale for k, v in vs.NOISE.items()}
    rng = random.Random(a.seed)
    lat0, lat1, lon0, lon1 = BBOX[1], BBOX[3], BBOX[0], BBOX[2]
    mlat = vs.MARGIN_M / 111320
    mlon = vs.MARGIN_M / (111320 * math.cos(math.radians((lat0 + lat1) / 2)))
    recs, tries = [], 0
    while len(recs) < a.n and tries < a.n * 50:
        tries += 1
        flat = rng.uniform(lat0 + mlat, lat1 - mlat)
        flon = rng.uniform(lon0 + mlon, lon1 - mlon)
        fx, fy = vs.to_5179(flon, flat)
        use = geo.landuse_at(fx, fy)
        fg = dem.height(fx, fy)
        if use not in GREEN or fg is None:
            continue
        fr = rng.choice(frames)
        D = fr["D"]
        bearing = rng.uniform(0, 360)
        dlat, dlon = vs.move(flat, flon, D * math.sin(math.radians(bearing)), D * math.cos(math.radians(bearing)))
        dg = dem.height(*vs.to_5179(dlon, dlat))
        if dg is None:
            continue
        heading = (bearing + 180) % 360
        dep = vs.dep_of(vs.ALT + dg - (fg + vs.FIRE_BASE_M), D)
        pitch = fr["aim"] - dep
        W, H = fr["W"], fr["H"]
        f = (W / 2) / math.tan(math.radians(fr["hfov"]) / 2)
        tx, ty = W / 2, H / 2 + f * math.tan(math.radians(dep + pitch))
        cam = {"lat": dlat, "lon": dlon, "alt_m": vs.ALT, "alt_ref": "지면", "heading_deg": heading,
               "pitch_deg": pitch, "hfov_deg": fr["hfov"], "source": "가상"}
        nlat, nlon = vs.move(dlat, dlon, rng.gauss(0, noise["h"]), rng.gauss(0, noise["h"]))
        cam.update(lat=nlat, lon=nlon, alt_m=vs.ALT + rng.gauss(0, noise["alt"]),
                   heading_deg=(heading + rng.gauss(0, noise["heading"])) % 360,
                   pitch_deg=pitch + rng.gauss(0, noise["pitch"]))
        frame = {"image_size": {"width": W, "height": H}, "camera": cam}
        dx, dy, _ = fr["V2"]
        det = {"polygon": None, "bbox": [tx + dx - 0.5, ty + dy - 1, 1, 1]}
        try:
            pos, info = vs.to_position(frame, det, dem=dem)
        except ValueError:
            continue
        recs.append({"true_lat": flat, "true_lon": flon, "true_use": use, "D": D,
                     "lat": pos["lat"], "lon": pos["lon"], "radius": round(pos["error_radius_m"], 1),
                     "shallow": info["shallow"],
                     "err_m": round(vs.meters(pos["lat"], pos["lon"], flat, flon), 1)})
    return recs


def score(geo, recs, buffer_m, points):
    rows = []
    for r in recs:
        truth = "높음" if near_people(geo, r["true_lat"], r["true_lon"], buffer_m, points) else "보통"
        lu, ev, qi = vs.query(geo, r["lat"], r["lon"], r["radius"], n_points=points)
        verdict, _ = vs.judge(lu, r["radius"], qi["landuse_coverage"], r["shallow"])
        alert = {"location_verdict": verdict, "land_use": lu, "evidence": ev, "observed": {"flame_visible": False}}
        pred = assess(alert)["level"]
        point = "높음" if near_people(geo, r["lat"], r["lon"], buffer_m, points) else "보통"
        rows.append({"truth": truth, "radius_rule": pred, "one_point": point, "verdict": verdict,
                     "D": r["D"], "err_m": r["err_m"], "radius": r["radius"]})
    return rows


def summarize(rows, key):
    n = len(rows)
    cm = Counter((r["truth"], r[key]) for r in rows)
    hi = sum(r["truth"] == "높음" for r in rows)
    lo = n - hi
    under = cm[("높음", "보통")]
    over = cm[("보통", "높음")]
    defer = sum(r[key] == "확인 필요" for r in rows)
    decided = n - defer
    right = cm[("높음", "높음")] + cm[("보통", "보통")]
    return {"n": n, "정답 높음": hi, "정답 보통": lo,
            "혼동행렬": {t: {p: cm[(t, p)] for p in LEVELS_P} for t in LEVELS_T},
            "과소평가(높음을 보통으로)": under, "과소평가율(정답 높음 중)": round(under / hi, 4) if hi else None,
            "과대평가(보통을 높음으로)": over, "과대평가율(정답 보통 중)": round(over / lo, 4) if lo else None,
            "확인 필요": defer, "확인 필요 비율": round(defer / n, 4),
            "판단한 것 중 정확도": round(right / decided, 4) if decided else None,
            "높음 놓침(높음인데 높음 아님)": hi - cm[("높음", "높음")]}


def main():
    ap = argparse.ArgumentParser(description="위험도 규칙을 시뮬레이션 정답으로 채점")
    ap.add_argument("--data-root", default=str(ROOT), help="sim/, data/ 가 있는 체크아웃")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--buffer", type=float, nargs="+", default=[50.0, 30.0, 100.0])
    ap.add_argument("--out", default="runs/risk_eval")
    a = ap.parse_args()
    data = Path(a.data_root).resolve()
    os.environ["SIM_SITE"] = "jagok"
    sim_terrain.ROOT = data
    geo = vs.GeoData(str(data / "data/geo_jagok/landuse_5179.geojson"), str(data / "data/geo_jagok/buildings_5179.geojson"))
    dem = vs.SimTerrain(str(data / "sim/terrain_jagok.json"))
    frames = vs.frame_offsets()
    print(f"프레임 {len(frames)}장, 건물 {len(geo.buildings)}개", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for ns in a.noise:
        recs = make_cases(a, geo, dem, frames, ns)
        with open(out / f"cases_n{ns:g}.jsonl", "w", encoding="utf-8") as fo:
            for r in recs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"잡음 {ns:g}배: 사례 {len(recs)}개, 오차 중앙 {sorted(r['err_m'] for r in recs)[len(recs) // 2]}m", flush=True)
        for b in a.buffer:
            rows = score(geo, recs, b, a.points)
            for key, name in (("radius_rule", "반경 규칙"), ("one_point", "한 점 조회")):
                k = f"잡음 {ns:g}배 / 정답 거리 {b:g}m / {name}"
                report[k] = summarize(rows, key)
                print(k, json.dumps(report[k], ensure_ascii=False), flush=True)
            if b == a.buffer[0]:
                with open(out / f"rows_n{ns:g}_b{b:g}.jsonl", "w", encoding="utf-8") as fo:
                    for r in rows:
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
