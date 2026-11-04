# -*- coding: utf-8 -*-
"""시뮬레이터 프레임의 카메라 자세에 잡음을 섞어 위치 오차와 오차 반경을 검증한다.

시뮬레이터는 카메라 위치와 자세를 오차 없이 준다. 실제 드론은 GNSS 위치, 고도,
기수, 짐벌 각도가 모두 조금씩 틀린다. 그래서 시뮬레이터로 잰 위치 오차는 실제보다
낙관적이다. 이 도구는 프레임 JSON 의 camera 값에 정규분포 잡음을 섞어 여러 번
좌표를 다시 내고, 정답(truth)과의 거리와 transform.to_position 이 낸 오차 반경
(error_radius_m) 안에 정답이 들어오는 비율(포함률)을 잰다.

    python tools/pose_noise.py sim/frames runs/smoke_1260_960/weights/best.pt
    python tools/pose_noise.py sim/frames best.pt --level 나쁨 --n 500

잡음 크기(1 시그마, 가정). --level 로 고르거나 하나씩 덮어쓴다.
    보통  수평 2.0m(반경 95% 약 5m, docs/gsd.md 의 GNSS 5m 와 맞춤), 고도 2m,
          기수 2도, 카메라 각도 1도
    나쁨  보통의 두 배

정답 픽셀(순수 기하)과 conf 최고 탐지의 밑동(지금 파이프라인) 두 가지를 잰다.
지형은 sim/terrain.json(시뮬레이터 지형)을 쓴다. V-World DEM 은 시뮬레이터 지형과
맞지 않는다(src/geo/sim_terrain.py).
"""
import argparse
import glob
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from transform import to_position, move  # noqa: E402

LEVELS = {
    "없음": dict(h=0.0, alt=0.0, heading=0.0, pitch=0.0),
    "보통": dict(h=2.0, alt=2.0, heading=2.0, pitch=1.0),
    "나쁨": dict(h=4.0, alt=4.0, heading=4.0, pitch=2.0),
}
FIRE_BASE_M = 2.0      # 시뮬레이터 연기가 나기 시작하는 높이 (index.html lightSmoke)


def meters(la1, lo1, la2, lo2):
    R = 6378137.0
    dy = math.radians(la2 - la1) * R
    dx = math.radians(lo2 - lo1) * R * math.cos(math.radians(la1))
    return math.hypot(dx, dy)


def true_pixel(meta, fire_ground):
    """정답 발화점이 화면에 찍히는 픽셀. sweep 는 기수를 발화점에 맞추므로 x 는 가운데다."""
    c, W, H = meta["camera"], meta["image_size"]["width"], meta["image_size"]["height"]
    f = (W / 2) / math.tan(math.radians(c["hfov_deg"]) / 2)
    dz = c["alt_m"] + meta["sweep"]["ground_m"] - (fire_ground + FIRE_BASE_M)
    dep = math.degrees(math.atan2(dz, meta["sweep"]["D"]))
    return W / 2, H / 2 + f * math.tan(math.radians(dep + c["pitch_deg"]))


def noisy_camera(cam, lv, rng):
    c = dict(cam)
    lat, lon = move(cam["lat"], cam["lon"], rng.gauss(0, lv["h"]), rng.gauss(0, lv["h"]))
    c["lat"], c["lon"] = lat, lon
    c["alt_m"] = cam["alt_m"] + rng.gauss(0, lv["alt"])
    c["heading_deg"] = (cam["heading_deg"] + rng.gauss(0, lv["heading"])) % 360
    c["pitch_deg"] = cam["pitch_deg"] + rng.gauss(0, lv["pitch"])
    return c


def main():
    ap = argparse.ArgumentParser(description="카메라 자세 잡음으로 위치 오차와 오차 반경 검증")
    ap.add_argument("frames")
    ap.add_argument("weights")
    ap.add_argument("--level", choices=list(LEVELS), default="보통")
    ap.add_argument("--n", type=int, default=300, help="프레임마다 잡음 표본 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pose_noise")
    a = ap.parse_args()

    from detect import Detector
    from sim_terrain import SimTerrain
    from query import to_5179
    ter = SimTerrain(str(ROOT / "sim" / "terrain.json"))
    det = Detector(a.weights)
    lv = LEVELS[a.level]
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"잡음 '{a.level}': 수평 {lv['h']}m, 고도 {lv['alt']}m, 기수 {lv['heading']}도, "
          f"카메라 각도 {lv['pitch']}도 (1 시그마), 표본 {a.n}")
    by_d = defaultdict(lambda: defaultdict(list))
    recs = []
    for i, png in enumerate(sorted(glob.glob(str(Path(a.frames) / "*.png")))):
        meta = json.load(open(Path(png).with_suffix(".json"), encoding="utf-8"))
        t = meta["truth"]
        fire_g = ter.height(*to_5179(t["fire_lon"], t["fire_lat"]))
        frame, _ = det.run(png, i)
        tx, ty = true_pixel(meta, fire_g)
        targets = {"정답 픽셀": {"polygon": None, "bbox": [tx - 0.5, ty - 1, 1, 1]}}
        if frame["detections"]:
            targets["지금 방식"] = max(frame["detections"], key=lambda d: d["score"])
        D = meta["sweep"]["D"]
        for name, d in targets.items():
            errs, radii, inside = [], [], 0
            for _ in range(a.n):
                f2 = dict(frame)
                f2["camera"] = noisy_camera(meta["camera"], lv, rng)
                try:
                    pos, _ = to_position(f2, d, dem=ter)
                except ValueError:
                    continue
                e = meters(pos["lat"], pos["lon"], t["fire_lat"], t["fire_lon"])
                errs.append(e)
                radii.append(pos["error_radius_m"])
                inside += e <= pos["error_radius_m"]
            if not errs:
                continue
            rec = {"image": Path(png).name, "D": D, "bearing": meta["sweep"]["bearing"], "target": name,
                   "err_med": round(st.median(errs), 1),
                   "err_p95": round(sorted(errs)[int(0.95 * (len(errs) - 1))], 1),
                   "radius_mean": round(st.mean(radii), 1), "coverage": round(inside / len(errs), 3)}
            recs.append(rec)
            by_d[name][D].append(rec)

    with open(out / f"pose_noise_{a.level}.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name in ("정답 픽셀", "지금 방식"):
        print(f"\n[{name}]")
        print(f"{'거리':>6}{'장':>4}{'오차 중앙':>10}{'오차 95%':>10}{'반경 평균':>10}{'포함률':>8}")
        for D in sorted(by_d[name]):
            rs = by_d[name][D]
            print(f"{D:>5}m{len(rs):>4}{st.mean(r['err_med'] for r in rs):>9.1f}m"
                  f"{st.mean(r['err_p95'] for r in rs):>9.1f}m{st.mean(r['radius_mean'] for r in rs):>9.1f}m"
                  f"{st.mean(r['coverage'] for r in rs):>8.0%}")
    print(f"\n결과: {out / f'pose_noise_{a.level}.jsonl'}")


if __name__ == "__main__":
    main()
