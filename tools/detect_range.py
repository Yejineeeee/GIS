# -*- coding: utf-8 -*-
"""시뮬레이터 프레임으로 탐지가 어느 거리까지 되는지 잰다.

시연 전에 거리와 고도를 정하려고 만들었다. 프레임마다 발화점을 화면에 투영해,
그 자리에 있는 탐지만 연기로 세고 나머지는 발화점 밖 탐지로 따로 센다.
빈 화면으로 캡처된 프레임은 세지 않는다.

    python tools/detect_range.py sim/frames runs/smoke_1260_960/weights/best.pt 44.7

마지막 인자는 발화점의 지면 높이(m)다. 시뮬레이터 콘솔에서
groundHeight(FIRE.lon, FIRE.lat) 로 읽는다.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "geo"))
import transform as tf  # noqa: E402

FRAMES = Path(sys.argv[1])
WEIGHTS = sys.argv[2]
FIRE_GROUND = float(sys.argv[3])
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
IMGSZ = 960
EARTH_R = 6378137.0
BLANK_MEAN = 5.0          # 이보다 어두우면 캡처가 빈 것으로 본다

# 발화점 자리로 인정하는 범위. 연기가 바람을 타고 기울므로 가로는 넉넉히 둔다
NEAR_X = 0.25             # 화면 폭 대비
NEAR_BELOW = 0.02         # 발화점 지면보다 이만큼 아래까지는 봐준다
NEAR_ABOVE = 0.45         # 발화점 지면보다 이만큼 위까지


def ground_dist(lat1, lon1, lat2, lon2):
    dn = math.radians(lat2 - lat1) * EARTH_R
    de = math.radians(lon2 - lon1) * EARTH_R * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dn, de)


def y_of_depression(dep_deg, w, h, pitch, hfov):
    """화면 가운데 세로줄에서 그 내려다보는 각이 되는 y 를 찾는다."""
    lo, hi = 0.0, float(h)
    for _ in range(60):
        mid = (lo + hi) / 2
        if tf.depression_deg(w / 2, mid, w, h, pitch, hfov) < dep_deg:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def poly_area(poly):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(poly, list(poly[1:]) + [poly[0]]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def main():
    model = YOLO(WEIGHTS)
    rows, blank = [], []

    for meta_path in sorted(FRAMES.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        img_path = FRAMES / meta_path.with_suffix(".png").name
        truth, sweep = meta.get("truth"), meta.get("sweep")
        if not img_path.exists() or not truth or not sweep:
            continue

        if np.asarray(Image.open(img_path).convert("L"), dtype=float).mean() < BLANK_MEAN:
            blank.append(meta_path.stem)
            continue

        cam = meta["camera"]
        w, h = meta["image_size"]["width"], meta["image_size"]["height"]
        cam_above_fire = sweep["ground_m"] + cam["alt_m"] - FIRE_GROUND
        dist = ground_dist(cam["lat"], cam["lon"], truth["fire_lat"], truth["fire_lon"])
        y_fire = y_of_depression(math.degrees(math.atan2(cam_above_fire, dist)),
                                 w, h, cam["pitch_deg"], cam["hfov_deg"])

        res = model.predict(str(img_path), conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        smoke, outside = None, 0
        if res.masks is not None:
            for i in range(len(res.masks)):
                poly = res.masks.xy[i]
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                at_fire = (abs((min(xs) + max(xs)) / 2 - w / 2) <= NEAR_X * w
                           and max(ys) <= y_fire + NEAR_BELOW * h
                           and max(ys) >= y_fire - NEAR_ABOVE * h)
                conf = float(res.boxes.conf[i])
                if at_fire:
                    if smoke is None or conf > smoke[0]:
                        smoke = (conf, poly_area(poly) / (w * h) * 100)
                else:
                    outside += 1

        rows.append({"거리": round(sweep["D"]), "고도": round(cam["alt_m"]),
                     "conf": round(smoke[0], 2) if smoke else None,
                     "면적비": round(smoke[1], 2) if smoke else None,
                     "밖": outside})

    if not rows:
        print("잴 프레임이 없다")
        return
    if blank:
        print(f"빈 화면으로 캡처돼 뺀 프레임: {', '.join(blank)}\n")

    by_dist = defaultdict(list)
    for r in rows:
        by_dist[r["거리"]].append(r)

    print(f"{'거리(m)':>7} {'고도(m)':>7} {'잡음':>7} {'conf':>12} {'면적비(%)':>12} {'밖':>5}")
    for d in sorted(by_dist):
        v = by_dist[d]
        hit = [x for x in v if x["conf"] is not None]
        confs = [x["conf"] for x in hit]
        areas = [x["면적비"] for x in hit]
        alt = v[0]["고도"]
        cs = f"{min(confs):.2f}~{max(confs):.2f}" if confs else "-"
        ar = f"{min(areas):.2f}~{max(areas):.2f}" if areas else "-"
        print(f"{d:>7} {alt:>7} {f'{len(hit)}/{len(v)}':>7} {cs:>12} {ar:>12} "
              f"{sum(x['밖'] for x in v):>5}")

    print("\n밖 = 발화점 자리가 아닌 곳에 잡힌 탐지 수. 같은 연기가 조각나서 세어진 것도 "
          "들어가므로, 숫자가 크면 프레임을 직접 봐야 한다.")


if __name__ == "__main__":
    main()
