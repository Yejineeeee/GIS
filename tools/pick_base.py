# -*- coding: utf-8 -*-
"""연기 밑동 후보를 번호 붙은 십자로 그린다 (WBS 4.7 ① 후보 지점 선택).

지금 파이프라인은 conf 가 가장 높은 탐지 하나의 밑동(transform.target_pixel)을
좌표로 바꾼다. 그런데 YOLO 는 한 기둥을 여러 조각으로 쪼개고(docs/simulator.md 7번)
밭이나 비닐하우스도 잡는다(6번). conf 최고 조각이 기둥 몸통이거나 헛탐지면 위치가
수백 m 틀린다. 이 도구는 탐지마다 밑동을 후보로 내고, 사진에 번호 붙은 십자로
그려 VLM 이 고를 수 있게 한다.

    python tools/pick_base.py sim/frames runs/smoke_1260_960/weights/best.pt
    python tools/pick_base.py sim/frames best.pt --out runs/pick_base --merge-px 15

출력 (--out, 기본 runs/pick_base)
    <프레임>_pick.png     후보 십자와 번호를 그린 사진. VLM 입력
    candidates.jsonl      프레임마다 후보 목록(번호, 픽셀, conf, 정답과의 거리)

정하고 간 것
    - 기본은 글자 표시(A, B, C), 2배 크기, 무작위 순서다. 11/04 에 사진 4장을 순서만
      바꿔 16번씩 물었더니 숫자는 "1" 에 쏠렸다(1.0배 10/16, 2.0배 13/16, 아무렇게나
      고를 때 4.1). 글자 2배만 첫 표시를 4/16 으로 골라 쏠림이 없었고, 순서를 바꿔도
      사진당 평균 1.5곳만 골랐다(runs/bias).
    - 내부 번호는 conf 높은 순이다(--order random 이면 섞는다). 묶기를 끄면(--merge-px 0) 1번이 지금 방식이 고르는
      점이고, 묶으면 1번 자리가 아래 점으로 바뀔 수 있다. 표의 "지금 방식" 칸은
      묶기와 상관없이 conf 최고 탐지로 따로 잰다
    - 고르는 대상은 박스가 아니라 십자(대표 픽셀)다. 번호도 십자에 붙인다.
      박스 모서리에 붙였더니 큰 박스의 번호가 가려졌다
    - 밑동끼리 --merge-px 안에 있으면 하나로 묶고 **더 아래 점을 남긴다**.
      밑동을 찾는 것이므로 아래가 맞다. conf 높은 쪽을 남겼더니 400m 에서 가장
      가까운 후보가 30.4m 에서 56.1m 로 나빠졌다.
      거리는 15px 로 둔다. 11/04 프레임에서 40px 은 300m 와 800m 의 최선 후보를
      묶어 없앴고(4.8m -> 10.2m, 12.2m -> 16.5m), 15px 은 겹친 십자만 합치고
      후보 중 최선을 하나도 바꾸지 않았다
    - 정답 거리는 프레임 JSON 에 truth 가 있을 때만 낸다. sim/terrain.json 이
      있으면 시뮬레이터 지형으로 잰다(src/geo/sim_terrain.py). V-World DEM 은
      시뮬레이터 지형과 맞지 않아 정답 픽셀로도 수십 m 가 틀린다
"""
import argparse
import random
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from transform import target_pixel, to_position  # noqa: E402

YELLOW = (255, 230, 0)
BLACK = (0, 0, 0)


def meters(la1, lo1, la2, lo2):
    R = 6378137.0
    dy = math.radians(la2 - la1) * R
    dx = math.radians(lo2 - lo1) * R * math.cos(math.radians(la1))
    return math.hypot(dx, dy)


def label_of(no, labels):
    """번호 1, 2, 3 ... 을 그림에 쓸 표시로 바꾼다. alpha 면 A, B, C ..."""
    return chr(64 + no) if labels == "alpha" else str(no)


def candidates(frame, merge_px, order="conf", rng=None):
    """탐지마다 밑동을 내고, 가까운 것끼리 묶어 아래 점을 남긴다.

    order="conf" 면 conf 높은 순으로 번호를 붙이고, "random" 이면 rng 로 섞어 붙인다.
    """
    raw = []
    for d in frame["detections"]:
        x, y = target_pixel(d)
        raw.append({"x": x, "y": y, "score": d["score"], "det": d, "members": 1})
    groups = []
    # conf 높은 것부터 자리를 잡되, 같은 묶음 안에서는 아래 점이 대표가 된다
    for c in sorted(raw, key=lambda c: -c["score"]):
        for g in groups:
            if math.hypot(c["x"] - g["x"], c["y"] - g["y"]) <= merge_px:
                g["members"] += 1
                g["score"] = max(g["score"], c["score"])
                if c["y"] > g["y"]:
                    g["x"], g["y"], g["det"] = c["x"], c["y"], c["det"]
                break
        else:
            groups.append(dict(c))
    groups.sort(key=lambda g: -g["score"])
    if order == "random":
        rng.shuffle(groups)
    for i, g in enumerate(groups, 1):
        g["no"] = i
    groups.sort(key=lambda g: -g["score"])     # 목록은 늘 conf 순. 번호만 바뀐다
    return groups


def _font(size):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(image_path, cands, out_path, scale, labels="num"):
    """탐지 박스(얇은 테두리), 밑동 십자, 십자 오른쪽의 노란 번호 칸을 그린다.

    scale 은 크기 배율이다. 1024 너비 기준 십자 팔 9px, 번호 칸 약 20x22px.
    """
    from PIL import Image, ImageDraw
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    s = max(1.0, W / 1024) * scale
    arm, lw, fs, pad = int(round(9 * s)), max(2, int(round(2 * s))), int(round(17 * s)), int(round(3 * s))
    font = _font(fs)
    dr = ImageDraw.Draw(im)

    # 박스를 먼저 다 그려야 번호 칸이 다른 박스 선에 가리지 않는다
    for c in cands:
        bx, by, bw, bh = c["det"]["bbox"]
        dr.rectangle([bx, by, bx + bw, by + bh], outline=YELLOW, width=max(1, int(round(s))))

    taken = []
    for c in cands:
        x, y = c["x"], c["y"]
        dr.line([(x - arm, y), (x + arm, y)], fill=YELLOW, width=lw)
        dr.line([(x, y - arm), (x, y + arm)], fill=YELLOW, width=lw)

        label = label_of(c["no"], labels)
        l, t, r, b = dr.textbbox((0, 0), label, font=font)
        bw_, bh_ = (r - l) + 2 * pad, (b - t) + 2 * pad
        # 십자 바로 오른쪽이 기본. 겹치면 왼쪽, 아래, 위 순으로 옮긴다
        spots = [(x + arm + 1, y - bh_ * 0.4), (x - arm - 1 - bw_, y - bh_ * 0.4),
                 (x - bw_ / 2, y + arm + 1), (x - bw_ / 2, y - arm - 1 - bh_)]
        pick = None
        for sx, sy in spots:
            sx = min(max(sx, 0), W - bw_)
            sy = min(max(sy, 0), H - bh_)
            box = (sx, sy, sx + bw_, sy + bh_)
            if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3]
                       for o in taken):
                pick = box
                break
        if pick is None:
            sx, sy = spots[0]
            sx, sy = min(max(sx, 0), W - bw_), min(max(sy, 0), H - bh_)
            pick = (sx, sy, sx + bw_, sy + bh_)
        taken.append(pick)
        dr.rectangle(pick, fill=YELLOW)
        dr.text((pick[0] + pad - l, pick[1] + pad - t), label, font=font, fill=BLACK)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="연기 밑동 후보를 번호 십자로 그린다 (4.7 ①)")
    ap.add_argument("frames", help="사진 폴더 (옆에 같은 이름의 JSON 이 있으면 카메라와 정답을 씀)")
    ap.add_argument("weights", help="분할 모델 가중치")
    ap.add_argument("--out", default="runs/pick_base")
    ap.add_argument("--merge-px", type=float, default=15.0,
                    help="밑동끼리 이 픽셀 안이면 하나로 묶는다 (0 이면 묶지 않음)")
    ap.add_argument("--scale", type=float, default=2.0, help="십자와 표시 크기 배율")
    ap.add_argument("--order", choices=["conf", "random"], default="random",
                    help="번호 붙이는 순서. random 은 VLM 이 숫자 자체에 쏠리는지 볼 때 쓴다")
    ap.add_argument("--seed", type=int, default=0, help="--order random 의 난수 씨앗")
    ap.add_argument("--labels", choices=["num", "alpha"], default="alpha",
                    help="표시를 숫자(1, 2, 3)로 할지 글자(A, B, C)로 할지")
    ap.add_argument("--terrain", default=str(ROOT / "sim" / "terrain.json"),
                    help="시뮬레이터 지형 격자. 없으면 지형 없이 잰다")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from detect import Detector
    ter = None
    if Path(a.terrain).is_file():
        from sim_terrain import SimTerrain
        ter = SimTerrain(a.terrain)
    det = Detector(a.weights)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"지형: {'시뮬레이터 격자' if ter else '없음 (평지 가정)'}  묶기: {a.merge_px:g}px")
    print(f"{'사진':<14}{'거리':>6}{'탐지':>5}{'후보':>5}{'지금 방식':>11}{'후보 중 최선':>13}{'최선':>9}")
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as fo:
        for i, png in enumerate(sorted(glob.glob(str(Path(a.frames) / "*.png")))):
            frame, truth = det.run(png, i)
            if not frame["detections"]:
                print(f"{Path(png).stem:<14} 탐지 없음")
                continue
            cands = candidates(frame, a.merge_px, a.order, rng)
            side = Path(png).with_suffix(".json")
            meta = json.load(open(side, encoding="utf-8")) if side.is_file() else {}
            t = meta.get("truth")

            def err(d):
                if not t:
                    return None
                try:
                    pos, _ = to_position(frame, d, dem=ter)
                except ValueError:
                    return float("inf")
                return round(meters(pos["lat"], pos["lon"], t["fire_lat"], t["fire_lon"]), 1)

            for c in cands:
                c["err_m"] = err(c["det"])
            # 지금 방식: 묶기와 상관없이 conf 최고 탐지의 밑동
            now_err = err(max(frame["detections"], key=lambda d: d["score"]))
            draw(png, cands, out / f"{Path(png).stem}_pick.png", a.scale, a.labels)

            dist = (meta.get("sweep") or {}).get("D")
            rec = {"image": Path(png).name, "pick_image": f"{Path(png).stem}_pick.png",
                   "distance_m": dist, "n_detections": len(frame["detections"]),
                   "camera": frame.get("camera"), "image_size": frame.get("image_size"),
                   "truth": t, "now_err_m": now_err, "order": a.order, "seed": a.seed,
                   "labels": a.labels, "scale": a.scale,
                   "candidates": [{"no": c["no"], "label": label_of(c["no"], a.labels), "x": round(c["x"], 1), "y": round(c["y"], 1),
                                   "score": c["score"], "members": c["members"],
                                   "err_m": c["err_m"], "det_id": c["det"]["id"]}
                                  for c in cands]}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if t:
                best = min(cands, key=lambda c: c["err_m"])
                print(f"{Path(png).stem:<14}{dist or '-':>5}m{len(frame['detections']):>5}"
                      f"{len(cands):>5}{now_err:>10.1f}m{best['err_m']:>12.1f}m"
                      f"{label_of(best['no'], a.labels):>8}")
            else:
                print(f"{Path(png).stem:<14}{'-':>6}{len(frame['detections']):>5}{len(cands):>5}")
    print(f"\n그림과 후보 목록: {out}")


if __name__ == "__main__":
    main()
