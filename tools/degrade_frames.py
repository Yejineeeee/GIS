# -*- coding: utf-8 -*-
"""시뮬레이터 프레임을 실제 드론 사진처럼 거칠게 만들어 탐지와 위치 오차를 다시 잰다.

시뮬레이터 캡처는 압축, 흐림, 노출 변화가 없는 깨끗한 그림이다. 실제 드론 사진은
전송 압축, 흔들림, 역광, 연무가 섞인다. 같은 프레임에 열화를 하나씩 넣어 YOLO 가
여전히 발화점의 연기를 잡는지, conf 최고 탐지(지금 파이프라인)의 위치 오차가
어떻게 바뀌는지 본다. 카메라 값과 정답은 그대로 둔다.

    python tools/degrade_frames.py sim/frames runs/smoke_1260_960/weights/best.pt

열화 종류 (--only 로 고를 수 있음)
    원본      그대로
    반해상도  0.5 배로 줄였다가 되돌림 (먼 거리 / 낮은 화소)
    JPEG30    품질 30 압축 (영상 전송)
    흐림      가우시안 반경 2px (흔들림, 초점)
    어둡게    밝기 0.6 배
    밝게      밝기 1.4 배 (역광, 과노출)
    연무      흰색 25% 섞기 (대기 연무)
    종합      반해상도 + JPEG40 + 흐림 1px

"연기 잡음" 은 탐지 중 하나라도 밑동이 정답에서 60m 안에 들어온 경우로 센다.
지형은 sim/terrain.json 을 쓴다.
"""
import argparse
import glob
import io
import json
import math
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from transform import to_position  # noqa: E402

FOUND_M = 60.0


def _jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _half(im):
    w, h = im.size
    return im.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)


VARIANTS = {
    "원본": lambda im: im,
    "반해상도": _half,
    "JPEG30": lambda im: _jpeg(im, 30),
    "흐림": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "어둡게": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "밝게": lambda im: ImageEnhance.Brightness(im).enhance(1.4),
    "연무": lambda im: Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.25),
    "종합": lambda im: _jpeg(_half(im), 40).filter(ImageFilter.GaussianBlur(1)),
}


def meters(la1, lo1, la2, lo2):
    R = 6378137.0
    dy = math.radians(la2 - la1) * R
    dx = math.radians(lo2 - lo1) * R * math.cos(math.radians(la1))
    return math.hypot(dx, dy)


def main():
    ap = argparse.ArgumentParser(description="프레임 화질 열화 후 탐지와 위치 오차")
    ap.add_argument("frames")
    ap.add_argument("weights")
    ap.add_argument("--out", default="runs/degrade")
    ap.add_argument("--only", nargs="*", help="이 열화만 돌린다")
    a = ap.parse_args()

    from detect import Detector
    from sim_terrain import SimTerrain
    ter = SimTerrain(str(ROOT / "sim" / "terrain.json"))
    det = Detector(a.weights)
    pngs = sorted(glob.glob(str(Path(a.frames) / "*.png")))
    names = a.only or list(VARIANTS)

    print(f"{'열화':<8}{'연기 잡음':>10}{'탐지 수':>8}{'conf 최고 평균':>14}{'지금 방식 오차 중앙':>18}{'평균':>9}")
    summary = {}
    for name in names:
        vd = Path(a.out) / name
        vd.mkdir(parents=True, exist_ok=True)
        found, ndet, tops, errs, rows = 0, [], [], [], []
        for i, png in enumerate(pngs):
            dst = vd / Path(png).name
            VARIANTS[name](Image.open(png).convert("RGB")).save(dst)
            shutil.copy(Path(png).with_suffix(".json"), dst.with_suffix(".json"))
            frame, _ = det.run(dst, i)
            t = json.load(open(dst.with_suffix(".json"), encoding="utf-8"))["truth"]
            ds = frame["detections"]
            ndet.append(len(ds))
            e_all = []
            for d in ds:
                try:
                    pos, _ = to_position(frame, d, dem=ter)
                    e_all.append(meters(pos["lat"], pos["lon"], t["fire_lat"], t["fire_lon"]))
                except ValueError:
                    e_all.append(float("inf"))
            hit = any(e <= FOUND_M for e in e_all)
            found += hit
            top_e = None
            if ds:
                k = max(range(len(ds)), key=lambda j: ds[j]["score"])
                tops.append(ds[k]["score"])
                top_e = e_all[k]
                errs.append(top_e)
            rows.append({"image": Path(png).name, "n_det": len(ds), "found": hit,
                         "top_score": ds[k]["score"] if ds else None,
                         "top_err_m": None if top_e is None else round(top_e, 1)})
        with open(vd / "result.jsonl", "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        fin = [e for e in errs if math.isfinite(e)]
        summary[name] = (found, len(pngs))
        print(f"{name:<8}{found:>6}/{len(pngs):<3}{st.mean(ndet):>8.1f}{st.mean(tops) if tops else 0:>14.2f}"
              f"{st.median(fin) if fin else float('nan'):>17.1f}m{st.mean(fin) if fin else float('nan'):>8.1f}m")
    print(f"\n열화 사진과 결과: {a.out}/<열화>/")


if __name__ == "__main__":
    main()
