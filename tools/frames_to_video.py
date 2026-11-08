# -*- coding: utf-8 -*-
"""시뮬레이터 비행 프레임(sim_0000.png ...)을 mp4 로 묶는다 (시연 영상용).

    python tools/frames_to_video.py sim/frames_flight flight.mp4 --fps 12
"""
import argparse
import glob
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(str(Path(a.frames) / "sim_*.png")))
    if not files:
        raise SystemExit("프레임이 없다")
    h, w = cv2.imread(files[0]).shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for f in files:
        img = cv2.imread(f)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()
    print(f"{len(files)}장 -> {a.out} ({len(files) / a.fps:.1f}초, {w}x{h})")


if __name__ == "__main__":
    main()
