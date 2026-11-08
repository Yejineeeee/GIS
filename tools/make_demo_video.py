# -*- coding: utf-8 -*-
"""시연 영상: 왼쪽 드론 화면, 오른쪽 대시보드. 드론이 다가갈수록 같은 알람이 갱신된다.

1. 드론 프레임을 run.py --no-merge 로 돌린 결과(src)에서 프레임마다 알람을 골라
   tools/replay_alerts.py 로 대시보드 폴더(dst)에 한 단계씩 쓴다
2. 단계마다 크롬 헤드리스로 대시보드(localhost:8800)를 찍는다
3. 드론 프레임과 대시보드 화면을 나란히 붙여 mp4 로 묶는다

대시보드는 미리 띄워 둔다.

    python src/dashboard/server.py --out results/demo_replay --frames sim/frames_approach3 \
        --geo-dir data/geo_jagok --center 37.4710 127.1034 --site-name "자곡동 · 서울 강남구"
    python tools/make_demo_video.py results/pipeline_approach3 results/demo_replay \
        sim/frames_approach3 results/demo_approach.mp4

탐지가 없는 앞쪽 프레임은 알람 없이 드론 화면만 보인다.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from replay_alerts import steps  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def shoot(url, out, w=1280, h=820):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={w},{h}", "--virtual-time-budget=5000",
                    f"--screenshot={out}", url], check=True, capture_output=True)


def put_text(img, text, org, scale=0.9):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8800/")
    ap.add_argument("--hold", type=float, default=0.5, help="한 프레임을 보여 줄 초")
    a = ap.parse_args()

    src, dst, frames = Path(a.src), Path(a.dst), Path(a.frames)
    shots = dst / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    S = steps(src)
    by_frame = {k: i for i, (k, _, _) in enumerate(S)}
    pngs = sorted(frames.glob("sim_*.png"))

    H = 820
    fps = 10
    hold = max(1, round(a.hold * fps))
    writer = None
    # 첫 알람 전에도 대시보드(지도, 신고 지점)를 보여 준다. dst 는 비어 있어야 한다
    first = shots / "dash_start.png"
    shoot(a.url, str(first.resolve()))
    last_dash = cv2.imread(str(first))
    for k, png in enumerate(pngs):
        meta = json.load(open(png.with_suffix(".json"), encoding="utf-8"))
        if k in by_frame:
            subprocess.run([sys.executable, str(ROOT / "tools" / "replay_alerts.py"), str(src), str(dst),
                            "--step", str(by_frame[k])], check=True, capture_output=True)
            shot = shots / f"dash_{k:04d}.png"
            shoot(a.url, str(shot.resolve()))
            last_dash = cv2.imread(str(shot))
        drone = cv2.imread(str(png))
        drone = cv2.resize(drone, (int(drone.shape[1] * H / drone.shape[0]), H))
        put_text(drone, meta.get("caption") or f"drone {meta['sweep']['D']:.0f} m", (20, 45))
        dash = last_dash if last_dash is not None else np.full((H, 1280, 3), 24, np.uint8)
        dash = cv2.resize(dash, (int(dash.shape[1] * H / dash.shape[0]), H))
        frame = np.hstack([drone, dash])
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        for _ in range(hold):
            writer.write(frame)
        print(k, meta["sweep"]["D"], "알람" if k in by_frame else "-", flush=True)
    writer.release()
    print(f"-> {a.out} ({len(pngs) * hold / fps:.1f}초)")


if __name__ == "__main__":
    main()
