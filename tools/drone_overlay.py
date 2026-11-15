# -*- coding: utf-8 -*-
"""시연 영상의 드론 화면에 탐지 윤곽과 한글 자막을 그린다.

파이프라인 결과 폴더(run.py --no-merge 의 --out)만 읽는다. 대시보드와는 따로라서,
대시보드를 어떻게 찍든(replay_alerts 로 한 장씩, 또는 실시간 녹화) 그대로 쓴다.

    from drone_overlay import Overlay
    ov = Overlay(results/pipeline_scenario)
    img = ov.draw(cv2.imread(png), meta)          # meta: 프레임 json (frame_id, caption, sweep.D)

드론 화면만 영상으로 미리 보기

    python tools/drone_overlay.py results/pipeline_scenario sim/frames_scenario results/drone_preview.mp4

자막

    1줄: 단계 (신고 지점 둘레 수색 / 추정 위치로 접근)
    2줄: 그 프레임의 알람 판정 (미확정, 모호, 확정과 오차 반경)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_alerts import held, steps  # noqa: E402

def _font(names):
    """글꼴 파일을 찾는다. 저장소 assets/fonts, 사용자 글꼴 폴더, 윈도우 글꼴 폴더 순."""
    import os
    dirs = [Path(__file__).resolve().parents[1] / "assets" / "fonts",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts", Path("C:/Windows/Fonts")]
    for n in names:
        for d in dirs:
            if (d / n).is_file():
                return str(d / n)
    raise FileNotFoundError(f"글꼴 없음: {names}")


# 프리텐다드, 없으면 나눔고딕, 맑은 고딕
FONT = _font(["Pretendard-Regular.otf", "NanumGothic-Regular.ttf", "NanumGothic.ttf", "malgun.ttf"])
FONT_B = _font(["Pretendard-Bold.otf", "NanumGothic-Bold.ttf", "NanumGothicBold.ttf", "malgunbd.ttf"])
STAGE = {"SEARCH": "1. 신고 지점 주변을 돌며 연기를 찾는 중", "APPROACH": "2. 추정 위치로 다가가는 중"}
# 판정 글자색 RGB (어두운 자막 띠 위)
TAG = {"확정": (110, 210, 140), "모호": (255, 196, 80), "미확정": (190, 196, 206)}


def stage_of(meta):
    cap = (meta.get("stage") or meta.get("caption") or "").split()
    return cap[0] if cap else ""


def verdict_line(alert):
    """판정 꼬리표 옆에 쓸 문장."""
    v = alert["location_verdict"]
    r = alert["position"]["error_radius_m"]
    lu = alert.get("land_use") or []
    if v == "확정":
        return f"{short(lu[0]['name']) if lu else '용도 모름'}, 오차 반경 {r:.0f}m"
    if v == "모호":
        mix = " / ".join(f"{short(x['name'])} {x['ratio'] * 100:.0f}%" for x in lu[:2])
        return f"오차 반경 {r:.0f}m, {mix}"
    return f"연기 발견, 오차 반경 {r:.0f}m라 용도 판단 보류"


def short(name):
    """'제3종일반주거지역' -> '제3종일반주거'."""
    return name[:-2] if name.endswith("지역") else name


class Overlay:
    def __init__(self, src, hold=2):
        """hold: 판정이 이만큼 연속으로 같아야 자막의 판정을 바꾼다(경계에서 깜빡임 막기). 1이면 장마다."""
        src = Path(src)
        S = steps(src)
        self.alert = {k: al for k, al, _ in S}
        # 자막에 보일 알람 (판정이 hold 장 연속일 때만 바뀜, replay_alerts.held)
        self.shown = {k: al for (k, _, _), al in zip(S, held(S, hold))}
        self.dets = {}
        for f in (src / "detections").glob("*.json"):
            fr = json.load(open(f, encoding="utf-8"))
            self.dets[int(fr["frame_id"])] = fr.get("detections") or []

    def draw(self, img, meta):
        k = int(meta["frame_id"])
        alert = self.alert.get(k)
        H, W = img.shape[:2]
        s = H / 820
        # 탐지 윤곽: 알람이 된 탐지는 굵은 빨강, 나머지는 얇은 주황
        for d in self.dets.get(k, []):
            poly = np.array(d.get("polygon") or [], np.int32)
            if len(poly) < 3:
                continue
            main = alert is not None and d["id"] == alert["detection_id"]
            cv2.polylines(img, [poly], True, (40, 40, 230) if main else (40, 160, 240),
                          max(1, round((3 if main else 1) * s)), cv2.LINE_AA)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        g = ImageDraw.Draw(pil, "RGBA")
        f_s = ImageFont.truetype(FONT, round(20 * s))
        pad = round(14 * s)
        # 왼쪽 위: 무엇을 보는 화면인지, 그 아래 실제 발화점까지 거리
        band = round(72 * s)          # 이 높이 아래로는 연기 점수 글씨만 쓴다
        D = (meta.get("sweep") or {}).get("D")
        if D is None and meta.get("truth") and meta.get("camera"):     # 실시간 장면은 거리를 따로 안 적는다
            c, t = meta["camera"], meta["truth"]
            D = math.hypot((t["fire_lat"] - c["lat"]) * 111320,
                           (t["fire_lon"] - c["lon"]) * 111320 * math.cos(math.radians(c["lat"])))
        head = ["드론 화면 (시뮬레이터)"] + ([f"실제 발화점까지 {D:.0f}m"] if D is not None else [])
        f_h = ImageFont.truetype(FONT_B, f_s.size)                 # 첫 줄은 굵게
        bw = max(g.textlength(t, font=f_h if i == 0 else f_s) for i, t in enumerate(head))
        g.rectangle([pad // 2, pad // 2, pad + bw + pad // 2, pad + round(28 * s) * len(head)],
                    fill=(255, 255, 255, 75))                      # 두 줄 뒤에만 흰 반투명 상자
        for i, t in enumerate(head):
            g.text((pad, pad + round(28 * s) * i), t, font=f_h if i == 0 else f_s, fill=(25, 25, 25))
        # 알람이 된 탐지 위에 점수
        if alert is not None:
            d = next((x for x in self.dets.get(k, []) if x["id"] == alert["detection_id"]), None)
            if d:
                t = f"연기 {d['score']:.2f}"
                tw = g.textlength(t, font=f_s)
                x = min(max(pad, d["bbox"][0]), W - pad - tw)
                y = d["bbox"][1] - 28 * s
                if y < band + 6 * s:           # 위쪽 흰 띠와 겹치면 윤곽 안쪽 위에 쓴다
                    y = d["bbox"][1] + 6 * s
                y = max(y, band + 6 * s)
                g.text((x, y), t, font=f_s, fill=(255, 80, 80))
        # 아래 자막 띠: 1줄 단계, 2줄 판정 꼬리표와 문장. 두 줄 글씨 크기는 같다
        f_t = ImageFont.truetype(FONT, round(21 * s))
        f_tag = ImageFont.truetype(FONT_B, round(21 * s))
        stage = STAGE.get(stage_of(meta), "")
        rows = (1 if stage else 0) + (1 if alert is not None else 0)
        if rows:
            lh = round(34 * s)
            top = H - pad - lh * rows - pad // 2
            g.rectangle([0, top, W, H], fill=(0, 0, 0, 140))
            y = top + pad // 2
            if stage:
                g.text((pad, y), stage, font=ImageFont.truetype(FONT_B, f_t.size), fill=(255, 255, 255))
                y += lh
            if alert is not None:
                shown = self.shown.get(k, alert)
                v = shown["location_verdict"]
                # 판정은 색 글씨로만 쓰고, 얇은 세로선으로 뒤 문장과 나눈다
                g.text((pad, y + lh / 2), v, font=f_tag, fill=TAG[v], anchor="lm")
                x = pad + g.textlength(v, font=f_tag) + round(12 * s)
                g.line([x, y + 8 * s, x, y + lh - 8 * s], fill=(255, 255, 255, 110), width=max(1, round(s)))
                x += round(12 * s)
                line, f = verdict_line(shown), f_t
                while g.textlength(line, font=f) > W - x - pad and f.size > 12:   # 넘치면 글씨를 줄인다
                    f = ImageFont.truetype(f.path, f.size - 1)
                g.text((x, y + lh / 2), line, font=f, fill=(255, 255, 255), anchor="lm")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="run.py --no-merge 결과 폴더")
    ap.add_argument("frames", help="드론 프레임 폴더 (sim_*.png, .json)")
    ap.add_argument("out", help="mp4")
    ap.add_argument("--hold", type=float, default=0.5)
    a = ap.parse_args()
    ov = Overlay(a.src)
    fps = 10
    hold = max(1, round(a.hold * fps))
    writer = None
    for png in sorted(Path(a.frames).glob("sim_*.png")):
        meta = json.load(open(png.with_suffix(".json"), encoding="utf-8"))
        img = ov.draw(cv2.imread(str(png)), meta)
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        for _ in range(hold):
            writer.write(img)
    writer.release()
    print("->", a.out)


if __name__ == "__main__":
    main()
