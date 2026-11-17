# -*- coding: utf-8 -*-
"""파이프라인 안에서 분할 모델로 탐지한다 (2.5 탐지 → 5.2 연결).

tools/predict_seg.py 와 같은 규칙으로 탐지 결과 JSON(docs/schema.md)을 만든다.
다른 점은 카메라 값을 어디서 가져오는가 하나다.

    - 사진 옆에 같은 이름의 JSON(시뮬레이터 프레임)이 있으면 그 camera 블록을 쓴다.
      truth(연기를 피운 정답 좌표)가 있으면 따로 돌려줘 오차를 잴 수 있게 한다.
    - 없으면 predict_seg.py 의 가정값(고도 120m, 기울기 30도 등)과 시연 지역 좌표를 쓴다.

추론 설정: imgsz 960, conf 0.10 (이예진 팀원 최종 설정, docs/train_results.md)
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import predict_seg as ps   # noqa: E402  가정값과 넓이 계산을 한 곳에서 가져온다

KST = timezone(timedelta(hours=9))
IMGSZ = 960
SITE = (37.736790, 128.906239)   # 모래고개입구 (predict_seg.py 기본값과 같음)


class Detector:
    def __init__(self, weights, conf=ps.CONF, imgsz=IMGSZ):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz
        # 첫 추론은 모델 준비 시간(약 10초)이 붙어 탐지 시간이 튄다. 빈 사진으로 한 번 미리 돌린다
        import numpy as np
        self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), conf=conf, imgsz=imgsz, verbose=False)

    def run(self, image_path, index, video_id="V0000"):
        """사진 한 장 → (탐지 결과 frame, truth 또는 None)."""
        image_path = Path(image_path)
        meta = {}
        side = image_path.with_suffix(".json")
        if side.is_file():
            with open(side, encoding="utf-8") as f:
                meta = json.load(f)

        t0 = time.perf_counter()
        r = self.model.predict(str(image_path), conf=self.conf, imgsz=self.imgsz,
                               verbose=False)[0]
        sec = time.perf_counter() - t0

        h, w = r.orig_shape
        vid = meta.get("video_id", video_id)
        fid = meta.get("frame_id", index)
        dets = []
        for j in range(len(r.boxes)):
            b = r.boxes[j]
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            poly = []
            if r.masks is not None and j < len(r.masks):
                poly = [[round(float(px), 1), round(float(py), 1)] for px, py in r.masks.xy[j]]
            if poly:   # predict_seg.py 와 같이 폴리곤을 감싸는 박스를 쓴다
                xs, ys = [q[0] for q in poly], [q[1] for q in poly]
                bx, by, bw, bh = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                area = ps._poly_area(poly)
            else:
                bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
                area = bw * bh
            area_px = int(round(area))
            dets.append({
                "id": f"{vid}-{fid}-{j}",
                "class": ps.CLASS_NAME,
                "score": round(float(b.conf[0]), 3),
                "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                "polygon": poly or None,
                "area_px": area_px,
                "area_ratio": round(area_px / (w * h), 5),
            })

        camera = meta.get("camera") or {
            "lat": SITE[0], "lon": SITE[1], "alt_m": ps.ALT_M, "alt_ref": ps.ALT_REF,
            "heading_deg": ps.HEADING_DEG, "pitch_deg": ps.PITCH_DEG,
            "hfov_deg": ps.HFOV_DEG, "source": "가상"}
        frame = {
            "schema_version": ps.SCHEMA_VERSION,
            "video_id": vid,
            "frame_id": fid,
            "time": datetime.now(KST).isoformat(timespec="milliseconds"),
            "image": str(image_path),
            "image_size": {"width": w, "height": h},
            "detections": dets,
            "camera": camera,
            "latency_ms": {"detect": round(sec * 1000)},
        }
        return frame, meta.get("truth")
