# -*- coding: utf-8 -*-
"""학습한 분할 모델로 탐지 결과 JSON을 만든다.

출력 형식은 schema/detection.example.json 과 같다. 카메라 좌표는 드론 메타에서
채워야 하므로 여기서는 인자로 받거나 비워 둔다.

    python predict_seg.py 가중치 이미지폴더 [출력폴더]

Windows 주의. main 가드를 지우면 데이터 로더가 이 파일을 다시 실행한다.
"""
import sys, os, json, time, statistics
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

CLASS_NAME = "연기"       # 스키마 v0.2 는 클래스를 한 종류로 둔다
CONF = float(os.environ.get("DET_CONF", "0.25"))
SCHEMA_VERSION = "0.2"


def main():
    from ultralytics import YOLO

    WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "runs/smoke/weights/best.pt"
    SRC = sys.argv[2] if len(sys.argv) > 2 else "data/pilot"
    OUT = sys.argv[3] if len(sys.argv) > 3 else "results/detections"
    VIDEO_ID = os.environ.get("VIDEO_ID", "V0000")
    os.makedirs(OUT, exist_ok=True)

    files = sorted(f for f in os.listdir(SRC)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"{SRC} 에 이미지가 없습니다.")
    print(f"이미지 {len(files)}장")

    model = YOLO(WEIGHTS)
    if len(model.names) > 1:
        print(f"  주의: 가중치의 클래스가 {len(model.names)}개다 {model.names}. "
              f"스키마는 한 종류이므로 모두 {CLASS_NAME} 으로 낸다.")
    times = []
    for i, name in enumerate(files):
        path = os.path.join(SRC, name)
        t0 = time.time()
        r = model.predict(path, conf=CONF, verbose=False)[0]
        sec = time.time() - t0
        times.append(sec)

        h, w = r.orig_shape
        dets = []
        for j in range(len(r.boxes)):
            b = r.boxes[j]
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            poly = []
            if r.masks is not None and j < len(r.masks):
                poly = [[round(float(px), 1), round(float(py), 1)]
                        for px, py in r.masks.xy[j]]
            area = ((x2 - x1) * (y2 - y1)) if not poly else _poly_area(poly)
            dets.append({
                "id": f"{VIDEO_ID}-{i}-{j}",
                "class": CLASS_NAME,
                "score": round(float(b.conf[0]), 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "polygon": poly or None,
                "area_px": round(area, 1),
                "area_ratio": round(area / (w * h), 5),
            })

        # 카메라 값은 드론 메타에서 채운다. 여기서는 자리만 두고 source 를 virtual 로 둔다.
        out = {
            "schema_version": SCHEMA_VERSION,
            "video_id": VIDEO_ID,
            "frame_id": i,
            "time": datetime.now(KST).isoformat(timespec="seconds"),
            "image": name,
            "image_size": {"width": w, "height": h},
            "detections": dets,
            "camera": {"lat": None, "lon": None, "alt_m": None, "alt_ref": None,
                       "heading_deg": None, "pitch_deg": None, "hfov_deg": None,
                       "source": "virtual"},
            "latency_ms": {"detect": round(sec * 1000)},
        }
        with open(os.path.join(OUT, os.path.splitext(name)[0] + ".json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  {name}  탐지 {len(dets)}개  {sec*1000:.0f}ms")

    # 첫 장은 모델을 데우는 시간이 섞이므로 따로 적는다.
    print("")
    print(f"첫 장 {times[0]*1000:.0f}ms")
    if len(times) > 1:
        rest = times[1:]
        print(f"이후 평균 {sum(rest)/len(rest)*1000:.0f}ms/장   "
              f"중앙값 {statistics.median(rest)*1000:.0f}ms")
    print(f"저장 {OUT}")


def _poly_area(poly):
    """신발끈 공식으로 폴리곤 넓이를 구한다."""
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


if __name__ == "__main__":
    main()
