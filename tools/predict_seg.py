# -*- coding: utf-8 -*-
"""학습한 분할 모델로 탐지 결과 JSON을 만든다.

출력 형식은 schema/detection.example.json 과 같다. 카메라 좌표는 드론 메타에서
채워야 하므로 여기서는 인자로 받거나 비워 둔다.

    python predict_seg.py 가중치 이미지폴더 [출력폴더]

Windows 주의. main 가드를 지우면 데이터 로더가 이 파일을 다시 실행한다.
"""
import sys, os, json, time, statistics
from datetime import datetime

CLASSES = {0: "흑색연기", 1: "백색및회색연기"}
CONF = float(os.environ.get("DET_CONF", "0.25"))


def main():
    from ultralytics import YOLO

    WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "runs/smoke/weights/best.pt"
    SRC = sys.argv[2] if len(sys.argv) > 2 else "data/pilot"
    OUT = sys.argv[3] if len(sys.argv) > 3 else "results/detections"
    os.makedirs(OUT, exist_ok=True)

    files = sorted(f for f in os.listdir(SRC)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"{SRC} 에 이미지가 없습니다.")
    print(f"이미지 {len(files)}장")

    model = YOLO(WEIGHTS)
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
                "id": f"d{j+1}",
                "class": CLASSES.get(int(b.cls[0]), str(int(b.cls[0]))),
                "score": round(float(b.conf[0]), 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "polygon": poly,
                "area_px": round(area, 1),
                "area_ratio": round(area / (w * h), 5),
            })

        out = {
            "frame_id": i,
            "time": datetime.now().isoformat(timespec="seconds"),
            "image": name,
            "detections": dets,
            "camera": {"lat": None, "lon": None, "alt_m": None, "angle_deg": None},
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
