# -*- coding: utf-8 -*-
"""학습한 분할 모델로 탐지 결과 JSON을 만든다 (WBS 2.5).

출력 형식은 schema/detection.example.json 과 같고 `tools/check_schema.py` 로
검사할 수 있다.

    python predict_seg.py 가중치 이미지폴더 [출력폴더] [배치위치csv]

카메라 값에 대하여. 71330 사진에는 촬영 좌표가 없다. 라벨의 gps 는 장소마다
하나씩이고 절반이 같은 기본값이라 쓸 수 없다(docs/data_71330.md). 그래서 배치
위치 csv 를 받아 좌표를 붙이고 `source` 를 `가상` 으로 둔다. 실제 드론 메타가
들어오면 그 값을 쓰고 `실제` 로 바꾼다.

좌표 말고 나머지(고도, 기수 방향, 화각)는 데이터에 없으므로 아래 가정값을 쓴다.
스키마가 null 을 받지 않으므로 비워 둘 수 없다. 어떤 가정을 썼는지 화면에 찍는다.
가정값은 팀 결정(10/17)과 src/geo/transform.py 와 같게 맞춘다.

Windows 주의. main 가드를 지우면 데이터 로더가 이 파일을 다시 실행한다.
"""
import sys, os, json, time, statistics
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

CLASS_NAME = "연기"       # 스키마는 연기 한 종류로 둔다
# 탐지 문턱. 0.25 가 아니라 0.10 이다. 재난 상황에서는 놓치는 쪽이 헛걸음보다
# 나쁘고, 헛탐지는 뒤의 지리 결합과 사람 확인으로 거를 수 있다. 0.25 에서 0.10 으로
# 내리면 작은 연기 탐지율이 77%에서 96%로 오르고, 늘어나는 헛탐지는 굴뚝연기와
# 안개연무뿐이다. 구름 26장은 한 장도 틀리지 않는다. docs/train_results.md 참고.
CONF = float(os.environ.get("DET_CONF", "0.10"))
SCHEMA_VERSION = "0.3"

# 카메라 가정값. 데이터에 없는 값이고 스키마가 null 을 받지 않아 채워 넣는다.
ALT_M = 120.0           # 지면 기준. 팀 결정(10/17)
ALT_REF = "지면"
PITCH_DEG = -60.0       # 수직이 -90. 수직에서 30도 기울인 값, 팀 결정(10/17)
HEADING_DEG = 0.0       # 기수 방향은 모른다. 진북으로 둔다
HFOV_DEG = 73.7         # docs/gsd.md 예시 카메라


def main():
    from ultralytics import YOLO

    WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "runs/smoke/weights/best.pt"
    SRC = sys.argv[2] if len(sys.argv) > 2 else "data/pilot"
    OUT = sys.argv[3] if len(sys.argv) > 3 else "results/detections"
    POS = sys.argv[4] if len(sys.argv) > 4 else "docs/demo_positions.csv"

    # 배치 위치를 읽는다. 없으면 시연 지역 중심 한 곳만 쓴다.
    spots = []
    if os.path.exists(POS):
        import csv as _csv
        with open(POS, encoding="utf-8-sig") as f:
            for r in _csv.DictReader(f):
                try:
                    spots.append((float(r["lat"]), float(r["lon"]),
                                  r.get("pos_id", "")))
                except (KeyError, TypeError, ValueError):
                    continue
    if not spots:
        spots = [(37.736790, 128.906239, "site")]   # 모래고개입구
    print(f"배치 위치 {len(spots)}곳 ({POS if os.path.exists(POS) else '기본값'})")
    print(f"가정: 고도 {ALT_M}m({ALT_REF}), 기울기 {90 + PITCH_DEG}도, "
          f"기수 {HEADING_DEG}도, 화각 {HFOV_DEG}도")
    print("가정: 사진에 촬영 좌표가 없어 배치 위치를 돌려 가며 붙인다. source 는 가상")
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
            # 박스와 마스크는 YOLO 가 따로 내므로 서로 조금 어긋난다. 폴리곤이
            # 있으면 그것을 감싸는 박스를 쓴다. 둘이 다르면 뒤에서 대표 픽셀을
            # 고를 때 어느 쪽을 썼느냐에 따라 좌표가 달라진다.
            if poly:
                xs = [q[0] for q in poly]
                ys = [q[1] for q in poly]
                bx, by = min(xs), min(ys)
                bw, bh = max(xs) - bx, max(ys) - by
            else:
                bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            area_px = int(round(area))          # 스키마는 정수로 받는다
            dets.append({
                "id": f"{VIDEO_ID}-{i}-{j}",
                "class": CLASS_NAME,
                "score": round(float(b.conf[0]), 3),
                "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                "polygon": poly or None,
                "area_px": area_px,
                "area_ratio": round(area_px / (w * h), 5),
            })

        spot = spots[i % len(spots)]
        out = {
            "schema_version": SCHEMA_VERSION,
            "video_id": VIDEO_ID,
            "frame_id": i,
            "time": datetime.now(KST).isoformat(timespec="milliseconds"),
            "image": name,
            "image_size": {"width": w, "height": h},
            "detections": dets,
            "camera": {"lat": spot[0], "lon": spot[1], "alt_m": ALT_M,
                       "alt_ref": ALT_REF, "heading_deg": HEADING_DEG,
                       "pitch_deg": PITCH_DEG, "hfov_deg": HFOV_DEG,
                       "source": "가상"},
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
