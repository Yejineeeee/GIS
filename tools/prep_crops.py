# -*- coding: utf-8 -*-
"""탐지한 연기 영역만 잘라낸 이미지를 만든다 (WBS 7.2 후속).

토큰 상한을 낮추면 10초에 닿지만 연기 사진 5장 중 2장을 놓쳤다. 사진 전체를
뭉개는 방식이라 작은 연기부터 사라진다. 대신 YOLO 가 낸 연기 영역만 잘라
넣으면 토큰은 줄이면서 연기는 크게 남는다.

탐지도 자르기도 원본(3840x2160)에서 한다. 1024x576 으로 줄인 것에서 탐지하면
작은 연기가 사라진다. 960px 를 고른 이유가 그것이다(docs/train_results.md).
자른 뒤에는 전체 프레임과 같은 방식으로 thumbnail(1024, 1024) 를 건다.

여백을 여러 배로 둔다. 여백이 없으면 연기만 꽉 차서 terrain 을 답할 근거가
사라질 수 있기 때문이다.

    python prep_crops.py 저장소경로 가중치 출력폴더 [여백배수...]
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

REPO = Path(sys.argv[1])
WEIGHTS = sys.argv[2]
OUT = Path(sys.argv[3])
MARGINS = [float(x) for x in sys.argv[4:]] or [1.0, 2.0, 3.0]

CONF = 0.10          # 운영 설정 (docs/train_results.md)
IMGSZ = 960
MIN_SIDE = 96        # 너무 작게 잘리면 모델이 아무것도 못 본다

names = []
with open(REPO / "results/vlm_measure.csv", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        if r["image"] not in names:
            names.append(r["image"])
pool = {os.path.basename(p): p
        for p in glob.glob(str(REPO / "data/**/*.jpg"), recursive=True)}

model = YOLO(WEIGHTS)
out_meta = {f"{m:g}": [] for m in MARGINS}

for i, n in enumerate(names, 1):
    src = pool[n]
    im = Image.open(src).convert("RGB")
    W, H = im.size
    res = model.predict(src, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    boxes = [] if res.boxes is None else [
        [float(v) for v in b] for b in res.boxes.xyxy.tolist()]

    if boxes:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
    else:
        # 탐지가 없으면 자르지 않는다. 파이프라인에서도 알람이 안 뜨는 경우다
        x0, y0, x1, y1 = 0, 0, W, H

    line = [f"img{i:02d}  원본 {W}x{H}  탐지 {len(boxes)}개  "
            f"박스 {int(x1 - x0)}x{int(y1 - y0)}"]
    for mar in MARGINS:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max((x1 - x0) * mar, MIN_SIDE)
        h = max((y1 - y0) * mar, MIN_SIDE)
        box = (max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
               min(W, int(cx + w / 2)), min(H, int(cy + h / 2)))
        crop = im.crop(box)
        area = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
        crop.thumbnail((1024, 1024))      # 전체 프레임과 같은 처리
        tag = f"{mar:g}"
        d = OUT / f"m{tag}"
        d.mkdir(parents=True, exist_ok=True)
        crop.save(d / f"img{i:02d}.jpg", quality=95)
        out_meta[tag].append({
            "id": f"img{i:02d}", "원본": n,
            "넣는크기": f"{crop.size[0]}x{crop.size[1]}",
            "탐지수": len(boxes), "원본대비넓이": round(area, 4),
            "잘랐나": bool(boxes)})
        line.append(f"m{tag} {crop.size[0]}x{crop.size[1]}({area:.1%})")
    print("  ".join(line))

print()
for tag, items in out_meta.items():
    d = OUT / f"m{tag}"
    json.dump(items, open(d / "images.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    cut = [it for it in items if it["잘랐나"]]
    area = sum(it["원본대비넓이"] for it in cut) / len(cut) if cut else 1.0
    print(f"여백 {tag}배  자른 {len(cut)}/{len(items)}장, "
          f"그 평균 넓이가 원본의 {area:.2%}  -> {d}")
