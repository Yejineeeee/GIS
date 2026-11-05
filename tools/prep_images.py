# -*- coding: utf-8 -*-
"""GGUF 측정에 쓸 이미지를 기존 측정과 같은 크기로 맞춰 둔다.

results/vlm_measure.csv 에 쓰인 6장을 그대로 쓴다. 기존 측정은 thumbnail(1024,
1024) 으로 줄인 뒤 넣었으므로 같게 맞춘다. 파일 이름이 한글이라 명령줄에서
문제가 생기지 않도록 ASCII 이름으로 바꿔 저장하고 대응표를 남긴다.

    python prep_images.py 저장소경로 출력폴더
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

from PIL import Image

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

names = []
with open(REPO / "results/vlm_measure.csv", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        if r["image"] not in names:
            names.append(r["image"])

pool = {os.path.basename(p): p
        for p in glob.glob(str(REPO / "data/**/*.jpg"), recursive=True)}

table = []
for i, n in enumerate(names, 1):
    im = Image.open(pool[n]).convert("RGB")
    before = im.size
    im.thumbnail((1024, 1024))
    dst = OUT / f"img{i:02d}.jpg"
    im.save(dst, quality=95)
    table.append({"id": f"img{i:02d}", "원본": n,
                  "원본크기": f"{before[0]}x{before[1]}",
                  "넣는크기": f"{im.size[0]}x{im.size[1]}"})
    print(f"img{i:02d}  {before[0]}x{before[1]} -> {im.size[0]}x{im.size[1]}  {n[:50]}")

json.dump(table, open(OUT / "images.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n{len(table)}장 저장 -> {OUT}")
