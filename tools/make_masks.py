# -*- coding: utf-8 -*-
"""YOLO 분할 라벨을 의미 분할용 마스크 PNG 로 바꾼다.

YOLO-seg 는 개체 분할이라 연기를 하나씩 센다. 의미 분할은 픽셀마다 연기인지만
정한다. 같은 데이터로 두 방식을 비교하려면 라벨 형식을 맞춰야 한다.

폴리곤이 여러 개여도 한 장에 모두 칠한다. 개체 구분이 사라지는 것이 의미 분할의
성질이고, 우리 쓰임새에서도 연기 개수는 쓰지 않는다.

    python make_masks.py yolo폴더 출력폴더 [크기]

출력
    <출력 폴더>/images/train/*.jpg   원본 사진 (크기를 줄여 저장)
    <출력 폴더>/masks/train/*.png    0 배경, 1 연기
"""
import glob
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/yolo_1260"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/seg_1260"
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 960


def main():
    from PIL import Image, ImageDraw

    for split in ("train", "val"):
        for kind in ("images", "masks"):
            os.makedirs(os.path.join(OUT, kind, split), exist_ok=True)

        n_img = n_poly = n_empty = 0
        for lp in sorted(glob.glob(os.path.join(SRC, "labels", split, "*.txt"))):
            stem = os.path.splitext(os.path.basename(lp))[0]
            ip = os.path.join(SRC, "images", split, stem + ".jpg")
            if not os.path.exists(ip):
                continue
            img = Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            img.save(os.path.join(OUT, "images", split, stem + ".jpg"), quality=92)

            mask = Image.new("L", (SIZE, SIZE), 0)
            d = ImageDraw.Draw(mask)
            lines = [l for l in open(lp, encoding="utf-8") if l.strip()]
            for line in lines:
                v = [float(x) for x in line.split()[1:]]
                pts = [(v[i] * SIZE, v[i + 1] * SIZE) for i in range(0, len(v) - 1, 2)]
                if len(pts) >= 3:
                    d.polygon(pts, fill=1)
                    n_poly += 1
            if not lines:
                n_empty += 1
            mask.save(os.path.join(OUT, "masks", split, stem + ".png"))
            n_img += 1
        print(f"  {split}: 사진 {n_img}장, 폴리곤 {n_poly}개, 배경 {n_empty}장")

    print(f"\n{SIZE}x{SIZE} 로 저장했다. 마스크는 0 배경, 1 연기다.")


if __name__ == "__main__":
    main()
