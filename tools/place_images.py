# -*- coding: utf-8 -*-
"""변환한 라벨에 맞춰 이미지를 넣고, 연기 없는 사진을 배경으로 더한다.

convert_labels.py 는 라벨만 만들고 이미지는 옮기지 않는다. 또 연기가 없는 사진
(구름, 안개연무, 굴뚝연기)은 연기 클래스가 없어 통째로 빠진다. 그 사진들은 2.2
에서 오탐률을 재려고 일부러 10% 넣은 것이므로, 빈 라벨을 붙여 배경으로 넣는다.

YOLO 는 라벨이 빈 사진을 배경으로 학습해 오탐을 줄인다. 학습에 쓴 사진으로
오탐률을 재면 안 되므로 배경도 같은 비율로 나눈다.

    python place_images.py 원본이미지폴더 yolo폴더
"""
import sys, os, glob, shutil, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/71330_300/images"
YOLO = sys.argv[2] if len(sys.argv) > 2 else "data/yolo"
VAL_RATIO = 0.2
SEED = 42


def main():
    images = {os.path.basename(p): p
              for p in glob.glob(os.path.join(SRC, "*", "*.jpg"))}
    print(f"원본 사진 {len(images)}장")

    # 1) 라벨이 있는 사진을 라벨과 같은 쪽으로 옮긴다
    placed = set()
    for split in ("train", "val"):
        ldir = os.path.join(YOLO, "labels", split)
        idir = os.path.join(YOLO, "images", split)
        os.makedirs(idir, exist_ok=True)
        n = 0
        for lp in glob.glob(os.path.join(ldir, "*.txt")):
            name = os.path.splitext(os.path.basename(lp))[0] + ".jpg"
            src = images.get(name)
            if src is None:
                print(f"  사진 없음: {name}")
                continue
            shutil.copyfile(src, os.path.join(idir, name))
            placed.add(name)
            n += 1
        print(f"  {split}: 연기 사진 {n}장")

    # 2) 남은 사진은 연기가 없는 것이다. 빈 라벨을 붙여 배경으로 넣는다
    rest = sorted(set(images) - placed)
    random.Random(SEED).shuffle(rest)
    n_val = round(len(rest) * VAL_RATIO)
    for i, name in enumerate(rest):
        split = "val" if i < n_val else "train"
        shutil.copyfile(images[name],
                        os.path.join(YOLO, "images", split, name))
        open(os.path.join(YOLO, "labels", split,
                          os.path.splitext(name)[0] + ".txt"),
             "w", encoding="utf-8").close()        # 빈 라벨 = 배경
    print(f"  배경(연기 없음) {len(rest)}장: 학습 {len(rest) - n_val}, 검증 {n_val}")

    print("\n최종")
    for split in ("train", "val"):
        im = len(glob.glob(os.path.join(YOLO, "images", split, "*.jpg")))
        lb = glob.glob(os.path.join(YOLO, "labels", split, "*.txt"))
        empty = sum(1 for p in lb if os.path.getsize(p) == 0)
        print(f"  {split:5} 사진 {im:4}  라벨 {len(lb):4} (그중 배경 {empty})")


if __name__ == "__main__":
    main()
