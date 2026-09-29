# -*- coding: utf-8 -*-
"""학습 코드를 점검하기 위한 가짜 데이터셋을 만든다.

실제 이미지가 오기 전에 학습이 끝까지 도달하는지 확인하는 용도다. 성능을 보려는
것이 아니므로 그림은 단순해도 된다. 라벨은 실제와 같은 YOLO 분할 형식으로 쓴다.

    python make_fake_dataset.py 출력폴더 [장수]
"""
import sys, os, random, math
from PIL import Image, ImageDraw, ImageFilter

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/fake"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
W, H = 640, 480
VAL_RATIO = 0.25
random.seed(42)

for split in ("train", "val"):
    for kind in ("images", "labels"):
        os.makedirs(os.path.join(OUT, kind, split), exist_ok=True)


def plume(draw, cx, base_y, height, width, color):
    """연기 기둥 모양의 폴리곤을 그리고 좌표를 돌려준다."""
    left, right = [], []
    steps = 8
    for i in range(steps + 1):
        t = i / steps
        y = base_y - height * t
        w = width * (0.4 + 1.3 * t) / 2
        wobble = math.sin(t * 5 + cx) * width * 0.15
        left.append((cx - w + wobble, y))
        right.append((cx + w + wobble, y))
    poly = left + right[::-1]
    draw.polygon(poly, fill=color)
    return poly


rows = []
for i in range(N):
    sky = random.randint(140, 185)
    img = Image.new("RGB", (W, H), (sky, sky + 20, sky + 45))
    d = ImageDraw.Draw(img)
    horizon = random.randint(240, 320)
    d.rectangle([0, horizon, W, H], fill=(60 + random.randint(0, 25),
                                          90 + random.randint(0, 30), 45))
    for _ in range(250):                       # 나무 질감
        x, y = random.randint(0, W), random.randint(horizon, H)
        r = random.randint(5, 14)
        d.ellipse([x, y, x + r, y + r],
                  fill=(45 + random.randint(0, 30), 75 + random.randint(0, 35), 38))

    lines = []
    for _ in range(random.randint(1, 2)):
        cls = random.choice([0, 1])            # 0 흑색연기, 1 백색및회색연기
        tone = random.randint(40, 80) if cls == 0 else random.randint(195, 235)
        cx = random.randint(90, W - 90)
        poly = plume(d, cx, horizon + random.randint(-20, 20),
                     random.randint(120, 230), random.randint(40, 80),
                     (tone, tone, tone))
        coords = []
        for x, y in poly:
            coords += [f"{min(max(x / W, 0), 1):.6f}", f"{min(max(y / H, 0), 1):.6f}"]
        lines.append(f"{cls} " + " ".join(coords))

    img = img.filter(ImageFilter.GaussianBlur(1.4))
    split = "val" if i < int(N * VAL_RATIO) else "train"
    stem = f"fake_{i:04d}"
    img.save(os.path.join(OUT, "images", split, stem + ".jpg"), quality=88)
    with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    rows.append(split)

with open(os.path.join(OUT, "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\n\n"
            "names:\n  0: 흑색연기\n  1: 백색및회색연기\n")

print(f"학습 {rows.count('train')}장, 검증 {rows.count('val')}장")
print(f"{OUT}/dataset.yaml")
