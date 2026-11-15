# -*- coding: utf-8 -*-
"""기존 학습셋(yolo_1260)에 큰 연기 사진(large_smoke)을 섞은 학습셋을 만든다.

큰 연기 사진은 extract_71330.py 로 꺼낸 원천 사진을 쓰고, 라벨은 VL.zip 에서 바로 읽어
YOLO 분할 형식으로 바꾼다. convert_labels.py 는 드론 사진만 남기고 나누기를 새로 하므로
쓰지 않는다. 지상 카메라 사진도 넣고, 나누기는 선별 목록(select_large_smoke.csv)의 split 을 따른다.

    python tools/make_mix_dataset.py docs/select_large_smoke.csv <VL.zip> data/large_smoke data/yolo_mix

기존 1,260장의 나누기(학습 1,008 / 검증 252)는 그대로 둔다. 그래서 검증 252장 점수로
옛 모델과 새 모델을 바로 비교할 수 있다.
"""
import csv
import json
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = {"흑색연기", "백색/회색연기"}


def fix(name):
    """zip 안 한글 이름 깨짐(cp437) 복구 + NFC."""
    try:
        name = name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize("NFC", name)


def polygon(ann):
    seg = ann.get("segmentation")
    if seg:
        flat = seg[0] if isinstance(seg[0], list) else seg
        if flat and isinstance(flat[0], (list, tuple)):
            return [(float(p[0]), float(p[1])) for p in flat]
        return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]
    return None


def to_lines(lab):
    cat = {c["id"]: c["name"] for c in lab["categories"]}
    W, H = lab["images"][0]["width"], lab["images"][0]["height"]
    lines = []
    for an in lab["annotations"]:
        if cat.get(an["category_id"]) not in SMOKE:
            continue
        poly = polygon(an)
        if not poly or len(poly) < 3:
            continue
        xy = []
        for x, y in poly:
            xy += [f"{min(max(x / W, 0), 1):.6f}", f"{min(max(y / H, 0), 1):.6f}"]
        lines.append("0 " + " ".join(xy))
    return lines


def main():
    sel, vl_zip, large_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    rows = list(csv.DictReader(open(sel, encoding="utf-8-sig")))
    want = {unicodedata.normalize("NFC", r["file_name"]): r["split"] for r in rows}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1) 기존 1,260장 그대로
    base = ROOT / "data/yolo_1260"
    n_base = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for img in (base / "images" / split).iterdir():
            # 라벨 파일이 없는 사진은 연기 없는 배경 사진이다. 그대로 넣는다
            lab = base / "labels" / split / (img.stem + ".txt")
            shutil.copy2(img, out / "images" / split / img.name)
            if lab.exists():
                shutil.copy2(lab, out / "labels" / split / lab.name)
            n_base[split] += 1

    # 2) 큰 연기 사진: 라벨은 VL.zip 에서
    n_new = {"train": 0, "val": 0}
    missing = []
    z = zipfile.ZipFile(vl_zip)
    for info in z.infolist():
        name = Path(fix(info.filename)).name
        if name not in want:
            continue
        split = want.pop(name)
        stem = Path(name).stem
        img = large_dir / "images" / split / (stem + ".jpg")
        if not img.exists():
            missing.append(stem)
            continue
        lines = to_lines(json.loads(z.read(info).decode("utf-8")))
        if not lines:
            missing.append(stem)
            continue
        # 이름 앞에 표시를 붙여 기존 사진과 구분한다
        shutil.copy2(img, out / "images" / split / ("L_" + img.name))
        (out / "labels" / split / ("L_" + stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_new[split] += 1

    (out / "dataset.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: 연기\n",
        encoding="utf-8")
    print(f"기존 학습 {n_base['train']} 검증 {n_base['val']}")
    print(f"큰 연기 학습 {n_new['train']} 검증 {n_new['val']}")
    print(f"라벨을 못 찾음 {len(want)}, 사진이나 연기 폴리곤 없음 {len(missing)}")
    print(f"-> {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
