# -*- coding: utf-8 -*-
"""71330 검증 라벨(VL.zip)에서 연기가 크게 찍힌 사진을 고른다 (탐지 모델 재학습 준비).

탐지 모델은 화면을 넓게 덮는 연기를 연기로 잡지 못했다(번진 산불 시험, 200m 장면).
학습 1,260장은 작은 초기 연기 위주였다. 연기 폴리곤 면적이 화면의 일정 비율 이상인
사진을 골라 원천 사진을 받을 목록을 만든다.

    python tools/select_large_smoke.py <VL.zip> docs/select_large_smoke.csv --min-ratio 0.08

원천 사진은 VS.zip(74GB, 한 파일로만 받을 수 있음)에서 목록만 꺼낸다.

    python tools/extract_71330.py docs/select_large_smoke.csv <VS.zip> <VL.zip> data/large_smoke

기준
    - PositiveRealDB, PositiveSyntheticDB 의 연기 사진 (흑색연기, 백색/회색연기)
    - 연기 폴리곤 면적 합 / 화면 면적 >= --min-ratio
    - 학습 1,260장, 평가셋 100장과 겹치는 사진, 시연·평가 지역(모래고개입구, 장현동)은 뺀다
    - 영상(camera_id + video_id) 하나에서 최대 --per-video 장. 비슷한 프레임이 몰리지 않게
"""
import argparse
import csv
import json
import random
import unicodedata
import zipfile
from collections import Counter, defaultdict
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


def used_names():
    names = set()
    for d in ["data/yolo_1260/images/train", "data/yolo_1260/images/val", "data/eval_100/images"]:
        p = ROOT / d
        if p.is_dir():
            names |= {unicodedata.normalize("NFC", f.stem) for f in p.iterdir()}
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vl_zip")
    ap.add_argument("out")
    ap.add_argument("--min-ratio", type=float, default=0.08)
    ap.add_argument("--per-video", type=int, default=3)
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    used = used_names()
    z = zipfile.ZipFile(a.vl_zip)
    by_video = defaultdict(list)
    ratios = []
    for info in z.infolist():
        if not info.filename.endswith(".json") or "__MACOSX" in info.filename:
            continue
        path = fix(info.filename)
        stem = Path(path).stem
        if not stem.startswith("Positive") or stem in used:
            continue
        if "모래고개입구" in stem or "장현동" in stem:
            continue
        lab = json.loads(z.read(info).decode("utf-8"))
        cat = {c["id"]: c["name"] for c in lab["categories"]}
        img = lab["images"][0]
        area = sum(an.get("area", 0) for an in lab["annotations"] if cat.get(an["category_id"]) in SMOKE)
        r = area / (img["width"] * img["height"])
        ratios.append(r)
        if r < a.min_ratio:
            continue
        cls = Counter(cat.get(an["category_id"]) for an in lab["annotations"]).most_common(1)[0][0]
        vid = lab["info"].get("camera_id", "") + "_" + lab["info"].get("video_id", "")
        by_video[vid].append({"image_file": stem + ".jpg", "file_name": stem + ".json", "ratio": round(r, 4),
                              "class": cls, "db": stem.split("_")[0], "video": vid,
                              "camera": "드론" if vid.startswith("RD") else "지상"})
    rnd = random.Random(a.seed)
    rows = []
    for vid in sorted(by_video):
        items = sorted(by_video[vid], key=lambda x: -x["ratio"])
        rows += items[: a.per_video]
    rnd.shuffle(rows)
    rows = sorted(rows[: a.max], key=lambda x: x["file_name"])
    # 학습/검증은 영상 단위로 80/20 (extract_71330.py 가 split 칸을 읽는다)
    vids = sorted({r["video"] for r in rows})
    rnd.shuffle(vids)
    val = set(vids[: max(1, len(vids) // 5)])
    for r in rows:
        r["split"] = "val" if r["video"] in val else "train"
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ratios.sort()
    q = lambda p: ratios[int(p * (len(ratios) - 1))]
    print(f"연기 사진 {len(ratios)}장, 면적비 중앙값 {q(0.5):.4f}, 90% {q(0.9):.4f}, 99% {q(0.99):.4f}")
    print(f"면적비 {a.min_ratio} 이상 영상 {len(by_video)}개 -> {len(rows)}장 -> {a.out}")
    print("카메라", Counter(r["camera"] for r in rows), "나누기", Counter(r["split"] for r in rows))


if __name__ == "__main__":
    main()
