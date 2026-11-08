# -*- coding: utf-8 -*-
"""학습 사진 수를 줄여 가며 같은 설정으로 다시 학습한다 (데이터 양 학습 곡선).

    python tools/data_size_curve.py              # 25, 50, 75% 전부
    python tools/data_size_curve.py 25 50        # 일부만

학습 1,008장에서 분류(백색회색연기, 흑색연기, 구름, 굴뚝연기, 안개연무) 비율을 지켜
뽑는다. 25% 는 50% 에, 50% 는 75% 에 포함되게 한 번 섞은 순서의 앞부분을 쓴다.
검증 252장은 그대로 둔다. 설정은 운영 모델과 같다(yolo11n-seg, 960px, 배치 8, 100에폭).

결과는 runs/size_{n}/ 에 쌓이고, 학습이 끝나면 val_252 와 eval_100 을 conf 0.10 으로
잰다(runs/size_{n}_val_252, runs/size_{n}_eval_100).
"""
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo_1260"
PY = sys.executable


def make_subset(pct):
    imgs = sorted((DATA / "images" / "train").glob("*.jpg"))
    by_cls = defaultdict(list)
    for p in imgs:
        by_cls[p.name.split("_")[1]].append(p)
    rng = random.Random(0)
    keep = []
    for cls in sorted(by_cls):
        lst = by_cls[cls][:]
        rng.shuffle(lst)
        keep += lst[:round(len(lst) * pct / 100)]
    lst_file = DATA / f"train_{pct}.txt"
    lst_file.write_text("\n".join(str(p).replace("\\", "/") for p in keep) + "\n", encoding="utf-8")
    yaml = DATA / f"dataset_{pct}.yaml"
    yaml.write_text(f"path: {DATA.as_posix()}\ntrain: train_{pct}.txt\nval: images/val\n\n"
                    "names:\n  0: 연기\n", encoding="utf-8")
    counts = {c: sum(1 for p in keep if p.name.split("_")[1] == c) for c in sorted(by_cls)}
    print(f"{pct}%: {len(keep)}장 {counts}", flush=True)
    return yaml


def train(pct, yaml):
    from ultralytics import YOLO
    name = f"size_{pct}"
    csv = ROOT / "runs" / name / "results.csv"
    if csv.is_file() and len(csv.read_text(encoding="utf-8").strip().splitlines()) > 100:
        print(f"{name} 이미 있음, 건너뜀", flush=True)
        return
    last = ROOT / "runs" / name / "weights" / "last.pt"
    if last.is_file():
        # 끊긴 학습은 마지막으로 저장된 에폭 다음부터 잇는다
        print(f"{name} 이어서 학습", flush=True)
        YOLO(str(last)).train(resume=True)
        return
    YOLO("yolo11n-seg.pt").train(data=str(yaml), epochs=100, imgsz=960, batch=8, device=0,
                                 workers=4, seed=0, deterministic=True,
                                 project=str(ROOT / "runs"), name=name, exist_ok=True,
                                 plots=False, verbose=False)


def evaluate(pct):
    w = ROOT / "runs" / f"size_{pct}" / "weights" / "best.pt"
    for ev in ("val_252", "eval_100"):
        subprocess.run([PY, "tools/eval_detect.py", f"data/{ev}", str(w), "--conf", "0.10",
                        "--out", f"runs/size_{pct}_{ev}"], cwd=ROOT, check=True)


def main():
    pcts = [int(x) for x in sys.argv[1:]] or [25, 50, 75]
    for pct in pcts:
        yaml = make_subset(pct)
        train(pct, yaml)
        evaluate(pct)


if __name__ == "__main__":
    main()
