# -*- coding: utf-8 -*-
"""학습 사진 수를 줄여 가며 같은 설정으로 다시 학습한다 (데이터 양 학습 곡선).

    python tools/data_size_curve.py              # 25, 50, 75% 전부
    python tools/data_size_curve.py 25 50        # 일부만
    python tools/data_size_curve.py plot         # 결과 그림 (results/data_size_curve.png)

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


def plot():
    """25, 50, 75% 와 운영 모델(100%)의 mAP50, 사진 재현율, 헛경보를 그린다."""
    import csv
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    def op_row(path, conf=0.10):
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        return min(rows, key=lambda r: abs(r["conf"] - conf))

    n = [252, 504, 756, 1008]
    seg, v_rec, v_fa, e_rec, e_fa = [], [], [], [], []
    for p in (25, 50, 75, 100):
        ev = "cmp_smoke_1260_960" if p == 100 else f"size_{p}"
        if p == 100:
            seg.append(0.559)  # 운영 모델 best.pt 88에폭
        else:
            rows = list(csv.DictReader(open(ROOT / "runs" / f"size_{p}" / "results.csv")))
            k = [c for c in rows[0] if "mAP50(M)" in c][0]
            seg.append(max(float(r[k]) for r in rows))
        v = op_row(ROOT / "runs" / f"{ev}_val_252" / "eval_detect.json")
        e = op_row(ROOT / "runs" / f"{ev}_eval_100" / "eval_detect.json")
        v_rec.append(100 * v["recall_img"])
        v_fa.append(100 * v["bg_alarm"][0] / v["bg_alarm"][1])
        e_rec.append(100 * e["recall_img"])
        e_fa.append(100 * e["bg_alarm"][0] / e["bg_alarm"][1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(n, seg, "o-", color="#2C2C2A")
    axes[0].set_title("검증 분할 mAP50 (val 252장)")
    for ax, (a, b), title in [(axes[1], (v_rec, e_rec), "연기 사진을 찾은 비율 (%)"),
                              (axes[2], (v_fa, e_fa), "연기 없는 사진 헛경보 (%)")]:
        ax.plot(n, a, "o-", color="#2C2C2A", label="val 252장")
        ax.plot(n, b, "s--", color="#888780", label="평가 100장")
        ax.set_title(title)
        ax.legend(fontsize=9)
    for ax, ys in zip(axes, [[seg], [v_rec, e_rec], [v_fa, e_fa]]):
        for i, y in enumerate(ys):
            for x, val in zip(n, y):
                ax.annotate(f"{val:.3f}" if ax is axes[0] else f"{val:.0f}", (x, val),
                            xytext=(0, 7 if i == 0 else -14), textcoords="offset points",
                            ha="center", fontsize=9)
        ax.set_xticks(n)
        ax.set_xlabel("학습 사진 수")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("학습 사진 수에 따른 성능 (같은 설정, conf 0.10)", fontsize=13)
    fig.tight_layout()
    out = ROOT / "results" / "data_size_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


def main():
    if sys.argv[1:] == ["plot"]:
        plot()
        return
    pcts = [int(x) for x in sys.argv[1:]] or [25, 50, 75]
    for pct in pcts:
        yaml = make_subset(pct)
        train(pct, yaml)
        evaluate(pct)


if __name__ == "__main__":
    main()
