# -*- coding: utf-8 -*-
"""분할 모델의 탐지 성능을 평가셋으로 잰다 (WBS 6.4: IoU, 재현율, 오탐률).

    python tools/eval_detect.py data/eval_100 runs/smoke_1260_960/weights/best.pt
    python tools/eval_detect.py data/eval_100 best.pt --conf 0.05 0.10 0.25 0.5

평가셋은 71330 원본 라벨(COCO 형식)을 그대로 쓴다. images/eval/*.jpg 와
labels/eval/*.json 이 같은 이름이어야 한다.

정답
    연기  라벨 분류 흑색연기, 백색/회색연기, 화염(1, 2, 3)의 폴리곤
    오탐 사진(NegativeDB)에는 연기가 없다. 구름, 안개, 굴뚝연기(4, 5, 6) 폴리곤은 헛탐지가
    어디에 떨어졌는지 볼 때만 쓴다

재는 것 (conf 문턱마다)
    IoU         연기 사진에서 연기 정답 마스크 전체와 탐지 마스크 전체의 IoU (사진 평균)
    재현율(사진) 연기 사진 중 연기와 겹치는 탐지가 하나라도 있는 비율
    재현율(연기) 연기 하나하나 중 IoU 0.5 이상으로 맞춘 탐지가 있는 비율, 0.25 도 함께
    오탐(사진)   오탐 사진 중 탐지가 하나라도 있는 비율 (헛경보)
    헛탐지 수    연기와 겹치지 않는 탐지. 연기 사진 안의 것과 오탐 사진의 것을 나눠 센다

마스크는 긴 변 960px 로 줄여 그린 뒤 잰다(운영 입력 크기와 같음). 탐지는 한 번만
가장 낮은 문턱으로 돌리고 문턱마다 걸러 쓴다.
"""
import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

SMOKE = {1, 2, 3}
NAMES = {1: "흑색연기", 2: "백색회색연기", 3: "화염", 4: "구름", 5: "안개연무", 6: "굴뚝연기"}
MASK_LONG = 960
SMALL_PCT = 0.2        # 연기 넓이(화면 대비 %)가 이보다 작으면 작은 연기


def poly_mask(polys, W, H):
    s = MASK_LONG / max(W, H)
    w, h = round(W * s), round(H * s)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        pts = [(p[k] * s, p[k + 1] * s) for k in range(0, len(p) - 1, 2)]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.array(im, dtype=bool)


def flat(poly_xy):
    return [v for pt in poly_xy for v in pt]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser(description="분할 모델 탐지 성능 (6.4)")
    ap.add_argument("eval_dir")
    ap.add_argument("weights")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.40, 0.60])
    ap.add_argument("--out", default="runs/eval_detect")
    a = ap.parse_args()

    from detect import Detector
    det = Detector(a.weights, conf=min(a.conf))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = []
    jpgs = sorted(glob.glob(str(Path(a.eval_dir) / "images" / "eval" / "*.jpg")))
    for i, jpg in enumerate(jpgs):
        lab = json.load(open(Path(a.eval_dir) / "labels" / "eval" / (Path(jpg).stem + ".json"),
                             encoding="utf-8"))
        W, H = lab["images"][0]["width"], lab["images"][0]["height"]
        positive = lab["info"]["name"].startswith("Positive")
        cls = Path(jpg).stem.split("_")[1]
        gts = [{"cat": an["category_id"], "mask": poly_mask(an["segmentation"], W, H)}
               for an in lab["annotations"]]
        frame, _ = det.run(jpg, i)
        dets = []
        for d in frame["detections"]:
            polys = [flat(d["polygon"])] if d.get("polygon") else \
                [[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1],
                  d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3], d["bbox"][0], d["bbox"][1] + d["bbox"][3]]]
            m = poly_mask(polys, W, H)
            smoke_ov = max((np.logical_and(m, g["mask"]).sum() / max(m.sum(), 1)
                            for g in gts if g["cat"] in SMOKE and positive), default=0.0)
            other = [g["cat"] for g in gts if g["cat"] not in SMOKE and np.logical_and(m, g["mask"]).any()]
            dets.append({"score": d["score"], "mask": m, "smoke_overlap": float(smoke_ov),
                         "on": NAMES.get(other[0]) if other else None})
        smoke_px = np.zeros_like(gts[0]["mask"]) if gts else None
        for g in gts:
            if g["cat"] in SMOKE and positive:
                smoke_px |= g["mask"]
        area = float(smoke_px.mean() * 100) if smoke_px is not None else 0.0
        recs.append({"image": Path(jpg).name, "positive": positive, "cls": cls, "area_pct": area,
                     "gts": [g for g in gts if g["cat"] in SMOKE] if positive else [],
                     "others": [g for g in gts if g["cat"] not in SMOKE], "dets": dets,
                     "shape": (H, W)})
        print(i, cls, len(dets), flush=True)

    rows = []
    print(f"\n평가셋 {len(recs)}장: 연기 {sum(r['positive'] for r in recs)}장, "
          f"오탐 유발 {sum(not r['positive'] for r in recs)}장. 모델 {a.weights}")
    print(f"{'conf':>6}{'IoU':>7}{'재현율 사진':>10}{'연기 IoU.5':>11}{'연기 IoU.25':>12}"
          f"{'헛경보 사진':>10}{'헛탐지(연기사진)':>15}{'헛탐지(오탐사진)':>15}")
    for c in sorted(a.conf):
        ious, hit_img, n_pos = [], 0, 0
        inst = inst5 = inst25 = 0
        alarm = n_neg = 0
        fp_pos = fp_neg = 0
        by_cls = defaultdict(lambda: [0, 0])
        for r in recs:
            ds = [d for d in r["dets"] if d["score"] >= c]
            if r["positive"]:
                n_pos += 1
                gt_all = np.zeros_like(r["gts"][0]["mask"]) if r["gts"] else None
                for g in r["gts"]:
                    gt_all |= g["mask"]
                det_all = np.zeros_like(gt_all) if gt_all is not None else None
                for d in ds:
                    det_all |= d["mask"]
                if gt_all is not None:
                    ious.append(iou(gt_all, det_all))
                hit = any(d["smoke_overlap"] > 0 for d in ds)
                hit_img += hit
                by_cls[r["cls"]][0] += hit
                by_cls[r["cls"]][1] += 1
                for g in r["gts"]:
                    inst += 1
                    best = max((iou(g["mask"], d["mask"]) for d in ds), default=0.0)
                    inst5 += best >= 0.5
                    inst25 += best >= 0.25
                fp_pos += sum(d["smoke_overlap"] == 0 for d in ds)
            else:
                n_neg += 1
                alarm += bool(ds)
                by_cls[r["cls"]][0] += bool(ds)
                by_cls[r["cls"]][1] += 1
                fp_neg += len(ds)
        row = {"conf": c, "iou": st.mean(ious), "recall_img": hit_img / n_pos,
               "recall_inst50": inst5 / inst, "recall_inst25": inst25 / inst,
               "alarm_neg": alarm / n_neg if n_neg else 0.0, "fp_in_pos": fp_pos, "fp_in_neg": fp_neg,
               "n_pos": n_pos, "n_neg": n_neg, "n_inst": inst,
               "by_class": {k: f"{v[0]}/{v[1]}" for k, v in by_cls.items()}}
        rows.append(row)
        print(f"{c:>6.2f}{row['iou']:>7.3f}{hit_img:>6}/{n_pos:<3}{inst5:>7}/{inst:<3}{inst25:>8}/{inst:<3}"
              f"{alarm:>6}/{n_neg:<3}{fp_pos:>11}{fp_neg:>15}")

    print("\n분류별 (재현율은 연기 사진에서 잡은 수, 오탐 사진은 알람이 뜬 수)")
    for row in rows:
        print(f"  conf {row['conf']:.2f}: " + ", ".join(f"{k} {v}" for k, v in sorted(row["by_class"].items())))

    print(f"\n작은 연기(넓이 {SMALL_PCT}% 미만)와 큰 연기, 배경 헛탐지 (docs/train_results.md 와 같은 기준)")
    for row in rows:
        c = row["conf"]
        small = [r for r in recs if r["positive"] and r["area_pct"] < SMALL_PCT]
        large = [r for r in recs if r["positive"] and r["area_pct"] >= SMALL_PCT]
        neg = [r for r in recs if not r["positive"]]

        def hits(rs):
            return sum(any(d["score"] >= c and d["smoke_overlap"] > 0 for d in r["dets"]) for r in rs)
        fa = sum(any(d["score"] >= c for d in r["dets"]) for r in neg)
        row["small"], row["large"], row["bg_alarm"] = [hits(small), len(small)], [hits(large), len(large)], [fa, len(neg)]
        print(f"  conf {c:.2f}: 작은 연기 {hits(small)}/{len(small)}, 큰 연기 {hits(large)}/{len(large)}, "
              f"배경 헛탐지 {fa}/{len(neg)}")

    op = min(rows, key=lambda r: abs(r["conf"] - 0.10))
    print(f"\n운영 conf {op['conf']:.2f} 에서")
    miss = [r["image"] for r in recs if r["positive"] and not any(
        d["score"] >= op["conf"] and d["smoke_overlap"] > 0 for d in r["dets"])]
    print(f"  놓친 연기 사진 {len(miss)}장")
    fp_on = defaultdict(int)
    for r in recs:
        for d in r["dets"]:
            if d["score"] >= op["conf"] and d["smoke_overlap"] == 0:
                key = d["on"] or ("연기 사진 속 다른 곳" if r["positive"] else "라벨 없는 곳")
                fp_on[key] += 1
    print("  헛탐지가 떨어진 곳: " + ", ".join(f"{k} {v}" for k, v in sorted(fp_on.items(), key=lambda x: -x[1])))

    with open(out / "eval_detect.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "missed_at_op": miss, "fp_on_at_op": fp_on,
                   "per_image": [{"image": r["image"], "positive": r["positive"], "cls": r["cls"],
                                 "area_pct": round(r["area_pct"], 3),
                                  "dets": [{"score": d["score"], "smoke_overlap": round(d["smoke_overlap"], 3),
                                            "on": d["on"]} for d in r["dets"]]} for r in recs]},
                  f, ensure_ascii=False, indent=1)
    print(f"\n결과: {out / 'eval_detect.json'}")


if __name__ == "__main__":
    main()
