# -*- coding: utf-8 -*-
"""탐지 정밀도·재현율·F1 과 연기 밑동 픽셀 오차를 평가셋으로 잰다.

tools/eval_detect.py(6.4)는 IoU 와 재현율, 헛경보를 쟀다. 여기서는
  1. 연기 하나하나를 맞춘 탐지로 정밀도, 재현율, F1 (IoU 0.5, 0.25)과 사진 단위 정밀도, 재현율, F1
  2. 위치 계산의 출발점인 연기 밑동(src/geo/transform.py target_pixel)이 정답 폴리곤의 밑동에서
     몇 픽셀 벗어나는지
를 잰다. eval_detect.py 와 같은 마스크(긴 변 960px), 같은 정답 분류(흑색연기, 백색/회색연기, 화염)를 쓴다.
탐지 문턱은 운영값 conf 0.10.

    python tools/eval_base.py data/eval_100_v2 runs/smoke_1260_960/weights/best.pt --out runs/eval_base

밑동 오차는 맞춘 쌍(IoU 0.25 이상)마다 잰다. 원본 해상도 픽셀과 사진 높이 대비 % 로 적고,
세로 오차는 부호를 남긴다(+ 는 탐지 밑동이 정답보다 아래). 세로 오차가 거리 오차로 이어진다.
"""
import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from eval_detect import SMOKE, poly_mask, flat, iou  # noqa: E402
from transform import target_pixel  # noqa: E402

CONF = 0.10


def pts(seg):
    """COCO segmentation(평평한 리스트 여러 개) → [[x, y], ...]"""
    return [[p[k], p[k + 1]] for p in seg for k in range(0, len(p) - 1, 2)]


def match(dets, gts, thr):
    """IoU 가 큰 쌍부터 하나씩 짝짓는다. [(det 번호, gt 번호, IoU)]"""
    pairs = sorted(((iou(d["mask"], g["mask"]), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)),
                   reverse=True)
    used_d, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i); used_g.add(j); out.append((i, j, v))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "정밀도": round(p, 3), "재현율": round(r, 3), "F1": round(f, 3)}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser(description="탐지 정밀도/재현율/F1, 연기 밑동 오차")
    ap.add_argument("eval_dir")
    ap.add_argument("weights")
    ap.add_argument("--out", default="runs/eval_base")
    a = ap.parse_args()

    from detect import Detector
    det = Detector(a.weights, conf=CONF)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = []
    for i, jpg in enumerate(sorted(glob.glob(str(Path(a.eval_dir) / "images" / "eval" / "*.jpg")))):
        lab = json.load(open(Path(a.eval_dir) / "labels" / "eval" / (Path(jpg).stem + ".json"), encoding="utf-8"))
        W, H = lab["images"][0]["width"], lab["images"][0]["height"]
        positive = lab["info"]["name"].startswith("Positive")
        gts = [{"mask": poly_mask(an["segmentation"], W, H), "poly": pts(an["segmentation"])}
               for an in lab["annotations"] if positive and an["category_id"] in SMOKE]
        frame, _ = det.run(jpg, i)
        dets = []
        for d in frame["detections"]:
            polys = [flat(d["polygon"])] if d.get("polygon") else []
            if not polys:
                x, y, w, h = d["bbox"]
                polys = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            dets.append({"mask": poly_mask(polys, W, H), "det": d})
        recs.append({"image": Path(jpg).name, "positive": positive, "W": W, "H": H, "gts": gts, "dets": dets})
        print(i, len(dets), len(gts), flush=True)

    report = {"conf": CONF, "사진": len(recs), "연기 사진": sum(r["positive"] for r in recs)}
    for thr in (0.5, 0.25):
        tp = fp = fn = 0
        for r in recs:
            m = match(r["dets"], r["gts"], thr)
            tp += len(m); fp += len(r["dets"]) - len(m); fn += len(r["gts"]) - len(m)
        report[f"연기 단위 IoU {thr}"] = prf(tp, fp, fn)
    tp = sum(r["positive"] and bool(r["dets"]) for r in recs)
    fp = sum((not r["positive"]) and bool(r["dets"]) for r in recs)
    fn = sum(r["positive"] and not r["dets"] for r in recs)
    report["사진 단위 (탐지가 하나라도 있으면 연기)"] = prf(tp, fp, fn)

    errs = []
    for r in recs:
        for i, j, v in match(r["dets"], r["gts"], 0.25):
            px, py = target_pixel(r["dets"][i]["det"])
            gx, gy = target_pixel({"polygon": r["gts"][j]["poly"]})
            errs.append({"image": r["image"], "iou": round(v, 3), "dx": round(px - gx, 1), "dy": round(py - gy, 1),
                         "dist": round(float(np.hypot(px - gx, py - gy)), 1), "H": r["H"],
                         "dy_pct": round((py - gy) / r["H"] * 100, 2), "dist_pct": round(float(np.hypot(px - gx, py - gy)) / r["H"] * 100, 2)})
    if errs:
        d = [e["dist"] for e in errs]; dp = [e["dist_pct"] for e in errs]
        dy = [abs(e["dy"]) for e in errs]; dyp = [abs(e["dy_pct"]) for e in errs]
        report["밑동 오차 (맞춘 쌍, IoU 0.25 이상)"] = {
            "쌍": len(errs),
            "거리 px 중앙/90%/최대": [st.median(d), q(d, 0.9), max(d)],
            "거리 % 사진 높이 중앙/90%/최대": [st.median(dp), q(dp, 0.9), max(dp)],
            "세로 px 중앙/90%": [st.median(dy), q(dy, 0.9)],
            "세로 % 중앙/90%": [st.median(dyp), q(dyp, 0.9)],
            "탐지 밑동이 정답보다 위": sum(e["dy"] < 0 for e in errs),
            "탐지 밑동이 정답보다 아래": sum(e["dy"] > 0 for e in errs),
        }
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out / "base_errors.jsonl", "w", encoding="utf-8") as f:
        for e in errs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
