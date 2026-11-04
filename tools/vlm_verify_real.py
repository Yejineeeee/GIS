# -*- coding: utf-8 -*-
"""VLM 후보 검증기를 71330 실제 사진(평가셋 100장)으로 확인한다 (WBS 4.7).

tools/vlm_verify.py 는 시뮬레이터에서 위치 오차로 쟀다. 실제 사진에는 카메라 값과
정답 좌표가 없으므로, 탐지 하나하나가 진짜 연기인지를 라벨로 정해 검증기가 그것을
구분하는지 본다.

    python tools/vlm_verify_real.py data/eval_100 runs/smoke_1260_960/weights/best.pt

탐지의 정답
    연기 사진(PositiveRealDB)  탐지 박스가 연기 라벨(흑색연기, 백색/회색연기, 화염)
                               박스와 겹치면(IoU 0.1 이상 또는 중심이 안에 있으면) 연기
    오탐 사진(NegativeDB)       모든 탐지가 헛탐지다(구름, 굴뚝연기 사진)

보는 것
    - 탐지 단위: P(예) 로 연기와 헛탐지를 구분하는 정도(AUC), 문턱별로 남는 비율
    - 사진 단위: 오탐 사진에 알람이 남는지(헛경보), 연기 사진에서 연기 탐지가
      남는지(놓침), conf 1등이 헛탐지인 사진이 고쳐지는지

질문과 자르기, P(예) 계산은 vlm_verify.py 와 같다.
"""
import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from vlm_verify import Verifier, crop_det  # noqa: E402

SMOKE_CATS = {1, 2, 3}       # 흑색연기, 백색/회색연기, 화염 (71330)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def inside(pt, b):
    return b[0] <= pt[0] <= b[0] + b[2] and b[1] <= pt[1] <= b[1] + b[3]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description="VLM 후보 검증기, 실제 사진 확인 (4.7)")
    ap.add_argument("eval_dir", help="images/eval 과 labels/eval 이 있는 폴더")
    ap.add_argument("weights")
    ap.add_argument("--out", default="runs/verify_real")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    a = ap.parse_args()

    from PIL import Image
    from detect import Detector
    det = Detector(a.weights)
    ver = Verifier()
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    recs = []
    imgs = sorted(glob.glob(str(Path(a.eval_dir) / "images" / "eval" / "*.jpg")))
    for i, jpg in enumerate(imgs):
        lab = json.load(open(Path(a.eval_dir) / "labels" / "eval" / (Path(jpg).stem + ".json"),
                             encoding="utf-8"))
        positive = lab["info"]["name"].startswith("Positive")
        smoke_boxes = [an["bbox"] for an in lab["annotations"] if an["category_id"] in SMOKE_CATS]
        frame, _ = det.run(jpg, i)
        img = Image.open(jpg).convert("RGB")
        dets = []
        for j, d in enumerate(frame["detections"]):
            b = d["bbox"]
            c = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            is_smoke = positive and any(iou(b, s) >= 0.1 or inside(c, s) for s in smoke_boxes)
            cr = crop_det(img, b, min_side=max(160, img.size[0] // 6))
            cr.save(out / "crops" / f"{i:03d}_{j}.png")
            dets.append({"j": j, "score": d["score"], "p_yes": round(ver.p_yes(cr), 4),
                         "smoke": is_smoke, "bbox": b})
        recs.append({"i": i, "image": Path(jpg).name, "positive": positive,
                     "n_smoke_labels": len(smoke_boxes), "dets": dets})
        print(i, "연기" if positive else "오탐", [(d["score"], d["p_yes"], d["smoke"]) for d in dets], flush=True)

    with open(out / "verify_real.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    alld = [d for r in recs for d in r["dets"]]
    pos = [d["p_yes"] for d in alld if d["smoke"]]
    neg = [d["p_yes"] for d in alld if not d["smoke"]]
    negimg = [r for r in recs if not r["positive"]]
    posimg = [r for r in recs if r["positive"]]
    print(f"\n탐지 {len(alld)}개: 연기 {len(pos)}, 헛탐지 {len(neg)}   P(예) 평균 연기 {st.mean(pos):.2f} 헛탐지 {st.mean(neg):.2f}   AUC {auc(pos, neg):.2f}")
    print(f"사진: 연기 {len(posimg)}장, 오탐 유발 {len(negimg)}장")
    print(f"\n{'문턱':>6}{'연기 탐지 남김':>14}{'헛탐지 걸러냄':>14}{'오탐 사진 알람':>14}{'연기 사진 놓침':>14}{'conf1등 헛탐지 사진':>18}")
    for tau in [None] + a.tau:
        keep = (lambda d: True) if tau is None else (lambda d, t=tau: d["p_yes"] >= t)
        kp = sum(keep(d) for d in alld if d["smoke"])
        rn = sum(not keep(d) for d in alld if not d["smoke"])
        alarm = sum(any(keep(d) for d in r["dets"]) for r in negimg)
        miss = sum(not any(keep(d) and d["smoke"] for d in r["dets"]) for r in posimg)
        topfp = 0
        for r in recs:
            ks = [d for d in r["dets"] if keep(d)]
            if ks and not max(ks, key=lambda d: d["score"])["smoke"]:
                topfp += 1
        name = "없음" if tau is None else f"{tau:.1f}"
        print(f"{name:>6}{kp:>9}/{len(pos):<4}{rn:>9}/{len(neg):<4}{alarm:>9}/{len(negimg):<4}"
              f"{miss:>9}/{len(posimg):<4}{topfp:>12}장")
    print(f"\n결과: {out / 'verify_real.jsonl'}")


if __name__ == "__main__":
    main()
