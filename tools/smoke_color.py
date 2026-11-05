# -*- coding: utf-8 -*-
"""연기색을 탐지 조각으로 재 본다 (WBS 4.7, docs/vlm_compare.md 후속).

사진 전체를 보여 주고 연기색을 물었을 때 VLM 은 21/70 으로, 늘 "흰색/회색"이라고
쓰는 것(40/70)보다 못했다. 탐지 조각을 잘라 보여 주면 나아지는지, 그리고 VLM 없이
마스크 안 밝기만으로 가를 수 있는지 같은 탐지로 비교한다.

    python tools/smoke_color.py data/eval_100 runs/smoke_1260_960/weights/best.pt

대상
    평가셋 연기 사진의 탐지 중 연기 라벨과 겹치는 것(vlm_verify_real.py 와 같은 기준).
    정답 색은 가장 많이 겹치는 연기 라벨의 분류다. 흑색연기 = 검은색, 백색/회색연기 =
    밝은 연기. 화염 라벨만 겹치는 탐지는 뺀다.
    사진 단위는 좌표에 쓰는 탐지(V2: 검증기 0.7 통과 중 conf 1등)가 연기인 사진만 본다.

방식
    VLM-ko  탐지 조각(vlm_verify.py 와 같은 자르기)에 "검은색, 흰색, 회색 중 하나"를 묻고
            첫 토큰 확률로 고른다
    VLM-en  같은 것을 영어로 묻는다 (black, white, gray)
    밝기    마스크 안 평균 밝기(0~255). 문턱은 사진 번호 홀짝으로 나눠 한쪽에서 정하고
            다른 쪽에 쓴다
    대비    마스크 안 밝기 - 마스크를 둘러싼 띠(박스를 25% 넓힌 곳 중 마스크 밖)의 밝기.
            문턱은 밝기와 같은 방법으로 정한다
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from vlm_verify import crop_det  # noqa: E402
from vlm_verify_real import iou, inside  # noqa: E402

DARK, LIGHT = "검은색", "밝은색"
CAT = {1: DARK, 2: LIGHT}
TAU = 0.7
PROMPTS = {
    "VLM-ko": ("이 사진 속 연기의 색은 무엇인가. 검은색, 흰색, 회색 중 하나로만 답하라.",
               {DARK: ["검"], LIGHT: ["흰", "회"]}),
    "VLM-en": ("What color is the smoke in this image? Answer with one word: black, white, or gray.",
               {DARK: ["black", "Black"], LIGHT: ["white", "White", "gray", "Gray", "grey", "Grey"]}),
}


def first_ids(tok, words):
    ids = set()
    for w in words:
        e = tok.encode(w, add_special_tokens=False)
        if e:
            ids.add(e[0])
    return sorted(ids)


def poly_mask(poly, W, H):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
    return np.array(im, dtype=bool)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def best_cut(vals):
    """vals: [(값, 정답)] -> 값 >= 문턱이면 밝은색으로 볼 때 가장 많이 맞는 문턱."""
    cands = sorted({v for v, _ in vals})
    best = (-1, 0.0)
    for c in cands:
        ok = sum((v >= c) == (g == LIGHT) for v, g in vals)
        best = max(best, (ok, c))
    return best[1]


def main():
    ap = argparse.ArgumentParser(description="탐지 조각으로 연기색 재기")
    ap.add_argument("eval_dir")
    ap.add_argument("weights")
    ap.add_argument("--out", default="runs/smoke_color")
    a = ap.parse_args()

    import torch
    from detect import Detector
    from vlm_verify import Verifier
    det = Detector(a.weights)
    ver = Verifier()
    tok = ver.p.processor.tokenizer
    ids = {k: {c: first_ids(tok, ws) for c, ws in m.items()} for k, (_, m) in PROMPTS.items()}
    print("첫 토큰:", ids, flush=True)
    vr = {json.loads(l)["i"]: json.loads(l) for l in open(ROOT / "runs/verify_real/verify_real.jsonl", encoding="utf-8")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def ask(image, key):
        prompt = PROMPTS[key][0]
        msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inp = ver.p.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt").to(ver.p.model.device)
        with torch.inference_mode():
            pr = torch.softmax(ver.p.model(**inp).logits[0, -1].float(), -1)
        p = {c: float(sum(pr[t] for t in ids[key][c])) for c in (DARK, LIGHT)}
        s = p[DARK] + p[LIGHT]
        return p[LIGHT] / s if s > 0 else 0.5, s

    recs = []
    imgs = sorted(glob.glob(str(Path(a.eval_dir) / "images" / "eval" / "*.jpg")))
    for i, jpg in enumerate(imgs):
        lab = json.load(open(Path(a.eval_dir) / "labels" / "eval" / (Path(jpg).stem + ".json"), encoding="utf-8"))
        if not lab["info"]["name"].startswith("Positive"):
            continue
        smoke = [an for an in lab["annotations"] if an["category_id"] in (1, 2, 3)]
        frame, _ = det.run(jpg, i)
        img = Image.open(jpg).convert("RGB")
        W, H = img.size
        lum = np.asarray(img.convert("L"), dtype=float)
        saved = vr[i]["dets"]
        assert len(saved) == len(frame["detections"]), f"{i}: 탐지 수가 저장된 것과 다르다"
        for j, d in enumerate(frame["detections"]):
            b = d["bbox"]
            c = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            hit = [an for an in smoke if iou(b, an["bbox"]) >= 0.1 or inside(c, an["bbox"])]
            if not hit:
                continue
            top = max(hit, key=lambda an: iou(b, an["bbox"]))
            if top["category_id"] not in CAT:
                continue
            if d.get("polygon"):
                m = poly_mask(d["polygon"], W, H)
            else:
                m = np.zeros((H, W), bool)
                m[int(b[1]):int(b[1] + b[3]), int(b[0]):int(b[0] + b[2])] = True
            if m.sum() < 10:
                continue
            ex = 0.25
            x0, y0 = max(int(b[0] - b[2] * ex), 0), max(int(b[1] - b[3] * ex), 0)
            x1, y1 = min(int(b[0] + b[2] * (1 + ex)), W), min(int(b[1] + b[3] * (1 + ex)), H)
            ring = np.zeros_like(m)
            ring[y0:y1, x0:x1] = True
            ring &= ~m
            inner = float(lum[m].mean())
            cr = crop_det(img, b, min_side=max(160, W // 6))
            rec = {"i": i, "j": j, "image": Path(jpg).name, "truth": CAT[top["category_id"]],
                   "score": d["score"], "p_yes": saved[j]["p_yes"],
                   "bright": round(inner, 1),
                   "contrast": round(inner - float(lum[ring].mean()), 1) if ring.any() else 0.0}
            for k in PROMPTS:
                pl, mass = ask(cr, k)
                rec[k] = round(pl, 4)
                rec[k + "_mass"] = round(mass, 3)
            recs.append(rec)
            print(i, j, rec["truth"], rec["bright"], rec["contrast"], rec["VLM-ko"], rec["VLM-en"], flush=True)

    with open(out / "smoke_color.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(recs)


def report(recs):
    # 사진 단위: 좌표에 쓰는 탐지(V2)가 연기인 사진
    by_img = {}
    for r in recs:
        by_img.setdefault(r["i"], []).append(r)
    vr = {json.loads(l)["i"]: json.loads(l) for l in open(ROOT / "runs/verify_real/verify_real.jsonl", encoding="utf-8")}
    img_recs = []
    for i, rs in by_img.items():
        dets = vr[i]["dets"]
        ok = [d for d in dets if d["p_yes"] >= TAU]
        ch = max(ok or dets, key=lambda d: d["score"])
        hit = [r for r in rs if r["j"] == ch["j"]]
        if hit:
            img_recs.append(hit[0])

    for name, rs in (("탐지 단위", recs), ("사진 단위 (좌표에 쓰는 탐지)", img_recs)):
        cnt = Counter(r["truth"] for r in rs)
        print(f"\n[{name}] {len(rs)}개: {dict(cnt)}. 늘 '{cnt.most_common(1)[0][0]}' 이라고 쓰면 "
              f"{cnt.most_common(1)[0][1]}/{len(rs)}")
        for k in ("VLM-ko", "VLM-en"):
            pos = [r[k] for r in rs if r["truth"] == LIGHT]
            neg = [r[k] for r in rs if r["truth"] == DARK]
            acc = sum((r[k] >= 0.5) == (r["truth"] == LIGHT) for r in rs)
            dark_ok = sum(r[k] < 0.5 for r in rs if r["truth"] == DARK)
            light_ok = sum(r[k] >= 0.5 for r in rs if r["truth"] == LIGHT)
            mass = np.mean([r[k + "_mass"] for r in rs])
            print(f"  {k:<7} 맞춤 {acc}/{len(rs)}  검은색 {dark_ok}/{cnt[DARK]}  밝은색 {light_ok}/{cnt[LIGHT]}"
                  f"  AUC {auc(pos, neg):.2f}  색 단어 확률 합 {mass:.2f}")
        for k in ("bright", "contrast"):
            pos = [r[k] for r in rs if r["truth"] == LIGHT]
            neg = [r[k] for r in rs if r["truth"] == DARK]
            acc = dark_ok = light_ok = 0
            for half in (0, 1):
                tr = [(r[k], r["truth"]) for r in rs if r["i"] % 2 != half]
                te = [r for r in rs if r["i"] % 2 == half]
                c = best_cut(tr)
                acc += sum((r[k] >= c) == (r["truth"] == LIGHT) for r in te)
                dark_ok += sum(r[k] < c for r in te if r["truth"] == DARK)
                light_ok += sum(r[k] >= c for r in te if r["truth"] == LIGHT)
            print(f"  {'밝기' if k == 'bright' else '대비':<7} 맞춤 {acc}/{len(rs)}  검은색 {dark_ok}/{cnt[DARK]}"
                  f"  밝은색 {light_ok}/{cnt[LIGHT]}  AUC {auc(pos, neg):.2f}  (문턱은 홀짝 교차)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report([json.loads(l) for l in open(ROOT / "runs/smoke_color/smoke_color.jsonl", encoding="utf-8")])
    else:
        main()
