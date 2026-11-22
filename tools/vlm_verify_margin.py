# -*- coding: utf-8 -*-
"""후보 검증기에 주는 조각을 넓히면 헛탐지를 더 잘 거르는지 잰다 (WBS 4.7 보충).

지금 검증기는 탐지 박스를 사방 25% 넓힌 정사각형을 448px 로 키워 "연기가 있는가"를 묻는다.
주변이 거의 안 보여서 밭, 맨땅 같은 헛탐지를 가를 근거가 적을 수 있다. 조각을 넓히거나
사진 전체를 함께 주면 달라지는지 같은 탐지로 비교한다.

탐지와 정답은 다시 만들지 않는다. vlm_verify_real.py 가 남긴 verify_real.jsonl 의 박스와
연기 여부를 그대로 읽는다. 입력은 모두 448px 로 맞추므로 조각을 넓혀도 토큰 수는 같다
(사진 두 장을 주는 조건만 두 배).

    python tools/vlm_verify_margin.py --dets runs/verify_real_v2/verify_real.jsonl \\
        --images data/eval_100_v2/images/eval --out runs/verify_margin/eval.jsonl

조건
    m0.25      사방 25% (지금 검증기. 저장된 p_yes 와 같아야 한다)
    m1         사방 100% (한 변이 박스의 3배)
    m2         사방 200% (한 변이 박스의 5배)
    m1box      m1 에 탐지 박스를 빨간 선으로 그리고 "빨간 상자 안에 연기가 있는가"
    full+crop  사진 전체(긴 변 448) + m0.25 조각, 두 장
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from vlm_verify import Verifier, PROMPT, YES, NO, OUT_PX  # noqa: E402

BOX_PROMPT = "사진에 빨간 상자가 그려져 있다. 빨간 상자 안에 연기가 있는가. 예 또는 아니오로만 답하라."
TWO_PROMPT = ("사진은 두 장이다. 두 번째 사진은 첫 번째 사진의 일부를 확대한 것이다. "
              "두 번째 사진에 연기가 있는가. 예 또는 아니오로만 답하라.")
CONDS = ["m0.25", "m1", "m2", "m1box", "full+crop"]


def fixed(name):
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unicodedata.normalize("NFC", name)


def crop(img, bbox, margin, min_side, draw_box=False):
    """vlm_client.crop_det 과 같은 방식. margin 만 바꾸고, 필요하면 박스를 그린다."""
    from PIL import Image, ImageDraw
    W, H = img.size
    x, y, w, h = bbox
    side = max(max(w, h) * (1 + 2 * margin), min_side)
    cx, cy, half = x + w / 2, y + h / 2, side / 2
    b = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    cv = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), (128, 128, 128))
    cv.paste(img.crop((max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H))),
             (max(-b[0], 0), max(-b[1], 0)))
    cv = cv.resize((OUT_PX, OUT_PX), Image.LANCZOS)
    if draw_box:
        s = OUT_PX / (b[2] - b[0])
        ImageDraw.Draw(cv).rectangle([(x - b[0]) * s, (y - b[1]) * s, (x + w - b[0]) * s, (y + h - b[1]) * s],
                                     outline=(255, 0, 0), width=3)
    return cv


def p_yes(ver, images, text):
    import torch
    msg = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": text}]}]
    inp = ver.p.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt").to(ver.p.model.device)
    with torch.inference_mode():
        pr = torch.softmax(ver.p.model(**inp).logits[0, -1].float(), -1)
    y, n = float(pr[YES]), float(pr[NO])
    return y / (y + n), int(inp["input_ids"].shape[1])


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report(rows, tau):
    by = {}
    for r in rows:
        by.setdefault(r["image"], {"positive": r["positive"], "dets": []})["dets"].append(r)
    print(f"\n{'조건':<10}{'AUC':>6}{'연기 남김':>12}{'헛탐지 거름':>13}{'  연기사진속':>10}{'  오탐사진':>9}"
          f"{'오탐 사진 알람':>13}{'새로 놓친 연기 사진':>16}{'토큰':>6}{'초':>6}")
    for c in CONDS:
        if c not in rows[0]["p"]:
            continue
        pos = [r["p"][c] for r in rows if r["smoke"]]
        neg = [r["p"][c] for r in rows if not r["smoke"]]
        neg_in = [r["p"][c] for r in rows if not r["smoke"] and r["positive"]]
        neg_out = [r["p"][c] for r in rows if not r["smoke"] and not r["positive"]]
        alarm = sum(any(d["p"][c] >= tau for d in v["dets"]) for v in by.values() if not v["positive"])
        n_neg_img = sum(1 for v in by.values() if not v["positive"])
        lost = sum(1 for v in by.values() if v["positive"] and any(d["smoke"] for d in v["dets"])
                   and not any(d["smoke"] and d["p"][c] >= tau for d in v["dets"]))
        tok = sorted(r["tok"][c] for r in rows)[len(rows) // 2]
        sec = sorted(r["sec"][c] for r in rows)[len(rows) // 2]
        print(f"{c:<10}{auc(pos, neg):>6.3f}{sum(p >= tau for p in pos):>7}/{len(pos):<4}"
              f"{sum(p < tau for p in neg):>8}/{len(neg):<4}{sum(p < tau for p in neg_in):>6}/{len(neg_in):<3}"
              f"{sum(p < tau for p in neg_out):>6}/{len(neg_out):<3}{alarm:>9}/{n_neg_img:<3}{lost:>12}장"
              f"{tok:>8}{sec:>6.2f}")
    print(f"(문턱 {tau}. 오탐 사진 알람의 분모는 탐지가 있는 오탐 사진 수)")


def main():
    ap = argparse.ArgumentParser(description="검증기 조각 넓이 비교")
    ap.add_argument("--dets", required=True, help="vlm_verify_real.py 의 verify_real.jsonl")
    ap.add_argument("--images", required=True, help="사진 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if a.report_only:
        report([json.loads(l) for l in open(a.out, encoding="utf-8")], a.tau)
        return

    from PIL import Image
    disk = {fixed(p.name): p for p in Path(a.images).glob("*.jpg")}
    recs = [json.loads(l) for l in open(a.dets, encoding="utf-8")]
    ver = Verifier()
    rows = []
    for r in recs:
        if not r["dets"]:
            continue
        img = Image.open(disk[fixed(r["image"])]).convert("RGB")
        ms = max(160, img.size[0] // 6)
        full = img.copy()
        full.thumbnail((OUT_PX, OUT_PX))
        for d in r["dets"]:
            row = {"image": fixed(r["image"]), "positive": r["positive"], "j": d["j"], "score": d["score"],
                   "smoke": d["smoke"], "saved": d["p_yes"], "p": {}, "tok": {}, "sec": {}}
            base = crop(img, d["bbox"], 0.25, ms)
            for c in a.conds:
                t0 = time.perf_counter()
                if c == "m0.25":
                    p, n = p_yes(ver, [base], PROMPT)
                elif c == "m1":
                    p, n = p_yes(ver, [crop(img, d["bbox"], 1.0, ms)], PROMPT)
                elif c == "m2":
                    p, n = p_yes(ver, [crop(img, d["bbox"], 2.0, ms)], PROMPT)
                elif c == "m1box":
                    p, n = p_yes(ver, [crop(img, d["bbox"], 1.0, ms, draw_box=True)], BOX_PROMPT)
                elif c == "full+crop":
                    p, n = p_yes(ver, [full, base], TWO_PROMPT)
                else:
                    raise SystemExit(f"모르는 조건: {c}")
                row["p"][c], row["tok"][c], row["sec"][c] = round(p, 4), n, round(time.perf_counter() - t0, 2)
            rows.append(row)
            print(len(rows), "연기" if d["smoke"] else "헛", "저장", d["p_yes"], row["p"], flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for row in rows:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    if "m0.25" in a.conds:
        diff = max(abs(r["p"]["m0.25"] - r["saved"]) for r in rows)
        print(f"\n지금 검증기 재현: 저장된 p_yes 와의 최대 차이 {diff:.4f}")
    report(rows, a.tau)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
