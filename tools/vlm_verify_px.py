# -*- coding: utf-8 -*-
"""후보 검증기의 조각 크기를 줄이면 정확도가 유지되는지 잰다 (WBS 4.7, 7.2 보충).

검증기는 탐지 조각을 448px 로 키워 묻는다(입력 약 295토큰). CPU 스레드 4개에서 한 번에 11.9초라
목표 10초를 넘는다(`docs/model_size.md`). 조각을 360, 300, 224px 로 줄였을 때 연기와 헛탐지를
얼마나 가르는지 본다. 자르는 범위(박스 사방 25%)와 질문, P(예) 계산은 검증기와 같고 크기만 바꾼다.

Qwen2.5-VL 의 이미지 처리기는 min_pixels(파이프라인은 256x256)보다 작은 그림을 키운다. 224px 를
제대로 재려고 이 도구에서는 min_pixels 를 낮춘다. 448, 360, 300 은 이 값과 상관없다.

    python tools/vlm_verify_px.py --dets runs/verify_real/verify_real.jsonl \\
        --images data/eval_100/images/eval --out results/verify_px_eval.jsonl
    python tools/vlm_verify_px.py --report-only --out results/verify_px_eval.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from vlm_verify_margin import fixed, auc  # noqa: E402

SIZES = [448, 360, 300, 224]


def crop(img, bbox, px, min_side):
    """src/pipeline/vlm_client.crop_det 과 같고, 마지막 크기만 px 로 받는다."""
    from PIL import Image
    W, H = img.size
    x, y, w, h = bbox
    side = max(max(w, h) * 1.5, min_side)
    cx, cy, half = x + w / 2, y + h / 2, side / 2
    b = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    cv = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), (128, 128, 128))
    cv.paste(img.crop((max(b[0], 0), max(b[1], 0), min(W, b[2]), min(H, b[3]))), (max(-b[0], 0), max(-b[1], 0)))
    return cv.resize((px, px), Image.LANCZOS)


def p_yes(ver, img, px):
    import torch
    from vlm_verify import PROMPT, YES, NO
    ip = ver.p.processor.image_processor
    ip.max_pixels, ip.min_pixels = px * px, 28 * 28 * 4
    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inp = ver.p.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt").to(ver.p.model.device)
    with torch.inference_mode():
        pr = torch.softmax(ver.p.model(**inp).logits[0, -1].float(), -1)
    y, n = float(pr[YES]), float(pr[NO])
    return y / (y + n), int(inp["input_ids"].shape[1])


def report(rows, tau):
    by = {}
    for r in rows:
        by.setdefault(r["image"], []).append(r)
    print(f"{'크기':>6}{'입력 토큰':>9}{'AUC':>7}{'연기 남김':>11}{'헛탐지 거름':>12}{'  연기 사진 속':>10}{'  굴뚝연기':>9}"
          f"{'새로 놓친 연기 사진':>13}{'95% 문턱':>9}{'그때 거름':>8}")
    for px in SIZES:
        k = str(px)
        if k not in rows[0]["p"]:
            continue
        pos = [r["p"][k] for r in rows if r["smoke"]]
        neg = [r for r in rows if not r["smoke"]]
        ni = [r["p"][k] for r in neg if r["positive"]]
        nc = [r["p"][k] for r in neg if not r["positive"]]
        lost = sum(1 for v in by.values() if v[0]["positive"] and any(d["smoke"] for d in v)
                   and not any(d["smoke"] and d["p"][k] >= tau for d in v))
        t95 = sorted(pos)[int(len(pos) * 0.05)]
        tok = sorted(r["tok"][k] for r in rows)[len(rows) // 2]
        print(f"{px:>6}{tok:>9}{auc(pos, [r['p'][k] for r in neg]):>7.3f}{sum(p >= tau for p in pos):>7}/{len(pos):<4}"
              f"{sum(r['p'][k] < tau for r in neg):>8}/{len(neg):<4}{sum(p < tau for p in ni):>6}/{len(ni):<3}"
              f"{sum(p < tau for p in nc):>6}/{len(nc):<3}{lost:>10}장{t95:>11.3f}{sum(r['p'][k] < t95 for r in neg):>6}/{len(neg)}")
    print(f"(문턱 {tau}. 95% 문턱은 연기 탐지의 95% 를 남기는 값)")


def main():
    ap = argparse.ArgumentParser(description="검증기 조각 크기별 정확도")
    ap.add_argument("--dets")
    ap.add_argument("--images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        from PIL import Image
        from vlm_verify import Verifier
        disk = {fixed(p.name): p for p in Path(a.images).glob("*.jpg")}
        ver = Verifier()
        rows = []
        for r in (json.loads(l) for l in open(a.dets, encoding="utf-8")):
            if not r["dets"]:
                continue
            img = Image.open(disk[fixed(r["image"])]).convert("RGB")
            ms = max(160, img.size[0] // 6)
            for d in r["dets"]:
                row = {"image": fixed(r["image"]), "positive": r["positive"], "j": d["j"], "smoke": d["smoke"],
                       "saved": d["p_yes"], "p": {}, "tok": {}, "sec": {}}
                for px in SIZES:
                    t0 = time.perf_counter()
                    p, n = p_yes(ver, crop(img, d["bbox"], px, ms), px)
                    row["p"][str(px)], row["tok"][str(px)], row["sec"][str(px)] = round(p, 4), n, round(time.perf_counter() - t0, 2)
                rows.append(row)
                print(len(rows), "연기" if d["smoke"] else "헛", "저장", d["p_yes"], row["p"], flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fo:
            for row in rows:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n448 재현: 저장값과의 최대 차이 {max(abs(r['p']['448'] - r['saved']) for r in rows):.4f}")
    report([json.loads(l) for l in open(a.out, encoding="utf-8")], a.tau)


if __name__ == "__main__":
    main()
