# -*- coding: utf-8 -*-
"""전체 사진과 연기 부분을 확대한 조각을 같이 보여 주면 VLM 인식이 나아지는지 잰다.

전체 사진만 주면 작은 연기를 못 보고(면적 0.5% 미만 17장 중 1장), 조각만 주면 주변이 잘려
굴뚝연기와 구름을 연기라고 한다. 둘을 함께 주면 양쪽 장점을 가져가는지 본다.
모델, 스키마, 질문은 생성 방식 (iii)(tools/run_structured.py)과 같다. 평가셋 100장.

    python tools/vlm_crop_views.py run B          조각만 (정답 라벨로 자름)
    python tools/vlm_crop_views.py run C          전체 + 조각 (정답 라벨로 자름)
    python tools/vlm_crop_views.py run D          전체 + 조각 (YOLO 탐지 상자로 자름)
    python tools/vlm_crop_views.py report         A~D 를 나란히 놓은 표

조건 A(전체 사진만)는 results/eval_method_iii.jsonl 을 그대로 읽는다.

자르는 법
    정답 라벨  연기 사진은 연기 라벨(흑색, 백색/회색) 상자를 모두 감싼 상자, 연기 아닌 사진은
               그 사진의 라벨(구름, 굴뚝연기) 상자를 모두 감싼 상자. 가로세로를 1.5배로 넓힌다
    YOLO       conf 1등 탐지 상자를 같은 방식으로 넓힌다. 탐지는 tools/vlm_verify_real.py 가
               남긴 runs/verify_real/verify_real.jsonl 에서 읽는다. 탐지가 없는 사진은 VLM 을
               부르지 않는다(파이프라인에서도 알람이 없다)

결과는 results/crop_views_<조건>.jsonl 에 장마다 바로 적는다. 끊기면 같은 명령으로 이어진다.
"""
import argparse
import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "results" / "eval_facts.jsonl"
A_FILE = ROOT / "results" / "eval_method_iii.jsonl"
DETS = ROOT / "runs" / "verify_real" / "verify_real.jsonl"
EVAL = ROOT / "data" / "eval_100"
SMOKE_CATS = {1, 2}          # 흑색연기, 백색/회색연기 (71330)
MARGIN = 1.5
MIN_SIDE = 160
TWO_NOTE = "\n\n사진은 두 장이다. 두 번째 사진은 첫 번째 사진의 일부를 확대한 것이다."
BANDS = [(0.005, "0.5% 미만"), (0.02, "0.5~2%"), (0.05, "2~5%"), (9, "5% 이상")]


def nfc(s):
    return unicodedata.normalize("NFC", s)


def fixed(name):
    """폴더에 깨진 채 풀린 한글 이름(UTF-8 을 cp437 로 읽은 꼴)을 되돌린다."""
    try:
        return nfc(name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return nfc(name)


def out_path(cond):
    return ROOT / "results" / f"crop_views_{cond}.jsonl"


def facts():
    return [json.loads(x) for x in open(FACTS, encoding="utf-8") if x.strip()]


def grow(box, W, H):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    w, h = max(w * MARGIN, MIN_SIDE), max(h * MARGIN, MIN_SIDE)
    return (max(int(cx - w / 2), 0), max(int(cy - h / 2), 0), min(int(cx + w / 2), W), min(int(cy + h / 2), H))


def union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def label_box(stem, positive):
    lab = json.load(open(EVAL / "labels" / "eval" / (stem + ".json"), encoding="utf-8"))
    anns = lab["annotations"]
    if positive:
        anns = [a for a in anns if a["category_id"] in SMOKE_CATS] or anns
    return union([a["bbox"] for a in anns]) if anns else None


def run(a):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from lmformatenforcer import TokenEnforcer, JsonSchemaParser
    sys.argv = sys.argv[:1]                       # run_structured 는 불러올 때 인자를 읽는다
    sys.path.insert(0, str(ROOT / "tools"))
    from run_structured import SCHEMA, PROMPT, build_tokenizer_data

    disk = {fixed(p.name): p for p in (EVAL / "images" / "eval").glob("*.jpg")}
    dets = {}
    if a.cond == "D":
        for line in open(DETS, encoding="utf-8"):
            r = json.loads(line)
            dets[fixed(r["image"])] = r["dets"]
    items = facts()
    if a.limit:
        items = items[::max(1, len(items) // a.limit)][:a.limit]
    path = Path(a.out) if a.out else out_path(a.cond)
    done = set()
    if path.is_file():
        done = {json.loads(x)["image"] for x in open(path, encoding="utf-8") if x.strip()}
    items = [f for f in items if f["image"] not in done]
    print(f"조건 {a.cond}: {len(items)}장 (이미 한 것 {len(done)}장)", flush=True)
    if not items:
        return

    mid = "Qwen/Qwen2.5-VL-3B-Instruct"
    dt = torch.bfloat16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(mid)
    processor.image_processor.max_pixels = 768 * 768
    processor.image_processor.min_pixels = 256 * 256
    model = AutoModelForImageTextToText.from_pretrained(
        mid, quantization_config=quant, device_map="cuda:0", dtype=dt)
    data = build_tokenizer_data(processor.tokenizer)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fo:
        for f in items:
            p = disk[nfc(f["image"])]
            positive = "연기" in f["class"]
            rec = {"image": f["image"], "cond": a.cond}
            img = Image.open(p).convert("RGB")
            W, H = img.size
            if a.cond == "D":
                ds = dets.get(nfc(f["image"])) or []
                if not ds:
                    rec["no_det"] = True
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fo.flush()
                    print(f["class"], "탐지 없음", flush=True)
                    continue
                top = max(ds, key=lambda d: d["score"])
                box = top["bbox"]
                rec["det"] = {"score": top["score"], "p_yes": top["p_yes"], "smoke": top["smoke"]}
            else:
                box = label_box(p.stem, positive)
            crop = img.crop(grow(box, W, H))
            crop.thumbnail((a.crop_px, a.crop_px))
            full = img.copy()
            full.thumbnail((1024, 1024))
            images = [crop] if a.cond == "B" else [full, crop]
            text = PROMPT if a.cond == "B" else PROMPT + TWO_NOTE
            msg = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                    + [{"type": "text", "text": text}]}]
            enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))

            def allow(batch_id, sent):
                allowed = enforcer.get_allowed_tokens(sent.tolist())
                return getattr(allowed, "allowed_tokens", allowed)

            inputs = processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                   prefix_allowed_tokens_fn=allow)
            rec["sec"] = round(time.time() - t0, 2)
            rec["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            rec["tokens_in"] = int(inputs["input_ids"].shape[1])
            raw = processor.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            try:
                rec["observed"] = json.loads(raw)
            except ValueError:
                rec["observed"] = None
                rec["raw"] = raw
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            ob = rec["observed"] or {}
            print(f["class"], f.get("area_ratio"), ob.get("smoke_present"), ob.get("smoke_color"),
                  ob.get("spread"), rec["sec"], rec["vram_gb"], rec["tokens_in"], flush=True)


# ---- 표 ----

def spread_of(ratio):
    """tools/score_descriptions.py 와 같은 구간."""
    return "좁음" if ratio < 0.05 else "보통" if ratio < 0.20 else "넓음"


def load(cond):
    p = A_FILE if cond == "A" else out_path(cond)
    if not p.is_file():
        return None
    return {nfc(r["image"]): r for r in (json.loads(x) for x in open(p, encoding="utf-8") if x.strip())}


def frac(k, n):
    return f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")


def report(a):
    F = facts()
    conds = [c for c in "ABCD" if load(c)]
    R = {c: load(c) for c in conds}
    rows = defaultdict(dict)
    for c in conds:
        pos = neg = chim = cloud = col = spr = 0
        npos = nneg = nch = ncl = 0
        band = defaultdict(lambda: [0, 0])
        secs, mems = [], []
        nodet_pos = nodet_neg = 0
        fp_total = fp_reject = 0          # D: 헛탐지를 잘라 준 사진, 그중 연기 아님이라고 한 것
        for f in F:
            r = R[c].get(nfc(f["image"]))
            if r is None:
                continue
            ob = r.get("observed") or {}
            said = bool(ob.get("smoke_present")) and not r.get("no_det")
            if r.get("sec"):
                secs.append(r["sec"])
            if r.get("vram_gb"):
                mems.append(r["vram_gb"])
            if r.get("det") and not r["det"]["smoke"]:
                fp_total += 1
                fp_reject += not said
            if "연기" in f["class"]:
                npos += 1
                pos += said
                nodet_pos += bool(r.get("no_det"))
                name = next(n for lim, n in BANDS if f["area_ratio"] < lim)
                band[name][1] += 1
                band[name][0] += said
                want = ("검은색",) if f["class"] == "흑색연기" else ("회색", "흰색")
                col += ob.get("smoke_color") in want
                spr += ob.get("spread") == spread_of(f["area_ratio"])
            else:
                nneg += 1
                neg += not said
                nodet_neg += bool(r.get("no_det"))
                if f["class"] == "굴뚝":
                    nch += 1
                    chim += not said
                else:
                    ncl += 1
                    cloud += not said
        rows["연기 사진을 연기라고 함"][c] = frac(pos, npos)
        rows["연기 아닌 사진을 아니라고 함"][c] = frac(neg, nneg)
        rows["  굴뚝연기"][c] = frac(chim, nch)
        rows["  구름"][c] = frac(cloud, ncl)
        for _, n in BANDS:
            rows[f"면적 {n}"][c] = frac(*band[n])
        rows["연기색 (연기 사진)"][c] = frac(col, npos)
        rows["퍼짐 (연기 사진)"][c] = frac(spr, npos)
        rows["장당 시간(초) 중앙값"][c] = f"{sorted(secs)[len(secs) // 2]:.1f}" if secs else "-"
        rows["GPU 메모리 최대(GB)"][c] = f"{max(mems):.2f}" if mems else "-"
        if c == "D":
            rows["탐지 없어 VLM 안 부름 (연기/아님)"][c] = f"{nodet_pos} / {nodet_neg}"
            rows["헛탐지 조각을 연기 아님이라고 함"][c] = frac(fp_reject, fp_total)
    names = {"A": "A 전체만", "B": "B 조각만", "C": "C 전체+조각", "D": "D 전체+YOLO 조각"}
    print("| 항목 | " + " | ".join(names[c] for c in conds) + " |")
    print("|---|" + "---|" * len(conds))
    for k, v in rows.items():
        print(f"| {k} | " + " | ".join(v.get(c, "-") for c in conds) + " |")
    print("\nD 의 탐지 없는 사진은 '연기 아님'으로 셌다(알람이 뜨지 않는다).")


def main():
    ap = argparse.ArgumentParser(description="전체 사진 + 확대 조각 입력 비교")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cond", choices=["B", "C", "D"])
    r.add_argument("--limit", type=int, help="고르게 이만큼만 (시험용)")
    r.add_argument("--out")
    r.add_argument("--crop-px", type=int, default=768, help="조각의 긴 변")
    sub.add_parser("report")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
