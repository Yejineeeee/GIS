# -*- coding: utf-8 -*-
"""후보 검증기를 OpenVINO GenAI 로 인텔 NPU, 내장 GPU, CPU 에서 돌린다 (WBS 7.6).

모델은 NPU 용으로 변환해 공개된 Qwen2.5-VL-3B-Instruct int4(llmware/Qwen2.5-VL-3B-Instruct-ov-int4-npu)다.
공식 배포가 아니고, 양자화(int4 대칭, AWQ)가 지금 쓰는 GGUF Q4_K_M, bitsandbytes nf4 와 다르다.

질문과 조각은 검증기와 같다(조각 448px, "이 사진에 연기가 있는가. 예 또는 아니오로만 답하라.").
VLMPipeline 은 다음 토큰의 확률을 바로 주지 않는다. 빔 2개로 두 후보의 점수를 받으면
P(예)/(P(예)+P(아))를 낼 수 있지만, NPU 는 빔 탐색을 지원하지 않는다(그리디와 샘플링만).
그래서 NPU 에서는 그리디로 한 토큰을 만들고, 고른 토큰의 확률 q 로 "예"면 q, "아"면 1 - q 를
P(예)의 어림값(p_approx)으로 쓴다. 예와 아니오 밖의 토큰에 간 확률을 무시하므로 정확한 값이 아니다.

    python tools/vlm_ov_devices.py --model <변환 모델 폴더> --device NPU \\
        --dets runs/verify_real/verify_real.jsonl --images data/eval_100/images/eval \\
        --out results/ov_verify_npu.jsonl
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from vlm_gguf_compare import crop_det, fixed, VERIFY_PROMPT, auc  # noqa: E402


YES = ("예", "네", "yes")       # 모델이 영어나 "네"로도 답한다
NO = ("아", "no")


def to_tensor(img):
    import openvino as ov
    a = np.array(img.convert("RGB"), dtype=np.uint8)
    return ov.Tensor(a[None])


def ask(pipe, img, beam):
    import openvino_genai as og
    cfg = og.GenerationConfig()
    cfg.max_new_tokens = 1
    if beam:
        cfg.num_beams = 2
        cfg.num_return_sequences = 2
        cfg.num_beam_groups = 1
    t0 = time.perf_counter()
    r = pipe.generate(VERIFY_PROMPT, image=to_tensor(img), generation_config=cfg)
    sec = time.perf_counter() - t0
    texts = [t.strip() for t in r.texts]
    scores = list(r.scores)
    y = n = None
    for t, s in zip(texts, scores):
        if t.lower().startswith(YES) and y is None:
            y = math.exp(s)
        elif t.lower().startswith(NO) and n is None:
            n = math.exp(s)
    if y is not None and n is not None:
        p = y / (y + n)
    elif texts and texts[0].lower().startswith(YES):
        p = None
    else:
        p = None
    q = math.exp(scores[0]) if scores else None
    approx = None
    if q is not None and texts:
        approx = q if texts[0].lower().startswith(YES) else (1 - q if texts[0].lower().startswith(NO) else None)
    return {"texts": texts, "scores": [round(s, 4) for s in scores], "p_yes": None if p is None else round(p, 4),
            "p_approx": None if approx is None else round(approx, 4),
            "answer_yes": bool(texts) and texts[0].lower().startswith(YES), "sec": round(sec, 2)}


def main():
    ap = argparse.ArgumentParser(description="OpenVINO 로 검증기를 칩마다")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--dets", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-beam", action="store_true", help="빔 없이 답 글자만")
    ap.add_argument("--threads", type=int, default=4, help="CPU 일 때 스레드 수")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        import openvino_genai as og
        from PIL import Image
        cfg = {"CACHE_DIR": str(Path(a.model) / "model_cache")}
        if a.device == "CPU":
            cfg["INFERENCE_NUM_THREADS"] = a.threads
        t0 = time.perf_counter()
        pipe = og.VLMPipeline(a.model, a.device, **cfg)
        load = time.perf_counter() - t0
        print(f"{a.device} 모델 준비 {load:.1f}초", flush=True)
        disk = {fixed(p.name): p for p in Path(a.images).glob("*.jpg")}
        recs = [json.loads(l) for l in open(a.dets, encoding="utf-8")]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(a.out, "w", encoding="utf-8") as fo:
            fo.write(json.dumps({"device": a.device, "load_sec": round(load, 1)}, ensure_ascii=False) + "\n")
            for r in recs:
                if not r["dets"]:
                    continue
                img = Image.open(disk[fixed(r["image"])]).convert("RGB")
                for d in r["dets"]:
                    cr = crop_det(img, d["bbox"], min_side=max(160, img.size[0] // 6))
                    res = ask(pipe, cr, not a.no_beam)
                    row = {"image": fixed(r["image"]), "positive": r["positive"], "j": d["j"], "smoke": d["smoke"],
                           "saved": d["p_yes"], **res}
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fo.flush()
                    n += 1
                    print(n, "연기" if d["smoke"] else "헛", d["p_yes"], "→", res["texts"], res["p_yes"], res["sec"], "초",
                          flush=True)
                    if a.limit and n >= a.limit:
                        break
                if a.limit and n >= a.limit:
                    break
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    head, rows = rows[0], rows[1:]
    pos = [r for r in rows if r["smoke"]]
    neg = [r for r in rows if not r["smoke"]]
    print(f"\n[{head['device']}] 탐지 {len(rows)}개 (연기 {len(pos)}, 헛탐지 {len(neg)}), 모델 준비 {head['load_sec']}초")
    print(f"  답 '예' 비율: 연기 {sum(r['answer_yes'] for r in pos)}/{len(pos)}, 헛탐지 {sum(r['answer_yes'] for r in neg)}/{len(neg)}")
    key = "p_yes" if any(r["p_yes"] is not None for r in rows) else "p_approx"
    got = [r for r in rows if r[key] is not None]
    print(f"  확률({key})을 읽은 것 {len(got)}/{len(rows)}")
    if got:
        pp = [r[key] for r in got if r["smoke"]]
        nn = [r[key] for r in got if not r["smoke"]]
        print(f"  AUC {auc(pp, nn):.3f}   문턱 0.7: 연기 남김 {sum(p >= 0.7 for p in pp)}/{len(pp)}, 헛탐지 거름 {sum(p < 0.7 for p in nn)}/{len(nn)}")
    secs = [r["sec"] for r in rows]
    print(f"  1회 중앙값 {statistics.median(secs):.2f}초 (최소 {min(secs):.2f} ~ 최대 {max(secs):.2f})")


if __name__ == "__main__":
    main()
