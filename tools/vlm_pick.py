# -*- coding: utf-8 -*-
"""VLM 에게 연기 밑동 십자 번호를 고르게 한다 (WBS 4.7 ① 후보 지점 선택).

tools/pick_base.py 가 그린 사진(번호 붙은 십자)과 후보 목록을 받아, Qwen 에게
"연기가 땅에서 피어오르기 시작하는 자리에 가장 가까운 십자" 를 묻는다. 고른 번호의
밑동을 좌표로 바꿨을 때 정답과 몇 m 떨어지는지를 지금 방식(conf 최고)과 상한(후보 중
최선)과 나란히 낸다.

    python tools/pick_base.py sim/frames runs/smoke_1260_960/weights/best.pt
    python tools/vlm_pick.py runs/pick_base [runs/pick_base_s15 ...]

방식은 src/pipeline/vlm_client.py 와 같다(구조화 출력, 그리디, 4bit nf4 이중 양자화,
입력 768x768 상한). 다른 점은 계산 dtype 을 bf16 으로 둔 것이다. fp16 에서는 어떤
입력에서 출력이 전부 NaN 이 된다(docs/generation_methods.md).

VLM 에는 고르기만 시키고 숫자 계산은 기하가 한다(WBS 4.7 비고). 답은 후보 번호 중
하나로 강제한다(lm-format-enforcer 정규식).

질문을 두 단계("연기 위에 있는 것만 남기고, 그중 가장 아래")로 쪼갰더니 11/04
첫 측정에서 더 나빠졌다(200m 26.6 -> 96.2m, 300m 41.3 -> 174.4m). 모델이 1단계를
못 하고 "가장 아래" 만 따라 바닥 쪽 헛탐지를 골랐다. 그래서 한 문장 질문을 쓴다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from vlm_client import MODEL_ID, _tokenizer_data  # noqa: E402

PROMPT = ("사진에 노란 십자와 번호가 있다. 연기가 땅에서 피어오르기 시작하는 자리에 "
          "가장 가까운 십자는 몇 번인가. 번호만 답하라.")
# 번호 대신 글자로 표시했을 때. 숫자 "1" 에 쏠리는지 보려고 둔다(runs/bias)
PROMPT_ALPHA = ("사진에 노란 십자와 글자가 있다. 연기가 땅에서 피어오르기 시작하는 자리에 "
                "가장 가까운 십자는 어느 글자인가. 글자만 답하라.")


class Picker:
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        gpu = torch.cuda.is_available()
        dtype = torch.bfloat16 if gpu else torch.float32
        self.device = "cuda:0" if gpu else "cpu"
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.image_processor.max_pixels = 768 * 768
        self.processor.image_processor.min_pixels = 256 * 256
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=quant, device_map=self.device, dtype=dtype)
        self.data = _tokenizer_data(self.processor.tokenizer)

    def pick(self, image_path, labels, alpha=False):
        """(고른 번호 또는 None, 걸린 ms, 원문)."""
        import torch
        from PIL import Image
        from lmformatenforcer import TokenEnforcer, RegexParser
        # 답은 후보 번호 하나로만 강제한다. JSON({"no": "3"})으로 강제했더니 모델이
        # 중괄호 뒤에 공백과 줄바꿈만 내다가 토큰이 끝나 5장 중 4장이 답을 못 냈다
        # 긴 표시부터 넣어야 "1" 이 "12" 를 가로채지 않는다
        pattern = "(" + "|".join(sorted(labels, key=lambda v: (-len(v), v))) + ")"
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((1024, 1024))               # vlm_client.py 와 같음
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": PROMPT_ALPHA if alpha else PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        enforcer = TokenEnforcer(self.data, RegexParser(pattern))

        def allow(batch_id, sent):
            allowed = enforcer.get_allowed_tokens(sent.tolist())
            return getattr(allowed, "allowed_tokens", allowed)

        with torch.inference_mode():
            o = self.model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                    prefix_allowed_tokens_fn=allow)
        raw = self.processor.decode(o[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        ms = round((time.perf_counter() - t0) * 1000)
        try:
            no = raw if raw in labels else None
        except Exception:
            no = None
        return no, ms, raw


def run_dir(picker, d, out):
    rows = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8") if l.strip()]
    print(f"\n== {d}")
    print(f"{'사진':<14}{'거리':>6}{'후보':>5}{'지금 방식':>11}{'VLM':>10}{'상한':>9}{'고른 번호':>9}{'최선':>5}{'ms':>7}")
    sums = {"now": 0.0, "vlm": 0.0, "best": 0.0}
    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            cands = r["candidates"]
            by_no = {c.get("label", str(c["no"])): c for c in cands}
            best = min(cands, key=lambda c: c["err_m"] if c["err_m"] is not None else float("inf"))
            top = max(cands, key=lambda c: c["score"])          # 지금 방식에 가장 가까운 점
            if len(cands) == 1:
                no, ms, raw = cands[0].get("label", str(cands[0]["no"])), 0, "(후보 하나)"
            else:
                no, ms, raw = picker.pick(d / r["pick_image"], sorted(by_no),
                                          alpha=r.get("labels") == "alpha")
            got = by_no.get(no)
            vlm_err = got["err_m"] if got else None
            rec = {"image": r["image"], "distance_m": r["distance_m"], "n_candidates": len(cands),
                   "order": r.get("order", "conf"), "seed": r.get("seed"),
                   "now_err_m": r.get("now_err_m"), "vlm_no": no, "vlm_err_m": vlm_err,
                   "best_no": best.get("label", str(best["no"])), "best_err_m": best["err_m"],
                   "top_conf_no": top.get("label", str(top["no"])),
                   "ms": ms, "raw": raw,
                   "picked_det_id": got["det_id"] if got else None,
                   "picked_xy": [got["x"], got["y"]] if got else None}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fmt = lambda v: "-" if v is None else f"{v:.1f}m"
            print(f"{Path(r['image']).stem:<14}{str(r['distance_m'] or '-'):>5}m{len(cands):>5}"
                  f"{fmt(r.get('now_err_m')):>11}{fmt(vlm_err):>10}{fmt(best['err_m']):>9}"
                  f"{str(no):>8}{best.get('label', str(best['no'])):>4}{ms:>7}")
            if None not in (r.get("now_err_m"), vlm_err, best["err_m"]):
                sums["now"] += r["now_err_m"]
                sums["vlm"] += vlm_err
                sums["best"] += best["err_m"]
                n += 1
    if n:
        print(f"평균 ({n}장)  지금 방식 {sums['now']/n:.1f}m   VLM {sums['vlm']/n:.1f}m   "
              f"상한 {sums['best']/n:.1f}m")
    print(f"결과: {out}")


def main():
    ap = argparse.ArgumentParser(description="VLM 으로 연기 밑동 십자 고르기 (4.7 ①)")
    ap.add_argument("pick_dirs", nargs="+",
                    help="pick_base.py 출력 폴더들 (candidates.jsonl 과 *_pick.png). 모델은 한 번만 올린다")
    a = ap.parse_args()
    picker = Picker()
    for d in a.pick_dirs:
        d = Path(d)
        run_dir(picker, d, d / "vlm_pick.jsonl")


if __name__ == "__main__":
    main()
