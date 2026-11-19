# -*- coding: utf-8 -*-
"""후보 검증기 자리에 공개 미세조정 모델을 넣어 원본과 비교한다 (WBS 4.7 ③ 보충).

검증기는 Qwen2.5-VL-3B 원본에 "이 사진에 연기가 있는가" 를 묻는다(tools/vlm_verify.py).
같은 원본을 산불 사진으로 미세조정한 공개 모델 ForestFireVLM-3B(leon-se/ForestFireVLM-3B,
Apache-2.0, Drones 2025 9(5) 347)가 이 자리에서 더 나은지 본다. 직접 미세조정할지를
정하기 전에 보는 값이다.

    python tools/vlm_verify_ffvlm.py
    python tools/vlm_verify_ffvlm.py --models base ffvlm --prompts ko en native
    python tools/vlm_verify_ffvlm.py --report-only

탐지와 자른 조각은 다시 만들지 않고 검증기 결과를 그대로 읽는다.
    runs/verify_real        평가셋 100장 (탐지 173개)
    runs/verify_situation   장소를 섞은 150장 (탐지 165개)
조각, 4bit nf4 이중 양자화, bf16, 448px 입력은 검증기와 같다.

묻는 방식 세 가지
    ko      검증기 질문 그대로. 첫 토큰 P(예)/(P(예)+P(아))
    en      같은 질문을 영어로. P(Yes)/(P(Yes)+P(No))
    native  ForestFireVLM 이 학습한 질문(11개 항목 JSON)을 주고, 답의 첫 항목
            forest_fire_smoke_visible 값 자리에서 P(Yes)/(P(Yes)+P(No))

미세조정 모델은 영어 JSON 으로 학습했으므로 ko 만 보면 불리하다. 그래서 두 모델 모두
세 방식으로 잰다. 결과는 조각 하나마다 한 줄씩 저장하고, 다시 돌리면 끝난 것은 건너뛴다.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

MODELS = {"base": "Qwen/Qwen2.5-VL-3B-Instruct", "ffvlm": "leon-se/ForestFireVLM-3B"}
SETS = {"eval100": "runs/verify_real", "mix150": "runs/verify_situation"}
OUT_PX = 448

PROMPT_KO = "이 사진에 연기가 있는가. 예 또는 아니오로만 답하라."
PROMPT_EN = "Is there smoke in this image? Answer only Yes or No."
# github.com/leon-seidel/ForestFireVLM templates/prompt.py 그대로
PROMPT_NATIVE = """
Analyze the aerial image from the UAV's camera and detect potential forest fires. Answer the following questions based on the image analysis
in a JSON schema. If a question cannot be determined from the image answer with 'Cannot be determined'.

  forest_fire_smoke_visible: Is smoke from a forest fire visible in the image? Answer with one of the following options: ['Yes', 'No']
  forest_fire_flames_visible: Are flames from a forest fire visible in the image? Answer with one of the following options: ['Yes', 'No']
  confirm_uncontrolled_forest_fire: Can you confirm that this is an uncontrolled forest fire? Answer with one of the following options: ['Yes', 'Closer investigation required', 'No forest fire visible']
  fire_state: What is the current state of the forest fire? Answer with one of the following options: ['Ignition Phase', 'Growth Phase', 'Fully Developed Phase', 'Decay Phase', 'Cannot be determined', 'No forest fire visible']
  fire_type: What type of fire is it? Answer with one of the following options: ['Ground Fire', 'Surface Fire', 'Crown Fire', 'Cannot be determined', 'No forest fire visible']
  fire_intensity: What is the intensity of the fire? Answer with one of the following options: ['Low', 'Moderate', 'High', 'Cannot be determined', 'No forest fire visible']
  fire_size: What is the size of the fire? Answer with one of the following options: ['Small', 'Medium', 'Large', 'Cannot be determined', 'No forest fire visible']
  fire_hotspots: Does the forest fire have multiple hotspots? Answer with one of the following options: ['Multiple hotspots', 'One hotspot', 'Cannot be determined', 'No forest fire visible']
  infrastructure_nearby: Is there infrastructure visible near the forest fire? Answer with one of the following options: ['Yes', 'No', 'Cannot be determined', 'No forest fire visible']
  people_nearby: Are there people visible near the forest fire? Answer with one of the following options: ['Yes', 'No', 'Cannot be determined', 'No forest fire visible']
  tree_vitality: Describe the vitality of the trees around the fire. Answer with one of the following options: ['Vital', 'Moderate Vitality', 'Declining', 'Dead', 'Cannot be determined', 'No forest fire visible']
"""
NATIVE_PREFIX = '{"forest_fire_smoke_visible": "'


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def load_dets():
    """[{set, i, j, crop, smoke, cls, base_p_yes}] 검증기 결과에서 읽는다."""
    rows = []
    for name, d in SETS.items():
        d = ROOT / d
        for line in open(d / "verify_real.jsonl", encoding="utf-8"):
            r = json.loads(line)
            name_ = r["image"]
            try:                       # 평가셋 100장 결과는 파일 이름이 cp437 로 깨져 저장돼 있다
                name_ = name_.encode("cp437").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            parts = name_.split("_")
            cls = parts[1] if len(parts) > 1 else "?"
            for det in r["dets"]:
                rows.append({"set": name, "i": r["i"], "j": det["j"],
                             "crop": str(d / "crops" / f"{r['i']:03d}_{det['j']}.png"),
                             "smoke": bool(det["smoke"]), "cls": cls, "positive": r["positive"],
                             "score": det["score"], "saved_p_yes": det["p_yes"]})
    return rows


class Scorer:
    def __init__(self, model_id):
        import torch
        from vlm_pick import Picker
        self.torch = torch
        self.p = Picker(model_id)
        self.p.processor.image_processor.max_pixels = OUT_PX * OUT_PX
        tok = self.p.processor.tokenizer
        first = lambda s: tok.encode(s, add_special_tokens=False)[0]
        self.ids = {"ko": (first("예"), first("아니오")),
                    "en": (first("Yes"), first("No")),
                    "native": (first("Yes"), first("No"))}
        self.native_prefix = NATIVE_PREFIX

    def _inputs(self, image, text, prefix=""):
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": text}]}]
        chat = self.p.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return self.p.processor(text=[chat + prefix], images=[image],
                                return_tensors="pt").to(self.p.model.device)

    def sample_native(self, image, n=40):
        """미세조정 모델이 실제로 내는 JSON 모양을 본다(공백, 줄바꿈). 접두어를 여기에 맞춘다."""
        inp = self._inputs(image, PROMPT_NATIVE)
        with self.torch.inference_mode():
            o = self.p.model.generate(**inp, max_new_tokens=n, do_sample=False)
        raw = self.p.processor.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.match(r'\s*(?:```json\s*)?\{\s*"forest_fire_smoke_visible"\s*:\s*"', raw)
        if m:
            self.native_prefix = m.group(0)
        return raw

    def p_yes(self, image, prompt):
        text = {"ko": PROMPT_KO, "en": PROMPT_EN, "native": PROMPT_NATIVE}[prompt]
        prefix = self.native_prefix if prompt == "native" else ""
        inp = self._inputs(image, text, prefix)
        with self.torch.inference_mode():
            pr = self.torch.softmax(self.p.model(**inp).logits[0, -1].float(), -1)
        y, n = (float(pr[t]) for t in self.ids[prompt])
        return y / (y + n + 1e-12), y + n


def run(a, rows, out):
    from PIL import Image
    import torch
    done = set()
    if out.is_file():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["prompt"], r["set"], r["i"], r["j"]))
    for mname in a.models:
        todo = [(r, p) for r in rows for p in a.prompts
                if (mname, p, r["set"], r["i"], r["j"]) not in done]
        if not todo:
            print(mname, "이미 끝남", flush=True)
            continue
        sc = Scorer(MODELS[mname])
        if "native" in a.prompts:
            raw = sc.sample_native(Image.open(rows[0]["crop"]).convert("RGB"))
            print(mname, "native 출력 예:", repr(raw[:160]), "/ 접두어:", repr(sc.native_prefix), flush=True)
        torch.cuda.reset_peak_memory_stats()
        with open(out, "a", encoding="utf-8") as fo:
            cache = (None, None)
            for k, (r, p) in enumerate(todo):
                if cache[0] != r["crop"]:
                    cache = (r["crop"], Image.open(r["crop"]).convert("RGB"))
                t0 = time.perf_counter()
                py, mass = sc.p_yes(cache[1], p)
                rec = {"model": mname, "prompt": p, "set": r["set"], "i": r["i"], "j": r["j"],
                       "smoke": r["smoke"], "cls": r["cls"], "positive": r["positive"],
                       "score": r["score"], "p_yes": round(py, 4), "mass": round(mass, 4),
                       "ms": round((time.perf_counter() - t0) * 1000)}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
                if k % 50 == 0:
                    print(mname, k, "/", len(todo), flush=True)
        print(mname, "GPU 최대", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GB", flush=True)
        del sc
        torch.cuda.empty_cache()


def report(rows, out, tau):
    import statistics as st
    recs = [json.loads(l) for l in open(out, encoding="utf-8")]
    print("\n검증기에 저장된 값(base, ko)으로 다시 계산한 AUC")
    for s in SETS:
        pos = [r["saved_p_yes"] for r in rows if r["set"] == s and r["smoke"]]
        neg = [r["saved_p_yes"] for r in rows if r["set"] == s and not r["smoke"]]
        print(f"  {s}: 연기 {len(pos)} 헛탐지 {len(neg)} AUC {auc(pos, neg):.3f}")
    print(f"\n{'세트':8}{'모델':7}{'질문':8}{'AUC':>7}{'연기 남김':>12}{'헛탐지 거름':>12}{'답 확률합':>10}{'ms':>6}")
    for s in SETS:
        for m in MODELS:
            for p in ("ko", "en", "native"):
                rs = [r for r in recs if r["set"] == s and r["model"] == m and r["prompt"] == p]
                if not rs:
                    continue
                pos = [r["p_yes"] for r in rs if r["smoke"]]
                neg = [r["p_yes"] for r in rs if not r["smoke"]]
                kp = sum(v >= tau for v in pos)
                rn = sum(v < tau for v in neg)
                print(f"{s:8}{m:7}{p:8}{auc(pos, neg):>7.3f}{kp:>7}/{len(pos):<4}{rn:>7}/{len(neg):<4}"
                      f"{st.mean(r['mass'] for r in rs):>10.2f}{st.median(r['ms'] for r in rs):>6.0f}")
    print(f"\n헛탐지를 종류별로 거른 수 (문턱 {tau})")
    for s in SETS:
        cls = sorted({r["cls"] for r in recs if r["set"] == s and not r["smoke"]})
        for m in MODELS:
            for p in ("ko", "en", "native"):
                rs = [r for r in recs if r["set"] == s and r["model"] == m and r["prompt"] == p and not r["smoke"]]
                if not rs:
                    continue
                cell = "  ".join(f"{c} {sum(r['p_yes'] < tau for r in rs if r['cls'] == c)}/{sum(r['cls'] == c for r in rs)}"
                                 for c in cls)
                print(f"  {s:8}{m:7}{p:8}{cell}")


def main():
    ap = argparse.ArgumentParser(description="검증기: 원본과 공개 미세조정 모델 비교 (4.7)")
    ap.add_argument("--models", nargs="+", default=["ffvlm", "base"], choices=list(MODELS))
    ap.add_argument("--prompts", nargs="+", default=["ko", "en", "native"])
    ap.add_argument("--out", default="runs/verify_ffvlm/scores.jsonl")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="조각 수 제한(동작 확인용)")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    rows = load_dets()
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print("조각", len(rows), "개", {s: sum(r["set"] == s for r in rows) for s in SETS}, flush=True)
    if not a.report_only:
        run(a, rows, out)
    report(rows, out, a.tau)


if __name__ == "__main__":
    main()
