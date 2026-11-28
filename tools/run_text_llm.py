# -*- coding: utf-8 -*-
"""생성 방식 (i) 를 글 전용 LLM 으로 돌린다.

질문: 모델이 직접 설명을 쓸 때 판정을 어기는 것이 사진을 봐서 생기는가.
방식 (i) 은 Qwen2.5-VL-3B 가 사진과 조건 D 프롬프트를 받아 문장을 쓴다. 여기서는
같은 계열의 글 전용 모델 Qwen2.5-3B-Instruct 에 같은 프롬프트를 넣고 사진만 뺀다.
프롬프트 첫 줄의 "사진" 표시도 사진이 없으니 지운다. 나머지(탐지 값, 알람, 지시,
예시, 4bit nf4, bf16, 그리디, repetition_penalty 1.15, 최대 140토큰)는 방식 (i) 과 같다.

입력은 사실 목록에서 만든다. 사진 파일은 읽지 않는다.

    python tools/run_text_llm.py [사실목록] [출력파일]

출력 한 줄의 형식은 방식 (i) 과 같아서 score_descriptions.py 로 그대로 채점한다.
    {"image":"a.jpg","model":"qwen-text","cond":"D","text":"...","sec":3.1,
     "tokens_in":420,"verdict":"확정"}
"""
import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "vlm"))
import eval_alert
import prompts as P

MODEL_ID = os.environ.get("TEXT_LLM", "Qwen/Qwen2.5-3B-Instruct")
FACTS = sys.argv[1] if len(sys.argv) > 1 else "results/eval_facts.jsonl"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "results/eval_method_i_text.jsonl"
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))


def items():
    with open(FACTS, encoding="utf-8") as f:
        facts = [json.loads(line) for line in f if line.strip()]
    out = [(fact["image"], fact, eval_alert.detection(fact["image"], i),
            eval_alert.alert(fact, i)) for i, fact in enumerate(facts)]
    return out[:LIMIT] if LIMIT else out


def prompt(det, alert):
    text = P.build("D", det, alert)
    head = "사진\n\n"
    assert text.startswith(head)
    return text[len(head):]


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if os.path.exists(OUTFILE):
        raise SystemExit(f"{OUTFILE} 이 이미 있습니다. 덮어쓰지 않습니다.")
    dt = torch.bfloat16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dt,
                               bnb_4bit_use_double_quant=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map="cuda:0", dtype=dt)
    print(f"로딩 {time.time()-t0:.0f}초  가중치 "
          f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    todo = items()
    print(f"{len(todo)}건", flush=True)
    os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as out:
        for name, fact, det, alert in todo:
            msg = [{"role": "user", "content": prompt(det, alert)}]
            inputs = tok.apply_chat_template(msg, add_generation_prompt=True,
                                             tokenize=True, return_dict=True,
                                             return_tensors="pt").to(model.device)
            ids = inputs["input_ids"]
            torch.cuda.synchronize(); t = time.time()
            with torch.inference_mode():
                o = model.generate(**inputs,
                                   max_new_tokens=140, do_sample=False,
                                   repetition_penalty=1.15,
                                   pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize(); sec = time.time() - t
            n_in = ids.shape[1]
            text = tok.decode(o[0][n_in:], skip_special_tokens=True).strip()
            rec = {"image": name, "model": "qwen-text", "cond": "D", "text": text,
                   "sec": round(sec, 1), "tokens_in": n_in,
                   "verdict": fact["verdict"]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {fact['verdict']:3} {sec:4.1f}초  {text[:60]}", flush=True)
    print(f"저장: {OUTFILE}  전체 {time.time()-t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
