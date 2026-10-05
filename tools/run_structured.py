# -*- coding: utf-8 -*-
"""생성 방식 (iii) 구조화 출력과 템플릿.

VLM 은 영상에서 본 사실만 JSON 으로 내고, 문장은 규칙이 조립한다. 디코딩 단계에서
스키마를 강제하므로 형식을 어길 수 없고, 문장을 코드가 만들므로 판정 규칙도 어길
수 없다. 그래도 구현에 문제가 없는지 보려고 규칙 검사를 함께 돌린다.

    python run_structured.py [이미지폴더] [출력파일]
"""
import sys, os, json, time, glob

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/pilot"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/method_iii.jsonl"

SCHEMA = {
    "type": "object",
    "properties": {
        "smoke_present": {"type": "boolean"},
        "smoke_color": {"type": "string", "enum": ["검은색", "회색", "흰색", "없음"]},
        "spread": {"type": "string", "enum": ["좁음", "보통", "넓음", "없음"]},
        "flame_visible": {"type": "boolean"},
        "terrain": {"type": "string",
                    "enum": ["산림", "농경지", "주거지", "도로", "물", "기타"]},
    },
    "required": ["smoke_present", "smoke_color", "spread", "flame_visible", "terrain"],
}

PROMPT = """사진을 보고 아래 항목만 채운다. 사진에서 보이는 것만 적고 추측하지 않는다.

smoke_present  연기가 보이는가
smoke_color    연기 색. 없으면 "없음"
spread         연기가 퍼진 정도. 없으면 "없음"
flame_visible  불꽃이 보이는가
terrain        주로 보이는 지형"""


def fake_alert(i):
    kind = i % 3
    if kind == 0:
        verdict, radius = "확정", 20
        land = [{"name": "보전녹지지역", "ratio": 0.85},
                {"name": "제1종일반주거지역", "ratio": 0.15}]
    elif kind == 1:
        verdict, radius = "모호", 60
        land = [{"name": "보전녹지지역", "ratio": 0.52},
                {"name": "제2종일반주거지역", "ratio": 0.41}]
    else:
        verdict, radius = "미확정", 110
        land = [{"name": "자연녹지지역", "ratio": 0.38},
                {"name": "제1종일반주거지역", "ratio": 0.33}]
    return {"alert_id": f"p-{i:04d}", "location_verdict": verdict,
            "position": {"lat": 37.74652, "lon": 128.88544, "error_radius_m": radius},
            "land_use": land, "evidence": {"buildings_in_radius": 12},
            "description": None}


def build_tokenizer_data(tokenizer):
    from lmformatenforcer import TokenEnforcerTokenizerData
    vocab = tokenizer.get_vocab()
    special = set(tokenizer.all_special_ids)
    regular = []
    for token, tid in vocab.items():
        if tid in special:
            continue
        decoded = tokenizer.convert_tokens_to_string([token])
        regular.append((tid, decoded, decoded.startswith(" ") or token.startswith("Ġ")))
    return TokenEnforcerTokenizerData(
        regular_tokens=regular,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        use_bitmask=False, vocab_size=len(vocab))


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from lmformatenforcer import TokenEnforcer, JsonSchemaParser
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
    import templates, verify

    files = (sorted(glob.glob(os.path.join(SRC, "smoke_*.jpg"))) +
             sorted(glob.glob(os.path.join(SRC, "nosmoke_*.jpg"))))
    if not files:
        raise SystemExit(f"{SRC} 에 이미지가 없습니다.")

    mid = "Qwen/Qwen2.5-VL-3B-Instruct"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(mid)
    processor.image_processor.max_pixels = 768 * 768
    processor.image_processor.min_pixels = 256 * 256
    model = AutoModelForImageTextToText.from_pretrained(
        mid, quantization_config=quant, device_map="cuda:0", dtype=torch.float16)
    data = build_tokenizer_data(processor.tokenizer)
    print(f"{mid} 올림. 이미지 {len(files)}장", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    schema_ok = rule_ok = rule_total = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for i, path in enumerate(files):
            name = os.path.basename(path)
            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024))
            alert = fake_alert(i)

            enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))

            def allow(batch_id, sent):
                allowed = enforcer.get_allowed_tokens(sent.tolist())
                return getattr(allowed, "allowed_tokens", allowed)

            msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text", "text": PROMPT}]}]
            inputs = processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                   prefix_allowed_tokens_fn=allow)
            sec = time.time() - t0
            raw = processor.decode(o[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            try:
                observed = json.loads(raw)
                ok_schema = set(observed) == set(SCHEMA["required"])
            except Exception:
                observed, ok_schema = None, False
            schema_ok += ok_schema

            desc = templates.build(observed, alert) if observed else None
            bad = verify.check(desc, alert, observed)
            counted = name.startswith("smoke")
            rule_total += counted
            rule_ok += counted and not bad

            out.write(json.dumps({
                "image": name, "model": "qwen", "cond": "D", "method": "구조화템플릿",
                "verdict": alert["location_verdict"],
                "radius": alert["position"]["error_radius_m"],
                "schema_ok": ok_schema, "observed": observed,
                "violations": bad, "sec": round(sec, 1),
                "description": desc}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {name:16} {alert['location_verdict']:4} {sec:5.1f}초  "
                  f"{'통과' if not bad else bad}", flush=True)
            print(f"     {desc}", flush=True)

    print(f"\n스키마 준수 {schema_ok}/{len(files)}   판정 준수 {rule_ok}/{len(files)}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
