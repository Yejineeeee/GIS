# -*- coding: utf-8 -*-
"""구조화 출력이 되는지 확인한다.

교수님 제안의 (iii) 방식은 VLM 이 영상에서 본 사실만 JSON 으로 내고, 문장은
규칙으로 조립한다. 그러려면 모델이 정해진 스키마를 반드시 지켜야 한다.
프롬프트로 부탁하는 것이 아니라 디코딩 단계에서 강제할 수 있는지 본다.

lm-format-enforcer 의 transformers 연동 모듈은 transformers 5.x 에서 깨진다.
핵심 부분은 모델과 무관하므로 연동만 직접 만든다.

    python structured_test.py [이미지폴더]
"""
import sys, os, json, time, glob


def build_tokenizer_data(tokenizer):
    """lm-format-enforcer 가 쓰는 토크나이저 정보를 직접 만든다."""
    from lmformatenforcer import TokenEnforcerTokenizerData
    vocab = tokenizer.get_vocab()
    regular, special = [], set(tokenizer.all_special_ids)
    for token, tid in vocab.items():
        if tid in special:
            continue
        decoded = tokenizer.convert_tokens_to_string([token])
        # 앞에 공백이 붙는 토큰인지 표시한다
        is_new_word = decoded.startswith(" ") or token.startswith("Ġ")
        regular.append((tid, decoded, is_new_word))
    return TokenEnforcerTokenizerData(
        regular_tokens=regular,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        use_bitmask=False,
        vocab_size=len(vocab),
    )


SCHEMA = {
    "type": "object",
    "properties": {
        "smoke_present": {"type": "boolean"},
        "smoke_color": {"type": "string", "enum": ["검은색", "회색", "흰색", "없음"]},
        "spread": {"type": "string", "enum": ["좁음", "보통", "넓음", "없음"]},
        "flame_visible": {"type": "boolean"},
        "terrain": {"type": "string", "enum": ["산림", "농경지", "주거지", "도로", "물", "기타"]},
    },
    "required": ["smoke_present", "smoke_color", "spread", "flame_visible", "terrain"],
}

PROMPT = """사진을 보고 아래 항목만 채운다. 사진에서 보이는 것만 적고 추측하지 않는다.

smoke_present  연기가 보이는가
smoke_color    연기 색. 없으면 "없음"
spread         연기가 퍼진 정도. 없으면 "없음"
flame_visible  불꽃이 보이는가
terrain        주로 보이는 지형"""


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from lmformatenforcer import TokenEnforcer, JsonSchemaParser

    SRC = sys.argv[1] if len(sys.argv) > 1 else "data/pilot"
    files = (sorted(glob.glob(os.path.join(SRC, "smoke_*.jpg")))[:4] +
             sorted(glob.glob(os.path.join(SRC, "nosmoke_*.jpg")))[:2])
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
    tok = processor.tokenizer

    print("토크나이저 정보 구성 중")
    t0 = time.time()
    data = build_tokenizer_data(tok)
    print(f"  {time.time()-t0:.0f}초")

    ok = 0
    for path in files:
        enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))

        def allow(batch_id, sent):
            # transformers 는 평범한 리스트를 기대하는데 TokenList 가 돌아온다
            allowed = enforcer.get_allowed_tokens(sent.tolist())
            return getattr(allowed, "allowed_tokens", allowed)

        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024))
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                 prefix_allowed_tokens_fn=allow)
        sec = time.time() - t0
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        try:
            parsed = json.loads(text)
            keys_ok = set(parsed) == set(SCHEMA["required"])
            ok += 1 if keys_ok else 0
            mark = "통과" if keys_ok else "키 불일치"
        except Exception as e:
            parsed, mark = None, f"파싱 실패 {e}"
        print(f"  {os.path.basename(path):16} {sec:5.1f}초  {mark}")
        print(f"     {text}")

    print(f"\n스키마를 지킨 건수: {ok}/{len(files)}")


if __name__ == "__main__":
    main()
