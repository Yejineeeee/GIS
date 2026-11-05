# -*- coding: utf-8 -*-
"""파이프라인에서 VLM을 부르는 부분 (방식 iii, 구조화템플릿).

VLM은 사진에서 본 사실(observed)만 내고, 문장은 src/vlm/templates.py가 조립한다.
질문 문구, 스키마, 형식 강제(lm-format-enforcer), 이미지 크기는
tools/run_structured.py와 같다. 바꾸면 비교 실험과 어긋나므로 같이 고친다.

모드
    none   VLM을 부르지 않는다. 알람만 나가고 설명은 null로 남는다
    fake   고정된 사실을 돌려준다. 연결 확인용이며 결과로 쓰면 안 된다
    qwen   Qwen2.5-VL-3B 4bit(nf4 이중 양자화). GPU가 있으면 GPU, 없으면 CPU
"""
import json
import time

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

FAKE_OBSERVED = {"smoke_present": True, "smoke_color": "회색", "spread": "보통",
                 "flame_visible": False, "terrain": "산림"}

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _tokenizer_data(tokenizer):
    """run_structured.py 와 같은 함수."""
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


class VLM:
    def __init__(self, mode="none", model_id=MODEL_ID, fake_sec=0.0):
        self.mode = mode
        self.device = None
        self.fake_sec = fake_sec      # fake 모드에서 VLM이 걸리는 시간을 흉내 낸다
        if mode == "qwen":
            self._load(model_id)
        elif mode not in ("none", "fake"):
            raise ValueError(f"알 수 없는 VLM 모드: {mode}")

    def _load(self, model_id):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        gpu = torch.cuda.is_available()
        dtype = torch.float16 if gpu else torch.float32   # CPU에서 float16은 느리거나 안 됨
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

    def observe(self, image_path):
        """(observed 또는 None, 걸린 ms, 원문)을 돌려준다."""
        t0 = time.perf_counter()
        if self.mode == "none":
            return None, None, ""
        if self.mode == "fake":
            if self.fake_sec:
                time.sleep(self.fake_sec)
            return dict(FAKE_OBSERVED), round((time.perf_counter() - t0) * 1000), "fake"

        import torch
        from PIL import Image
        from lmformatenforcer import TokenEnforcer, JsonSchemaParser
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((1024, 1024))              # run_structured.py 와 같음
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        enforcer = TokenEnforcer(self.data, JsonSchemaParser(SCHEMA))

        def allow(batch_id, sent):
            allowed = enforcer.get_allowed_tokens(sent.tolist())
            return getattr(allowed, "allowed_tokens", allowed)

        with torch.inference_mode():
            o = self.model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                    prefix_allowed_tokens_fn=allow)
        raw = self.processor.decode(o[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        ms = round((time.perf_counter() - t0) * 1000)
        try:
            ob = json.loads(raw)
            if set(ob) != set(SCHEMA["required"]):
                ob = None
        except Exception:
            ob = None
        return ob, ms, raw
