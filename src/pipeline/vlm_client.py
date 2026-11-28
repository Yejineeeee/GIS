# -*- coding: utf-8 -*-
"""파이프라인에서 VLM을 부르는 부분 (방식 iii, 구조화템플릿).

VLM은 사진에서 본 사실(observed)만 내고, 문장은 src/vlm/templates.py가 조립한다.
질문 문구, 스키마, 형식 강제(lm-format-enforcer), 이미지 크기는
tools/run_structured.py와 같다. 바꾸면 비교 실험과 어긋나므로 같이 고친다.

모드
    none   VLM을 부르지 않는다. 알람만 나가고 설명은 null로 남는다
    fake   고정된 사실을 돌려준다. 연결 확인용이며 결과로 쓰면 안 된다
    qwen   Qwen2.5-VL-3B 4bit(nf4 이중 양자화). GPU가 있으면 GPU, 없으면 CPU
    gguf   llama-server(llama.cpp)에 올린 Qwen2.5-VL-3B Q4_K_M 에 HTTP 로 묻는다 (--vlm-url).
           현장 기기(CPU)의 구성이다. nf4 는 CPU 에서 1장 180초, GGUF 는 조각 360 에서 9초쯤
           (docs/vlm_speed.md, docs/inputs_cpu.md). 질문, 스키마, 조각, 검증 질문은 qwen 모드와 같다

설명 입력 (11/19 회의, docs/inputs_cpu.md)
    observe() 는 사진 전체가 아니라 탐지 조각을 넣는다. 탐지 상자 그대로(여백 없음, 최소 96px)
    자르고 긴 변을 360px 로 줄인다(crop_px). CPU 4코어에서 사진 전체 48초 → 조각 360 중앙값 9초쯤,
    연기 인식은 유지(67/70). 연기가 아닌 것을 거르는 일은 검증기가 맡는다. crop_px=0 이면 예전처럼 전체

후보 검증기 (WBS 4.7, docs/vlm_base_pick.md)
    p_smoke() 는 탐지 박스 주변을 잘라 "연기가 있는가"를 묻고 P(예)를 돌려준다.
    설명과 같은 모델을 쓴다. tools/vlm_verify.py 와 질문, 자르기, 계산이 같다.
"""
import base64
import io
import json
import threading
import time
import urllib.request

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

VERIFY_PROMPT = "이 사진에 연기가 있는가. 예 또는 아니오로만 답하라."
VERIFY_YES, VERIFY_NO = 127027, 52959    # "예", "아"(아니오의 첫 토큰), Qwen2.5-VL 토크나이저
VERIFY_PX = 448

# 불꽃은 따로 묻는다 (docs/vlm_flame_obs.md). 구조화 질문에 섞어 물으면 불꽃 19장 중 0장을 "있다"고
# 했지만, 같은 조각에 이 질문만 하면 AUC 0.86 이었다(docs/vlm_compare.md). 문턱은 검증기와 같은 0.7
FLAME_PROMPT = "이 사진에 불꽃이 보이는가. 예 또는 아니오로만 답하라."
FLAME_TAU = 0.7
# gguf(Q4_K_M, CPU 현장 하한 구성)는 확률이 전체로 높게 나와 문턱 0.7 에서 헛말이 38장 중 12장이고,
# 캐시가 맞지 않으면 질문 한 번에 16초가 더 든다. 그래서 qwen(nf4, 소형 GPU 기기)에서만 묻는다
FLAME_MODES = ("qwen", "fake")

# 굴뚝은 사진 전체를 보여 주고 넷 중 하나를 고르게 한다 (tools/vlm_situation.py 와 같은 질문, docs/vlm_situation.md).
# "2" 가 가장 높으면 굴뚝으로 본다. 산불 사진을 굴뚝이라고 한 적은 130장 중 0장이지만, 공장 지대 밖의 굴뚝은
# 잘 못 찾는다(15장 중 1장). 사진 전체는 CPU 에서 40초가 넘어 불꽃과 같이 GPU 구성에서만 묻는다
SITUATION_PROMPT = ("드론으로 찍은 사진이다. 사진에 보이는 것을 아래에서 하나 골라 번호만 답하라.\n"
                    "1 산이나 들에서 난 불의 연기\n"
                    "2 굴뚝이나 공장, 건물 설비에서 나오는 연기나 수증기\n"
                    "3 구름이나 안개\n"
                    "4 연기가 없다")
SITUATION_CHOICES = ("산불", "굴뚝", "구름", "없음")
CHIMNEY_MODES = ("qwen", "fake")
DESC_MIN_SIDE = 96       # 설명 조각이 너무 작으면 모델이 아무것도 못 본다 (tools/prep_crops.py 와 같음)


def crop_desc(img, bbox, crop_px=360):
    """설명용 탐지 조각. 상자 그대로(최소 96px) 자르고 긴 변을 crop_px 로 줄인다
    (tools/measure_inputs_cpu.py 의 조각과 같음. 거기서는 프레임의 탐지 상자를 모두 감쌌다)."""
    W, H = img.size
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    w, h = max(w, DESC_MIN_SIDE), max(h, DESC_MIN_SIDE)
    crop = img.crop((max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
                     min(W, int(cx + w / 2)), min(H, int(cy + h / 2))))
    crop.thumbnail((crop_px, crop_px))
    return crop


def crop_det(img, bbox, margin=0.25, min_side=160):
    """탐지 박스를 사방 25% 넓혀 정사각형으로 잘라 448px 로 키운다 (tools/vlm_verify.py 와 같음)."""
    from PIL import Image
    W, H = img.size
    x, y, w, h = bbox
    side = max(max(w, h) * (1 + 2 * margin), min_side)
    cx, cy, half = x + w / 2, y + h / 2, side / 2
    b = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    cv = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), (128, 128, 128))
    cv.paste(img.crop((max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H))),
             (max(-b[0], 0), max(-b[1], 0)))
    return cv.resize((VERIFY_PX, VERIFY_PX), Image.LANCZOS)


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
    def __init__(self, mode="none", model_id=MODEL_ID, fake_sec=0.0, crop_px=360, url="http://127.0.0.1:8099"):
        self.mode = mode
        self.crop_px = crop_px        # 설명에 넣는 탐지 조각의 긴 변. 0 이면 사진 전체
        self.url = url.rstrip("/")    # gguf 모드의 llama-server 주소
        self.device = None
        self.fake_sec = fake_sec      # fake 모드에서 VLM이 걸리는 시간을 흉내 낸다
        self.lock = threading.Lock()  # 설명 스레드와 검증기가 같은 모델을 번갈아 쓴다
        if mode == "qwen":
            self._load(model_id)
        elif mode == "gguf":
            self.device = f"llama-server {self.url}"
            try:
                urllib.request.urlopen(self.url + "/health", timeout=10).read()
            except Exception as e:
                raise RuntimeError(f"llama-server 에 연결하지 못했다 ({self.url}): {e}") from e
        elif mode not in ("none", "fake"):
            raise ValueError(f"알 수 없는 VLM 모드: {mode}")

    def _load(self, model_id):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        gpu = torch.cuda.is_available()
        # GPU에서 float16 계산은 어떤 입력에서 출력이 전부 NaN이 된다(docs/generation_methods.md).
        # bf16은 메모리와 속도 차이 없이 이 문제가 없다(docs/quantization.md). CPU는 float32.
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

    def observe(self, image_path, bbox=None):
        """(observed 또는 None, 걸린 ms, 원문)을 돌려준다. bbox 가 있고 crop_px 가 0 이 아니면 탐지 조각을 넣는다."""
        t0 = time.perf_counter()
        if self.mode == "none":
            return None, None, ""
        if self.mode == "fake":
            if self.fake_sec:
                time.sleep(self.fake_sec)
            return dict(FAKE_OBSERVED), round((time.perf_counter() - t0) * 1000), "fake"

        if self.mode == "gguf":
            return self._observe_gguf(image_path, bbox, t0)
        import torch
        from PIL import Image
        from lmformatenforcer import TokenEnforcer, JsonSchemaParser
        image = Image.open(image_path).convert("RGB")
        if bbox and self.crop_px:
            image = crop_desc(image, bbox, self.crop_px)
        else:
            image.thumbnail((1024, 1024))          # run_structured.py 와 같음
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": PROMPT}]}]
        enforcer = TokenEnforcer(self.data, JsonSchemaParser(SCHEMA))

        def allow(batch_id, sent):
            allowed = enforcer.get_allowed_tokens(sent.tolist())
            return getattr(allowed, "allowed_tokens", allowed)

        with self.lock, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
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

    def p_smoke(self, image_path, bbox):
        """탐지 하나가 연기인지 P(예)/(P(예)+P(아니오)). (확률, 걸린 ms). none 이면 (None, None)."""
        return self.p_yes(image_path, bbox, VERIFY_PROMPT)

    def p_flame(self, image_path, bbox):
        """같은 조각에 불꽃이 보이는지 따로 묻는다. (확률, 걸린 ms). fake 는 0.0.
        FLAME_MODES 밖(gguf, none)이면 묻지 않고 (None, None) 을 돌려준다."""
        if self.mode not in FLAME_MODES:
            return None, None
        if self.mode == "fake":
            return 0.0, 0
        return self.p_yes(image_path, bbox, FLAME_PROMPT)

    def situation(self, image_path):
        """사진 전체(긴 변 1024)에 상황 질문을 하고 넷의 확률을 돌려준다. ({산불, 굴뚝, 구름, 없음: 확률}, ms).
        CHIMNEY_MODES 밖이면 (None, None). fake 는 늘 산불."""
        t0 = time.perf_counter()
        if self.mode not in CHIMNEY_MODES:
            return None, None
        if self.mode == "fake":
            return {"산불": 1.0, "굴뚝": 0.0, "구름": 0.0, "없음": 0.0}, 0
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((1024, 1024))
        tok = self.processor.tokenizer
        ids = [tok.encode(n, add_special_tokens=False)[0] for n in "1234"]
        with self.lock:
            msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                                {"type": "text", "text": SITUATION_PROMPT}]}]
            inp = self.processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                pr = torch.softmax(self.model(**inp).logits[0, -1].float(), -1)
        v = [float(pr[i]) for i in ids]
        tot = sum(v) or 1.0
        return {c: round(x / tot, 4) for c, x in zip(SITUATION_CHOICES, v)}, round((time.perf_counter() - t0) * 1000)

    def is_chimney(self, image_path):
        """굴뚝·설비 연기로 보이는지. (True/False 또는 None, 확률 dict, ms)"""
        probs, ms = self.situation(image_path)
        if probs is None:
            return None, None, None
        return max(probs, key=probs.get) == "굴뚝", probs, ms

    def p_yes(self, image_path, bbox, prompt):
        """검증기와 같은 조각(박스 사방 25%, 448px)에 예/아니오 질문을 하고 첫 토큰 P(예)/(P(예)+P(아니오))."""
        t0 = time.perf_counter()
        if self.mode == "none":
            return None, None
        if self.mode == "fake":
            return 1.0, 0
        if self.mode == "gguf":
            return self._p_yes_gguf(image_path, bbox, prompt, t0)
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        crop = crop_det(img, bbox, min_side=max(160, img.size[0] // 6))
        ip = self.processor.image_processor
        with self.lock:
            old = (ip.max_pixels, ip.min_pixels)
            ip.max_pixels, ip.min_pixels = VERIFY_PX * VERIFY_PX, 256 * 256
            try:
                msg = [{"role": "user", "content": [{"type": "image", "image": crop},
                                                    {"type": "text", "text": prompt}]}]
                inp = self.processor.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(self.model.device)
                with torch.inference_mode():
                    pr = torch.softmax(self.model(**inp).logits[0, -1].float(), -1)
            finally:
                ip.max_pixels, ip.min_pixels = old
        y, n = float(pr[VERIFY_YES]), float(pr[VERIFY_NO])
        return y / (y + n), round((time.perf_counter() - t0) * 1000)

    # ---------- gguf (llama-server) ----------
    def _post(self, body, timeout=600):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _b64(img):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _observe_gguf(self, image_path, bbox, t0):
        """tools/measure_inputs_cpu.py 의 ask 와 같은 요청. 스키마는 llama-server 의 json_schema 로 강제한다."""
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        if bbox and self.crop_px:
            image = crop_desc(image, bbox, self.crop_px)
        else:
            image.thumbnail((1024, 1024))
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._b64(image)}},
                    {"type": "text", "text": PROMPT}]}],
                "max_tokens": 120, "temperature": 0, "top_k": 1, "repeat_penalty": 1.15, "cache_prompt": True,
                "response_format": {"type": "json_schema", "json_schema": {"name": "obs", "schema": SCHEMA}}}
        with self.lock:
            d = self._post(body)
        raw = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
        ms = round((time.perf_counter() - t0) * 1000)
        try:
            ob = json.loads(raw)
            if set(ob) != set(SCHEMA["required"]):
                ob = None
        except Exception:
            ob = None
        return ob, ms, raw

    def _p_yes_gguf(self, image_path, bbox, prompt, t0):
        """검증 질문의 첫 토큰 확률. llama-server 의 top_logprobs 에서 "예"와 "아(니오)"를 찾는다.
        둘 다 없으면 답한 글자로 1 또는 0 을 준다."""
        import math
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        crop = crop_det(img, bbox, min_side=max(160, img.size[0] // 6))
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._b64(crop)}},
                    {"type": "text", "text": prompt}]}],
                "max_tokens": 1, "temperature": 0, "top_k": 1, "logprobs": True, "top_logprobs": 20}
        with self.lock:
            d = self._post(body)
        ch = (d.get("choices") or [{}])[0]
        tops = (((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs")) or []
        y = sum(math.exp(t["logprob"]) for t in tops if t.get("token", "").strip().startswith("예"))
        n = sum(math.exp(t["logprob"]) for t in tops if t.get("token", "").strip().startswith("아"))
        if y + n > 0:
            p = y / (y + n)
        else:
            p = 1.0 if (ch.get("message") or {}).get("content", "").strip().startswith("예") else 0.0
        return p, round((time.perf_counter() - t0) * 1000)
