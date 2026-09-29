# -*- coding: utf-8 -*-
"""경량 VLM 후보를 4bit로 올려 보고 메모리와 처리 시간을 잰다.

모델이 올라가는 것만으로는 판단할 수 없다. Windows는 VRAM이 모자라면 시스템
메모리로 넘겨서 할당을 성공시키기 때문에, 처리 시간을 같이 봐야 한다.

    python vlm_smoke.py 모델이름 이미지경로 [조건]
"""
import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
import prompts as P

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "OpenGVLab/InternVL3-2B-hf"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "data/pilot/synthetic_smoke.jpg"
COND = sys.argv[3] if len(sys.argv) > 3 else "A"
SIZE = int(os.environ.get("VLM_SIZE", "0"))        # 긴 변을 이 크기로 줄인다. 0이면 원본
REPPEN = float(os.environ.get("VLM_REPPEN", "1.0"))  # 반복 페널티

det = json.load(open("schema/detection.example.json", encoding="utf-8"))
alert = json.load(open("schema/alert.example.json", encoding="utf-8"))
prompt = P.build(COND, det, alert)

print(f"모델: {MODEL}")
print(f"이미지: {IMAGE}")
print(f"조건: {COND}  리사이즈: {SIZE or '원본'}  반복페널티: {REPPEN}")

free_before = torch.cuda.mem_get_info()[0] / 1024**3
quant = BitsAndBytesConfig(load_in_4bit=True,
                          bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True)

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, quantization_config=quant, device_map="cuda:0", dtype=torch.float16)
load_s = time.time() - t0

free_after = torch.cuda.mem_get_info()[0] / 1024**3
print(f"\n로딩 {load_s:.1f}초")
print(f"VRAM 사용 {free_before - free_after:.2f}GB   남은 여유 {free_after:.2f}GB")

image = Image.open(IMAGE).convert("RGB")
if SIZE:
    r = SIZE / max(image.size)
    image = image.resize((int(image.width*r), int(image.height*r)), Image.LANCZOS)
print(f"입력 이미지 {image.size[0]}x{image.size[1]}")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": prompt},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt").to(model.device)
print(f"입력 토큰 {inputs['input_ids'].shape[1]}개")

torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                         repetition_penalty=REPPEN)
torch.cuda.synchronize()
gen_s = time.time() - t0

text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n생성 {gen_s:.1f}초   새 토큰 {out.shape[1] - inputs['input_ids'].shape[1]}개")
print(f"최대 VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
print("\n---- 출력 ----")
print(text.strip())

try:
    parsed = json.loads(text.strip().strip("`").removeprefix("json").strip())
    print(f"\n형식 준수: 예   키 {list(parsed.keys())}")
except Exception as e:
    print(f"\n형식 준수: 아니오 ({e})")
