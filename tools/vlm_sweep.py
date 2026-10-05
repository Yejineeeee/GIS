# -*- coding: utf-8 -*-
"""모델을 한 번 올려 두고 입력 크기를 바꿔 가며 잰다.

드론 원본은 가로 5,000픽셀이 넘는다. 크게 넣으면 이미지 토큰이 늘어 메모리가
먼저 막히고, 많이 줄이면 멀리 있는 연기가 사라진다. 그 경계를 찾는다.

    python vlm_sweep.py 모델이름 이미지경로
"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
import prompts as P
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "OpenGVLab/InternVL3-2B-hf"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "data/pilot/synthetic_smoke.jpg"
PATCHES = [12, 8, 6, 4, 2, 1]   # 이미지를 몇 타일로 자를지. 타일당 256토큰
REPPEN = 1.15

det = json.load(open("schema/detection.example.json", encoding="utf-8"))
alert = json.load(open("schema/alert.example.json", encoding="utf-8"))

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
free0 = torch.cuda.mem_get_info()[0] / 1024**3
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, quantization_config=quant, device_map="cuda:0", dtype=torch.float16)
print(f"{MODEL}  로딩 {time.time()-t0:.1f}초  "
      f"가중치 {free0 - torch.cuda.mem_get_info()[0]/1024**3:.2f}GB")

base = Image.open(IMAGE).convert("RGB")
rows = []
for np_ in PATCHES:
    processor.image_processor.max_patches = np_
    img = base
    msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                        {"type": "text", "text": P.build("A", det, alert)}]}]
    inputs = processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                             repetition_penalty=REPPEN)
    torch.cuda.synchronize(); sec = time.time() - t0
    n_in = inputs["input_ids"].shape[1]
    text = processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
    try:
        desc = json.loads(text.strip("`").removeprefix("json").strip())["description"]
        ok = "예"
    except Exception:
        desc, ok = text, "아니오"
    peak = torch.cuda.max_memory_allocated()/1024**3
    print(f"\n── {img.width}x{img.height}  토큰 {n_in}  {sec:.1f}초  최대 {peak:.2f}GB  형식 {ok}")
    print(f"   {desc}")
    rows.append([np_, n_in, round(sec,1), round(peak,2), ok, desc])

os.makedirs("results", exist_ok=True)
with open("results/vlm_patch_sweep.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["타일수", "입력토큰", "생성초", "최대VRAM_GB", "형식준수", "설명"])
    w.writerows(rows)
print("\n저장: results/vlm_patch_sweep.csv")
