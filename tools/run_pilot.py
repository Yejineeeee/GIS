# -*- coding: utf-8 -*-
"""파일럿 실험. 모델 두 개에 조건 A와 B를 걸어 설명을 받는다.

합격선을 판단하기 위한 것이다. 본 실험이 아니므로 이미지는 위키미디어 공용에서
받은 것을 쓰고, 조건 B에 넣는 탐지 결과는 가짜 값이다. 형식만 실제와 같다.
설명이 한국어로 나오는지, 형식을 지키는지, 조건을 바꾸면 달라지는지를 본다.

    python run_pilot.py [이미지폴더] [출력파일]
"""
import sys, os, json, time, glob, subprocess

MODELS = {"internvl": "OpenGVLab/InternVL3-2B-hf",
          "qwen": "Qwen/Qwen2.5-VL-3B-Instruct"}
CONDS = ["A", "B"]
MAX_PATCHES = 6
IMAGES = sys.argv[1] if len(sys.argv) > 1 else "data/pilot"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "results/pilot_outputs.jsonl"


def fake_detection(name, i):
    """조건 B 용 탐지 결과. 파일럿이라 값은 가짜고 형식만 맞춘다."""
    smoke = name.startswith("smoke")
    return {"schema_version": "0.2", "video_id": "P000", "frame_id": i,
            "image": name, "image_size": {"width": 1280, "height": 900},
            "detections": [{"id": f"P000-{i}-0", "class": "연기",
                            "score": 0.88 if smoke else 0.31,
                            "bbox": [320, 180, 420, 300], "polygon": None,
                            "area_px": 126000,
                            "area_ratio": 0.109 if smoke else 0.012}],
            "camera": {"lat": None, "lon": None, "alt_m": 120.0, "alt_ref": "AGL",
                       "heading_deg": None, "pitch_deg": None, "hfov_deg": None,
                       "source": "virtual"}}


def run_model(key, model_id, files, out):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
    import prompts as P

    alert = json.load(open("schema/alert.example.json", encoding="utf-8"))
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        # InternVL 은 타일 수로, Qwen 은 픽셀 수로 입력 크기를 정한다.
        if hasattr(ip, "max_patches"):
            ip.max_patches = MAX_PATCHES
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = 768 * 768          # 4GB 안에 들어오게 상한을 건다
            ip.min_pixels = 256 * 256
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=quant, device_map="cuda:0", dtype=torch.float16)
    print(f"[{key}] 로딩 {time.time()-t0:.0f}초  "
          f"가중치 {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    for i, path in enumerate(files):
        name = os.path.basename(path)
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024))
        det = fake_detection(name, i)
        for cond in CONDS:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": P.build(cond, det, alert)}]}]
            inputs = processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            torch.cuda.synchronize(); t = time.time()
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=140, do_sample=False,
                                   repetition_penalty=1.15)
            torch.cuda.synchronize(); sec = time.time() - t
            n_in = inputs["input_ids"].shape[1]
            text = processor.decode(o[0][n_in:], skip_special_tokens=True).strip()
            out.write(json.dumps({"image": name, "model": key, "cond": cond,
                                  "text": text, "sec": round(sec, 1),
                                  "tokens_in": n_in}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {name} {cond} {sec:4.1f}초  {text[:60]}", flush=True)


def main():
    files = sorted(glob.glob(os.path.join(IMAGES, "smoke_*.jpg"))) + \
            sorted(glob.glob(os.path.join(IMAGES, "nosmoke_*.jpg")))
    if not files:
        raise SystemExit(f"{IMAGES} 에 이미지가 없습니다.")
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    print(f"이미지 {len(files)}장 × 모델 {len(MODELS)}개 × 조건 {len(CONDS)}개 "
          f"= {len(files)*len(MODELS)*len(CONDS)}건", flush=True)
    mode = "a" if os.path.exists(OUTFILE) else "w"
    with open(OUTFILE, mode, encoding="utf-8") as out:
        for key, mid in MODELS.items():
            # 모델마다 프로세스를 새로 띄워야 메모리가 섞이지 않는다
            if os.environ.get("PILOT_ONE") in (None, key):
                run_model(key, mid, files, out)
                if os.environ.get("PILOT_ONE"):
                    break


if __name__ == "__main__":
    main()
