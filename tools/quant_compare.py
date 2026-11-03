# -*- coding: utf-8 -*-
"""같은 모델을 양자화 설정별로 올려 메모리와 속도, 출력을 비교한다.

4bit를 쓰기로 한 근거를 숫자로 남기기 위한 것이다. VRAM이 모자라면 Windows가
시스템 메모리로 넘겨 버리므로, 올라갔는지가 아니라 처리 시간으로 판단한다.

한 프로세스에서 여러 설정을 재면 앞 모델이 완전히 비워지지 않아 값이 섞인다.
그래서 설정마다 프로세스를 새로 띄운다.

    python quant_compare.py [모델이름] [이미지경로] [출력파일]
"""
import sys, os, json, csv, subprocess

SETTINGS = ["fp16", "8bit", "4bit-fp4", "4bit-nf4", "4bit-nf4-double",
            "4bit-nf4-double-bf16"]
MAX_PATCHES = 6


def run_one(setting, model_id, image_path):
    import time, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
    import prompts as P

    det = json.load(open("schema/detection.example.json", encoding="utf-8"))
    alert = json.load(open("schema/alert.example.json", encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))

    if setting == "fp16":
        quant = None
    elif setting == "8bit":
        quant = BitsAndBytesConfig(load_in_8bit=True)
    else:
        compute = torch.bfloat16 if "bf16" in setting else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute,
            bnb_4bit_quant_type="fp4" if "fp4" in setting else "nf4",
            bnb_4bit_use_double_quant="double" in setting)

    free0 = torch.cuda.mem_get_info()[0] / 1024**3
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        # InternVL 은 타일 수로, Qwen 은 픽셀 수로 입력 크기를 정한다.
        # 다른 실행기(run_pilot.py 등)와 같은 상한을 써야 숫자가 비교된다.
        if hasattr(ip, "max_patches"):
            ip.max_patches = MAX_PATCHES
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = 768 * 768
            ip.min_pixels = 256 * 256
    dt = torch.bfloat16 if "bf16" in setting else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=quant, device_map="cuda:0", dtype=dt)
    load_s = time.time() - t0
    weights = free0 - torch.cuda.mem_get_info()[0] / 1024**3

    msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": P.build("A", det, alert)}]}]
    inputs = processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                             repetition_penalty=1.15)
    torch.cuda.synchronize(); gen_s = time.time() - t0

    n_in = inputs["input_ids"].shape[1]
    n_new = out.shape[1] - n_in
    text = processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
    try:
        desc, ok = json.loads(text.strip("`").removeprefix("json").strip())["description"], "예"
    except Exception:
        desc, ok = text, "아니오"

    print("RESULT " + json.dumps({
        "설정": setting, "가중치GB": round(weights, 2),
        "최대VRAM_GB": round(torch.cuda.max_memory_allocated()/1024**3, 2),
        "로딩초": round(load_s), "생성초": round(gen_s, 1),
        "새토큰": n_new, "초당토큰": round(n_new/gen_s, 1),
        "형식준수": ok, "출력": desc}, ensure_ascii=False))


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "OpenGVLab/InternVL3-2B-hf"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "data/pilot/synthetic_smoke.jpg"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "results/quant_compare.csv"
    rows = []
    for setting in SETTINGS:
        print(f"── {setting}")
        p = subprocess.run([sys.executable, __file__, "--one", setting, model_id, image_path],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((l for l in (p.stdout or "").splitlines() if l.startswith("RESULT ")), None)
        if not line:
            err = (p.stderr or "").strip().splitlines()
            print(f"   실패: {err[-1][:140] if err else '출력 없음'}")
            rows.append({"설정": setting, "형식준수": "실패"})
            continue
        r = json.loads(line[7:])
        print(f"   가중치 {r['가중치GB']}GB  최대 {r['최대VRAM_GB']}GB  "
              f"생성 {r['생성초']}초  초당 {r['초당토큰']}토큰  형식 {r['형식준수']}")
        print(f"   {r['출력'][:78]}")
        rows.append(r)

    cols = ["설정", "가중치GB", "최대VRAM_GB", "로딩초", "생성초", "새토큰",
            "초당토큰", "형식준수", "출력"]
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run_one(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
