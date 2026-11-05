# -*- coding: utf-8 -*-
"""파일럿 실험. 모델 두 개에 조건 A와 B를 걸어 설명을 받는다.

합격선을 판단하기 위한 것이다. 본 실험이 아니므로 이미지는 위키미디어 공용에서
받은 것을 쓰고, 조건 B에 넣는 탐지 결과는 가짜 값이다. 형식만 실제와 같다.
설명이 한국어로 나오는지, 형식을 지키는지, 조건을 바꾸면 달라지는지를 본다.

환경 변수 EVAL_FACTS 에 사실 목록 경로를 주면 파일럿 대신 평가셋(6.1)으로 돌린다.
생성 방식 비교(4.6)가 이 길을 쓴다. 입력을 만드는 규칙은 src/vlm/eval_alert.py 에 있다.

    python run_pilot.py [이미지폴더] [출력파일]
"""
import sys, os, json, time, glob, subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "vlm"))
from fake_alert import fake_alert as base_alert
import eval_alert

MODELS = {"internvl": "OpenGVLab/InternVL3-2B-hf",
          "qwen": "Qwen/Qwen2.5-VL-3B-Instruct"}
CONDS = os.environ.get("PILOT_CONDS", "A,B").split(",")
MAX_PATCHES = 6
IMAGES = sys.argv[1] if len(sys.argv) > 1 else "data/pilot"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "results/pilot_outputs.jsonl"
FACTS = os.environ.get("EVAL_FACTS")
DTYPE = os.environ.get("EVAL_DTYPE", "bf16")
RESUME = bool(os.environ.get("EVAL_RESUME"))


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


def fake_alert(i):
    """조건 C와 D 용 알람. 판정 세 가지가 고루 나오게 돌려 가며 만든다.
    판정이 바뀔 때 설명이 실제로 달라지는지 보려는 것이다.

    용도 이름은 src/vlm/fake_alert.py 한 곳에서 가져온다. 프롬프트 예시와 겹치지
    않아야 출력에 그 이름이 나왔을 때 입력을 쓴 것인지 예시를 베낀 것인지 가려진다.
    """
    a = base_alert(i)
    a["detection_id"] = f"P000-{i}-0"
    a["evidence"]["slope_deg"] = None
    a["latency_ms"] = {}
    return a


def build_items():
    """[(경로, 이름, 탐지, 알람, 판정)] 을 만든다. 평가셋이면 사실 목록에서 가져온다."""
    if FACTS:
        return [(p, n, det, al, f["verdict"])
                for p, n, f, det, al in eval_alert.load(IMAGES, FACTS)]
    files = (sorted(glob.glob(os.path.join(IMAGES, "smoke_*.jpg"))) +
             sorted(glob.glob(os.path.join(IMAGES, "nosmoke_*.jpg"))))
    if not files:
        raise SystemExit(f"{IMAGES} 에 이미지가 없습니다.")
    return [(p, os.path.basename(p), fake_detection(os.path.basename(p), i),
             fake_alert(i), None) for i, p in enumerate(files)]


def run_model(key, model_id, items, out):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
    import prompts as P

    dt = torch.bfloat16 if DTYPE == "bf16" else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dt,
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
        model_id, quantization_config=quant, device_map="cuda:0", dtype=dt)
    print(f"[{key}] 로딩 {time.time()-t0:.0f}초  "
          f"가중치 {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    for path, name, det, alert, verdict in items:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024))
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
            rec = {"image": name, "model": key, "cond": cond, "text": text,
                   "sec": round(sec, 1), "tokens_in": n_in}
            if verdict is not None:
                rec["verdict"] = verdict      # 채점기가 알람이 섞였는지 본다
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {name[:44]} {cond} {sec:4.1f}초  {text[:46]}", flush=True)


def main():
    items = build_items()
    if RESUME and os.path.exists(OUTFILE):
        with open(OUTFILE, encoding="utf-8") as f:
            done = {json.loads(line)["image"] for line in f if line.strip()}
        items = [it for it in items if it[1] not in done]
        print(f"{len(done)}장은 이미 있어 건너뜁니다.", flush=True)
        if not items:
            return
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    n_model = sum(1 for k in MODELS if os.environ.get("PILOT_ONE") in (None, k))
    print(f"{'평가셋' if FACTS else '파일럿'} {len(items)}장 × 모델 {n_model}개 "
          f"× 조건 {len(CONDS)}개 = {len(items)*n_model*len(CONDS)}건", flush=True)
    mode = "a" if os.path.exists(OUTFILE) else "w"
    with open(OUTFILE, mode, encoding="utf-8") as out:
        for key, mid in MODELS.items():
            # 모델마다 프로세스를 새로 띄워야 메모리가 섞이지 않는다
            if os.environ.get("PILOT_ONE") in (None, key):
                run_model(key, mid, items, out)
                if os.environ.get("PILOT_ONE"):
                    break


if __name__ == "__main__":
    main()
