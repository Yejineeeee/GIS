# -*- coding: utf-8 -*-
"""생성 방식 (ii) 규칙 검증 후 재생성.

조건 D로 설명을 만들고 판정 규칙으로 검사한다. 어긴 것이 있으면 무엇을 어겼는지
알려 주고 다시 쓰게 한다. 정해진 횟수까지 반복하고, 몇 번 만에 통과했는지를
기록한다. 재시도 횟수가 이 방식의 비용이다.

첫 시도는 방식 (i)과 같게 그리디 디코딩으로 한다. 재시도부터는 샘플링을 쓴다.
그리디 디코딩은 같은 입력에 같은 답을 내므로 다시 써도 바뀌지 않는다.

    python run_verify_retry.py [이미지폴더] [출력파일]
"""
import sys, os, json, time, glob

MAX_RETRY = 3
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/pilot"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/method_ii.jsonl"


def fake_detection(name, i):
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
            "land_use": land, "description": None}


def parse(text):
    s = (text or "").strip().strip("`")
    if s.lower().startswith("json"):
        s = s[4:].strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1]).get("description")
    except Exception:
        return None


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vlm"))
    import prompts as P
    import verify

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
    print(f"{mid} 올림. 이미지 {len(files)}장, 최대 재시도 {MAX_RETRY}회", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    passed = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for i, path in enumerate(files):
            name = os.path.basename(path)
            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024))
            det, alert = fake_detection(name, i), fake_alert(i)
            base = P.build("D", det, alert)

            tries, desc, bad, total = 0, None, ["시작 전"], 0.0
            while tries <= MAX_RETRY and bad:
                text_in = base if tries == 0 else base + "\n\n" + verify.feedback(bad)
                msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text", "text": text_in}]}]
                inputs = processor.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(model.device)
                t0 = time.time()
                with torch.inference_mode():
                    o = model.generate(
                        **inputs, max_new_tokens=160, repetition_penalty=1.15,
                        **({"do_sample": False} if tries == 0
                           else {"do_sample": True, "temperature": 0.7, "top_p": 0.9}))
                total += time.time() - t0
                desc = parse(processor.decode(o[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True))
                bad = verify.check(desc, alert)
                tries += 1

            ok = not bad
            passed += ok
            out.write(json.dumps({
                "image": name, "model": "qwen", "cond": "D", "method": "검증재생성",
                "verdict": alert["location_verdict"],
                "radius": alert["position"]["error_radius_m"],
                "attempts": tries, "passed": ok, "violations": bad,
                "sec": round(total, 1), "description": desc}, ensure_ascii=False) + "\n")
            out.flush()
            mark = f"{tries}회" if ok else f"{tries}회 실패"
            print(f"  {name:16} {alert['location_verdict']:4} {mark:8} {total:5.1f}초  "
                  f"{(desc or '')[:46]}", flush=True)

    print(f"\n통과 {passed}/{len(files)}  →  {OUT}")


if __name__ == "__main__":
    main()
