# -*- coding: utf-8 -*-
"""VLM 이 사진만 보고 촬영 거리를 가늠할 수 있는지 재 본다.

교수님 제안(10/13)에서 나온 확인이다. 지금 설계는 촬영 각도를 몰라 수직 촬영으로
가정하는데, 라벨의 distance 를 보면 중앙값이 170m 이고 82%가 100m 를 넘는다.
연기가 드론 바로 아래 있다고 치면 그만큼 틀린다.

사진만으로는 작게 보이는 것이 멀리 있는 큰 것인지 가까이 있는 작은 것인지
구분되지 않는다. VLM 이 주변 물체와의 크기 비교나 원근 같은 문맥을 써서 이를
가늠할 수 있다면 오차 반경을 조정하는 데 쓸 수 있다.

71330 라벨에 distance 가 실제 값(m)으로 들어 있어 정답과 대조할 수 있다.

    python distance_probe.py 이미지폴더 라벨폴더 [장수]
"""
import sys, os, json, glob, time, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/71330_300/images"
LAB = sys.argv[2] if len(sys.argv) > 2 else "data/71330_300/labels"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
OUT = "results/distance_probe.jsonl"

# 라벨 distance 를 구간으로 나눈다. 71330 설명의 구분을 따른다.
BANDS = [("근거리", 0, 50), ("중거리", 50, 300), ("원거리", 300, 10 ** 9)]

SCHEMA = {
    "type": "object",
    "properties": {
        "band": {"type": "string", "enum": ["근거리", "중거리", "원거리"]},
        "meters": {"type": "integer"},
        "cue": {"type": "string",
                "enum": ["주변 건물과 비교", "나무 크기와 비교", "도로 폭과 비교",
                         "흐릿한 정도", "지평선 위치", "단서 없음"]},
    },
    "required": ["band", "meters", "cue"],
}

PROMPT = """드론이 찍은 사진이다. 카메라에서 연기까지의 거리를 가늠한다.

사진에 보이는 것만 근거로 삼는다. 주변 건물, 나무, 도로처럼 크기를 아는 것과
견주거나, 흐릿한 정도와 원근을 본다.

band     근거리(50m 미만), 중거리(50~300m), 원거리(300m 이상) 중 하나
meters   거리를 미터로 어림한 값
cue      무엇을 보고 그렇게 판단했는지"""


def band_of(m):
    for name, lo, hi in BANDS:
        if lo <= m < hi:
            return name
    return "원거리"


def truth(label_path):
    with open(label_path, encoding="utf-8") as f:
        lab = json.load(f)
    env = lab.get("environment", {})
    if isinstance(env, list):
        env = env[0] if env else {}
    try:
        return float(env.get("distance"))
    except Exception:
        return None


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

    files = sorted(glob.glob(os.path.join(SRC, "*", "*.jpg")))
    if not files:
        raise SystemExit(f"{SRC} 에 이미지가 없습니다.")
    random.seed(42)
    random.shuffle(files)

    pairs = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        split = os.path.basename(os.path.dirname(f))
        lp = os.path.join(LAB, split, name + ".json")
        if not os.path.exists(lp):
            continue
        d = truth(lp)
        if d is not None:
            pairs.append((f, d))
        if len(pairs) == N:
            break
    print(f"{len(pairs)}장으로 잰다", flush=True)

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
    print(f"{mid} 올림", flush=True)

    os.makedirs("results", exist_ok=True)
    hit = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for i, (path, real) in enumerate(pairs):
            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024))
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
                o = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                   prefix_allowed_tokens_fn=allow)
            sec = time.time() - t0
            raw = processor.decode(o[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            try:
                said = json.loads(raw)
            except Exception:
                said = None

            real_band = band_of(real)
            ok = bool(said) and said.get("band") == real_band
            hit += ok
            out.write(json.dumps({
                "image": os.path.basename(path), "real_m": real,
                "real_band": real_band, "said": said, "band_ok": ok,
                "sec": round(sec, 1)}, ensure_ascii=False) + "\n")
            out.flush()
            if said:
                print(f"  {i + 1:3} 정답 {real:>5.0f}m({real_band})  "
                      f"추정 {said['meters']:>5}m({said['band']})  "
                      f"{'맞음' if ok else '틀림'}  단서: {said['cue']}", flush=True)

    print(f"\n구간 적중 {hit}/{len(pairs)} = {hit / max(1, len(pairs)):.0%}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
