# -*- coding: utf-8 -*-
"""VLM 에 참조값을 함께 주고 촬영 거리를 가늠하게 한다.

1차 측정(tools/distance_probe.py)에서 사진만 주었더니 50장 전부 "100m 중거리"로
같은 답을 냈다. 적중률 90%는 표본이 중거리에 쏠린 탓이고, 다수 구간만 답하는
전략과 차이가 0이었다(상관계수 -0.03).

두 가지를 바꿔 다시 잰다.
- 표본을 근거리, 중거리, 원거리에서 같은 수로 뽑는다. 쏠린 표본에서는 무엇을
  바꿔도 결과를 읽을 수 없다.
- 연기가 화면에서 차지하는 비율과 화면 크기를 함께 준다. 같은 연기라도 멀면 작게
  보이므로, 실제 크기를 가늠할 수 있다면 이 값이 거리로 이어진다.

    python distance_probe2.py 이미지폴더 라벨폴더 [구간당 장수]
"""
import sys, os, json, glob, time, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/71330_300/images"
LAB = sys.argv[2] if len(sys.argv) > 2 else "data/71330_300/labels"
PER = int(sys.argv[3]) if len(sys.argv) > 3 else 15
OUT = "results/distance_probe2.jsonl"

BANDS = [("근거리", 0, 50), ("중거리", 50, 300), ("원거리", 300, 10 ** 9)]
SMOKE_IDS = {1, 2}          # 흑색연기, 백색/회색연기

SCHEMA = {
    "type": "object",
    "properties": {
        "real_width_m": {"type": "integer"},
        "band": {"type": "string", "enum": ["근거리", "중거리", "원거리"]},
        "meters": {"type": "integer"},
        "cue": {"type": "string",
                "enum": ["주변 건물과 비교", "나무 크기와 비교", "도로 폭과 비교",
                         "흐릿한 정도", "지평선 위치", "단서 없음"]},
    },
    "required": ["real_width_m", "band", "meters", "cue"],
}

PROMPT = """드론이 찍은 사진이다. 카메라에서 연기까지의 거리를 가늠한다.

연기가 화면에서 차지하는 비율: {ratio}
연기를 둘러싼 상자의 화면 대비 가로 비율: {wratio}

같은 크기의 연기라도 멀리 있으면 화면에서 작게 보인다. 그러니 먼저 이 연기가
실제로 몇 미터쯤 되는지 가늠한다. 주변 건물, 나무, 도로처럼 크기를 아는 것과
견주면 된다. 그 다음 화면에서 차지하는 비율과 견주어 거리를 어림한다.

real_width_m  연기의 실제 가로 길이를 미터로 어림한 값
band          근거리(50m 미만), 중거리(50~300m), 원거리(300m 이상) 중 하나
meters        카메라에서 연기까지의 거리를 미터로 어림한 값
cue           실제 크기를 무엇과 견주어 판단했는지"""


def band_of(m):
    for name, lo, hi in BANDS:
        if lo <= m < hi:
            return name
    return "원거리"


def read_label(path):
    """거리와 연기 면적 비율, 상자 가로 비율을 꺼낸다."""
    with open(path, encoding="utf-8") as f:
        lab = json.load(f)
    env = lab.get("environment", {})
    if isinstance(env, list):
        env = env[0] if env else {}
    try:
        dist = float(env.get("distance"))
    except Exception:
        return None
    img = (lab.get("images") or [{}])[0]
    W, H = img.get("width"), img.get("height")
    if not W or not H:
        return None
    area = 0.0
    wmax = 0.0
    for a in lab.get("annotations", []):
        if a.get("category_id") not in SMOKE_IDS:
            continue
        area += float(a.get("area") or 0)
        bb = a.get("bbox") or [0, 0, 0, 0]
        wmax = max(wmax, float(bb[2]))
    if area <= 0:
        return None
    return {"dist": dist, "ratio": round(area / (W * H), 4),
            "wratio": round(wmax / W, 3)}


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
    pool = {name: [] for name, _, _ in BANDS}
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        split = os.path.basename(os.path.dirname(f))
        lp = os.path.join(LAB, split, base + ".json")
        if not os.path.exists(lp):
            continue
        info = read_label(lp)
        if info is None:
            continue
        pool[band_of(info["dist"])].append((f, info))

    print("구간별로 쓸 수 있는 사진")
    for k, v in pool.items():
        print(f"   {k:6} {len(v):4}장")

    random.seed(42)
    pairs = []
    for name, _, _ in BANDS:
        random.shuffle(pool[name])
        pairs.extend(pool[name][:PER])
    random.shuffle(pairs)
    if not pairs:
        raise SystemExit("쓸 수 있는 사진이 없습니다.")
    print(f"\n{len(pairs)}장으로 잰다", flush=True)

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
    print(f"{mid} 올림\n", flush=True)

    os.makedirs("results", exist_ok=True)
    hit = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for i, (path, info) in enumerate(pairs):
            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024))
            enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))

            def allow(batch_id, sent):
                allowed = enforcer.get_allowed_tokens(sent.tolist())
                return getattr(allowed, "allowed_tokens", allowed)

            text = PROMPT.format(ratio=info["ratio"], wratio=info["wratio"])
            msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text", "text": text}]}]
            inputs = processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=90, do_sample=False,
                                   prefix_allowed_tokens_fn=allow)
            sec = time.time() - t0
            raw = processor.decode(o[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            try:
                said = json.loads(raw)
            except Exception:
                said = None

            real_band = band_of(info["dist"])
            ok = bool(said) and said.get("band") == real_band
            hit += ok
            out.write(json.dumps({
                "image": os.path.basename(path), "real_m": info["dist"],
                "real_band": real_band, "ratio": info["ratio"],
                "wratio": info["wratio"], "said": said, "band_ok": ok,
                "sec": round(sec, 1)}, ensure_ascii=False) + "\n")
            out.flush()
            if said:
                print(f"  {i + 1:3} 정답 {info['dist']:>5.0f}m({real_band})  "
                      f"추정 {said['meters']:>5}m({said['band']})  "
                      f"연기폭 {said['real_width_m']:>4}m  화면비 {info['wratio']:.2f}  "
                      f"{'맞음' if ok else '틀림'}", flush=True)

    print(f"\n구간 적중 {hit}/{len(pairs)} = {hit / max(1, len(pairs)):.0%}")
    print(f"(구간을 같은 수로 뽑았으므로 찍기 기준선은 33%)")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
