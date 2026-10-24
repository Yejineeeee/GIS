"""선정 VLM 현장 기기 측정 (Qwen2.5-VL-3B, 4bit nf4 이중 양자화, CPU).

현장 기기 기준 환경(docs/device_spec.md: CPU 4코어, 메모리 8GB, 스왑 없음, GPU 없음)
안에서 실행해 메모리, 이미지 1장 처리 시간, 설명 표시 시간을 잰다.

생성 방식은 run_structured.py(방식 iii, 구조화템플릿)와 같게 맞췄다.
- 질문 문구(PROMPT), JSON 스키마(SCHEMA), lm-format-enforcer 로 디코딩 단계 스키마 강제
- 이미지: thumbnail(1024, 1024) 후 처리기 max_pixels 768*768, min_pixels 256*256
- max_new_tokens 120, 탐욕 디코딩(do_sample=False)
- 문장은 src/vlm/templates.py 의 build(observed, alert)로 조립 (없으면 측정용 최소 템플릿)
다른 점: GPU 대신 CPU, compute dtype float16 대신 float32 (CPU에서 float16은 느리거나 미지원)

측정 항목
- 메모리: 모델 적재 후 RSS, 측정 전체의 최대 RSS, 모델 크기(get_memory_footprint)
- 이미지 1장 처리 시간(vlm_s): 입력 준비 + 생성. 첫 토큰까지 시간도 남긴다
- 설명 표시 시간(description_s): VLM 호출 시작 -> JSON 해석 -> 템플릿 문장 완성
- 스키마 준수(schema_ok): run_structured.py 와 같은 기준(키 5개)
- 일관성(consistent): 연기 있음/없음과 색, 퍼짐 값이 서로 맞는지 (추가 기준)

준비 (제한 환경 안에서)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install "transformers>=4.49" accelerate "bitsandbytes>=0.48" pillow lm-format-enforcer

사용 예
    CUDA_VISIBLE_DEVICES="" python tools/measure_vlm.py \
        --images data/frames/sample --runs 3 --out results/vlm_measure.csv
"""
import argparse
import csv
import json
import os
import platform
import resource
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser(description="선정 VLM 현장 기기 측정")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
ap.add_argument("--images", required=True, help="이미지 폴더 또는 파일")
ap.add_argument("--runs", type=int, default=3, help="이미지마다 반복 횟수")
ap.add_argument("--warmup", type=int, default=1, help="측정 전 버리는 실행 횟수")
ap.add_argument("--threads", type=int, default=4, help="스레드 수 (기준 4)")
ap.add_argument("--compute-dtype", default="float32", choices=["float32", "bfloat16"])
ap.add_argument("--no-enforce", action="store_true",
                help="스키마 강제를 끈다 (비교 확인용. 기본은 run_structured.py 처럼 강제)")
ap.add_argument("--out", default="results/vlm_measure.csv")
args = ap.parse_args()

# 스레드 수는 torch를 불러오기 전에 맞춘다
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = str(args.threads)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (AutoModelForImageTextToText, AutoProcessor,  # noqa: E402
                          BitsAndBytesConfig, TextIteratorStreamer)

torch.set_num_threads(args.threads)
DTYPE = getattr(torch, args.compute_dtype)

# ── run_structured.py 와 같은 값 (바꾸면 비교가 깨진다)
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

MAX_NEW_TOKENS = 120
FAKE_ALERT = {  # 설명 표시 시간 측정용 알람 (run_structured.py fake_alert(0)와 같음)
    "alert_id": "p-0000", "location_verdict": "확정",
    "position": {"lat": 37.74652, "lon": 128.88544, "error_radius_m": 20},
    "land_use": [{"name": "보전녹지지역", "ratio": 0.85},
                 {"name": "제1종일반주거지역", "ratio": 0.15}],
    "evidence": {"buildings_in_radius": 12}, "description": None,
}

# 팀 템플릿이 있으면 그대로 쓰고, 없으면 측정용 최소 템플릿
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "vlm"))
try:
    import templates as team_templates
    TEMPLATE_SRC = "src/vlm/templates.py"
except Exception:
    team_templates = None
    TEMPLATE_SRC = "측정용 최소 템플릿"


def build_description(ob):
    if team_templates is not None:
        return team_templates.build(ob, FAKE_ALERT)
    if not ob["smoke_present"]:
        return f"{ob['terrain']} 지역에서 연기가 보이지 않습니다."
    return f"{ob['terrain']} 지역에서 {ob['smoke_color']} 연기가 보입니다 (퍼짐: {ob['spread']})."


def consistent(ob):
    if ob["smoke_present"] is False:
        return ob["smoke_color"] == "없음" and ob["spread"] == "없음"
    return ob["smoke_color"] != "없음" and ob["spread"] != "없음"


def build_tokenizer_data(tokenizer):
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


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024


p = Path(args.images)
images = sorted(x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
    if p.is_dir() else [p]
if not images:
    sys.exit(f"이미지가 없다: {p}")

# ── 환경 기록
env = {
    "nproc": os.cpu_count(),
    "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "threads": torch.get_num_threads(), "torch": torch.__version__,
    "cuda": torch.cuda.is_available(), "machine": platform.machine(),
}
try:
    import bitsandbytes
    import transformers
    import lmformatenforcer
    env["bitsandbytes"] = bitsandbytes.__version__
    env["transformers"] = transformers.__version__
    env["lmformatenforcer"] = getattr(lmformatenforcer, "__version__", "?")
except Exception as e:
    if not args.no_enforce:
        sys.exit(f"패키지 확인 실패: {e}\n  pip install lm-format-enforcer")
print("[환경]", env)
print(f"[설정] 템플릿: {TEMPLATE_SRC}, 스키마 강제: {not args.no_enforce}, compute {args.compute_dtype}")
if env["cuda"]:
    print("[주의] GPU가 보입니다. 기준 측정은 CUDA_VISIBLE_DEVICES='' 로 GPU 없이 합니다.")
if env["affinity"] and env["affinity"] != args.threads:
    print(f"[주의] 사용 가능한 코어 {env['affinity']}개 != 기준 {args.threads}개")

# ── 모델 적재 (원본을 읽으면서 바로 nf4로 양자화)
rss0 = rss_mb()
t = time.perf_counter()
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(args.model)
processor.image_processor.max_pixels = 768 * 768
processor.image_processor.min_pixels = 256 * 256
model = AutoModelForImageTextToText.from_pretrained(
    args.model, quantization_config=quant, device_map="cpu", dtype=DTYPE)
model.eval()
data = None if args.no_enforce else build_tokenizer_data(processor.tokenizer)
load_s = time.perf_counter() - t
rss_load = rss_mb()
footprint = model.get_memory_footprint() / 1024**3
print(f"[적재] {load_s:.1f}s, 모델 크기 {footprint:.2f}GB, RSS {rss_load:.0f}MB (적재 전 {rss0:.0f}MB)")


def run_once(img_path):
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((1024, 1024))          # run_structured.py 와 같음
    size = image.size
    t0 = time.perf_counter()
    msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    kw = dict(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if data is not None:
        from lmformatenforcer import TokenEnforcer, JsonSchemaParser
        enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))

        def allow(batch_id, sent):
            allowed = enforcer.get_allowed_tokens(sent.tolist())
            return getattr(allowed, "allowed_tokens", allowed)
        kw["prefix_allowed_tokens_fn"] = allow
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    kw["streamer"] = streamer
    th = threading.Thread(target=lambda: model.generate(**kw))
    th.start()
    first, chunks = None, []
    for piece in streamer:
        if first is None and piece:
            first = time.perf_counter() - t0
        chunks.append(piece)
    th.join()
    t_gen = time.perf_counter() - t0
    raw = "".join(chunks).strip()
    try:
        ob = json.loads(raw)
        schema_ok = set(ob) == set(SCHEMA["required"])
    except Exception:
        ob, schema_ok = None, False
    desc = None
    if ob and schema_ok:
        try:
            desc = build_description(ob)
        except Exception as e:
            desc = None
            raw += f"  [템플릿 오류: {e}]"
    t_desc = time.perf_counter() - t0 if desc else None
    return {
        "size": f"{size[0]}x{size[1]}", "input_tokens": inputs["input_ids"].shape[1],
        "first_token_s": round(first, 2) if first else "", "vlm_s": round(t_gen, 2),
        "description_s": round(t_desc, 2) if t_desc else "",
        "schema_ok": schema_ok, "consistent": bool(ob and schema_ok and consistent(ob)),
        "output": raw.replace("\n", " "), "description": desc or "",
    }


for i in range(args.warmup):
    r = run_once(images[0])
    print(f"[워밍업 {i + 1}/{args.warmup}] {images[0].name} 입력 {r['size']}, 토큰 {r['input_tokens']}, "
          f"vlm {r['vlm_s']}s, RSS {rss_mb():.0f}MB")

rows = []
for img_path in images:
    for k in range(args.runs):
        r = run_once(img_path)
        r = {"image": img_path.name, "run": k + 1, **r, "rss_mb": round(rss_mb())}
        rows.append(r)
        print(f"  {img_path.name[:40]} #{k + 1}: vlm {r['vlm_s']}s, 첫 토큰 {r['first_token_s']}s, "
              f"스키마 {'O' if r['schema_ok'] else 'X'}, 일관성 {'O' if r['consistent'] else 'X'}")
        print(f"     {r['output']}")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 1)


vlm_med = median([x["vlm_s"] for x in rows])
first_med = median([x["first_token_s"] for x in rows if x["first_token_s"] != ""])
desc_med = median([x["description_s"] for x in rows if x["description_s"] != ""])
schema_n = sum(x["schema_ok"] for x in rows)
cons_n = sum(x["consistent"] for x in rows)
peak = peak_rss_mb()

print("\n[요약]")
print(f"  이미지 {len(images)}장 x {args.runs}회, 입력 {rows[0]['size']}, 스레드 {args.threads}, "
      f"compute {args.compute_dtype}, 스키마 강제 {not args.no_enforce}")
print(f"  메모리: 모델 {footprint:.2f}GB, 적재 후 RSS {rss_load / 1024:.2f}GB, 최대 RSS {peak / 1024:.2f}GB")
print(f"  첫 토큰까지(중앙값): {first_med}s")
print(f"  이미지 1장 처리 시간(중앙값): {vlm_med}s")
print(f"  설명 표시 시간(중앙값): {desc_med}s" if desc_med else "  설명 표시 시간: 없음")
print(f"  스키마 준수: {schema_n}/{len(rows)}, 일관성: {cons_n}/{len(rows)}")
print(f"  결과 파일: {out}")
print("\n[vlm_candidates.md 실측 비교 표에 붙일 줄]")
print(f"| Qwen2.5-VL-3B (nf4 이중 양자화, CPU {args.threads}코어, {rows[0]['size']}) "
      f"| 최대 {peak / 1024:.2f}GB | {vlm_med}초 (설명 {desc_med}초) | (사람 확인) "
      f"| 스키마 {schema_n}/{len(rows)}, 일관성 {cons_n}/{len(rows)} |")
