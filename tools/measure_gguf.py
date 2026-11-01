# -*- coding: utf-8 -*-
"""GGUF 4bit(llama.cpp)로 설명 1장 처리 시간을 잰다 (WBS 7.2).

`tools/measure_vlm.py` 가 잰 180.7초는 bitsandbytes nf4 를 CPU 에서 돌린 값이다.
bitsandbytes 는 CUDA 를 보고 만든 것이라 CPU 에서는 느리다. 현장 기기 기준이
GPU 없음이므로 CPU 로 만든 llama.cpp 의 GGUF 4bit 로 다시 잰다.

llama-server 에 모델을 한 번 올려 두고 이미지를 하나씩 보낸다. 모델을 매번
올리지 않는 점이 기존 측정과 같다.

비교가 깨지지 않도록 아래를 같게 맞췄다.
    이미지   results/vlm_measure.csv 의 6장, thumbnail(1024,1024) 적용
    질문     measure_vlm.py 의 PROMPT 그대로
    스키마   같은 5개 항목. lm-format-enforcer 대신 llama-server 의 json_schema
    생성     최대 120토큰, 탐욕 디코딩, 반복 벌점 1.15
    스레드   4 (기준 환경의 4코어)
다른 점은 기기다. 기존 값은 박서연 노트북 WSL2 에서 쟀고 이것은 이예진 노트북
윈도우에서 잰다. 메모리 상한 8GB 는 걸지 않았다.

    python measure_gguf.py --url http://127.0.0.1:8099 --images 폴더 --out 결과.csv

sweep_tokens.py 가 아래 SCHEMA, PROMPT, consistent 를 그대로 가져다 쓴다.
"""
import argparse
import base64
import csv
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

# measure_vlm.py 와 같은 값. 바꾸면 비교가 깨진다.
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


def consistent(ob):
    """연기 있음/없음과 색, 퍼짐이 서로 맞는지. measure_vlm.py 와 같은 기준."""
    if ob.get("smoke_present") is True:
        return ob.get("smoke_color") != "없음" and ob.get("spread") != "없음"
    return ob.get("smoke_color") == "없음" and ob.get("spread") == "없음"


def clear_cache(url, slot=0):
    """슬롯의 KV 캐시를 비운다.

    같은 사진을 여러 번 보내면 llama-server 가 이미지 임베딩을 재사용해 입력
    토큰이 1개로 떨어진다. 그 값은 캐시 적중 시간이지 처리 시간이 아니다.
    기존 측정(measure_vlm.py)에는 캐시가 없으므로 매번 비우고 잰다.
    서버를 --slot-save-path 와 함께 띄워야 이 호출이 먹는다.
    """
    req = urllib.request.Request(f"{url}/slots/{slot}?action=erase",
                                 data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def ask(url, img_path):
    """이미지 한 장을 보내고 첫 토큰까지 시간과 전체 시간을 잰다."""
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": MAX_NEW_TOKENS, "temperature": 0, "top_k": 1,
        "repeat_penalty": 1.15, "stream": True,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "obs", "schema": SCHEMA}},
    }
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    first, text, timings = None, [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            piece = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter() - t0
                text.append(piece)
            if d.get("timings"):
                timings = d["timings"]
    return first, time.perf_counter() - t0, "".join(text), timings


def main():
    ap = argparse.ArgumentParser(description="GGUF 4bit 설명 시간 측정")
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    meta = json.load(open(Path(args.images) / "images.json", encoding="utf-8"))
    rows = []

    for run in range(1, args.runs + 1):
        for m in meta:
            clear_cache(args.url)
            first, total, text, tm = ask(args.url,
                                         Path(args.images) / f"{m['id']}.jpg")
            ob, ok, cons = None, False, False
            try:
                ob = json.loads(text)
                ok = set(ob) == set(SCHEMA["required"])
                cons = consistent(ob)
            except Exception:
                pass
            rows.append({
                "image": m["원본"], "id": m["id"], "run": run,
                "size": m["넣는크기"],
                "input_tokens": tm.get("prompt_n"),
                "output_tokens": tm.get("predicted_n"),
                "prompt_ms": round(tm.get("prompt_ms", 0), 1),
                "predicted_ms": round(tm.get("predicted_ms", 0), 1),
                "first_token_s": round(first, 2) if first else None,
                "vlm_s": round(total, 2),
                "schema_ok": ok, "consistent": cons,
                "output": json.dumps(ob, ensure_ascii=False) if ob else text[:200]})
            r = rows[-1]
            print(f"run{run} {m['id']}  첫토큰 {r['first_token_s']}s  "
                  f"1장 {r['vlm_s']}s  입력 {r['input_tokens']}tok  "
                  f"출력 {r['output_tokens']}tok  스키마 {ok}  일관성 {cons}")
            sys.stdout.flush()

    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 워밍업 1회를 빼고 중앙값. measure_vlm.py 와 같은 방식이다.
    kept = [r for r in rows if r["run"] > 1]
    print(f"\n저장 -> {args.out}")
    print(f"워밍업 1회를 뺀 {len(kept)}건")
    for k, label in (("first_token_s", "첫 토큰"), ("vlm_s", "1장 처리")):
        v = [r[k] for r in kept if r[k]]
        print(f"  {label:10} 중앙값 {statistics.median(v):7.2f}s   "
              f"최소 {min(v):7.2f}s   최대 {max(v):7.2f}s")
    tok = [r["input_tokens"] for r in kept if r["input_tokens"]]
    print(f"  입력 토큰   중앙값 {statistics.median(tok):.0f}")
    print(f"  스키마 준수 {sum(r['schema_ok'] for r in kept)}/{len(kept)}")
    print(f"  일관성     {sum(r['consistent'] for r in kept)}/{len(kept)}")


if __name__ == "__main__":
    main()
