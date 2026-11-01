# -*- coding: utf-8 -*-
"""질문에서 지형과 퍼짐을 빼고 다시 잰다 (WBS 7.2 후속).

잘라 넣기 실험에서 항목마다 결과가 갈렸다. 연기 유무와 색과 불꽃은 자를수록
유지되거나 좋아졌는데, 퍼짐과 지형은 틀어졌다. 연기만 잘라내면 얼마나 퍼졌는지
견줄 대상이 없고 주변을 버렸으니 지형을 볼 근거가 없으니 당연하다.

그 둘은 VLM 이 답할 필요가 없다.
    지형  land_use 의 용도지역 비율이 답한다. 폴리곤이 사진보다 정확하다
    퍼짐  탐지 마스크의 화면 면적비로 낸다. 스키마에 evidence.area_ratio 자리가 있다

그래서 세 항목만 묻고 시간과 답을 다시 잰다. measure_gguf.py 와 같은 조건이고
스키마와 질문에서 두 항목만 뺐다.

    python measure_short.py --url http://127.0.0.1:8099 --images 폴더 --out 결과.csv
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

SCHEMA = {
    "type": "object",
    "properties": {
        "smoke_present": {"type": "boolean"},
        "smoke_color": {"type": "string", "enum": ["검은색", "회색", "흰색", "없음"]},
        "flame_visible": {"type": "boolean"},
    },
    "required": ["smoke_present", "smoke_color", "flame_visible"],
}

PROMPT = """사진을 보고 아래 항목만 채운다. 사진에서 보이는 것만 적고 추측하지 않는다.

smoke_present  연기가 보이는가
smoke_color    연기 색. 없으면 "없음"
flame_visible  불꽃이 보이는가"""

MAX_NEW_TOKENS = 120


def consistent(ob):
    """연기 있음/없음과 색이 서로 맞는지. 퍼짐이 빠졌으므로 색만 본다."""
    if ob.get("smoke_present") is True:
        return ob.get("smoke_color") != "없음"
    return ob.get("smoke_color") == "없음"


def clear_cache(url, slot=0):
    req = urllib.request.Request(f"{url}/slots/{slot}?action=erase",
                                 data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def ask(url, img_path):
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
    ap = argparse.ArgumentParser(description="세 항목만 묻고 재기")
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
                  f"출력 {r['output_tokens']}tok  일관성 {cons}")
            sys.stdout.flush()

    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    kept = [r for r in rows if r["run"] > 1]
    print(f"\n저장 -> {args.out}")
    for k, label in (("first_token_s", "첫 토큰"), ("vlm_s", "1장 처리"),
                     ("output_tokens", "출력 토큰")):
        v = [r[k] for r in kept if r[k]]
        print(f"  {label:10} 중앙값 {statistics.median(v):7.2f}   "
              f"최소 {min(v):7.2f}   최대 {max(v):7.2f}")


if __name__ == "__main__":
    main()
