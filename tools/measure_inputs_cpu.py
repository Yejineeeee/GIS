# -*- coding: utf-8 -*-
"""설명 단계의 입력을 바꿨을 때 CPU 에서 몇 초 걸리는지 잰다.

GPU 에서는 전체 사진과 탐지 조각을 함께 넣으면 연기 인식이 오르고(docs/vlm_crop_views.md)
불꽃은 따로 물어야 가려진다(docs/vlm_compare.md). 현장 기기 기준은 CPU 4코어, 목표는
탐지에서 설명까지 10초다. 그 조합들이 CPU 에서 목표 안에 드는지 본다.

tools/measure_short.py 와 같은 방식이다. llama-server(GGUF Q4_K_M, 스레드 4)에 모델을 한 번
올려 두고 사진을 보낸다. 조건마다 슬롯 캐시를 비우고 잰다.

    llama-server -m Q4_K_M.gguf --mmproj mmproj.gguf -t 4 -tb 4 -c 8192 -np 1 \\
        --slot-save-path 슬롯폴더 --host 127.0.0.1 --port 8099
    python tools/measure_inputs_cpu.py --eval data/eval_100 --dets runs/verify_real/verify_real.jsonl \\
        --out results/inputs_cpu.csv

조건
    crop        탐지 조각만, 세 항목 JSON (지금 CPU 설정. docs/vlm_speed.md 의 9.2초)
    full        사진 전체만, 세 항목 JSON
    full+crop   사진 전체 + 탐지 조각, 세 항목 JSON
    small+crop  긴 변 448 로 줄인 전체 + 탐지 조각, 세 항목 JSON
    flame       탐지 조각에 "불꽃이 보이는가" 예/아니오 한 토큰 (캐시 비우고)
    flame+      crop 을 한 직후 같은 조각에 불꽃 질문 (캐시 유지). 설명 뒤에 덧붙일 때 드는 시간

조각은 탐지 상자를 모두 감싼 상자 그대로다(tools/prep_crops.py 의 여백 1배, 최소 96px).
--crop-px 로 조각의 긴 변 상한을 준다(기본 1024). 긴 변 360 을 잴 때:

    python tools/measure_inputs_cpu.py ... --conds crop --crop-px 360 --n 12 --runs 1
"""
import argparse
import base64
import csv
import io
import json
import statistics
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_short import SCHEMA, PROMPT          # 세 항목 질문 (연기 유무, 색, 불꽃)

TWO_NOTE = "\n\n사진은 두 장이다. 두 번째 사진은 첫 번째 사진의 일부를 확대한 것이다."
FLAME_Q = "이 사진에 불꽃이 보이는가. 예 또는 아니오로만 답하라."
MIN_SIDE = 96


def fixed(name):
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unicodedata.normalize("NFC", name)


def b64(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def post(url, path, body=None, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def ask(url, images, text, schema=None, max_tokens=120):
    body = {"messages": [{"role": "user", "content":
                          [{"type": "image_url", "image_url": {"url": b64(im)}} for im in images]
                          + [{"type": "text", "text": text}]}],
            "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "repeat_penalty": 1.15,
            "cache_prompt": True}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "obs", "schema": schema}}
    t0 = time.perf_counter()
    d = post(url, "/v1/chat/completions", body)
    sec = time.perf_counter() - t0
    tm = d.get("timings", {})
    return {"sec": round(sec, 2), "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
            "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
            "text": (d.get("choices") or [{}])[0].get("message", {}).get("content", "")}


def server_rss_gb():
    try:
        import psutil
    except ImportError:
        return None
    best = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        if "llama-server" in (p.info["name"] or ""):
            mi = p.info["memory_info"]
            best = max(best, getattr(mi, "peak_wset", mi.rss))
    return round(best / 2 ** 30, 2) if best else None


def main():
    ap = argparse.ArgumentParser(description="설명 입력 조합별 CPU 시간")
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--eval", required=True, help="images/eval 이 있는 폴더")
    ap.add_argument("--dets", required=True, help="vlm_verify_real.py 의 verify_real.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="사진 수 (탐지가 있는 연기 사진에서 고르게)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--conds", nargs="+",
                    default=["crop", "flame+", "flame", "small+crop", "full", "full+crop"])
    ap.add_argument("--crop-px", type=int, default=1024, help="조각의 긴 변 상한")
    a = ap.parse_args()

    from PIL import Image
    disk = {fixed(p.name): p for p in (Path(a.eval) / "images" / "eval").glob("*.jpg")}
    cand = []
    for line in open(a.dets, encoding="utf-8"):
        r = json.loads(line)
        if r["positive"] and r["dets"]:
            cand.append(r)
    cand.sort(key=lambda r: fixed(r["image"]))
    cand = cand[::max(1, len(cand) // a.n)][:a.n]

    rows = []
    for run in range(1, a.runs + 1):
        for i, r in enumerate(cand):
            img = Image.open(disk[fixed(r["image"])]).convert("RGB")
            W, H = img.size
            bs = [d["bbox"] for d in r["dets"]]
            x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
            x1, y1 = max(b[0] + b[2] for b in bs), max(b[1] + b[3] for b in bs)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = max(x1 - x0, MIN_SIDE), max(y1 - y0, MIN_SIDE)
            crop = img.crop((max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
                             min(W, int(cx + w / 2)), min(H, int(cy + h / 2))))
            crop.thumbnail((a.crop_px, a.crop_px))
            full = img.copy()
            full.thumbnail((1024, 1024))
            small = img.copy()
            small.thumbnail((448, 448))
            for c in a.conds:
                if c != "flame+":
                    post(a.url, "/slots/0?action=erase")
                if c == "crop":
                    res = ask(a.url, [crop], PROMPT, SCHEMA)
                elif c == "full":
                    res = ask(a.url, [full], PROMPT, SCHEMA)
                elif c == "full+crop":
                    res = ask(a.url, [full, crop], PROMPT + TWO_NOTE, SCHEMA)
                elif c == "small+crop":
                    res = ask(a.url, [small, crop], PROMPT + TWO_NOTE, SCHEMA)
                elif c in ("flame", "flame+"):
                    res = ask(a.url, [crop], FLAME_Q, max_tokens=1)
                else:
                    raise SystemExit(f"모르는 조건: {c}")
                res.update(run=run, img=i, cond=c, crop=f"{crop.size[0]}x{crop.size[1]}", rss_gb=server_rss_gb())
                rows.append(res)
                print(run, i, c, res["sec"], "초  입력", res["prompt_n"], "출력", res["predicted_n"],
                      "| 메모리", res["rss_gb"], "|", res["text"][:60].replace("\n", " "), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["run", "img", "cond", "crop", "sec", "prompt_n", "prompt_ms",
                                           "predicted_n", "predicted_ms", "rss_gb", "text"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'조건':<12}{'중앙값(초)':>10}{'최소':>8}{'최대':>8}{'입력 토큰':>10}{'출력 토큰':>10}")
    for c in a.conds:
        s = [r["sec"] for r in rows if r["cond"] == c]
        pn = [r["prompt_n"] for r in rows if r["cond"] == c and r["prompt_n"] is not None]
        on = [r["predicted_n"] for r in rows if r["cond"] == c and r["predicted_n"] is not None]
        print(f"{c:<12}{statistics.median(s):>10.1f}{min(s):>8.1f}{max(s):>8.1f}"
              f"{(statistics.median(pn) if pn else 0):>10.0f}{(statistics.median(on) if on else 0):>10.0f}")
    print("메모리 최대(GB):", max((r["rss_gb"] or 0) for r in rows))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
