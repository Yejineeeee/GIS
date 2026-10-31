# -*- coding: utf-8 -*-
"""이미지 토큰 상한을 줄여가며 설명 시간과 정확도를 잰다 (WBS 7.2 후속).

GGUF 4bit 로 1장에 58초가 나왔는데 그중 프롬프트 처리가 51.6초다. 입력 826토큰
가운데 대부분이 이미지 토큰이므로, 이미지 토큰을 줄이면 시간이 같이 줄어든다.
어디까지 줄여야 10초 목표에 닿는지, 그때 답이 망가지는지 본다.

상한마다 llama-server 를 다시 띄운다. 토큰 상한은 모델을 올릴 때 정해진다.

    python sweep_tokens.py --server llama-server.exe --model m.gguf --mmproj p.gguf
                           --images 폴더 --out 결과.csv --budgets 826 512 256 128
"""
import argparse
import base64
import csv
import json
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser(description="이미지 토큰 상한 훑기")
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--mmproj", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--budgets", type=int, nargs="+", default=[1024, 512, 256, 128])
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_gguf import SCHEMA, PROMPT, MAX_NEW_TOKENS, consistent  # noqa: E402

URL = f"http://127.0.0.1:{args.port}"


def wait_ready(proc, timeout=300):
    """서버가 뜰 때까지 기다린다. 죽으면 바로 알린다."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"서버가 종료됐다 (코드 {proc.returncode})")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
            pass
    raise SystemExit("서버가 뜨지 않는다")


def ask(img_path):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": MAX_NEW_TOKENS, "temperature": 0, "top_k": 1,
        "repeat_penalty": 1.15,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "obs", "schema": SCHEMA}},
    }
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return time.perf_counter() - t0, d


meta = json.load(open(Path(args.images) / "images.json", encoding="utf-8"))
rows = []

for budget in args.budgets:
    cmd = [args.server, "-m", args.model, "--mmproj", args.mmproj,
           "-t", str(args.threads), "-tb", str(args.threads), "-c", "8192",
           "--host", "127.0.0.1", "--port", str(args.port),
           "--image-max-tokens", str(budget)]
    log = open(Path(args.out).parent / f"server_{budget}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc)
        # 워밍업 1장은 버린다
        ask(Path(args.images) / f"{meta[0]['id']}.jpg")
        for m in meta:
            wall, d = ask(Path(args.images) / f"{m['id']}.jpg")
            tm, us = d.get("timings", {}), d.get("usage", {})
            text = d["choices"][0]["message"]["content"]
            ob, ok, cons = None, False, False
            try:
                ob = json.loads(text)
                ok = set(ob) == set(SCHEMA["required"])
                cons = consistent(ob)
            except Exception:
                pass
            rows.append({
                "budget": budget, "image": m["원본"], "id": m["id"],
                "input_tokens": us.get("prompt_tokens"),
                "output_tokens": us.get("completion_tokens"),
                "prompt_s": round(tm.get("prompt_ms", 0) / 1000, 2),
                "gen_s": round(tm.get("predicted_ms", 0) / 1000, 2),
                "vlm_s": round(wall, 2), "schema_ok": ok, "consistent": cons,
                "output": json.dumps(ob, ensure_ascii=False) if ob else text[:200]})
            r = rows[-1]
            print(f"상한 {budget:5} {m['id']}  입력 {r['input_tokens']:5}tok  "
                  f"1장 {r['vlm_s']:7.2f}s  스키마 {ok}  일관성 {cons}")
            sys.stdout.flush()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n저장 -> {args.out}\n")
print(f"{'상한':>6}{'입력토큰':>9}{'1장 중앙값':>12}{'프롬프트':>10}{'생성':>8}"
      f"{'스키마':>8}{'일관성':>8}")
for b in args.budgets:
    g = [r for r in rows if r["budget"] == b]
    if not g:
        continue
    print(f"{b:>6}{statistics.median([r['input_tokens'] for r in g]):>9.0f}"
          f"{statistics.median([r['vlm_s'] for r in g]):>11.2f}s"
          f"{statistics.median([r['prompt_s'] for r in g]):>9.2f}s"
          f"{statistics.median([r['gen_s'] for r in g]):>7.2f}s"
          f"{sum(r['schema_ok'] for r in g):>5}/{len(g)}"
          f"{sum(r['consistent'] for r in g):>5}/{len(g)}")
