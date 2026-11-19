# -*- coding: utf-8 -*-
"""CPU 실행 방식(llama.cpp GGUF)으로 모델 크기를 비교한다 (WBS 7.5).

7B 는 개발 노트북 GPU(4GB)에 올라가지 않는다. 그래서 3B 와 7B 를 둘 다 llama-server(GGUF
Q4_K_M)로 돌려 같은 사진, 같은 질문으로 정확도와 시간을 잰다. 모델만 바꿔 서버를 띄우고
이 도구를 두 번 돌린다.

    llama-server -m <모델>.gguf --mmproj <mmproj>.gguf -t 8 -tb 8 -c 8192 -np 1 --port 8099
    python tools/vlm_gguf_compare.py verify --dets runs/verify_real/verify_real.jsonl \\
        --images data/eval_100/images/eval --out results/model_size_verify_3b.jsonl
    python tools/vlm_gguf_compare.py observe --facts results/eval_facts.jsonl \\
        --images data/eval_100/images/eval --out results/model_size_observe_3b.jsonl
    python tools/vlm_gguf_compare.py report --verify results/model_size_verify_3b.jsonl \\
        --observe results/model_size_observe_3b.jsonl

verify   후보 검증기. vlm_verify_real.py 가 남긴 박스를 읽어 조각을 자르고 "연기가 있는가"를
         묻는다. 자르는 법과 질문은 src/pipeline/vlm_client.py 와 같다. 답은 한 토큰만 받고
         첫 토큰 후보의 확률에서 P(예)/(P(예)+P(아니오)) 를 읽는다
observe  사진 전체(긴 변 1024)에 세 항목(연기 유무, 연기색, 불꽃)을 JSON 으로 묻는다.
         tools/measure_short.py 의 질문과 스키마다

중간에 끊겨도 이어서 돈다(이미 있는 줄은 건너뛴다).
"""
import argparse
import base64
import io
import json
import math
import statistics
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from measure_short import SCHEMA, PROMPT as OBS_PROMPT  # noqa: E402

VERIFY_PROMPT = "이 사진에 연기가 있는가. 예 또는 아니오로만 답하라."
VERIFY_PX = 448
N_PROBS = 40
COLOR = {"흑색연기": "검은색", "백색회색연기": "흰색/회색"}


def fixed(name):
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unicodedata.normalize("NFC", name)


def crop_det(img, bbox, margin=0.25, min_side=160):
    """src/pipeline/vlm_client.py 의 crop_det 과 같다."""
    from PIL import Image
    W, H = img.size
    x, y, w, h = bbox
    side = max(max(w, h) * (1 + 2 * margin), min_side)
    cx, cy, half = x + w / 2, y + h / 2, side / 2
    b = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    cv = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), (128, 128, 128))
    cv.paste(img.crop((max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H))),
             (max(-b[0], 0), max(-b[1], 0)))
    return cv.resize((VERIFY_PX, VERIFY_PX), Image.LANCZOS)


def b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def chat(url, img, text, **extra):
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(img)}},
                                                      {"type": "text", "text": text}]}],
            "temperature": 0, "top_k": 1, "cache_prompt": False}
    body.update(extra)
    t0 = time.perf_counter()
    d = post(url, "/v1/chat/completions", body)
    return d, round(time.perf_counter() - t0, 2)


def p_yes(d):
    """첫 토큰 후보에서 '예'로 시작하는 것과 '아'로 시작하는 것의 확률을 모은다."""
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    y = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().startswith("예"))
    n = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().startswith("아"))
    return (y / (y + n) if y + n > 0 else None), round(y + n, 4)


def done_keys(path, key):
    if not Path(path).exists():
        return [], set()
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows, {key(r) for r in rows}


def run_verify(a):
    from PIL import Image
    disk = {fixed(p.name): p for p in Path(a.images).glob("*.jpg")}
    recs = [json.loads(l) for l in open(a.dets, encoding="utf-8")]
    rows, seen = done_keys(a.out, lambda r: (r["image"], r["j"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in recs:
            name = fixed(r["image"])
            todo = [d for d in r["dets"] if (name, d["j"]) not in seen]
            if not todo:
                continue
            img = Image.open(disk[name]).convert("RGB")
            for d in todo:
                cr = crop_det(img, d["bbox"], min_side=max(160, img.size[0] // 6))
                if a.px != VERIFY_PX:
                    cr = cr.resize((a.px, a.px), Image.LANCZOS)
                res, sec = chat(a.url, cr, VERIFY_PROMPT, max_tokens=1, logprobs=True, top_logprobs=N_PROBS)
                p, mass = p_yes(res)
                row = {"image": name, "positive": r["positive"], "j": d["j"], "score": d["score"],
                       "smoke": d["smoke"], "saved": d["p_yes"], "p_yes": None if p is None else round(p, 4),
                       "mass": mass, "sec": sec, "prompt_n": res.get("timings", {}).get("prompt_n")}
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                fo.flush()
                n += 1
                print(len(seen) + n, "연기" if d["smoke"] else "헛", "GPU 저장값", d["p_yes"], "→", row["p_yes"],
                      "합", mass, sec, "초", flush=True)
                if a.limit and n >= a.limit:
                    return


def run_observe(a):
    from PIL import Image
    disk = {fixed(p.name): p for p in Path(a.images).glob("*.jpg")}
    facts = [json.loads(l) for l in open(a.facts, encoding="utf-8")]
    rows, seen = done_keys(a.out, lambda r: r["image"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for f in facts:
            name = fixed(f["image"])
            if name in seen:
                continue
            img = Image.open(disk[name]).convert("RGB")
            img.thumbnail((1024, 1024))
            res, sec = chat(a.url, img, OBS_PROMPT, max_tokens=120, repeat_penalty=1.15,
                            response_format={"type": "json_schema", "json_schema": {"name": "obs", "schema": SCHEMA}})
            raw = res["choices"][0]["message"]["content"]
            try:
                ob = json.loads(raw)
            except ValueError:
                ob = None
            row = {"image": name, "label_class": f["label_class"], "observed": ob, "raw": raw if ob is None else None,
                   "sec": sec, "prompt_n": res.get("timings", {}).get("prompt_n")}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            print(len(seen) + n, f["label_class"], ob, sec, "초", flush=True)
            if a.limit and n >= a.limit:
                return


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def report(a):
    if a.verify:
        R = [json.loads(l) for l in open(a.verify, encoding="utf-8")]
        ok = [r for r in R if r["p_yes"] is not None]
        pos = [r["p_yes"] for r in ok if r["smoke"]]
        neg = [r for r in ok if not r["smoke"]]
        print(f"[검증기] 탐지 {len(R)}개 (확률을 못 읽은 것 {len(R) - len(ok)}개), 연기 {len(pos)}, 헛탐지 {len(neg)}")
        print(f"  AUC {auc(pos, [r['p_yes'] for r in neg]):.3f}   GPU 저장값으로 낸 AUC "
              f"{auc([r['saved'] for r in ok if r['smoke']], [r['saved'] for r in neg]):.3f}")
        for tau in a.tau:
            line = f"  문턱 {tau}: 연기 남김 {sum(p >= tau for p in pos)}/{len(pos)}, 헛탐지 거름 {sum(r['p_yes'] < tau for r in neg)}/{len(neg)}"
            kinds = {}
            for r in neg:
                k = "연기 사진 속" if r["positive"] else r["image"].split("_")[1]
                kinds.setdefault(k, []).append(r["p_yes"])
            line += " (" + ", ".join(f"{k} {sum(p < tau for p in v)}/{len(v)}" for k, v in kinds.items()) + ")"
            print(line)
        secs = [r["sec"] for r in R]
        print(f"  1회 시간 중앙값 {med(secs):.1f}초 (최소 {min(secs):.1f} ~ 최대 {max(secs):.1f}), 입력 토큰 {med([r['prompt_n'] for r in R if r['prompt_n']]):.0f}")
        print(f"  확률 합(예+아니오) 중앙값 {med([r['mass'] for r in R]):.3f}")
    if a.observe:
        R = [json.loads(l) for l in open(a.observe, encoding="utf-8")]
        ok = [r for r in R if r["observed"]]
        smoke = [r for r in ok if r["label_class"] in COLOR]
        other = [r for r in ok if r["label_class"] not in COLOR]
        print(f"[사진 전체] {len(R)}장 (JSON 을 못 읽은 것 {len(R) - len(ok)}장)")
        print(f"  연기 사진을 연기라고 함 {sum(r['observed']['smoke_present'] is True for r in smoke)}/{len(smoke)}")
        print(f"  연기 아닌 사진을 아니라고 함 {sum(r['observed']['smoke_present'] is False for r in other)}/{len(other)}")
        hit = 0
        dist = {}
        for r in smoke:
            c = r["observed"]["smoke_color"]
            dist[c] = dist.get(c, 0) + 1
            want = COLOR[r["label_class"]]
            hit += (c == "검은색") if want == "검은색" else (c in ("흰색", "회색"))
        print(f"  연기색 맞춤 {hit}/{len(smoke)}, 답한 색 {dist}")
        print(f"  불꽃이 있다고 함 {sum(r['observed']['flame_visible'] is True for r in ok)}/{len(ok)} (평가셋에는 불꽃이 없다)")
        secs = [r["sec"] for r in R]
        print(f"  1장 시간 중앙값 {med(secs):.1f}초 (최소 {min(secs):.1f} ~ 최대 {max(secs):.1f}), 입력 토큰 {med([r['prompt_n'] for r in R if r['prompt_n']]):.0f}")


def main():
    ap = argparse.ArgumentParser(description="CPU 실행 방식으로 모델 크기 비교")
    ap.add_argument("mode", choices=["verify", "observe", "report"])
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--dets")
    ap.add_argument("--facts")
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="이번에 돌릴 건수(0 은 끝까지)")
    ap.add_argument("--verify", help="report: verify 결과 파일")
    ap.add_argument("--observe", help="report: observe 결과 파일")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--px", type=int, default=VERIFY_PX, help="verify: 조각 크기 (기본 448)")
    a = ap.parse_args()
    {"verify": run_verify, "observe": run_observe, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
