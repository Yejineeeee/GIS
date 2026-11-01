# -*- coding: utf-8 -*-
"""건물 위 연기와 산림 연기를 VLM 이 가릴 수 있는지 본다.

같은 연기를 건물 옥상과 산림 한가운데에 두고 찍은 프레임을 넣는다. 프레임은
`sim/index.html` 의 `scenes3()` 으로 뽑는다.

두 가지를 묻는다.

  파이프라인  실제로 쓰는 스키마 그대로. 연기 유무와 지형을 받는다
  출처        연기가 건물에서 났는지 산림에서 났는지 바로 묻는다

출처 질문은 파이프라인에 없는 것이다. VLM 이 이것을 할 수 있으면 건물 위치를
따로 조회할 필요가 없으므로, 할 수 있는지부터 재는 것이다.

    python chimney_probe.py --frames 프레임폴더 --out 결과.csv

llama-server 를 --slot-save-path 와 함께 띄워 두어야 한다. 같은 사진을 거듭
넣으면 서버가 그림 임베딩을 재활용해 입력 토큰이 1 로 떨어지고 시간이 거짓으로
짧게 나온다. 그래서 한 장마다 캐시를 비운다.
"""
import argparse
import base64
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

# run_structured.py 와 같은 값. 파이프라인이 실제로 쓰는 것이다.
PIPE_SCHEMA = {
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

PIPE_PROMPT = """사진을 보고 아래 항목만 채운다. 사진에서 보이는 것만 적고 추측하지 않는다.

smoke_present  연기가 보이는가
smoke_color    연기 색
spread         연기가 퍼진 정도
flame_visible  불꽃이 보이는가
terrain        연기가 난 자리의 지형"""

ORIGIN_SCHEMA = {
    "type": "object",
    "properties": {
        "origin": {"type": "string", "enum": ["건물", "산림", "알 수 없음"]},
        "building_under_smoke": {"type": "boolean"},
        "cue": {"type": "string",
                "enum": ["연기 아래 건물이 보임", "연기 아래 나무가 보임",
                         "굴뚝이나 지붕이 보임", "주변에 건물이 많음",
                         "주변이 숲임", "단서 없음"]},
    },
    "required": ["origin", "building_under_smoke", "cue"],
}

ORIGIN_PROMPT = """드론이 찍은 사진이다. 연기가 어디에서 나는지 본다.

사진에 보이는 것만 근거로 삼는다. 연기 기둥의 밑동이 무엇 위에 있는지 본다.

origin                건물에서 나면 건물, 산림에서 나면 산림, 가릴 수 없으면 알 수 없음
building_under_smoke  연기 밑동에 건물이 있는가
cue                   무엇을 보고 그렇게 판단했는지"""

ASKS = [("파이프라인", PIPE_PROMPT, PIPE_SCHEMA, "pipe"),
        ("출처", ORIGIN_PROMPT, ORIGIN_SCHEMA, "origin")]


def clear_cache(url, slot=0):
    req = urllib.request.Request(f"{url}/slots/{slot}?action=erase",
                                 data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def ask(url, img_path, prompt, schema, name):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        "max_tokens": 160, "temperature": 0, "top_k": 1, "repeat_penalty": 1.15,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": name, "schema": schema}},
    }
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return (d["choices"][0]["message"]["content"], time.perf_counter() - t0,
            d.get("usage", {}).get("prompt_tokens"))


def main():
    ap = argparse.ArgumentParser(description="건물 위 연기와 산림 연기 비교")
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="",
                    help="자른 사진처럼 이름 뒤에 붙은 것이 있으면 적는다")
    args = ap.parse_args()

    rows = []
    for meta_path in sorted(Path(args.frames).glob("sim_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sid = (meta.get("scene") or {}).get("id")
        img = meta_path.with_name(meta_path.stem + args.suffix + ".png")
        if not sid or not img.exists():
            continue
        row = {"frame": meta_path.stem, "scene": sid}
        for label, prompt, schema, name in ASKS:
            clear_cache(args.url)
            text, sec, tok = ask(args.url, img, prompt, schema, name)
            try:
                said = json.loads(text)
            except Exception:
                said = {}
            for k, v in said.items():
                row[k] = v
            row[label + "_초"] = round(sec, 1)
            row[label + "_입력토큰"] = tok
        rows.append(row)
        print(f"{sid:5} {row.get('smoke_present')!s:5} 지형 {row.get('terrain','-'):4} "
              f"출처 {row.get('origin','-'):6} 밑동건물 {row.get('building_under_smoke')!s:5} "
              f"근거 {row.get('cue','-')}")
        sys.stdout.flush()

    if not rows:
        raise SystemExit("프레임을 찾지 못했다")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장 -> {args.out}")


if __name__ == "__main__":
    main()
