# -*- coding: utf-8 -*-
"""설명 결과를 조건과 모델을 가린 채점표로 만든다.

채점자가 어느 조건에서 나온 설명인지 알면 점수가 한쪽으로 쏠린다. 순서를 섞고
조건 열을 빼서 채점표를 만들고, 정답 대응표는 따로 저장한다.

    python make_score_sheet.py 결과파일.jsonl [채점자수]

결과파일은 한 줄에 하나씩 아래 형태의 JSON이다.
    {"image": "img_03.jpg", "model": "qwen", "cond": "D", "description": "..."}
"""
import sys, os, json, csv, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/vlm_outputs.jsonl"
N_RATER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SEED = 42

rows = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise SystemExit(f"{SRC} 에 결과가 없습니다.")

random.seed(SEED)
random.shuffle(rows)

os.makedirs("results", exist_ok=True)
COLS = ["case_id", "설명", "환각(0/1)", "환각문장수", "단정표현(0/1)",
        "무엇(0/1)", "어디(0/1)", "위험대상(0/1)", "불확실성(0/1)", "형식준수(0/1)", "메모"]

for i in range(1, N_RATER + 1):
    out = f"results/score_sheet_{i}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for n, r in enumerate(rows, 1):
            w.writerow([f"c{n:03d}", r["description"]] + [""] * (len(COLS) - 2))
    print("채점표:", out)

with open("results/score_key.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "image", "model", "cond"])
    for n, r in enumerate(rows, 1):
        w.writerow([f"c{n:03d}", r["image"], r["model"], r["cond"]])
print("대응표: results/score_key.csv  (채점이 끝나기 전에는 열지 않는다)")
print(f"총 {len(rows)}건")
