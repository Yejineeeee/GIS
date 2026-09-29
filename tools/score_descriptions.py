# -*- coding: utf-8 -*-
"""설명을 채점한다. 기준은 docs/eval_rubric.md 와 같다.

자동으로 셀 수 있는 것만 센다. 환각의 일부는 사람이 봐야 하므로, 자동 판정과
사람이 볼 항목을 나눠서 내놓는다. 사람이 볼 건수를 줄이는 것이 목적이지
사람을 빼는 것이 아니다.

    python score_descriptions.py 정답파일.jsonl 출력파일.jsonl

정답 한 줄
    {"image":"a.jpg","class":"흑색연기","land_use":[{"name":"보전산지","ratio":0.57}],
     "verdict":"모호","error_radius_m":55,"risk_targets":["산림","주거지"]}
출력 한 줄
    {"image":"a.jpg","model":"internvl","cond":"D","text":"모델이 낸 원문"}
"""
import sys, os, json, csv, re
from collections import defaultdict

CLASSES = ["흑색연기", "백색및회색연기", "백색 및 회색연기", "연기", "화염", "불꽃", "불길"]
SMOKE = ("연기",)
RISK_WORDS = ["산림", "임야", "숲", "주거지", "주택", "아파트", "건물", "도로",
              "농경지", "논", "밭", "시설", "학교", "공장"]
HEDGE = ["추정", "가능성", "보인다", "보입니다", "확인이 필요", "중 하나",
         "일 수 있", "것으로 판단", "불확실"]


def parse(text):
    """모델 원문에서 description 을 꺼낸다. 못 꺼내면 형식 미준수다."""
    s = (text or "").strip().strip("`")
    if s.lower().startswith("json"):
        s = s[4:].strip()
    try:
        v = json.loads(s)
    except Exception:
        return None
    if isinstance(v, dict) and isinstance(v.get("description"), str):
        return v["description"].strip()
    return None


def score_one(fact, desc):
    """한 건을 채점한다. 자동으로 못 정하는 것은 None 으로 둔다."""
    names = [u["name"] for u in fact.get("land_use", [])]
    top = names[0] if names else None
    radius = fact.get("error_radius_m")
    verdict = fact.get("verdict")
    truth = fact.get("class", "")

    said_right = truth and truth in desc
    said_wrong = [c for c in CLASSES
                  if c in desc and c != truth
                  and not (any(k in c for k in SMOKE) and any(k in truth for k in SMOKE))]

    e_what = bool(said_right)
    e_where = bool(top and top in desc) or bool(re.search(r"\d{2,3}\.\d{3,}", desc))
    e_risk = any(w in desc for w in RISK_WORDS)
    e_unc = (bool(radius) and str(radius) in desc) or any(h in desc for h in HEDGE)
    elements = sum([e_what, e_where, e_risk, e_unc])

    # 과신은 위치가 확실하지 않은데 용도 이름을 유보 없이 말한 경우로 본다.
    # 단정 표현 목록으로 잡으면 "~이다" 가 걸려 거의 전부 과신이 된다.
    overconf = None
    if verdict in ("모호", "미확정"):
        mentioned = any(n and n in desc for n in names)
        overconf = bool(mentioned) and not any(h in desc for h in HEDGE)

    return {
        "무엇": int(e_what), "어디": int(e_where),
        "위험대상": int(e_risk), "불확실성": int(e_unc),
        "요소수": elements,
        "회피": int(elements <= 1),
        "잘못봄": int(bool(said_wrong)),
        "잘못본단어": ",".join(said_wrong),
        "과신": "" if overconf is None else int(overconf),
        "사람확인필요": int(True),      # 없는 것과 근거 없는 추론은 자동으로 못 센다
    }


def main():
    FACTS = sys.argv[1] if len(sys.argv) > 1 else "results/eval_facts.jsonl"
    OUTS = sys.argv[2] if len(sys.argv) > 2 else "results/vlm_outputs.jsonl"

    facts = {}
    for line in open(FACTS, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            facts[r["image"]] = r
    rows = []
    for line in open(OUTS, encoding="utf-8"):
        if not line.strip():
            continue
        o = json.loads(line)
        fact = facts.get(o["image"])
        if fact is None:
            print(f"  정답 없음: {o['image']}")
            continue
        desc = parse(o.get("text", o.get("description", "")))
        base = {"image": o["image"], "model": o.get("model", ""),
                "cond": o.get("cond", ""), "형식준수": int(desc is not None),
                "설명": desc if desc is not None else (o.get("text") or "")[:120]}
        # 형식을 안 지킨 건은 다른 지표의 분모에서 뺀다.
        rows.append({**base, **(score_one(fact, desc) if desc is not None else {})})

    os.makedirs("results", exist_ok=True)
    cols = ["image", "model", "cond", "형식준수", "무엇", "어디", "위험대상",
            "불확실성", "요소수", "회피", "잘못봄", "잘못본단어", "과신",
            "사람확인필요", "설명"]
    with open("results/scores.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    print(f"\n{'조건':<6}{'건수':>5}{'형식준수':>9}{'제외':>6}{'요소포함률':>11}"
          f"{'회피':>6}{'잘못봄':>8}{'과신률':>9}")
    for cond in sorted(by):
        rs = by[cond]
        ok = [r for r in rs if r["형식준수"]]
        n = len(ok) or 1
        over = [r["과신"] for r in ok if r.get("과신") != ""]
        print(f"{cond:<6}{len(rs):>5}{len(ok)/len(rs):>8.0%}{len(rs)-len(ok):>6}"
              f"{sum(r['요소수'] for r in ok)/(4*n):>10.0%}"
              f"{sum(r['회피'] for r in ok):>6}{sum(r['잘못봄'] for r in ok):>8}"
              f"{(sum(over)/len(over) if over else 0):>8.0%}")

    print(f"\n저장: results/scores.csv")
    print("환각 중 '없는 것'과 '근거 없는 추론'은 자동으로 셀 수 없다.")
    print("scores.csv 의 설명 칸을 두 사람이 읽고 환각 칸을 채운다.")
    print("'잘못봄'은 자동으로 채워져 있으니 그 부분은 다시 보지 않아도 된다.")


if __name__ == "__main__":
    main()
