# -*- coding: utf-8 -*-
"""설명 문장에 네 요소(무엇, 어디, 위험 대상, 불확실성)가 얼마나 들어가는지 다시 센다.

docs/generation_methods.md 의 요소 포함 45%, 회피 25건은 방식 (iii)을 4.6 실험 설정으로
돌린 값이다. 그때는 VLM 이 연기가 없다고 하면 문장이 "연기가 관측되지 않았다"로 끝났다.
지금 파이프라인은 연기 유무를 탐지와 검증기가 정하고 문장은 "연기가 탐지되었다"로 시작한다
(src/vlm/templates.py). 같은 100장, 같은 VLM 답으로 지금 템플릿의 값을 낸다.

VLM 을 다시 돌리지 않는다. results/eval_method_iii.jsonl 의 observed 를 그대로 쓴다.
채점은 tools/score_descriptions.py 의 score_one 을 그대로 부른다. 기존 결과 파일은 건드리지 않는다.

    python tools/element_coverage.py                      지금 템플릿과 4.6 설정을 나란히
    python tools/element_coverage.py --out 결과.jsonl     지금 템플릿 문장을 파일로도 남김
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vlm"))
sys.path.insert(0, str(ROOT / "tools"))

ELEMENTS = [("규칙.무엇", "무엇"), ("규칙.어디", "어디"), ("규칙.위험대상", "위험 대상"), ("규칙.불확실성", "불확실성")]


def nfc(s):
    return unicodedata.normalize("NFC", s)


def run(facts, observed, build, label):
    import eval_alert
    from score_descriptions import score_one, has_smoke
    rows = []
    for i, f in enumerate(facts):
        ob = observed.get(nfc(f["image"])) or {}
        al = eval_alert.alert(f, i)
        desc = build(ob, al)
        s = score_one(f, desc, ob)
        s.update(image=f["image"], verdict=f["verdict"], smoke=has_smoke(f), description=desc)
        rows.append(s)
    sm = [r for r in rows if r["smoke"]]
    print(f"\n[{label}] 연기 사진 {len(sm)}장")
    print(f"{'판정':<6}{'장수':>5}" + "".join(f"{n:>10}" for _, n in ELEMENTS) + f"{'요소 포함':>10}{'회피':>6}{'판정 준수':>10}{'과신':>6}")
    for v in ("확정", "모호", "미확정", "전체"):
        S = sm if v == "전체" else [r for r in sm if r["verdict"] == v]
        if not S:
            continue
        cells = "".join(f"{sum(r[k] for r in S):>10}" for k, _ in ELEMENTS)
        inc = sum(r["규칙.요소수"] for r in S) / (4 * len(S))
        over = [r["규칙.과신"] for r in S if r["규칙.과신"] != ""]
        print(f"{v:<6}{len(S):>5}{cells}{inc:>10.0%}{sum(r['규칙.회피'] for r in S):>6}"
              f"{sum(r['규칙.판정준수'] for r in S):>7}/{len(S):<3}"
              f"{(str(sum(over)) + '/' + str(len(over))) if over else '-':>6}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="요소 포함률 다시 세기")
    ap.add_argument("--facts", default=str(ROOT / "results" / "eval_facts.jsonl"))
    ap.add_argument("--observed", default=str(ROOT / "results" / "eval_method_iii.jsonl"))
    ap.add_argument("--out", help="지금 템플릿 문장과 채점을 남길 jsonl")
    ap.add_argument("--show", type=int, default=3, help="판정마다 예시 문장 수")
    ap.add_argument("--sheet", help="짧은 문장과 늘린 문장을 나란히 놓은 평가지(.md)를 만든다")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    import templates
    facts = [json.loads(x) for x in open(a.facts, encoding="utf-8") if x.strip()]
    observed = {nfc(r["image"]): r.get("observed") for r in
                (json.loads(x) for x in open(a.observed, encoding="utf-8") if x.strip())}

    run(facts, observed, lambda ob, al: templates.build(ob, al, use_vlm_facts=True), "4.6 설정 (VLM 이 연기 유무를 정함)")
    short = run(facts, observed, lambda ob, al: templates.build(ob, al, rich=False), "짧은 문장 (연기는 탐지와 검증기가 정함)")
    now = run(facts, observed, lambda ob, al: templates.build(ob, al), "늘린 문장 (용도 비율, 좌표, 경사, 위험도)")

    # 채점기의 "무엇"은 문장에 클래스 이름(흑색연기 등)이 있어야 한다. 색을 쓰지 않기로 해서
    # 늘 0 이 된다. 연기라고 말했는지를 따로 센다
    for label, rows in (("짧은 문장", short), ("늘린 문장", now)):
        sm = [r for r in rows if r["smoke"]]
        what = sum("연기" in r["description"] for r in sm)
        other = sum(r["규칙.어디"] + r["규칙.위험대상"] + r["규칙.불확실성"] for r in sm)
        n = sum(len(r["description"]) for r in sm) / len(sm)
        print(f"{label}: 연기라고 말함 {what}/{len(sm)}, 그것을 '무엇'으로 치면 요소 포함 "
              f"{(what + other) / (4 * len(sm)):.0%}, 문장 길이 평균 {n:.0f}자")

    print("\n문장 예 (앞: 짧은 문장, 뒤: 늘린 문장)")
    for v in ("확정", "모호", "미확정"):
        idx = [i for i, r in enumerate(now) if r["smoke"] and r["verdict"] == v][:a.show]
        for i in idx:
            print(f"  [{v}] {short[i]['description']}\n      -> {now[i]['description']}")
    miss = Counter(n for r in now if r["smoke"] for k, n in ELEMENTS if not r[k])
    print("\n늘린 문장에서 빠진 요소(연기 사진):", dict(miss))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fo:
            for s, r in zip(short, now):
                r = dict(r, description_short=s["description"])
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {a.out}")
    if a.sheet:
        sheet(short, now, facts, a.sheet)


def sheet(short, now, facts, path, per=(4, 3, 3), seed=42):
    """짧은 문장과 늘린 문장을 나란히 놓은 평가지. 어느 쪽이 어느 것인지는 끝에 적는다."""
    import random
    rng = random.Random(seed)
    L = ["# 알람 설명 비교 평가지", "",
         "산불 연기 알람에 붙는 설명 문장 두 가지입니다. 관제 화면에서 이 알람을 받았다고 생각하고,",
         "어느 쪽이 상황을 판단하는 데 더 도움이 되는지 골라 주세요. 이유를 한 줄 적어 주시면 더 좋습니다.", ""]
    key = []
    k = 0
    for v, n in zip(("확정", "모호", "미확정"), per):
        idx = [i for i, r in enumerate(now) if r["smoke"] and r["verdict"] == v]
        for i in rng.sample(idx, min(n, len(idx))):
            k += 1
            first_is_new = rng.random() < 0.5
            a_txt, b_txt = (now[i], short[i]) if first_is_new else (short[i], now[i])
            f = facts[i]
            L += [f"## {k}. 위치 판정: {v} (오차 반경 {f['error_radius_m']}m)", "",
                  f"- 가: {a_txt['description']}", f"- 나: {b_txt['description']}", "",
                  "더 도움이 되는 쪽: (가 / 나 / 비슷함)", "이유:", ""]
            key.append(f"{k}번: 늘린 문장은 {'가' if first_is_new else '나'}")
    L += ["---", "", "평가자에게 보이지 않게 떼어 둘 것", ""] + [f"- {x}" for x in key]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {path} ({k}건)")


if __name__ == "__main__":
    main()
