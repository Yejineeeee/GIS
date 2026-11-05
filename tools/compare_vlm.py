# -*- coding: utf-8 -*-
"""VLM 이 있을 때와 없을 때를 같은 사진으로 비교한다 (WBS 4.7).

"YOLO 와 템플릿만으로도 같은 결과가 나오지 않느냐"에 답하려고 만들었다. VLM 이
맡는 일을 셋으로 나눠, 각각 VLM 없이 하는 방식과 나란히 잰다. 모델은 다시 돌리지
않고 이미 저장된 결과를 읽는다.

    python tools/compare_vlm.py

1. 좌표에 쓸 탐지 고르기
   없음  YOLO conf 1등 (지금 방식, V0)
   있음  후보 검증기 P(예) 0.7 이상 중 conf 1등, 통과한 것이 없으면 conf 1등 (V2)
   실제 사진 100장   좌표에 쓴 탐지가 헛탐지인 사진 수    runs/verify_real/verify_real.jsonl
   시뮬레이터 31장   좌표 오차(m)                         runs/verify/verify.jsonl
2. 알람 (이 사진에 연기가 있는가)
   없음  YOLO 탐지가 하나라도 있으면 알람
   있음  (a) 검증기를 통과한 탐지가 있으면 알람  (b) VLM 이 사진 전체를 보고 답한 연기 유무
3. 설명에 들어가는 사실 (연기색, 불꽃)
   없음  그 항목을 문장에서 뺀다, 또는 가장 흔한 값으로 늘 같게 쓴다
   있음  VLM 이 사진 전체를 보고 답한 값                  results/eval_method_iii.jsonl

정답은 results/eval_facts.jsonl(71330 라벨)이다.
탐지의 연기 여부는 vlm_verify_real.py 의 기준(연기 라벨 박스와 IoU 0.1 이상 또는 중심이
안)을 따른다. tools/eval_detect.py 는 마스크가 겹치는지로 세므로 놓친 사진 수가 하나
다르다(9 대 10).
"""
import json
import statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAU = 0.7


def load(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8")]


def fix_name(s):
    # verify_real.jsonl 의 이름은 utf-8 바이트를 cp437 로 읽은 채 저장됐다
    try:
        return s.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def pick(dets, use_vlm):
    if not dets:
        return None
    if use_vlm:
        ok = [d for d in dets if d["p_yes"] >= TAU]
        if ok:
            return max(ok, key=lambda d: d["score"])
    return max(dets, key=lambda d: d["score"])


def mcnemar(b, c):
    """짝지은 비교의 양측 정확 검정. b: 없음만 맞음, c: 있음만 맞음."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    real = load("runs/verify_real/verify_real.jsonl")
    sim = load("runs/verify/verify.jsonl")
    facts = {f["image"]: f for f in load("results/eval_facts.jsonl")}
    vlm = {r["image"]: r for r in load("results/eval_method_iii.jsonl")}
    for r in real:
        r["image"] = fix_name(r["image"])
    assert all(r["image"] in facts for r in real), "이름이 맞지 않는 사진이 있다"

    print("1. 좌표에 쓸 탐지 고르기\n")
    pos = [r for r in real if r["positive"] and r["dets"]]
    bad0 = {r["image"] for r in pos if not pick(r["dets"], False)["smoke"]}
    bad1 = {r["image"] for r in pos if not pick(r["dets"], True)["smoke"]}
    fixed, broke = len(bad0 - bad1), len(bad1 - bad0)
    print(f"  실제 사진: 탐지가 있는 연기 사진 {len(pos)}장 중 좌표에 쓴 탐지가 헛탐지인 사진")
    print(f"    VLM 없음 {len(bad0)}장, VLM 있음 {len(bad1)}장 (고침 {fixed}, 망침 {broke}, "
          f"McNemar p={mcnemar(broke, fixed):.3f})")
    neg = [r for r in real if not r["positive"] and r["dets"]]
    print(f"  오탐 사진 중 탐지가 있는 {len(neg)}장은 어느 탐지를 골라도 헛탐지다")

    e0 = [pick(r["dets"], False)["err_m"] for r in sim if r["dets"]]
    e1 = [pick(r["dets"], True)["err_m"] for r in sim if r["dets"]]
    better = sum(b < a - 1 for a, b in zip(e0, e1))
    worse = sum(b > a + 1 for a, b in zip(e0, e1))
    print(f"\n  시뮬레이터 {len(e0)}장 좌표 오차")
    print(f"    VLM 없음 평균 {st.mean(e0):.1f}m, 중앙 {st.median(e0):.1f}m, 100m 초과 {sum(e > 100 for e in e0)}장")
    print(f"    VLM 있음 평균 {st.mean(e1):.1f}m, 중앙 {st.median(e1):.1f}m, 100m 초과 {sum(e > 100 for e in e1)}장")
    print(f"    1m 넘게 좋아진 사진 {better}, 나빠진 사진 {worse}, 같음 {len(e0) - better - worse}")
    for a, b, r in zip(e0, e1, [r for r in sim if r["dets"]]):
        if abs(a - b) > 1:
            print(f"      {r['image']} {r['bearing']}도 {r['D']}m: {a:.1f}m -> {b:.1f}m")

    print("\n2. 알람 (사진에 연기가 있는가)\n")
    gt = {r["image"]: r["positive"] for r in real}

    def score(name, pred):
        tp = sum(pred[k] and gt[k] for k in gt)
        fp = sum(pred[k] and not gt[k] for k in gt)
        fn = sum(not pred[k] and gt[k] for k in gt)
        tn = len(gt) - tp - fp - fn
        print(f"  {name:<34} 맞춤 {tp + tn:>3}/100  연기 사진 잡음 {tp}/70  오탐 사진 알람 {fp}/30")
        return pred

    n_pos = sum(gt.values())
    score("항상 '있다' (기준선)", {k: True for k in gt})
    a0 = score("VLM 없음: YOLO 탐지 있음", {r["image"]: bool(r["dets"]) for r in real})
    a1 = score(f"VLM 있음 (a): 검증기 통과 탐지 있음", {r["image"]: any(d["p_yes"] >= TAU for d in r["dets"]) for r in real})
    a2 = score("VLM 있음 (b): 사진 전체 VLM 판단", {k: bool(vlm[k]["observed"].get("smoke_present")) for k in gt})
    assert n_pos == 70

    def pair(x, y):
        b = sum(x[k] == gt[k] and y[k] != gt[k] for k in gt)
        c = sum(x[k] != gt[k] and y[k] == gt[k] for k in gt)
        return b, c, mcnemar(b, c)
    b, c, p = pair(a0, a1)
    print(f"  YOLO 대 (a): YOLO만 맞음 {b}, (a)만 맞음 {c}, p={p:.3f}")
    b, c, p = pair(a0, a2)
    print(f"  YOLO 대 (b): YOLO만 맞음 {b}, (b)만 맞음 {c}, p={p:.3f}")

    print("\n3. 설명에 들어가는 사실 (VLM 이 사진 전체를 보고 답한 값)\n")
    smoke = [k for k in gt if gt[k]]
    color_gt = {k: "검은색" if facts[k]["label_class"] == "흑색연기" else "흰색/회색" for k in smoke}
    said = {k: vlm[k]["observed"].get("smoke_color") for k in smoke}
    said_n = {k: ("흰색/회색" if v in ("흰색", "회색") else v) for k, v in said.items()}
    from collections import Counter
    print(f"  연기색 정답 분포 {dict(Counter(color_gt.values()))}")
    print(f"  VLM 이 답한 색   {dict(Counter(said.values()))}")
    ok = sum(said_n[k] == color_gt[k] for k in smoke)
    major = Counter(color_gt.values()).most_common(1)[0]
    print(f"  VLM 맞춤 {ok}/70, 늘 '{major[0]}' 이라고 쓰면 {major[1]}/70")
    fl = Counter((facts[k]["flame"], bool(vlm[k]["observed"].get("flame_visible"))) for k in gt)
    print(f"  불꽃 (정답, VLM): {dict(fl)}. 평가셋에 불꽃 사진이 없어 이 평가셋으로는 잴 수 없다")


if __name__ == "__main__":
    main()
