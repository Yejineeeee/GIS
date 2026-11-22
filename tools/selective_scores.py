# -*- coding: utf-8 -*-
"""탐지 점수와 검증기 점수를 보정하고, 합쳐서 보류(선택적 예측)를 판단한다.

지금 파이프라인은 탐지 conf 와 검증기 p_smoke 를 따로 쓴다. p_smoke 가 0.7 이상이면 연기로
인정하고 나머지는 거른다. 여기서는 두 가지를 잰다.

1. 보정(calibration): 점수가 "연기일 확률"과 맞는가. ECE(10칸)로 재고, Platt 스케일링
   (점수의 logit 에 로지스틱 회귀)으로 고친 뒤 다시 잰다
2. 선택적 예측: 점수가 높은 탐지부터 남길 때, 남긴 것 중 헛탐지 비율(위험)이 남긴 비율
   (포함률)에 따라 어떻게 변하는가. 위험-포함률 곡선 아래 넓이(AURC, 작을수록 좋다)로 견준다

비교하는 점수
    conf          탐지기(YOLO) 점수만
    p_smoke       검증기(VLM) 점수만 (지금 쓰는 것)
    합침          logit(conf), logit(p_smoke) 두 개로 로지스틱 회귀 (사진 단위 5겹 교차검증)

데이터는 이미 잰 결과 파일을 읽기만 한다. VLM 을 다시 돌리지 않는다.
    results/verify_margin_eval.jsonl    평가셋 탐지 177개 (score=conf, saved=p_smoke, smoke=정답)
    results/verify_margin_mix150.jsonl  장소를 섞은 150장 탐지 165개

    python tools/selective_scores.py --json runs/selective/scores.json

평가셋 안 교차검증과, 한 세트로 맞추고 다른 세트에 적용하는 옮김 시험을 둘 다 낸다.
신뢰구간은 사진 단위 부트스트랩 2000번(95%)이다.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = {"평가셋": ROOT / "results/verify_margin_eval.jsonl",
        "150장": ROOT / "results/verify_margin_mix150.jsonl"}
TAU = 0.7
EPS = 1e-4


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    X = np.array([[logit(r["score"]), logit(r["saved"])] for r in rows])
    y = np.array([1 if r["smoke"] else 0 for r in rows])
    g = [r["image"] for r in rows]
    return X, y, g, np.array([r["saved"] for r in rows]), np.array([r["score"] for r in rows])


def fit_logreg(X, y, l2=1e-2, iters=50):
    """뉴턴법 로지스틱 회귀 (절편 포함, 약한 L2)."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    R = l2 * np.eye(A.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        H = A.T @ (A * (p * (1 - p))[:, None]) + R
        w -= np.linalg.solve(H, A.T @ (p - y) + R @ w)
    return w


def predict(w, X):
    A = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-A @ w))


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def ece(p, y, bins=10):
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & ((p < (b + 1) / bins) if b < bins - 1 else (p <= 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def risk_coverage(s, y):
    """점수 내림차순으로 남길 때 (포함률, 위험) 목록과 AURC."""
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    n = len(y)
    kept_bad = np.cumsum(1 - yy)
    cov = np.arange(1, n + 1) / n
    risk = kept_bad / np.arange(1, n + 1)
    return cov, risk, float(risk.mean())


def risk_at(s, y, k):
    """점수가 높은 k 개를 남길 때 위험."""
    order = np.argsort(-s, kind="stable")[:k]
    return float(1 - y[order].mean()) if k else float("nan")


def cv_scores(X, y, g, cols, folds=5, seed=0):
    imgs = sorted(set(g))
    random.Random(seed).shuffle(imgs)
    fold_of = {im: i % folds for i, im in enumerate(imgs)}
    out = np.zeros(len(y))
    for f in range(folds):
        te = np.array([fold_of[x] == f for x in g])
        w = fit_logreg(X[~te][:, cols], y[~te])
        out[te] = predict(w, X[te][:, cols])
    return out


def boot(fn, g, n=2000, seed=1):
    """사진 단위로 다시 뽑아 fn(idx) 의 95% 구간."""
    rng = random.Random(seed)
    imgs = sorted(set(g))
    by = {im: [i for i, x in enumerate(g) if x == im] for im in imgs}
    vals = []
    for _ in range(n):
        idx = [i for im in (rng.choice(imgs) for _ in imgs) for i in by[im]]
        v = fn(np.array(idx))
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def evaluate(name, scores, y, g, p_raw, k_rule):
    print(f"\n[{name}] 탐지 {len(y)}개, 연기 {int(y.sum())}, 헛탐지 {int((1 - y).sum())}, "
          f"지금 규칙(p>={TAU})이 남기는 수 {k_rule}")
    print("  점수        AUC  [95%]              AURC   [95%]            ECE    같은 수 남길 때 위험")
    res = {}
    for key, s in scores.items():
        a = auc(s, y)
        _, _, au = risk_coverage(s, y)
        ci_a = boot(lambda i: auc(s[i], y[i]), g)
        ci_r = boot(lambda i: risk_coverage(s[i], y[i])[2], g)
        e = ece(s, y) if s.min() >= 0 and s.max() <= 1 else float("nan")
        r = risk_at(s, y, k_rule)
        res[key] = {"auc": a, "auc_ci": ci_a, "aurc": au, "aurc_ci": ci_r, "ece": e, "risk_at_rule_k": r}
        print(f"  {key:<10}{a:.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}]   {au:.3f} [{ci_r[0]:.3f}, {ci_r[1]:.3f}]"
              f"   {e:.3f}   {r:.1%}")
    if "합침" in scores:
        s1, s0 = scores["합침"], scores["p_smoke"]
        d_auc = boot(lambda i: auc(s1[i], y[i]) - auc(s0[i], y[i]), g)
        d_aurc = boot(lambda i: risk_coverage(s1[i], y[i])[2] - risk_coverage(s0[i], y[i])[2], g)
        print(f"  합침 - p_smoke 차이 95% 구간: AUC [{d_auc[0]:+.3f}, {d_auc[1]:+.3f}], "
              f"AURC [{d_aurc[0]:+.3f}, {d_aurc[1]:+.3f}]")
        res["diff_vs_p_smoke"] = {"auc_ci": d_auc, "aurc_ci": d_aurc}
    rule_risk = float(1 - y[p_raw >= TAU].mean())
    print(f"  지금 규칙 그대로: 남김 {k_rule}/{len(y)}, 위험 {rule_risk:.1%}")
    res["rule"] = {"kept": k_rule, "risk": rule_risk}
    return res


def main():
    ap = argparse.ArgumentParser(description="탐지, 검증기 점수 보정과 선택적 예측")
    ap.add_argument("--json")
    a = ap.parse_args()
    data = {k: load(v) for k, v in SETS.items()}
    out = {}

    # 1) 세트 안 교차검증
    for name, (X, y, g, p, c) in data.items():
        platt = cv_scores(X, y, g, [1])
        both = cv_scores(X, y, g, [0, 1])
        k_rule = int((p >= TAU).sum())
        out[name + " 교차검증"] = evaluate(name + " 안 5겹 교차검증", {
            "conf": c, "p_smoke": p, "p_smoke 보정": platt, "합침": both}, y, g, p, k_rule)

    # 2) 옮김 시험: 한 세트로 맞추고 다른 세트에 적용
    for src, dst in (("평가셋", "150장"), ("150장", "평가셋")):
        Xs, ys, _, _, _ = data[src]
        Xd, yd, gd, pd, cd = data[dst]
        w1 = fit_logreg(Xs[:, [1]], ys)
        w2 = fit_logreg(Xs[:, [0, 1]], ys)
        print(f"\n  {src}로 맞춘 계수: p_smoke 보정 {np.round(w1, 3).tolist()}, 합침 {np.round(w2, 3).tolist()}"
              " (절편, logit conf, logit p_smoke 순)")
        k_rule = int((pd >= TAU).sum())
        out[f"{src}->{dst}"] = evaluate(f"{src}로 맞추고 {dst}에 적용", {
            "conf": cd, "p_smoke": pd, "p_smoke 보정": predict(w1, Xd[:, [1]]),
            "합침": predict(w2, Xd[:, [0, 1]])}, yd, gd, pd, k_rule)
        out[f"{src}->{dst}"]["coef"] = {"platt": w1.tolist(), "both": w2.tolist()}

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n요약: {a.json}")


if __name__ == "__main__":
    main()
