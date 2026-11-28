# -*- coding: utf-8 -*-
"""용도 확정을 규칙 대신 학습한 점수로 고르면 어떤가 (위치 쪽 선택적 예측).

지금 판정 규칙(src/pipeline/verdict.py)은 반경 안 1순위 용도 비율이 문턱을 넘고, 내려다보는 각이
얕지 않으면 확정한다. 여기서는 같은 정보로 "1순위 용도가 맞을 확률"을 로지스틱 회귀로 배우고,
점수가 높은 사례부터 확정할 때 틀린 단정이 어떻게 변하는지 규칙과 같은 확정 비율에서 견준다.

특징 (모두 파이프라인이 실제로 아는 값이다. 정답은 쓰지 않는다)
    p1, p2        반경 안 1, 2순위 용도 비율
    coverage      반경 안에서 용도지역이 있는 비율
    shallow       내려다보는 각이 얕은가
    log 반경, 거리 D

사례는 tools/conformal_radius.py make 로 만든 파일을 쓴다. 보정용(시드 7)으로 맞추고
시험용(시드 42)에 적용한다. 시험 사례는 잡음 1배와 2배 둘 다 본다.

    python tools/selective_location.py <landuse_5179.geojson> --train runs/conformal/calib_n1.jsonl \\
        --test runs/conformal/test_n1.jsonl runs/conformal/test_n2.jsonl --json runs/selective/location.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verdict_coverage as vc  # noqa: E402
from selective_scores import fit_logreg, predict, auc, risk_coverage, ece  # noqa: E402


def features(geo, recs, points, scale=1.0):
    X, y, rule = [], [], []
    for r in recs:
        R = r["radius"] * scale
        lu, _, qi = vc.query(geo, r["lat"], r["lon"], R, n_points=points)
        kind, _ = vc.judge(lu, R, qi["landuse_coverage"], r["shallow"])
        p1 = lu[0]["ratio"] if lu else 0.0
        p2 = lu[1]["ratio"] if len(lu) > 1 else 0.0
        X.append([p1, p2, qi["landuse_coverage"], 1.0 if r["shallow"] else 0.0, math.log(R), r["D"] / 1000])
        y.append(1 if lu and lu[0]["name"] == r["truth"] else 0)
        rule.append(kind == "확정")
    return np.array(X), np.array(y), np.array(rule)


def at_coverage(s, y, k):
    order = np.argsort(-s, kind="stable")[:k]
    return float(y[order].mean())


def main():
    ap = argparse.ArgumentParser(description="위치 쪽 선택적 예측")
    ap.add_argument("landuse")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--json")
    a = ap.parse_args()
    geo = vc.GeoData(a.landuse)
    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    Xtr, ytr, _ = features(geo, load(a.train), a.points)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = fit_logreg((Xtr - mu) / sd, ytr)
    print(f"학습 {len(ytr)}개 ({a.train}), 계수(표준화): "
          + ", ".join(f"{n} {v:+.2f}" for n, v in zip(["절편", "p1", "p2", "coverage", "shallow", "log r", "D"], w)))
    out = {"coef": w.tolist(), "tests": {}}
    for tp in a.test:
        recs = load(tp)
        X, y, rule = features(geo, recs, a.points)
        s = predict(w, (X - mu) / sd)
        n, k = len(y), int(rule.sum())
        rule_acc = float(y[rule].mean())
        p1_acc = at_coverage(X[:, 0] - 10 * X[:, 3], y, k)   # 얕은 각은 뒤로 미루고 p1 순
        lr_acc = at_coverage(s, y, k)
        res = {"n": n, "rule_confirm": k / n, "rule_acc": rule_acc, "p1_acc": p1_acc, "learned_acc": lr_acc,
               "learned_auc": auc(s, y), "learned_ece": ece(s, y), "learned_aurc": risk_coverage(s, y)[2],
               "curve": {}}
        print(f"\n== {tp} (잡음 x{recs[0]['noise_scale']}), 사례 {n}개")
        print(f"  규칙: 확정 {k / n:.0%}, 확정 정확도 {rule_acc:.1%}, 틀린 단정 {(1 - rule_acc) * k / n:.1%}")
        print(f"  같은 {k}개를 p1 순으로:   정확도 {p1_acc:.1%}")
        print(f"  같은 {k}개를 학습 점수 순으로: 정확도 {lr_acc:.1%}  (AUC {res['learned_auc']:.3f}, "
              f"ECE {res['learned_ece']:.3f})")
        print("  학습 점수로 확정 비율을 바꾸면:  확정 비율  확정 정확도  틀린 단정")
        for f in (0.3, 0.4, 0.48, 0.55, 0.6):
            kk = round(f * n)
            acc = at_coverage(s, y, kk)
            res["curve"][f] = acc
            print(f"    {f:>28.0%}{acc:>12.1%}{(1 - acc) * kk / n:>10.1%}")
        # 사례 단위 부트스트랩 (모델은 그대로, 시험 사례만 다시 뽑는다). 학습 점수는 규칙과 같은 수를 확정
        rng = np.random.default_rng(1)
        bs = {"rule_wrong": [], "learned_wrong": [], "diff_wrong": [], "rule_acc": [], "learned_acc": []}
        for _ in range(2000):
            i = rng.integers(0, n, n)
            yy, rr, ss = y[i], rule[i], s[i]
            kk = int(rr.sum())
            ra = yy[rr].mean()
            la = at_coverage(ss, yy, kk)
            rw, lw = (1 - ra) * kk / n, (1 - la) * kk / n
            for key, v in (("rule_wrong", rw), ("learned_wrong", lw), ("diff_wrong", lw - rw),
                           ("rule_acc", ra), ("learned_acc", la)):
                bs[key].append(v)
        res["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
        res["learned_wrong"] = (1 - lr_acc) * k / n
        res["rule_wrong"] = (1 - rule_acc) * k / n
        c = res["ci"]
        print(f"  95% 구간 (부트스트랩 2000번): 틀린 단정 규칙 {res['rule_wrong']:.1%} "
              f"[{c['rule_wrong'][0]:.1%}, {c['rule_wrong'][1]:.1%}], 학습 {res['learned_wrong']:.1%} "
              f"[{c['learned_wrong'][0]:.1%}, {c['learned_wrong'][1]:.1%}], 차이 "
              f"[{c['diff_wrong'][0]:+.1%}, {c['diff_wrong'][1]:+.1%}]")
        print(f"                              확정 정확도 규칙 [{c['rule_acc'][0]:.1%}, {c['rule_acc'][1]:.1%}], "
              f"학습 [{c['learned_acc'][0]:.1%}, {c['learned_acc'][1]:.1%}]")
        res["cases"] = {"y": y.tolist(), "rule": rule.astype(int).tolist(), "score": s.tolist()}
        out["tests"][tp] = res
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n요약: {a.json}")


if __name__ == "__main__":
    main()
