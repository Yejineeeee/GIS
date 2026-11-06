# -*- coding: utf-8 -*-
"""확정, 모호, 미확정 3단계 위치 판정 (WBS 3.6).

오차 반경 안의 용도 비율(3.4 query.py)을 보고, 그 용도를 알람에 단정해도 되는지 정한다.
임계값은 3.5 모의 실험 결과(docs/threshold.md)를 쓴다.

    from judge import judge
    verdict, threshold = judge(land_use, radius_m, coverage, shallow)

판정 순서 (위에서 걸리면 거기서 끝)
    1. 미확정: 조회 결과가 없음
    2. 미확정: 내려다보는 각이 얕음 (transform info.shallow). 위치 자체를 믿기 어렵다
    3. 미확정: 용도지역이 원의 60% 미만만 덮음 (나머지가 바다나 자료 밖)
    4. 미확정: 반경이 임계값 표보다 큼 (100m 초과). 3.5에서 재지 않은 반경이라 근거가 없다
    5. 확정:   1순위 비율 >= 반경별 임계값 (최다비율 규칙)
    6. 모호:   용도가 2개 이상이고 1·2순위 합 >= 0.9 ("A 또는 B")
    7. 미확정: 그 밖

임계값 표 사이 반경은 가까운 두 값 사이를 직선으로 잇는다. 10m보다 작으면 10m 값을 쓴다.
외부 패키지 없이 동작한다. 직접 실행하면 예시 판정을 확인한다.

    python3 src/geo/judge.py
"""

RULE = "최다비율"

# 3.5 결과, 최다비율 규칙. 목표 정확도별 (반경 m, 임계값). docs/threshold.md
THRESHOLDS = {
    90: [(10, 0.00), (30, 0.00), (50, 0.56), (100, 0.64)],
    95: [(10, 0.00), (30, 0.57), (50, 0.72), (100, 0.80)],
    99: [(10, 0.77), (30, 0.83), (50, 0.87), (100, 0.92)],
}
TARGET = 95             # 기본 목표 정확도 (팀 결정 전까지 95%)
AMBIGUOUS_SUM = 0.9     # 모호: 1·2순위 합 기준
MIN_COVERAGE = 0.6      # 용도지역이 원 안을 이만큼은 덮어야 판정한다

# 5.2 임시 판정(src/pipeline/verdict.py)이 쓰던 이름. 그대로 두어 다른 코드가 깨지지 않게 한다
THRESHOLDS_95 = THRESHOLDS[95]


def threshold_for(radius_m, target=TARGET):
    """반경에 맞는 임계값. 표보다 큰 반경이면 표의 마지막 값 (판정은 judge에서 미확정)."""
    table = THRESHOLDS[target]
    if radius_m <= table[0][0]:
        return table[0][1]
    for (r0, t0), (r1, t1) in zip(table, table[1:]):
        if radius_m <= r1:
            return round(t0 + (t1 - t0) * (radius_m - r0) / (r1 - r0), 3)
    return table[-1][1]


def judge_detail(land_use, radius_m, coverage=1.0, shallow=False, target=TARGET):
    """판정과 그 이유를 dict로 돌려준다. 디버그 기록과 문서용."""
    t = threshold_for(radius_m, target)
    p1 = land_use[0]["ratio"] if land_use else 0.0
    p2 = land_use[1]["ratio"] if land_use and len(land_use) > 1 else 0.0
    out = {"verdict": "미확정", "threshold": t, "top1": p1, "top2": p2,
           "radius_m": radius_m, "target": target}

    if not land_use:
        out["reason"] = "조회 결과 없음"
    elif shallow:
        out["reason"] = "내려다보는 각이 얕음"
    elif coverage < MIN_COVERAGE:
        out["reason"] = f"용도지역이 원의 {coverage:.0%}만 덮음 (기준 {MIN_COVERAGE:.0%})"
    elif radius_m > THRESHOLDS[target][-1][0]:
        out["reason"] = f"반경 {radius_m}m가 임계값 표({THRESHOLDS[target][-1][0]}m)보다 큼"
    elif p1 >= t:
        out["verdict"] = "확정"
        out["reason"] = f"1순위 {p1:.2f} >= 임계값 {t:.2f}"
    elif len(land_use) > 1 and p1 + p2 >= AMBIGUOUS_SUM:
        out["verdict"] = "모호"
        out["reason"] = f"1·2순위 합 {p1 + p2:.2f} >= {AMBIGUOUS_SUM}"
    else:
        out["reason"] = f"1순위 {p1:.2f} < 임계값 {t:.2f}, 1·2순위 합 {p1 + p2:.2f} < {AMBIGUOUS_SUM}"
    return out


def judge(land_use, radius_m, coverage=1.0, shallow=False, target=TARGET):
    """(판정, 임계값)을 돌려준다. 5.2 임시 judge()와 입출력이 같다.

    land_use  query()가 낸 [{"name", "ratio"}, ...] (비율 내림차순)
    radius_m  오차 반경 (transform의 position.error_radius_m)
    coverage  query() info.landuse_coverage
    shallow   transform() info.shallow
    임계값은 알람 verdict_rule.threshold에 들어간다. 표보다 큰 반경이면 표의 마지막 값을 적는다.
    """
    d = judge_detail(land_use, radius_m, coverage, shallow, target)
    t = d["threshold"] if d["threshold"] is not None else THRESHOLDS[target][-1][1]
    return d["verdict"], t


def _demo():
    A = [{"name": "보전녹지지역", "ratio": 0.85}, {"name": "자연녹지지역", "ratio": 0.15}]
    B = [{"name": "보전녹지지역", "ratio": 0.60}, {"name": "제1종일반주거지역", "ratio": 0.35},
         {"name": "자연녹지지역", "ratio": 0.05}]
    C = [{"name": "보전녹지지역", "ratio": 0.45}, {"name": "제1종일반주거지역", "ratio": 0.30},
         {"name": "자연녹지지역", "ratio": 0.25}]
    cases = [
        ("한 용도 85%, 반경 50m", A, 50, 1.0, False, "확정"),
        ("한 용도 85%, 반경 100m", A, 100, 1.0, False, "확정"),
        ("60%+35%, 반경 50m", B, 50, 1.0, False, "모호"),
        ("60%+35%, 반경 20m (임계값 0.285)", B, 20, 1.0, False, "확정"),
        ("45/30/25%, 반경 50m", C, 50, 1.0, False, "미확정"),
        ("한 용도 85%, 반경 165m", A, 165, 1.0, False, "미확정"),
        ("한 용도 85%, 얕은 각", A, 30, 1.0, True, "미확정"),
        ("한 용도 85%, 원의 40%만 자료", A, 30, 0.4, False, "미확정"),
        ("조회 결과 없음", [], 30, 0.0, False, "미확정"),
        ("한 용도 100%, 반경 5m", [{"name": "자연녹지지역", "ratio": 1.0}], 5, 1.0, False, "확정"),
    ]
    bad = 0
    for name, lu, r, cov, sh, want in cases:
        d = judge_detail(lu, r, cov, sh)
        ok = d["verdict"] == want
        bad += not ok
        print(f"  {'OK ' if ok else '틀림'} {name:<30} → {d['verdict']:<3} ({d['reason']})")
    print(f"\n{len(cases) - bad}/{len(cases)} 통과")
    return bad


if __name__ == "__main__":
    raise SystemExit(_demo())
