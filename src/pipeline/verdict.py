# -*- coding: utf-8 -*-
"""임시 3단계 판정 (5.2 파이프라인 첫 연결용).

정식 판정(WBS 3.6, 이수진, 10주차)이 들어오기 전까지 파이프라인을 끝까지 이어 보려고
둔 자리다. 3.6이 완성되면 `judge()`만 그 구현으로 바꾸면 되도록 입출력을 맞췄다.

규칙 (docs/threshold.md 결과를 그대로 씀)
    - 최다비율 규칙, 목표 정확도 95%의 반경별 임계값 표
      10m 0.00 / 30m 0.57 / 50m 0.72 / 100m 0.80
      표 사이 반경은 가까운 두 값 사이를 직선으로 잇고, 100m 넘으면 100m 값을 쓴다.
    - 1순위 비율 >= 임계값 이면 확정
    - 아니고 1·2순위 합 >= 0.9 이면 모호 (threshold.md '남은 것'의 현재 기준)
    - 그 밖, 조회 결과 없음, 용도지역이 원의 60% 미만만 덮음, 내려다보는 각이 얕음 → 미확정

목표 정확도(90/95/99%)와 모호 기준은 팀 결정 전이라 바뀔 수 있다.
"""

# (반경 m, 임계값) 목표 정확도 95%, 최다비율 규칙 (docs/threshold.md)
THRESHOLDS_95 = [(10, 0.00), (30, 0.57), (50, 0.72), (100, 0.80)]
AMBIGUOUS_SUM = 0.9      # 모호: 1·2순위 합 기준
MIN_COVERAGE = 0.6       # 용도지역이 원 안을 이만큼은 덮어야 판정한다
RULE = "최다비율"


def threshold_for(radius_m, table=THRESHOLDS_95):
    """반경에 맞는 임계값. 표 사이는 직선으로 잇는다."""
    if radius_m <= table[0][0]:
        return table[0][1]
    for (r0, t0), (r1, t1) in zip(table, table[1:]):
        if radius_m <= r1:
            return round(t0 + (t1 - t0) * (radius_m - r0) / (r1 - r0), 3)
    return table[-1][1]


def judge(land_use, radius_m, coverage=1.0, shallow=False):
    """(판정, 임계값)을 돌려준다.

    land_use  query()가 낸 [{"name", "ratio"}, ...] (비율 내림차순)
    radius_m  오차 반경
    coverage  query() info.landuse_coverage
    shallow   transform() info.shallow (내려다보는 각이 얕음)
    """
    t = threshold_for(radius_m)
    if shallow or not land_use or coverage < MIN_COVERAGE:
        return "미확정", t
    p1 = land_use[0]["ratio"]
    p2 = land_use[1]["ratio"] if len(land_use) > 1 else 0.0
    if p1 >= t:
        return "확정", t
    if len(land_use) > 1 and p1 + p2 >= AMBIGUOUS_SUM:
        return "모호", t
    return "미확정", t
