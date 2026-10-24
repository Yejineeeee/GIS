# -*- coding: utf-8 -*-
"""생성한 설명이 판정에 맞는 표현을 썼는지 검사한다.

생성 방식 (ii) 규칙 검증 후 재생성에 쓰고, 평가의 판정 준수율에도 같은 규칙을
쓴다. 한쪽만 고치면 검사와 채점이 어긋나므로 여기 한 곳에만 둔다.
기준은 docs/eval_rubric.md 의 판정 준수율과 같다.
"""
import re

HEDGE = ("추정", "가능성", "보인다", "보입니다", "확인이 필요", "중 하나",
         "일 수 있", "것으로 판단", "불확실", "알 수 없")


def _has_radius(text, radius):
    """숫자 앞에 다른 숫자가 붙은 경우(고도 120미터)를 거른다."""
    return bool(re.search(rf"(?<![0-9]){radius}\s*(미터|m(?![a-zA-Z]))", text))


def check(description, alert, observed=None):
    """지키지 않은 규칙을 목록으로 돌려준다. 빈 목록이면 통과다.

    observed 를 주고 연기가 없다고 되어 있으면 판정 규칙을 적용하지 않는다.
    연기가 없으면 알람이 뜰 상황이 아니므로 위치를 설명할 이유가 없다.
    """
    if not description:
        return ["설명이 비어 있다"]
    if observed is not None and not observed.get("smoke_present"):
        return []

    land = alert.get("land_use") or []
    first = land[0]["name"] if land else None
    second = land[1]["name"] if len(land) > 1 else None
    verdict = alert.get("location_verdict")
    radius = (alert.get("position") or {}).get("error_radius_m")

    names = [u["name"] for u in land if u.get("name") and u["name"] != "없음"]
    mentioned = [n for n in names if n in description]
    hedged = any(h in description for h in HEDGE)

    bad = []
    if verdict == "확정":
        if first and first not in description:
            bad.append(f"확정인데 1순위 용도({first})를 쓰지 않았다")
    elif verdict == "모호":
        if first and first not in description:
            bad.append(f"모호인데 1순위 용도({first})를 쓰지 않았다")
        if second and second != "없음" and second not in description:
            bad.append(f"모호인데 2순위 용도({second})를 쓰지 않았다")
        if not hedged:
            bad.append("모호인데 유보 표현이 없다")
    elif verdict == "미확정":
        if mentioned:
            bad.append(f"미확정인데 용도 이름({', '.join(mentioned)})을 썼다")
        if radius and not _has_radius(description, radius):
            bad.append(f"미확정인데 오차 반경({radius}미터)을 쓰지 않았다")

    return bad


def feedback(violations):
    """재생성할 때 모델에게 덧붙일 문장을 만든다."""
    lines = "\n".join(f"- {v}" for v in violations)
    return ("앞서 쓴 설명이 아래를 지키지 않았다. 고쳐서 다시 쓴다.\n"
            f"{lines}")
