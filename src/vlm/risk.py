# -*- coding: utf-8 -*-
"""위험도 규칙 (초안).

VLM 에게 위험도를 묻지 않는다. VLM 이 본 것(굴뚝으로 보임, 불꽃)과 공간정보(위치 판정,
반경 안 위험 대상, 건물 수)를 규칙으로 합친다. 근거와 한계는 docs/vlm_situation.md 에 있다.
평가셋에 위험도 정답이 없어 정확도를 잰 것이 아니다. 등급과 순서는 팀 논의로 정한다.

    from risk import assess
    r = assess(alert, chimney=False, flame=True)
    r["level"], r["reason"]

    python src/vlm/risk.py          # 규칙표의 예를 돌려 확인

아직 파이프라인에 연결하지 않았다. 알람에 넣으려면 alert.py 와 docs/schema.md 를 고쳐야 한다.
"""

LEVELS = ("높음", "보통", "낮음", "확인 필요")
PEOPLE = ("주거지", "건물", "시설")          # 반경 안에 있으면 사람과 재산이 가깝다고 본다

# 정하지 않은 것. 논의 뒤에 여기만 고치면 된다
MIN_BUILDINGS = 1                 # "건물이 있다"의 기준
FLAME_OVER_UNCERTAIN = False      # True 면 불꽃이 보일 때 위치가 미확정이어도 높음으로 올린다


def _has_buildings(alert):
    ev = alert.get("evidence") or {}
    n = ev.get("buildings_in_radius")
    if n is not None:
        return n >= MIN_BUILDINGS
    return "건물" in (ev.get("risk_targets") or [])


def _near_people(alert):
    """반경 안에 건물이 있거나 주거지역이 걸려 있는지.

    평가셋 알람에는 risk_targets 가 있지만 파이프라인 알람에는 없다. 그래서 용도지역 이름에
    "주거"가 들어 있는지도 본다(제1종일반주거지역 등).
    """
    ev = alert.get("evidence") or {}
    if _has_buildings(alert) or any(t in PEOPLE for t in (ev.get("risk_targets") or [])):
        return True
    return any("주거" in (u.get("name") or "") and (u.get("ratio") or 0) > 0
               for u in (alert.get("land_use") or []))


def _uses(alert):
    return [u["name"] for u in (alert.get("land_use") or [])
            if u.get("name") and u["name"] not in ("없음", "미지정")][:2]


def assess(alert, chimney=False, flame=None):
    """알람과 VLM 이 본 것으로 위험도를 낸다. 위에서부터 먼저 맞는 줄을 쓴다.

    chimney  2단계 답이 "굴뚝이나 공장 설비의 연기"인지 (docs/vlm_situation.md)
    flame    불꽃이 보이는지. None 이면 알람의 observed.flame_visible 을 쓴다
    돌려주는 것: {"level", "reason", "rule"}  rule 은 맞은 줄 번호(1~6)
    """
    if flame is None:
        flame = bool((alert.get("observed") or {}).get("flame_visible"))
    verdict = alert.get("location_verdict")
    buildings = _has_buildings(alert)

    # 불꽃이 보이면 굴뚝으로 보여도 낮추지 않는다. 건물 화재일 수 있다
    if chimney and buildings and not flame:
        return {"level": "낮음", "rule": 1, "reason": "건물 설비의 연기로 보인다. 확인을 권한다"}
    if chimney and not flame:
        return {"level": "확인 필요", "rule": 2,
                "reason": "사진은 설비의 연기로 보이나 반경 안에 건물이 없다"}
    if flame and FLAME_OVER_UNCERTAIN:
        return {"level": "높음", "rule": 4, "reason": "불꽃이 보인다"}
    if verdict == "미확정":
        return {"level": "확인 필요", "rule": 3,
                "reason": "위치가 불확실하다. 접근해서 다시 판정한다"
                          + (". 불꽃이 보인다" if flame else "")}
    if flame:
        return {"level": "높음", "rule": 4, "reason": "불꽃이 보인다"}
    if _near_people(alert):
        return {"level": "높음", "rule": 5, "reason": "반경 안에 주거지나 건물이 있다"}
    uses = _uses(alert)
    return {"level": "보통", "rule": 6,
            "reason": ("반경 안은 " + ", ".join(uses) + "이다") if uses else "반경 안에 주거지나 건물이 없다"}


def _alert(verdict, targets=(), buildings=0, uses=("보전녹지지역",), flame=False):
    return {"location_verdict": verdict,
            "land_use": [{"name": u, "ratio": round(1 / len(uses), 2)} for u in uses],
            "evidence": {"risk_targets": list(targets), "buildings_in_radius": buildings},
            "observed": {"flame_visible": flame}}


def _cases():
    """규칙표 여섯 줄과 경계 경우."""
    forest = _alert("확정", ["산림"])
    homes = _alert("모호", ["산림", "주거지"], buildings=4, uses=("보전녹지지역", "제1종일반주거지역"))
    unsure = _alert("미확정", ["산림", "주거지"], buildings=2)
    cases = [
        ("굴뚝으로 보이고 건물이 있다", assess(homes, chimney=True), "낮음", 1),
        ("굴뚝으로 보이는데 건물이 없다", assess(forest, chimney=True), "확인 필요", 2),
        ("위치 미확정", assess(unsure), "확인 필요", 3),
        ("위치 미확정, 불꽃 보임", assess(unsure, flame=True), "확인 필요", 3),
        ("확정, 산림, 불꽃 보임", assess(forest, flame=True), "높음", 4),
        ("모호, 주거지 인접", assess(homes), "높음", 5),
        ("확정, 산림만", assess(forest), "보통", 6),
        ("알람의 observed 에서 불꽃 읽기", assess(_alert("확정", ["산림"], flame=True)), "높음", 4),
        ("굴뚝으로 보이고 건물이 있어도 불꽃이 보이면 높음", assess(homes, chimney=True, flame=True), "높음", 4),
    ]
    ok = True
    for name, got, level, rule in cases:
        good = got["level"] == level and got["rule"] == rule
        ok &= good
        print(("  " if good else "X ") + f"{name}: {got['level']} ({got['rule']}번) - {got['reason']}")
    return ok


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(0 if _cases() else 1)
