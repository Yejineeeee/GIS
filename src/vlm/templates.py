# -*- coding: utf-8 -*-
"""VLM이 낸 사실과 지리 판정으로 한국어 알람 문장을 조립한다.

생성 방식 (iii)에 쓴다. 문장을 모델이 쓰지 않고 규칙이 만들기 때문에, 판정에 따라
무엇을 쓰고 무엇을 쓰지 않을지가 코드로 정해진다. 지시를 어길 수가 없다.

입력은 두 가지다.
- observed  VLM 이 영상에서 본 사실 (tools/structured_test.py 의 스키마)
- alert     지리 조회와 판정 결과 (schema/alert.example.json)
"""

SPREAD = {"좁음": "좁게 퍼져", "보통": "퍼져", "넓음": "넓게 퍼져"}


def _josa(word, with_jong, without_jong):
    """받침 유무에 따라 조사를 고른다. 와/과, 은/는 같은 것."""
    if not word:
        return without_jong
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return without_jong
    return with_jong if (ord(last) - 0xAC00) % 28 else without_jong
TERRAIN_NEAR = {
    "산림": "주변이 산림이다",
    "농경지": "주변이 농경지다",
    "주거지": "주거지가 가깝다",
    "도로": "도로가 가깝다",
    "물": "주변에 물이 있다",
    "기타": None,
}


def _what(observed):
    """무엇이 보이는가. 연기가 없으면 그렇게 쓴다."""
    if not observed.get("smoke_present"):
        return "연기가 관측되지 않았다"
    color = observed.get("smoke_color")
    spread = SPREAD.get(observed.get("spread"), "퍼져")
    head = f"{color} 연기가 {spread} 있다" if color and color != "없음" \
        else f"연기가 {spread} 있다"
    if observed.get("flame_visible"):
        head += ". 불꽃도 함께 보인다"
    return head


def _where(alert):
    """어디인가. 판정에 따라 쓸 것이 정해진다."""
    land = [u for u in (alert.get("land_use") or [])
            if u.get("name") and u["name"] not in ("없음", "미지정")]
    first = land[0]["name"] if land else None
    second = land[1]["name"] if len(land) > 1 else None
    radius = (alert.get("position") or {}).get("error_radius_m")
    verdict = alert.get("location_verdict")

    if verdict == "확정":
        base = f"{first}에서 발생했다" if first else "위치가 확인되었다"
        return f"{base}. 좌표 오차 반경은 {radius}미터다" if radius else base
    if verdict == "모호":
        if first and second:
            body = f"{first}{_josa(first, '과', '와')} {second} 중 하나로 보인다"
        elif first:
            body = f"{first}{_josa(first, '으로', '로')} 보이나 확인이 필요하다"
        else:
            body = "용도를 특정하기 어렵다"
        return f"오차 반경 {radius}미터 안이라 {body}" if radius else body
    # 미확정. 용도 이름을 쓰지 않는다.
    if radius:
        return f"좌표 오차 반경이 {radius}미터여서 정확한 용도는 알 수 없다"
    return "정확한 용도는 알 수 없다"


def _near(observed, alert):
    """가까이에 무엇이 있는가."""
    parts = []
    near = TERRAIN_NEAR.get(observed.get("terrain"))
    if near:
        parts.append(near)
    n = (alert.get("evidence") or {}).get("buildings_in_radius")
    if n:
        parts.append(f"반경 안 건물은 {n}개다")
    return ". ".join(parts) if parts else None


def build(observed, alert):
    """알람 설명 문장을 만든다.

    연기가 없으면 위치와 주변 설명을 붙이지 않는다. 연기가 없는데 용도를 말하면
    이상한 문장이 된다. 애초에 그런 경우는 알람이 뜰 상황이 아니다.
    """
    observed = observed or {}
    if observed and not observed.get("smoke_present"):
        return _what(observed) + "."
    pieces = [_what(observed), _where(alert)]
    near = _near(observed, alert)
    if near:
        pieces.append(near)
    return ". ".join(p.rstrip(".") for p in pieces if p) + "."
