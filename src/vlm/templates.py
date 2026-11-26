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


def _what(observed, use_vlm_facts=False):
    """무엇이 보이는가.

    알람은 탐지(YOLO)와 후보 검증기를 거쳐 나온 것이라 연기는 있다고 쓴다. VLM 이 사진
    전체를 보고 답한 연기 유무(62%), 색(21/70), 퍼짐(29%)은 늘 같은 답보다도 못해 문장에
    쓰지 않는다(docs/vlm_compare.md). 비교 실험을 재현할 때만 use_vlm_facts=True 로 쓴다.
    """
    if use_vlm_facts:
        if not observed.get("smoke_present"):
            return "연기가 관측되지 않았다"
        color = observed.get("smoke_color")
        spread = SPREAD.get(observed.get("spread"), "퍼져")
        head = f"{color} 연기가 {spread} 있다" if color and color != "없음" \
            else f"연기가 {spread} 있다"
    else:
        head = "연기가 탐지되었다"
    if observed.get("flame_visible"):      # VLM 이 불꽃이 보인다고 했을 때 (따로 묻는 질문, 열에 둘은 틀림)
        head += ". 불꽃으로 보이는 것이 있다"
    return head


def _named(use, rich):
    """용도 이름. rich 면 반경 안 비율을 붙인다: 자연녹지지역(58%)."""
    ratio = use.get("ratio")
    if rich and ratio is not None:
        return f"{use['name']}({round(ratio * 100)}%)"
    return use["name"]


def _where(alert, rich=False):
    """어디인가. 판정에 따라 쓸 것이 정해진다."""
    land = [u for u in (alert.get("land_use") or [])
            if u.get("name") and u["name"] not in ("없음", "미지정")]
    first = land[0]["name"] if land else None
    second = land[1]["name"] if len(land) > 1 else None
    pos = alert.get("position") or {}
    radius = pos.get("error_radius_m")
    verdict = alert.get("location_verdict")

    if verdict == "확정":
        base = f"{_named(land[0], rich)}에서 발생했다" if first else "위치가 확인되었다"
        return f"{base}. 좌표 오차 반경은 {radius}미터다" if radius else base
    if verdict == "모호":
        if first and second:
            body = (f"{_named(land[0], rich)}{_josa(first, '과', '와')} "
                    f"{_named(land[1], rich)} 중 하나로 보인다")
        elif first:
            body = f"{_named(land[0], rich)}{_josa(first, '으로', '로')} 보이나 확인이 필요하다"
        else:
            body = "용도를 특정하기 어렵다"
        return f"오차 반경 {radius}미터 안이라 {body}" if radius else body
    # 미확정. 용도 이름을 쓰지 않는다. 대신 좌표로 어디인지 알린다.
    text = (f"좌표 오차 반경이 {radius}미터여서 정확한 용도는 알 수 없다" if radius
            else "정확한 용도는 알 수 없다")
    if rich and pos.get("lat") is not None and pos.get("lon") is not None:
        text += f". 추정 위치는 위도 {pos['lat']:.4f}, 경도 {pos['lon']:.4f} 부근이다"
    return text


def _near(observed, alert, rich=False):
    """가까이에 무엇이 있는가.

    rich 면 VLM 이 답한 지형("주변이 산림이다")을 쓰지 않는다. 검증하지 않은 답이고, 용도지역과
    어긋나는 문장이 나온다. 대신 알람에 있는 값(건물 수, 가장 가까운 건물, 경사)을 쓴다.
    """
    ev = alert.get("evidence") or {}
    n = ev.get("buildings_in_radius")
    if not rich:
        parts = []
        near = TERRAIN_NEAR.get(observed.get("terrain"))
        if near:
            parts.append(near)
        if n:
            parts.append(f"반경 안 건물은 {n}개다")
        return ". ".join(parts) if parts else None

    slope = ev.get("terrain_slope_deg", ev.get("slope_deg"))
    tail = f"경사는 {round(slope)}도다" if slope is not None else None
    if n:
        text = f"반경 안 건물은 {n}개"
        d = ev.get("nearest_building_m")          # 지리 조회가 이 값을 내면 쓴다
        if d is not None:
            text += (", 추정 위치가 건물과 겹친다" if d < 1 else f", 가장 가까운 건물까지 약 {round(d)}미터")
        return f"{text}, {tail}" if tail else text + "다"
    if n == 0:
        return f"반경 안에 건물이 없고 {tail}" if tail else "반경 안에 건물이 없다"
    return tail


def _risk(observed, alert):
    """위험도와 그 근거, 권고. 규칙은 risk.py 에 있다(초안)."""
    import risk
    r = risk.assess(alert, flame=bool(observed.get("flame_visible")))
    text = f"위험도 {r['level']}: {r['reason']}"
    # 권고는 위치가 확실하지 않을 때만. 3번 줄의 근거에는 이미 들어 있다
    if alert.get("location_verdict") == "모호":
        text += ". 접근해 위치를 다시 확인할 것"
    return text


def build(observed, alert, use_vlm_facts=False, rich=True):
    """알람 설명 문장을 만든다.

    rich=True (기본)  용도 비율, 미확정일 때 좌표, 경사, 위험도와 근거, 권고를 담는다.
                      전부 알람에 있는 계산값이다. VLM 이 답한 지형은 쓰지 않는다
    rich=False        그 전의 짧은 문장. 전후 비교용(tools/element_coverage.py)
    use_vlm_facts=True 는 4.6 비교 실험(tools/run_structured.py) 재현용이다. 그때는 VLM 이
    연기가 없다고 하면 위치와 주변 설명을 붙이지 않고, 문장도 그때 그대로다.
    """
    observed = observed or {}
    if use_vlm_facts:
        rich = False
    if use_vlm_facts and observed and not observed.get("smoke_present"):
        return _what(observed, True) + "."
    what = _what(observed, use_vlm_facts)
    if rich and (alert.get("verifier") or {}).get("passed"):     # 알람에 이 칸이 생기면 쓴다
        what = what.replace("연기가 탐지되었다", "연기가 탐지되었다(영상 검증 통과)", 1)
    pieces = [what, _where(alert, rich)]
    near = _near(observed, alert, rich)
    if near:
        pieces.append(near)
    if rich:
        pieces.append(_risk(observed, alert))
    return ". ".join(p.rstrip(".") for p in pieces if p) + "."
