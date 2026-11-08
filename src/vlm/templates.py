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


# ---------- 관제 카드: 제목 / 판정 / 위험 / 행동 네 줄 ----------
# 숫자(반경, 경사, 비율)는 대시보드 카드 칸이 보여 주므로, 문장은 "무슨 상황이고 다음에
# 무엇을 할지"를 말한다. 판정 규칙(src/vlm/verify.py)은 네 줄을 이어 붙인 글로 검사한다.

def _kind(name):
    """용도지역 이름 → 녹지 / 시가지 / 기타."""
    if not name:
        return "기타"
    if any(k in name for k in ("녹지", "보전", "농림", "자연환경")):
        return "녹지"
    if any(k in name for k in ("주거", "상업", "공업")):
        return "시가지"
    return "기타"


def _share(land, kind):
    return sum(u.get("ratio") or 0 for u in land if _kind(u.get("name")) == kind)


def _m(v):
    """거리를 10m 단위로. 오차가 수십 m 라 그보다 잘게 쓰지 않는다."""
    return f"약 {max(10, int(round(v / 10.0)) * 10)}m"


def card(observed, alert, site=None):
    """알람 하나를 관제 카드 네 줄로 만든다. {"title", "verdict", "risk", "action"}.

    site 는 지역 이름(대시보드 --site-name). 없으면 제목에서 뺀다.
    evidence 에 nearest_building_m(가장 가까운 건물까지), residential_m(주거지역 경계까지)이
    있으면 위험 줄에 거리를 쓰고, 없으면 오차 반경 안의 용도 비율과 건물 수로 쓴다.
    """
    observed = observed or {}
    land = [u for u in (alert.get("land_use") or [])
            if u.get("name") and u["name"] not in ("없음", "미지정")]
    first = land[0]["name"] if land else None
    second = land[1]["name"] if len(land) > 1 else None
    k1 = _kind(first)
    verdict = alert.get("location_verdict")
    radius = (alert.get("position") or {}).get("error_radius_m")
    ev = alert.get("evidence") or {}
    n_bd = ev.get("buildings_in_radius")
    urban = _share(land, "시가지")
    where = f" · {site}" if site else ""

    # 제목
    if verdict == "확정" and k1 == "녹지":
        title = f"산불{where}"
    elif verdict == "확정" and k1 == "시가지":
        title = f"화재 의심{where}"
    else:
        title = f"연기 탐지{where}"
    if observed.get("flame_visible"):
        title += " · 불꽃 보임"

    # 판정
    if verdict == "확정":
        vline = f"{first}에서 발생" if first else "위치 확인됨"
    elif verdict == "모호":
        if first and second:
            vline = f"{first}{_josa(first, '과', '와')} {second} 중 하나로 보임"
        elif first:
            vline = f"{first}{_josa(first, '으로', '로')} 보이나 확인이 필요"
        else:
            vline = "용도를 특정하기 어려움"
    else:
        vline = "위치 불확실, 용도 판단 보류"

    # 위험
    risk = [f"오차 반경 {radius}m"] if radius else []
    near_bd, near_res = ev.get("nearest_building_m"), ev.get("residential_m")
    if verdict == "미확정":
        # 추정점을 믿기 어려우므로 추정점에서 잰 거리는 쓰지 않는다
        near_bd = near_res = None
        if n_bd:
            risk.append(f"오차 범위 안 건물 {n_bd}동")
    if near_res is not None and k1 != "시가지":
        risk.append("주거지역과 맞닿음" if near_res == 0 else f"주거지역 {_m(near_res)}")
    if near_bd is not None:
        risk.append("건물과 겹침" if near_bd == 0 else f"가장 가까운 건물 {_m(near_bd)}")
    if near_bd is None and verdict != "미확정":
        if k1 != "시가지" and urban >= 0.05:
            risk.append(f"반경 안 {int(round(urban * 100))}%가 시가지")
        elif n_bd:
            risk.append(f"반경 안 건물 {n_bd}동")
    rline = " · ".join(risk) if risk else "주변 정보 없음"

    # 다음 행동
    homes_near = (near_res is not None and near_res <= 100) or urban >= 0.05 or bool(n_bd)
    if verdict == "미확정":
        aline = "접근해 위치 확인 필요"
    elif verdict == "모호":
        aline = "위치를 좁히면 판정 가능"
        if urban >= 0.05 or _kind(second) == "시가지":
            aline += " · 주거지 확산 대비"
    elif k1 == "시가지":
        aline = "건물 화재 가능 · 119 통보 검토"
    elif k1 == "녹지" and homes_near:
        aline = "확산 시 민가 위험 · 출동 검토"
    else:
        aline = "출동 검토"

    return {"title": title, "verdict": vline, "risk": rline, "action": aline}


def card_text(c):
    """카드 네 줄을 한 글로 (판정 규칙 검사와 로그용)."""
    return "\n".join(c[k] for k in ("title", "verdict", "risk", "action"))


def build(observed, alert, use_vlm_facts=False):
    """알람 설명 문장을 만든다.

    use_vlm_facts=True 는 4.6 비교 실험(tools/run_structured.py) 재현용이다. 그때는 VLM 이
    연기가 없다고 하면 위치와 주변 설명을 붙이지 않는다.
    """
    observed = observed or {}
    if use_vlm_facts and observed and not observed.get("smoke_present"):
        return _what(observed, True) + "."
    pieces = [_what(observed, use_vlm_facts), _where(alert)]
    near = _near(observed, alert)
    if near:
        pieces.append(near)
    return ". ".join(p.rstrip(".") for p in pieces if p) + "."
