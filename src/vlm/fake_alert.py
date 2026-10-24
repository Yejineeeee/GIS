# -*- coding: utf-8 -*-
"""실험용 가짜 알람을 만든다.

지리 조회(3.4)가 붙기 전까지 VLM 실험을 돌리려면 알람 값이 필요하다. 판정 세
가지가 고르게 나오도록 돌려 가며 만든다.

용도 이름은 prompts.py 의 예시에 쓰인 이름과 겹치지 않게 고른다. 겹치면 출력에
그 이름이 나왔을 때 입력을 쓴 것인지 예시를 베낀 것인지 구분할 수 없다. 조건 A 와
B 는 용도 정보를 받지 않으므로 예시를 베꼈는지 바로 알 수 있지만, C 와 D 는
이름이 겹치면 알 수 없다.

    예시에 쓰는 이름   보전녹지지역, 제1종일반주거지역
    알람에 쓰는 이름   자연녹지지역, 제2종일반주거지역, 농림지역, 계획관리지역,
                      보전관리지역, 제3종일반주거지역

모두 시연 지역(모래고개입구)에 실제로 있는 용도지역이다. docs/landuse.md 참고.

세 파일이 같은 알람을 써야 비교가 되므로 여기 한 곳에만 둔다.
    tools/run_pilot.py        방식 (i)
    tools/run_verify_retry.py 방식 (ii)
    tools/run_structured.py   방식 (iii)
"""

# (판정, 오차 반경, 용도 목록)
PLAN = [
    ("확정", 20, [{"name": "자연녹지지역", "ratio": 0.85},
                  {"name": "제2종일반주거지역", "ratio": 0.15}]),
    ("모호", 60, [{"name": "농림지역", "ratio": 0.52},
                  {"name": "계획관리지역", "ratio": 0.41}]),
    ("미확정", 110, [{"name": "보전관리지역", "ratio": 0.38},
                     {"name": "제3종일반주거지역", "ratio": 0.33}]),
]

# 좌표는 시연 지역 안의 한 지점으로 고정한다. 가짜 값이므로 알람마다 바꿀 이유가 없다
LAT, LON = 37.74652, 128.88544


def fake_alert(i, buildings=12):
    """i 번째 알람을 만든다. 판정 세 가지가 고르게 돌아가게 한다."""
    verdict, radius, land = PLAN[i % len(PLAN)]
    return {
        "alert_id": f"p-{i:04d}",
        "location_verdict": verdict,
        "position": {"lat": LAT, "lon": LON, "error_radius_m": radius},
        "land_use": land,
        "evidence": {"buildings_in_radius": buildings},
        "description": None,
    }
