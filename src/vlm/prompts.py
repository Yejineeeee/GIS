# -*- coding: utf-8 -*-
"""입력 조건 네 가지의 프롬프트. 조건 사이에 지시문이 달라지지 않게 한 곳에 모아 둔다.

바뀌는 것은 모델에게 주는 정보의 양뿐이고, 역할 설명과 출력 형식은 네 조건이 모두
같다. 이 부분이 조건마다 다르면 결과 차이가 정보 때문인지 지시문 때문인지 알 수 없다.
"""

COMMON = """너는 산불 관제 화면에 올릴 상황 설명을 쓴다.
주어진 정보만 사용한다. 주어지지 않은 지명, 수치, 원인은 쓰지 않는다.
한국어 두 문장에서 세 문장으로 쓴다.
아래 형식의 JSON만 출력한다. 다른 키를 넣지 않는다.

{{"description": "..."}}"""

A = COMMON + """

사진을 보고 상황을 설명한다."""

B = COMMON + """

사진과 탐지 결과를 보고 상황을 설명한다.

탐지 결과
- 종류: {cls}
- 탐지 신뢰도: {score}
- 화면에서 차지하는 비율: {area_ratio}
- 촬영 고도: {alt_m}미터"""

C = B + """

추정 위치의 토지 용도
- 1순위: {top_name} ({top_ratio})"""

D = C + """

위치 판정
- 판정: {verdict}
- 좌표 오차 반경: {error_radius_m}미터
- 2순위 용도: {second_name} ({second_ratio})

판정에 따라 이렇게 쓴다.
- 확정: 용도 이름을 그대로 쓴다.
- 모호: 1순위와 2순위를 같이 쓰고 둘 중 하나로 보인다고 쓴다.
- 미확정: 용도 이름을 쓰지 않고 오차 반경만 쓴다.
어느 경우에도 오차 반경 수치를 넣는다."""

PROMPTS = {"A": A, "B": B, "C": C, "D": D}


def build(cond, detection=None, alert=None):
    """조건 이름과 JSON을 받아 프롬프트 문자열을 만든다."""
    if cond == "A":
        return A.format()

    d = (detection or {})["detections"][0]
    values = {
        "cls": d["class"],
        "score": d["score"],
        "area_ratio": d["area_ratio"],
        "alt_m": detection["camera"]["alt_m"],
    }
    if cond == "B":
        return B.format(**values)

    land = (alert or {}).get("land_use", [])
    first = land[0] if land else {"name": "없음", "ratio": 0}
    second = land[1] if len(land) > 1 else {"name": "없음", "ratio": 0}
    values.update(top_name=first["name"], top_ratio=first["ratio"])
    if cond == "C":
        return C.format(**values)

    values.update(
        verdict=alert["location_verdict"],
        error_radius_m=alert["position"]["error_radius_m"],
        second_name=second["name"],
        second_ratio=second["ratio"],
    )
    return D.format(**values)


if __name__ == "__main__":
    import json, os, sys
    root = os.path.join(os.path.dirname(__file__), "..", "..", "schema")
    det = json.load(open(os.path.join(root, "detection.example.json"), encoding="utf-8"))
    alt = json.load(open(os.path.join(root, "alert.example.json"), encoding="utf-8"))
    for name in sys.argv[1:] or ["A", "B", "C", "D"]:
        print(f"---- 조건 {name}\n{build(name, det, alt)}\n")
