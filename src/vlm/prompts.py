# -*- coding: utf-8 -*-
"""입력 조건 네 가지의 프롬프트.

바뀌는 것은 모델에게 주는 정보의 양뿐이고, 지시와 예시는 네 조건이 모두 같다.
이 부분이 조건마다 다르면 결과 차이가 정보 때문인지 지시 때문인지 알 수 없다.

v2 에서 바꾼 것 (v1 파일럿 결과 반영)
- 정보를 지시보다 앞에 둔다. v1 은 정보가 맨 뒤라 모델이 거의 읽지 않았다.
- 넣을 항목을 번호로 지시한다. 주어지지 않은 항목은 건너뛰게 한다.
- 예시를 하나 보여 준다. 네 조건이 같은 예시를 쓴다.
"""

VERSION = "v2"

# 네 조건이 공유하는 부분. 정보 블록 뒤에 붙는다.
TAIL = """

너는 산불 관제 화면에 올릴 상황 설명을 쓴다.
위에 주어진 정보만 사용한다. 주어지지 않은 지명, 수치, 원인은 쓰지 않는다.

아래 네 가지를 순서대로 넣는다. 위에 주어지지 않은 항목은 건너뛴다.
1. 무엇이 보이는가
2. 어디인가. 용도 이름과 오차 반경 수치를 주어진 그대로 쓴다
3. 가까이에 무엇이 있는가
4. 확실하지 않은 부분

한국어 두 문장에서 세 문장으로 쓴다.

예시
{{"description": "보전녹지지역에서 연기가 관측된다. 오차 반경 55미터 안이라 제1종일반주거지역일 가능성도 있다. 주거지가 가깝다."}}

아래 형식의 JSON만 출력한다. 다른 키를 넣지 않는다.
{{"description": "..."}}"""

A = "사진" + TAIL

B = """사진

탐지 결과
- 종류: {cls}
- 탐지 신뢰도: {score}
- 화면에서 차지하는 비율: {area_ratio}
- 촬영 고도: {alt_m}미터""" + TAIL

C = """사진

탐지 결과
- 종류: {cls}
- 탐지 신뢰도: {score}
- 화면에서 차지하는 비율: {area_ratio}
- 촬영 고도: {alt_m}미터

추정 위치의 토지 용도
- 1순위: {top_name} ({top_ratio})""" + TAIL

D = """사진

탐지 결과
- 종류: {cls}
- 탐지 신뢰도: {score}
- 화면에서 차지하는 비율: {area_ratio}
- 촬영 고도: {alt_m}미터

추정 위치의 토지 용도
- 1순위: {top_name} ({top_ratio})
- 2순위: {second_name} ({second_ratio})

위치 판정
- 판정: {verdict}
- 좌표 오차 반경: {error_radius_m}미터

판정에 따라 이렇게 쓴다.
- 확정: 1순위 용도 이름을 그대로 쓴다.
- 모호: 1순위와 2순위를 같이 쓰고 둘 중 하나로 보인다고 쓴다.
- 미확정: 용도 이름을 쓰지 않고 오차 반경만 쓴다.""" + TAIL

PROMPTS = {"A": A, "B": B, "C": C, "D": D}

# 스키마 v0.2 는 클래스와 판정을 영문으로 둘 수 있다. 프롬프트에는 한국어로 넣는다.
CLASS_KO = {"smoke": "연기", "fire": "화염"}
VERDICT_KO = {"confirmed": "확정", "ambiguous": "모호", "undetermined": "미확정"}


def build(cond, detection=None, alert=None):
    """조건 이름과 JSON을 받아 프롬프트 문자열을 만든다."""
    if cond == "A":
        return A.format()

    d = (detection or {})["detections"][0]
    values = {
        "cls": CLASS_KO.get(d["class"], d["class"]),
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
        verdict=VERDICT_KO.get(alert["location_verdict"], alert["location_verdict"]),
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
        print(f"---- 조건 {name} ({VERSION})\n{build(name, det, alt)}\n")
