# 입출력 형식

단계 사이에 주고받는 JSON 형식을 정의한다. 이 형식이 고정되어야 탐지, 지리 조회,
알람, 대시보드를 따로 만들 수 있다. 필드 이름은 나중에 바꾸기 어려우므로 초안을
검토한 뒤 확정한다.

담당: 박서연 / 상태: 초안

## 탐지 결과

프레임 한 장을 처리한 결과다. 예시는 `schema/detection.example.json`에 있다.

| 필드 | 설명 |
|---|---|
| `frame_id` | 프레임 번호 |
| `time` | 프레임 시각 |
| `image` | 프레임 이미지 경로 |
| `detections[].id` | 탐지 식별자 |
| `detections[].class` | 클래스 이름 |
| `detections[].score` | 탐지 신뢰도 |
| `detections[].bbox` | 픽셀 기준 x, y, 너비, 높이 |
| `detections[].polygon` | 픽셀 좌표 목록 |
| `detections[].area_px` | 영역 넓이(픽셀) |
| `detections[].area_ratio` | 화면에서 차지하는 비율 |
| `camera` | 촬영 위치(위도, 경도), 고도, 촬영 각도 |

## 알람

관제 화면에 표시하는 단위다. 예시는 `schema/alert.example.json`에 있다.

| 필드 | 설명 |
|---|---|
| `alert_id` | 알람 식별자 |
| `detection_id` | 근거가 된 탐지 식별자 |
| `position` | 추정 좌표와 오차 반경(m) |
| `location_verdict` | 확정, 모호, 미확정 |
| `land_use[]` | 오차 반경 안 용도와 비율 |
| `evidence` | 판단 근거 값(반경 내 건물 수, 경사 등) |
| `description` | 한국어 상황 설명. 생성 전에는 null |
| `latency_ms` | 탐지에서 알람까지, 알람에서 설명까지 걸린 시간 |

## 확정하면서 정할 것

- 위도와 경도를 어떤 이름으로 쓸지 통일한다.
- 폴리곤 좌표가 픽셀 기준인지 실제 좌표 기준인지 명시한다.
- 설명이 아직 없을 때 `description`을 null로 둘지 빈 문자열로 둘지 정한다.
