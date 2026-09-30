# 입출력 형식

단계 사이에 주고받는 JSON 형식을 정의한다. 이 형식이 고정되어야 탐지, 지리 조회,
알람, 대시보드를 따로 만들 수 있다. 필드 이름은 나중에 바꾸기 어려우므로 초안을
검토한 뒤 확정한다.

담당: 박서연 / 상태: 검토 요청 (v0.2)

## 공통 규칙

- 위도, 경도는 모든 곳에서 `lat`, `lon`으로 쓴다. 좌표계는 WGS84(EPSG:4326)다.
- 픽셀 좌표는 이미지 좌상단이 원점이다. `bbox`, `polygon`은 항상 픽셀 기준이다.
- 실제 좌표는 알람의 `position`에만 둔다.
- 단위는 거리 m, 각도 도(deg), 시간 ms다. 단위가 붙는 필드는 이름 끝에 `_m`, `_deg`를 붙인다.
- 시각은 ISO 8601, 한국 시간(+09:00)으로 쓴다.
- 모든 메시지에 `schema_version`을 둔다.

## 탐지 결과

프레임 한 장을 처리한 결과다. 예시는 `schema/detection.example.json`에 있다.

| 필드 | 설명 |
|---|---|
| `video_id` | 영상 번호. 학습/평가 분할 단위 |
| `frame_id` | 프레임 번호 |
| `time` | 프레임 시각 |
| `image` | 프레임 이미지 경로 |
| `image_size` | 원본 너비, 높이(px). 위치 변환에 필요 |
| `detections[].id` | 탐지 식별자. `{video_id}-{frame_id}-{순번}` |
| `detections[].class` | 클래스 이름. 현재는 `smoke`만 |
| `detections[].score` | 탐지 신뢰도(0~1) |
| `detections[].bbox` | 픽셀 기준 x, y, 너비, 높이 |
| `detections[].polygon` | 픽셀 좌표 목록. 없으면 null |
| `detections[].area_px` | 영역 넓이(픽셀) |
| `detections[].area_ratio` | 화면에서 차지하는 비율 |
| `camera.lat`, `camera.lon` | 드론 위치 |
| `camera.alt_m`, `camera.alt_ref` | 고도와 기준. `AGL`(지면 기준) 또는 `MSL`(해발) |
| `camera.heading_deg` | 기수 방향. 진북 0도, 시계 방향 |
| `camera.pitch_deg` | 촬영 각도. -90이 수직 하향, 0이 수평 |
| `camera.hfov_deg` | 수평 화각 |
| `camera.source` | `exif`(실제 값) 또는 `virtual`(가상 배치) |
| `latency_ms.detect` | 탐지에 걸린 시간 |

## 알람

관제 화면에 표시하는 단위다. 예시는 `schema/alert.example.json`에 있다.
탐지 직후 `description`이 null인 알람을 먼저 보내고, 설명이 생성되면 같은
`alert_id`로 다시 보낸다.

| 필드 | 설명 |
|---|---|
| `alert_id` | 알람 식별자 |
| `time` | 알람 생성 시각 |
| `detection_id` | 근거가 된 탐지 식별자 |
| `position.lat`, `position.lon` | 추정 좌표 |
| `position.error_radius_m` | 오차 반경(m) |
| `position.virtual` | 가상 배치 여부. 결과표 표시용 |
| `location_verdict` | `confirmed`(확정), `ambiguous`(모호), `undetermined`(미확정) |
| `land_use[]` | 오차 반경 안 용도(`name`, V-World 원문)와 비율(`ratio`). 비율 내림차순, 합계 1 |
| `verdict_rule` | 적용한 판정 규칙, 임계값, 조회 지점 수 |
| `evidence.buildings_in_radius` | 반경 내 주거 건물 수 |
| `evidence.slope_deg` | 반경 내 평균 경사 |
| `description` | 한국어 상황 설명. 생성 전에는 null |
| `latency_ms` | 단계별 소요 시간(아래 표). 아직 안 끝난 단계는 null |

### latency_ms

| 필드 | 설명 |
|---|---|
| `detect` | 탐지 |
| `geo_transform` | 픽셀 → 실제 좌표 변환 |
| `geo_query` | V-World, DEM 조회 |
| `verdict` | 3단계 판정 |
| `vlm` | VLM 응답 |
| `detect_to_alert` | 탐지 → 알람 출력. 목표: 즉시 |
| `alert_to_description` | 알람 → 설명 출력. `detect_to_alert`와 더한 값이 10초 이내 |

## 확정한 것

- 위도와 경도는 `lat`, `lon`으로 통일한다.
- 폴리곤은 픽셀 기준이다. 실제 좌표는 알람 `position`에만 둔다.
- 설명이 아직 없으면 `description`은 null로 둔다. 빈 문자열은 쓰지 않는다.

## 남은 것

- `camera.alt_ref`: AI Hub 71330의 elevation이 지면 기준인지 해발 기준인지는
  `tools/count_labels.py` 결과로 확인한다. 확인 전까지는 두 값을 모두 허용한다.
- 미확정일 때 `evidence`를 null로 둘지 정한다. 미확정이면 주변 시설 정보를 제공하지 않는다.
