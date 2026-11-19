# 입출력 형식

단계 사이에 주고받는 JSON 형식을 정의한다. 이 형식이 고정되어야 탐지, 지리 조회,
알람, 대시보드를 따로 만들 수 있다. 필드 이름은 나중에 바꾸기 어려우므로 초안을
검토한 뒤 확정한다.

담당: 박서연 / 상태: 탐지 v0.3, 알람 v0.4 (11/19 회의 반영)

## 공통 규칙

- 위도, 경도는 모든 곳에서 `lat`, `lon`으로 쓴다. 좌표계는 WGS84(EPSG:4326)다.
- 픽셀 좌표는 이미지 좌상단이 원점이다. `bbox`, `polygon`은 항상 픽셀 기준이다.
- 실제 좌표는 알람의 `position`에만 둔다.
- 단위는 거리 m, 각도 도(deg), 시간 ms다. 단위가 붙는 필드는 이름 끝에 `_m`, `_deg`를 붙인다.
- 시각은 ISO 8601, 한국 시간(+09:00)으로 쓴다.
- 모든 메시지에 `schema_version`을 둔다.
- 키는 영문, 값은 한글로 쓴다. 정해진 값 목록이 있는 필드는 목록 밖의 값을 쓰지 않는다.
- 시각은 항상 밀리초까지 쓴다 (`14:03:21.000+09:00`).

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
| `detections[].class` | 클래스 이름. 현재는 `연기`만 |
| `detections[].score` | 탐지 신뢰도(0~1) |
| `detections[].bbox` | 픽셀 기준 x, y, 너비, 높이 |
| `detections[].polygon` | 픽셀 좌표 목록. 없으면 null |
| `detections[].area_px` | 영역 넓이(픽셀) |
| `detections[].area_ratio` | 화면에서 차지하는 비율 |
| `camera.lat`, `camera.lon` | 드론 위치 |
| `camera.alt_m`, `camera.alt_ref` | 고도와 기준. `지면` 또는 `해발` |
| `camera.heading_deg` | 기수 방향. 진북 0도, 시계 방향 |
| `camera.pitch_deg` | 촬영 각도. -90이 수직 하향, 0이 수평 |
| `camera.hfov_deg` | 수평 화각 |
| `camera.source` | `실제`(촬영 메타 값) 또는 `가상`(가상 배치) |
| `latency_ms.detect` | 탐지에 걸린 시간 |

## 알람

관제 화면에 표시하는 단위다. 예시는 `schema/alert.example.json`에 있다. `schema_version`은 `"0.4"`다.
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
| `location_verdict` | `확정`, `모호`, `미확정` |
| `land_use[]` | 오차 반경 안 용도(`name`, V-World 원문)와 비율(`ratio`). 비율 내림차순, 합계 1. 아래 규칙 참고 |
| `verdict_rule` | 적용한 판정 규칙(`최다비율` 또는 `격차`), 임계값, 조회 지점 수 |
| `evidence.buildings_in_radius` | 반경 내 건물 수 (건물 자료에 용도가 없어 주거만 따로 세지 못함) |
| `evidence.nearest_building_m` | 추정 좌표(원 중심)에서 가장 가까운 건물 외곽까지 거리(m). 건물 안이면 0. 300m(반경이 더 크면 반경) 안에 없으면 null. v0.4 |
| `evidence.terrain_slope_deg` | 추정 좌표 주변 270m 범위의 경사. 아래 규칙 참고 |
| `evidence.dem_resolution_m` | 사용한 DEM 격자 크기(m) |
| `risk.level` | 위험도. `높음`, `보통`, `낮음`, `확인 필요`. 아래 규칙 참고. v0.4 |
| `risk.reason` | 위험도 근거 한 줄 |
| `verifier.p_smoke` | 검증기(4.7)가 탐지 조각에 "연기가 있는가"를 물었을 때 P(예). 0~1. v0.4 |
| `verifier.passed` | `p_smoke`가 0.7 이상인가. 대시보드의 "영상 검증 통과 / 헛탐지 의심"이 이 값을 쓴다 |
| `observed` | VLM이 영상에서 본 사실. 아래 표. 생성 전에는 null |
| `description` | 한국어 상황 설명. 생성 전에는 null |
| `description_method` | 설명을 만든 방식. `프롬프트`, `검증재생성`, `구조화템플릿`. 생성 전에는 null |
| `latency_ms` | 단계별 소요 시간(아래 표). 아직 안 끝난 단계는 null |

### land_use 규칙

- 항상 배열로 둔다. 용도가 1개뿐이어도 `[{"name": "보전산지", "ratio": 1.0}]`처럼 배열에 담는다.
- 용도가 1개면 비율이 1.0이라 판정은 `확정`이다. `모호`는 용도가 2개 이상일 때만 나온다.
- 조회 실패나 데이터 없음이면 빈 배열 `[]`로 두고 판정은 `미확정`이다.

### terrain_slope_deg 규칙

공개 DEM은 격자가 90m라서 오차 반경(약 5~110m) 안의 평균 경사를 구할 수 없다.
반경이 격자 한 칸보다 작은 경우가 많기 때문이다. 그래서 추정 좌표가 속한 격자를
중심으로 3×3 격자(270m × 270m) 범위의 경사를 쓰고, 격자 크기를
`dem_resolution_m`에 함께 남긴다. 해설과 대시보드에서는 "반경 내 경사"가 아니라
"주변 경사"로 표시한다. 값을 못 구하면 null이다.

### observed

VLM이 정해진 질문에 답한 사실 항목이다. 키 5개는 항상 모두 채운다.

| 필드 | 값 |
|---|---|
| `smoke_present` | true / false |
| `smoke_color` | `검은색`, `회색`, `흰색`, `없음` |
| `spread` | `좁음`, `보통`, `넓음`, `없음` |
| `flame_visible` | true / false |
| `terrain` | `산림`, `농경지`, `주거지`, `도로`, `물`, `기타` |

`smoke_present`가 false면 `smoke_color`와 `spread`는 `없음`이다.

시스템 기본 방식은 `구조화템플릿`이다 (세 방식 비교에서 판정 준수 15/15, 평균 5.9초).
`구조화템플릿`이면 `observed`는 반드시 채운다. `프롬프트`, `검증재생성`은 비교 실험
기록용이며 `observed`가 null일 수 있다.

### risk, verifier (v0.4)

- `risk`는 알람을 낼 때마다 `src/vlm/risk.py`의 `assess`로 다시 계산한다. 판정, 반경 안 건물, 검증기, 불꽃(설명이
  붙은 뒤)에 따라 바뀐다. 위에서부터 먼저 맞는 줄을 쓴다
  0. 검증기 미통과(`verifier.passed` false) → 확인 필요 (헛탐지 의심)
  1~2. 굴뚝·설비 연기로 보임 → 낮음 / 확인 필요 (파이프라인은 아직 굴뚝을 묻지 않음)
  3. 위치 미확정 → 확인 필요
  4. 불꽃이 보임 → 높음
  5. 반경 안에 주거지나 건물 → 높음
  6. 그 밖 → 보통
- `verifier`는 검증 전이거나 검증기를 끄면(`--no-verify`, `--vlm none`) null이다. 알람이 갱신되면(같은 사건,
  더 좁은 반경) 새 탐지를 다시 검증하므로 그동안 null로 돌아간다
- 설명용 VLM의 `observed.smoke_present`는 사진 일부만 보고 답해 연기가 아닌 것을 거르는 힘이 약하다
  (`docs/inputs_cpu.md`). 헛탐지 판단은 `verifier`로 한다

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
- 키는 영문, 값은 한글로 통일한다. (v0.3)
- VLM은 사실 항목(`observed`)만 내고, 문장은 코드가 템플릿으로 조립한다. 기본 방식은 `구조화템플릿`이다. (v0.3)
- 알람에 `risk`, `verifier`, `evidence.nearest_building_m`를 넣는다. 탐지 결과 형식은 그대로 v0.3이다. (v0.4, 11/19 회의)

## 남은 것

- `camera.alt_ref`: AI Hub 71330의 elevation이 지면 기준인지 해발 기준인지는
  `tools/count_labels.py` 결과로 확인한다. 확인 전까지는 두 값을 모두 허용한다.
- 미확정일 때 `evidence`를 null로 둘지 정한다. 미확정이면 주변 시설 정보를 제공하지 않는다.

## 검사

`tools/check_schema.py`로 JSON 파일이 이 형식을 지키는지 검사한다.

```bash
python tools/check_schema.py schema/*.example.json
python tools/check_schema.py results/pred/*.json
```
