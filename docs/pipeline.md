# 전체 파이프라인 연결 (WBS 5.2)

탐지 결과부터 설명이 붙은 알람까지 한 번에 돌린다.

담당: 박서연 / 상태: 첫 연결 (판정은 임시 규칙)

## 흐름

```
탐지 결과 JSON (tools/predict_seg.py 또는 tools/replay.py)
 → 좌표 변환 src/geo/transform.py (3.3)
 → 지리 조회 src/geo/query.py (3.4)
 → 판정 src/pipeline/verdict.py (임시. 3.6이 끝나면 judge()만 교체)
 → 알람 발행 (설명 null)
 → VLM 사실 추출 src/pipeline/vlm_client.py (방식 iii, run_structured.py와 같은 설정)
 → 문장 조립 src/vlm/templates.py
 → 같은 alert_id로 설명을 붙여 다시 발행
```

## 쓰는 법

```bash
# 1) 탐지 결과 만들기
python tools/predict_seg.py 가중치.pt 이미지폴더 results/detections

# 2) 파이프라인
python src/pipeline/run.py --detections results/detections --frames 이미지폴더 \
    --geo-dir data/geo_moraegogae --vlm fake
```

| 옵션 | 내용 |
|---|---|
| `--geo-dir` | `landuse_5179.geojson`, `buildings_5179.geojson`, DEM `*.img`가 있는 폴더 |
| `--vlm none` | 설명 없이 알람만 |
| `--vlm fake` | 고정된 사실로 설명을 만든다. 연결 확인용, 결과로 쓰지 않는다 |
| `--vlm qwen` | Qwen2.5-VL-3B 4bit. GPU가 있으면 GPU |
| `--interval 1.5` | 프레임 사이를 실제처럼 띄운다 |
| `--no-merge` | 같은 사건 묶기를 끈다 |

출력은 `results/pipeline/`에 쌓인다.

| 파일 | 내용 |
|---|---|
| `alerts/<alert_id>.json` | 알람의 최신 상태. 대시보드(5.5)가 읽는다 |
| `alerts.jsonl` | 발행 기록. 설명 전과 후가 차례로 쌓인다 |
| `debug.jsonl` | 형식에 없는 중간 값(대표 픽셀, 가정값, 조회 지점 수, 규칙 위반) |

끝나면 `tools/check_schema.py`로 `alerts/*.json`을 자동 검사한다.

## 임시 판정 규칙

`docs/threshold.md`의 결과를 그대로 쓴다.

- 최다비율 규칙, 목표 정확도 95% 임계값: 10m 0.00 / 30m 0.57 / 50m 0.72 / 100m 0.80 (사이는 직선, 100m 초과는 0.80)
- 모호: 1·2순위 합 0.9 이상
- 미확정: 조회 결과 없음, 용도지역이 원의 60% 미만만 덮음, 내려다보는 각이 얕음(transform의 `shallow`)

## 같은 사건 묶기

새 탐지의 위치가 이미 낸 알람의 오차 원과 겹치면 같은 사건으로 보고 알람을 새로 내지 않는다.
알람 갱신과 해제(생명주기)는 향후 과제다.

## 확인한 것

가짜 지리 자료와 `tools/replay.py` 입력으로 돌려 보았다.

- 8프레임 → 알람 7개, 형식 검사 7/7 통과 (`--vlm fake`)
- 같은 탐지 3프레임 → 알람 1개, 묶음 2
- 판정 세 가지가 모두 나오는 것 확인 (임계값 보간, 커버리지, 얕은 각)

## 남은 것

- 실제 지리 자료(`data/geo_moraegogae`)와 탐지 가중치로 돌리기
- `--vlm qwen` 실제 실행 확인 (GPU 기기)
- VLM이 도는 동안 다음 프레임이 기다린다. 알람과 설명을 따로 돌리는 것은 5.4
- `evidence.buildings_in_radius`는 주거 건물이 아니라 전체 건물 수다 (건물 자료에 용도 정보 없음)
- 실제 건물 1.5만 채로 조회하면 건물 근접 계산이 느릴 수 있다. 시간이 길면 건물 목록을 미리 범위로 걸러 둔다
