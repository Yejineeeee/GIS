# 보고서 그림

`python tools/report_figs.py` 로 다시 만든다(`val_operational/` 은 아래 명령).

| 파일 | 내용 | 근거 |
|---|---|---|
| vlm_pick.png | ① 후보 밑동에 글자를 붙여 VLM 이 고르게 한 실험. 예시 두 장과 평균 오차(지금 69.7m, VLM 77.8m, 상한 51.5m) | docs/vlm_base_pick.md ① |
| vlm_verify.png | ③ 후보 검증기. 연기를 남기고 헛탐지를 거른 예, 놓친 가는 연기와 못 거른 굴뚝 연기 | docs/vlm_base_pick.md ③ |
| detect_confusion.png | 운영 문턱 conf 0.10 사진 단위 혼동행렬. val_252 184/190, 헛경보 7/62. eval_100 61/70, 5/30 | docs/detect_eval.md |
| detect_examples.png | eval_100 맞힘, 놓침, 헛경보 예시. 빨강 탐지, 노란 점선 정답 | docs/detect_eval.md |
| train_curves.png | 운영 모델 학습 곡선(분할·박스 손실, 검증 mAP50). 최고 88에폭 분할 mAP50 0.559 | docs/train_results.md |
| data_size_curve.png | 학습 사진 252/504/756/1,008장의 mAP50, 재현율, 헛경보. 756장부터 거의 평평 | docs/train_results.md |
| retrain_curves.png | 큰 연기 재학습 4가지의 손실과 mAP50 곡선 (채택 안 함) | docs/train_results.md |
| scenario.png | 신고 시나리오 세 장면, 드론 화면과 대시보드 | docs/simulator.md |
| dashboard.png | 대시보드 마지막 화면 (확정, 반경 29m) | |
| chart_risk_coverage.png, chart_rule_compare.png | 반경 판정 효과 차트 | docs/verdict_compare.md |
| chimney_compare.png | 굴뚝 연기 비교 | docs/chimney_demo.md |
| demo_area_candidates.png | 시연 지역 후보 | docs/demo_area.md |
| val_operational/ | 운영 모델 val_252 검증 그림(PR, F1 곡선, 예측 배치). 혼동행렬은 conf 0.001 기준이라 헛탐지가 크게 나오므로 detect_confusion.png 를 쓴다 | |

운영 모델의 results.csv 는 PC 초기화로 잃었지만, best.pt 안의 `train_results` 에 100에폭 기록이 그대로 있어 그것으로 그린다.

```
python -c "from ultralytics import YOLO; YOLO('runs/smoke_1260_960/weights/best.pt').val(data='data/yolo_1260/dataset.yaml', imgsz=960, plots=True, project='results/report_figs', name='val_operational', exist_ok=True)"
```

ultralytics 가 `runs/segment/results/report_figs/val_operational` 에 저장하면 이 폴더로 옮긴다.

VLM 은 학습하지 않았다(Qwen2.5-VL-3B 를 4bit 로 줄여 그대로 쓴다). 그래서 VLM 학습 곡선은 없다.
