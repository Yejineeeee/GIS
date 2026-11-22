# 프레임워크와 칩 호환성 (WBS 7.6)

요청서 유의사항 "AI 프레임워크(PyTorch/TensorFlow 등)와 칩 간 호환성 분석"에 답한다. 우리 두 모델
(탐지 YOLO11n-seg, 설명과 검증 Qwen2.5-VL-3B)을 어떤 실행 방식으로 어떤 칩에서 돌릴 수 있는지,
직접 잰 것과 문서로만 확인한 것을 나눠 적는다.

담당: 이예진 / 상태: 정리 완료(11/18). 탐지 모델과 VLM 검증기 모두 인텔 내장 GPU 와 NPU 까지 실측함. Jetson 은 문서로만

칸마다 근거를 셋으로 나눈다.

| 표시 | 뜻 |
|---|---|
| **잼** | 우리가 직접 돌려 숫자를 냈다 |
| 문서 | 공식 문서나 배포처에서 지원한다고 적은 것을 확인했다. 돌려 보지는 않았다 |
| 모름 | 문서에서도 확인하지 못했다 |

## 1. 대상 기기

현장 기기 기준은 두 단으로 둔다(11/17 결정, `docs/device_spec.md` 는 합의 후 고침).

| 단 | 기기 예 | CPU | 가속기 | 메모리 |
|---|---|---|---|---|
| 하한 (가속기 없음) | 노트북 CPU 4코어 | x86 4코어 | 없음 | 8GB |
| 대표 기기 | NVIDIA Jetson Orin Nano 8GB | ARM 6코어 | GPU (CUDA 1024코어), 최대 67 TOPS | 8GB (CPU 와 GPU 가 같이 씀) |
| 참고 | DJI Manifold 3 (드론 탑재 컴퓨터) | - | 100 TOPS (판매처 정보) | 16GB (판매처 정보) |
| 참고 | 인텔 Core Ultra 노트북 (개발 노트북) | x86 16코어 | 내장 GPU (Arc), NPU (AI Boost) | 32GB |
| 개발 (참고값) | RTX 3050 노트북 GPU | - | GPU 4GB | - |

드론 산불 감시 연구에서는 Jetson Nano, Orin Nano 급에 YOLO 를 올리는 구성이 흔하다. 회사가 실제로 쓰는
기기는 확인하지 못했다.

## 2. 직접 잰 것

### VLM (Qwen2.5-VL-3B)

| 실행 방식 | 칩 | 양자화 | 시간 | 메모리 | 출처 |
|---|---|---|---|---|---|
| PyTorch + transformers + bitsandbytes | RTX 3050 (4GB) | 4bit nf4 이중 양자화, bf16 계산 | 설명 1장 6.5초 | VRAM 최대 2.51GB | `docs/quantization.md` |
| 같음 | 같음 | fp16 (양자화 없음) | 219.8초 | 7.32GB (4GB 를 넘어 시스템 메모리로 넘김) | `docs/quantization.md` |
| 같음 | 같음 | 8bit | 924.9초 | 4.07GB | `docs/quantization.md` |
| 같음 | 노트북 CPU 4코어 (WSL2, 8GB 제한) | 4bit nf4 | 설명 1장 180.7초 | 최대 7.09GB | `docs/vlm_speed.md` |
| llama.cpp (GGUF) | 노트북 CPU 스레드 4개 (윈도우) | Q4_K_M | 사진 전체 41.2초, 검증 1회 11.9초, 조각 설명 9초쯤 | 최대 5.4GB | `docs/model_size.md`, `docs/inputs_cpu.md` |
| 같음 (7B) | 같음 | Q4_K_M | 사진 전체 60.3초, 검증 1회 19.0초 | 최대 9.7GB | `docs/model_size.md` |

### 탐지 (YOLO11n-seg)

| 실행 방식 | 칩 | 입력 | 시간 | 출처 |
|---|---|---|---|---|
| PyTorch (Ultralytics) | 노트북 CPU 4코어 | 640 | 82ms | `docs/seg_candidates.md` |
| 같음 | 같음 | 960 (운영) | 133ms | `docs/seg_candidates.md`, `docs/device_spec.md` |
| 같음 | RTX 3050 | 640 | 18ms | `docs/seg_candidates.md` |
| PyTorch (SegFormer-B0, 비교 후보) | 노트북 CPU 4코어 | 640 | 340ms | `docs/seg_candidates.md` |

### 탐지 모델을 칩 네 가지에서 (11/18)

`tools/compare_runtimes.py`. 같은 가중치를 ONNX, OpenVINO, TorchScript 로 내보내(`yolo export ... imgsz=960`)
평가셋 100장을 같은 크기(960x960)로 넣었다. 모델 계산 시간만 쟀고(앞 5장 빼고 95장의 중앙값),
CPU 는 스레드 4개다. 개발 노트북(Core Ultra 7 155H), 전원 연결.

| 실행 방식 | 칩 | 1장 계산 | 모델 준비 |
|---|---|---|---|
| PyTorch (Conv 와 BN 합침) | CPU 스레드 4 | 185ms | - |
| TorchScript | CPU 스레드 4 | 188ms | - |
| ONNX Runtime | CPU 스레드 4 | 132ms | - |
| OpenVINO | CPU 스레드 4 | 134ms | 0.3초 |
| **OpenVINO** | **인텔 내장 GPU (Arc)** | **16ms** | 0.7초 (첫 실행 5.3초) |
| **OpenVINO** | **인텔 NPU (AI Boost)** | **35ms** | 0.1초 (첫 실행 3.3초) |
| PyTorch | NVIDIA RTX 3050 | 22ms | - |
| OpenVINO | NVIDIA RTX 3050 | 287ms | 1.1초 |

결과가 같은지는 Ultralytics 로 끝까지 예측해 conf 0.10 에서 견줬다. 기준은 같은 960x960 입력을 쓰는
TorchScript 다.

| 실행 방식 / 칩 | 연기 사진에 탐지 있음 | 헛경보 | 알람 일치 | 탐지 수 일치 | 마스크 IoU |
|---|---|---|---|---|---|
| TorchScript / CPU (기준) | 66/70 | 6/30 | - | - | - |
| ONNX Runtime / CPU | 66/70 | 6/30 | 100/100 | 100/100 | 1.000 |
| OpenVINO / CPU | 66/70 | 6/30 | 100/100 | 100/100 | 1.000 |
| OpenVINO / 인텔 내장 GPU | 66/70 | 6/30 | 100/100 | 96/100 | 0.989 |
| OpenVINO / 인텔 NPU | 66/70 | 6/30 | 100/100 | 98/100 | 0.995 |
| PyTorch .pt / CPU (지금 파이프라인) | 65/70 | 5/30 | 92/100 | 64/100 | 0.793 |

- **인텔 내장 GPU 와 NPU 에서 탐지가 돈다.** 내장 GPU 16ms, NPU 35ms 로 CPU(134ms)보다 4~8배 빠르다.
  알람은 100장 모두 같고, 마스크도 거의 같다(IoU 0.99 이상). 내장 GPU 와 NPU 는 안에서 정밀도를 낮춰
  계산해 작은 탐지 몇 개가 생기거나 빠진다(탐지 수 96/100, 98/100)
- **CPU 에서는 ONNX Runtime 과 OpenVINO 가 PyTorch 보다 1.4배 빠르다.** 결과는 완전히 같다
- **OpenVINO 를 NVIDIA GPU 에서 돌리면 CPU 보다 느리다(287ms).** OpenVINO 가 장치로 잡기는 하지만
  NVIDIA 용으로 맞춘 경로가 아니다. NVIDIA 에서는 PyTorch 나 TensorRT 를 쓴다
- **입력의 여백 방식만 바꿔도 알람 8장이 갈린다.** 지금 파이프라인(.pt)은 사진 비율에 맞춰 여백을
  줄인 입력(예: 960x544)을 쓰고, 내보낸 모델은 960x960 고정 입력을 쓴다. 칩 때문이 아니라 입력 모양
  때문이다. 모델을 내보내 다른 칩에 올리면 평가셋 수치를 그 형식으로 다시 재야 한다
- 처음 한 번은 칩에 맞게 모델을 준비하는 시간이 든다(내장 GPU 5.3초, NPU 3.3초). 두 번째부터는
  캐시를 써서 1초 안이다

### VLM 검증기를 인텔 칩에서 (11/18)

`tools/vlm_ov_devices.py`. OpenVINO GenAI 2026.4.1 의 `VLMPipeline` 으로 검증기(탐지 조각 448px,
"이 사진에 연기가 있는가")를 돌렸다. 모델은 NPU 용으로 변환해 공개된
`llmware/Qwen2.5-VL-3B-Instruct-ov-int4-npu`(2.7GB)다. 공식 배포가 아니고, 언어 모델을 int4(대칭, 채널 단위,
AWQ)로 줄인 것이라 지금 쓰는 GGUF Q4_K_M, bitsandbytes nf4 와 양자화가 다르다. 평가셋 확정본의 탐지 177개.

| 실행 방식 | 칩 | 검증 1회 | 모델 준비 |
|---|---|---|---|
| OpenVINO GenAI (int4) | **인텔 내장 GPU (Arc)** | **0.94초** | 3.9초 (첫 실행 25.8초) |
| OpenVINO GenAI (int4) | **인텔 NPU (AI Boost)** | **6.0초** | 3.4초 (첫 실행 33.0초) |
| OpenVINO GenAI (int4) | CPU 스레드 4 (20개) | 8.6초 | 5.4초 |
| llama.cpp GGUF Q4_K_M | CPU 스레드 4 | 11.9초 | - |
| transformers + bitsandbytes nf4 | NVIDIA RTX 3050 | 2.8초 (파이프라인 안에서) | - |

OpenVINO 칸의 시간은 배터리로 잰 값이다(전원을 꽂으면 더 빠를 수 있다).

| 실행 방식 / 칩 | 연기 탐지 남김 | 헛탐지 거름 | 새로 놓친 연기 사진 | 오탐 사진 알람 |
|---|---|---|---|---|
| bitsandbytes / NVIDIA GPU (지금, 문턱 0.7) | 127/131 | 41/46 | 1장 | 3/5 |
| GGUF / CPU (문턱 0.7) | 130/131 | 38/46 | 0장 | - |
| OpenVINO int4 / NPU (답 그대로) | 111/131 | 41/46 | 5장 | 3/5 |
| OpenVINO int4 / 내장 GPU (답 그대로) | 104/131 | 41/46 | - | - |

- **우리 VLM 이 인텔 NPU 와 내장 GPU 에서 돈다.** 내장 GPU 는 검증 1회 0.94초로 CPU(11.9초)보다 12배,
  NPU 는 6.0초로 2배 빠르다
- **헛탐지를 거르는 힘은 같다(41/46).** 굴뚝연기 2/6, 연기 사진 속 다른 곳 39/40 으로 지금 검증기와 같다
- **그러나 연기를 "아니오"로 답하는 것이 늘었다.** 연기 탐지를 남긴 수가 127 → 111(NPU), 104(내장 GPU)다.
  NPU 와 내장 GPU 의 답은 177개 중 170개가 같아서, 칩 때문이 아니라 변환된 모델(양자화 방식) 때문이다.
  이 모델을 쓰려면 직접 변환하거나 문턱을 다시 맞춰야 한다
- **NPU 는 빔 탐색을 지원하지 않는다**(그리디와 샘플링만). 그래서 "예"와 "아니오"의 확률을 둘 다 받지
  못하고 답 글자만 썼다. 위 표의 OpenVINO 줄은 문턱 없이 답 그대로 센 값이고, 고른 답의 확률로 어림한 AUC 는
  NPU 0.87, 내장 GPU 0.84 다(지금 검증기 0.92)
- **한국어로 물어도 영어로 답한다.** 답이 "Yes" 85, "No" 41, "네" 30, "no" 20, "yes" 1 이었다. 예나 아니오 밖의
  답은 없었다. 같은 모델을 llama.cpp, transformers 로 돌릴 때는 한국어로 답했다

### 실행 방식을 바꿨을 때 답이 달라지는가 (VLM)

같은 3B 를 GPU(bitsandbytes 4bit)와 CPU(GGUF Q4_K_M)로 돌려 검증기 177개를 견줬다(`docs/model_size.md`).

- 순서는 같다. AUC 0.916 대 0.917 이다. 탐지마다의 P(예) 차이는 중앙값 0.011 이다
- 그러나 문턱 근처에서 갈린다. 177개 중 6개가 0.7 을 사이에 두고 판정이 바뀌어, 헛탐지를 거른 수가
  41 → 38 로 준다. **실행 방식을 바꾸면 문턱을 다시 맞춰야 한다.** CPU 방식에서는 0.83 쯤이 맞다
- 사진 전체 관찰은 연기 인식 51/70(GPU, 다섯 항목) 대 48/70(CPU, 세 항목)으로 질문도 달라 바로
  견줄 수 없다

## 3. 실행 방식과 칩 호환성 표

### VLM

| 실행 방식 | x86 CPU | NVIDIA GPU (노트북) | Jetson (ARM + CUDA) | 인텔 내장 GPU | 인텔 NPU |
|---|---|---|---|---|---|
| transformers + bitsandbytes 4bit | **잼** (180.7초, 느림) | **잼** (6.5초) | 모름 | 문서 (XPU 백엔드) | 모름 |
| llama.cpp GGUF | **잼** (스레드 4개 9~41초) | 문서 (CUDA 빌드) | 문서 (CUDA 빌드. 비슷한 2B VLM 의 사례 있음) | 문서 (SYCL, Vulkan 빌드) | 문서 (OpenVINO 백엔드) |
| OpenVINO GenAI | **잼** (8.6초) | - | - | **잼** (0.94초) | **잼** (6.0초) |

- **bitsandbytes 는 CUDA 를 보고 만든 것이라 CPU 에서 느리다.** 같은 모델을 llama.cpp 로 바꾸자 180.7초가
  35.1초가 됐다(`docs/vlm_speed.md`). GPU 가 없으면 bitsandbytes 를 쓰지 않는다
- bitsandbytes 문서는 CUDA, 인텔 XPU, 인텔 Gaudi, CPU(리눅스 x86-64, 리눅스 aarch64, 윈도우 x86-64)를
  지원한다고 적는다. Jetson(ARM + CUDA)에 대한 언급은 찾지 못했다
- llama.cpp 의 빌드 문서는 CPU, BLAS, Metal, SYCL, CUDA, MUSA, HIP, Vulkan, CANN, OpenCL, OpenVINO,
  Hexagon 등을 다룬다. 우리가 받은 것은 윈도우 CPU 빌드라 GPU 로는 돌려 보지 않았다
- OpenVINO 2025 릴리스 노트가 NPU 에서 Qwen2.5-VL-3B-Instruct 를 지원한다고 적는다. 7B 는 변환에
  문제가 있었다는 인텔 안내문이 있다

### 탐지 (YOLO11n-seg)

Ultralytics 는 학습한 모델을 여러 형식으로 내보낼 수 있다. 칩마다 맞는 형식이 다르다.

| 칩 | 내보낼 형식 | 근거 |
|---|---|---|
| x86 CPU | PyTorch (**잼**, 185ms), ONNX Runtime (**잼**, 132ms), OpenVINO (**잼**, 134ms). 960x960, 스레드 4 | 잼 |
| NVIDIA GPU, Jetson | TensorRT (`format=engine`). Jetson 의 DLA 도 지정 가능 | 문서 |
| 인텔 내장 GPU, NPU | OpenVINO (**잼**, 내장 GPU 16ms, NPU 35ms). NPU 용 INT8 내보내기 옵션도 있다 | 잼 |
| 휴대폰, ARM 보드 | NCNN, LiteRT(TFLite), CoreML(애플) | 문서 |
| 전용 AI 칩 | RKNN(락칩), Hailo, Edge TPU, Qualcomm QNN 등 | 문서 |

내보내기 표에 작업(탐지, 분할)별 지원 여부가 따로 없다. 우리 분할 모델은 ONNX, OpenVINO, TorchScript 로
내보내 돌려 봤다. TensorRT, NCNN 등은 해 보지 않았다.

## 4. 겪은 호환성 문제

실제로 부딪힌 것들이다. 칩과 프레임워크를 바꿀 때 같은 일이 생길 수 있다.

| 문제 | 어디서 | 어떻게 했나 | 출처 |
|---|---|---|---|
| fp16 계산에서 출력이 모두 NaN 이 됨 | transformers, RTX 3050 | 계산 dtype 을 bf16 으로 바꿈 | `docs/generation_methods.md` |
| 4bit 가 CPU 에서 180초 | bitsandbytes, CPU | llama.cpp GGUF 로 바꿈 | `docs/vlm_speed.md` |
| 8bit 가 4bit 보다 100배 느림 | bitsandbytes, RTX 3050 | 4bit 를 씀 | `docs/quantization.md` |
| VRAM 이 모자라도 오류 없이 시스템 메모리로 넘겨 느려짐 | 윈도우 GPU 드라이버 | 최대 VRAM 을 따로 재서 확인 | `docs/vlm_candidates.md` |
| 형식 강제 도구가 transformers 5.x 에서 깨짐 | lm-format-enforcer | 토크나이저 연동만 직접 만듦. llama.cpp 에서는 서버의 json_schema 를 씀 | `docs/generation_methods.md`, `docs/vlm_speed.md` |
| 실행 방식을 바꾸면 검증기 확률이 조금 바뀜 | bitsandbytes 대 GGUF | 문턱을 실행 방식마다 다시 맞춤 | `docs/model_size.md` |
| 내보낸 탐지 모델은 입력 여백 방식이 달라 알람 8장이 갈림 | Ultralytics .pt 대 ONNX, OpenVINO | 내보낸 형식으로 평가셋을 다시 재야 함 | 위 2절 |
| OpenVINO 가 NVIDIA GPU 도 장치로 잡지만 CPU 보다 느림 | OpenVINO, RTX 3050 | NVIDIA 에서는 쓰지 않음 | 위 2절 |
| 내보낼 때 Ultralytics 가 필요한 패키지(onnx)를 스스로 설치함 | Ultralytics | 설치 기록을 남김. 현장 기기는 미리 설치해 둬야 함 | 11/18 |
| NPU 에서 빔 탐색이 안 돼 "예"의 확률을 바로 못 읽음 | OpenVINO GenAI, 인텔 NPU | 그리디로 답 글자만 씀 | 위 2절 |
| 변환된 VLM 이 연기를 "아니오"로 답하는 것이 늘어남(127 → 111) | 공개 변환 모델(int4 AWQ) | 칩이 아니라 변환 탓. 직접 변환이나 문턱 재조정이 필요 | 위 2절 |
| 한국어로 물어도 영어로 답함 | OpenVINO GenAI | 답을 Yes, No, 네 모두 받게 고침 | 위 2절 |
| 7B 가 4GB GPU 에 안 들어감 | RTX 3050 | CPU(GGUF)로 비교함. 8GB 기기에도 안 들어감(최대 9.7GB) | `docs/model_size.md` |
| 같은 이미지를 다시 보내면 입력 토큰이 줄어 시간이 짧게 나옴 | llama-server 캐시 | 잴 때마다 슬롯 캐시를 비움 | `docs/vlm_speed.md` |

## 5. 기기별로 권하는 구성

| 기기 | 탐지 | VLM | 근거와 남은 것 |
|---|---|---|---|
| 가속기 없는 x86 CPU (하한) | OpenVINO 또는 ONNX Runtime (134ms) | llama.cpp GGUF Q4_K_M | 잼. 검증 1회 11.9초로 10초를 넘는다. 조각을 줄이는 실험이 남음 |
| Jetson Orin Nano 급 (대표 기기) | TensorRT | llama.cpp CUDA 빌드 GGUF, 또는 transformers | 문서. 3B 는 메모리에 들어가고(5.4GB), 7B 는 안 들어간다(9.7GB). 실기기로는 재지 않음 |
| 인텔 NPU 나 내장 GPU 가 있는 노트북 | OpenVINO (내장 GPU 16ms, NPU 35ms) | OpenVINO GenAI (내장 GPU 0.94초, NPU 6.0초) | 둘 다 잼. VLM 은 공개 변환 모델이라 연기 인식이 줄어 직접 변환이 필요 |

- **같은 모델이라도 실행 방식이 속도를 정한다.** VLM 은 CPU 에서 실행 방식만 바꿔 180.7초가 35.1초가 됐다.
  칩을 고르기 전에 그 칩에서 도는 실행 방식부터 정해야 한다
- **실행 방식을 바꾸면 정확도 문턱도 다시 맞춘다.** 순서(AUC)는 같아도 확률값이 조금 움직인다
- PyTorch 로 학습한 모델을 그대로 쓰는 것은 x86 CPU 와 NVIDIA GPU 뿐이다. 다른 칩은 내보내기(ONNX,
  OpenVINO, TensorRT) 또는 다른 실행 방식(llama.cpp)을 거친다

## 6. 라이선스

현장 적용에는 칩만큼 라이선스도 걸린다(`docs/vlm_candidates.md`, `docs/seg_candidates.md`).

- Qwen2.5-VL-3B-Instruct 는 Qwen Research License 로 비상업 한정이다. 같은 계열 7B 는 Apache 2.0 이다
- Ultralytics YOLO 는 AGPL-3.0 이라 상업 서비스에 쓰려면 기업 라이선스가 필요하다

## 한계와 남은 것

- VLM 의 인텔 칩 실측은 공개 변환 모델 하나로 했다. 직접 변환하면 정확도가 달라질 수 있다
- OpenVINO 쪽 VLM 시간은 배터리로 잰 값이다. CPU(OpenVINO)는 20개로만 쟀다
- Jetson 은 실기기가 없어 하나도 실측하지 않았다. 표의 "문서" 칸은 공식 문서나 배포처의 설명이다
- 칩 비교 시간은 개발 노트북의 내장 GPU 와 NPU 값이다. 다른 기기의 같은 이름 칩과는 다를 수 있다
- 내장 GPU 와 NPU 는 FP32 로 내보낸 모델을 넣었고, 정밀도는 각 칩이 알아서 낮췄다. INT8 로 줄이는 것은
  해 보지 않았다(보정용 패키지가 더 필요하다)
- 메모리 8GB 제한은 WSL2 에서 잰 180.7초 측정 하나에만 걸었다. 나머지 CPU 값은 상한 없이 쟀다
- TensorFlow 는 쓰지 않았다. 우리 모델은 모두 PyTorch 로 학습했다

## 참고

- bitsandbytes 설치 문서: https://huggingface.co/docs/bitsandbytes/installation
- llama.cpp 빌드 문서: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Ultralytics 내보내기: https://docs.ultralytics.com/modes/export/
- OpenVINO 릴리스 노트: https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html
- Jetson Orin Nano 에서 VLM 을 돌린 사례(NVIDIA 포럼): https://forums.developer.nvidia.com/t/performance-inquiry-optimizing-qwen3-vl-2b-inference-for-2-qps-target-on-orin-nano-super/359639
