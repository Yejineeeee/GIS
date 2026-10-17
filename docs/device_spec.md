# 현장 기기 기준 환경

현장 기기에서 동작하는지 측정하기 위한 기준 환경을 정한다. 팀원 기기 사양을 모으고,
모든 측정은 아래 기준으로 제한한 환경에서 한다. GPU 결과는 참고용으로만 남긴다.

담당: 박서연 / 상태: 검토 요청. 처리 시간 목표는 미정 (아래)

## 기준

| 항목 | 기준 |
|---|---|
| CPU | 4코어 |
| 메모리 | 8GB (스왑 없음) |
| GPU | 사용 안 함 |
| 처리 시간 | **정해져 있지 않음. 아래 참고** |

### 처리 시간 목표가 없다

하드웨어만 정해져 있고 "한 프레임을 몇 ms 안에 처리해야 하는가"가 없다.

`seg_candidates.md`에 "후보 중 유일하게 100ms 아래"라고 적힌 것은 모델을 고를 때의
관찰이지 기준이 아니다. 실제로 이것 때문에 판단이 한 번 흔들렸다. SegFormer-B0이
CPU 4코어에서 340ms가 나왔을 때 100ms를 넘겼다는 이유로 탈락시켰다가, 기준이
없다는 것을 확인하고 학습까지 진행했다.

지금 고른 설정도 100ms를 넘는다. YOLO11n-seg를 960px로 쓰기로 했는데 CPU 4코어에서
133ms다. 640px은 82ms지만 작은 연기 탐지율이 77%에서 62%로 떨어진다.

정해야 할 것은 세 가지다.

| | 왜 필요한가 |
|---|---|
| 드론이 프레임을 몇 초 간격으로 내는가 | 탐지가 이 간격 안에 끝나야 한다 |
| 탐지부터 알람까지 몇 초 안에 떠야 하는가 | 지리 조회 시간까지 포함한 예산 |
| VLM 설명을 이 예산에 넣을 것인가 | 설명 생성이 10초에서 15초 걸린다. 알람과 별도로 두기로 했으나 목표값은 없다 |

프레임 간격이 1초에서 2초라면 지금 측정한 값은 모두 여유가 있다. 그 경우 100ms는
의미 없는 선이 되고, SegFormer를 탈락시킨 근거도 속도가 아니라 정확도가 된다.

측정값은 `docs/seg_candidates.md`와 `docs/train_results.md`에 있다.

## 팀원 기기

| 이름 | 기기 | CPU 모델 | 코어(물리/논리) | 메모리 | GPU | GPU 메모리 | OS |
|---|---|---|---|---|---|---|---|
| 이예진 | 노트북 | Intel Core Ultra 7 155H | 16 / 22 | 32GB | NVIDIA GeForce RTX 3050 Laptop + Intel Arc (내장), NPU: Intel AI Boost | 전용 4GB (RTX 3050), 공유 15.8GB | Windows |
| 이예진 | 폰 | Apple A16 (iPhone 15) | 6 / 6 | 6GB | 내장 | - | iOS |
| 박서연 | 노트북 | Intel Core Ultra 7 255U | 12 / 14 | 32GB | Intel Graphics (내장), NPU: Intel AI Boost | 전용 없음, 공유 최대 17.9GB | Windows |
| 박서연 | 폰 | Apple A18 (iPhone 16) | 6 / 6 | 8GB | 내장 | - | iOS |
| 이수진 | 노트북 | Intel Core i5-8210Y | 2 / 4 | 8GB | Intel UHD Graphics 617 (내장) | 전용 없음, 동적 최대 1.5GB | macOS |
| 이수진 | 폰 | Apple A16 (iPhone 14 Pro) | 6 / 6 | 6GB | 내장 | - | iOS |

- 이예진 노트북: 팀에서 유일한 NVIDIA GPU. GPU 참고 측정은 이 기기에서 한다. 전용 메모리 4GB라 큰 모델은 GPU에 다 올라가지 않을 수 있다.
- 박서연 노트북: 하이브리드 CPU(성능 2 + 효율 2 + 저전력 효율 8 구조 계열). 4코어 제한 시 어떤 코어가 잡히는지에 따라 측정값이 달라질 수 있다.
- 이수진 노트북: 기준(4코어, 8GB) 미만. 기준 측정에는 쓰지 않고, 저사양 참고용으로만 쓴다.
- 기준의 "4코어"는 논리 코어 4개(nproc 4) 기준이다.
  
### 사양 확인 방법

Windows (PowerShell):

```powershell
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
nvidia-smi   # NVIDIA GPU 이름과 메모리
```

GPU 메모리는 `nvidia-smi`가 없으면 작업 관리자 > 성능 > GPU의 "전용 GPU 메모리"를 적는다.

macOS (터미널):

```bash
sysctl -n machdep.cpu.brand_string
sysctl -n hw.physicalcpu hw.logicalcpu
echo $(( $(sysctl -n hw.memsize) / 1073741824 ))GB
```

Apple Silicon은 GPU 메모리를 따로 두지 않으므로 "통합 메모리"로 적는다.

폰: 설정 > 휴대전화 정보(Android) 또는 설정 > 일반 > 정보(iPhone)에서 모델명을 확인하고,
Android는 CPU-Z 앱으로 CPU와 메모리를 확인한다. 폰은 기록만 하고 측정은 하지 않는다.

## 제한해서 실행하는 방법

### 방법 1: Docker

```bash
docker run --rm -it \
  --cpuset-cpus=0-3 \
  --memory=8g --memory-swap=8g \
  -v "$PWD":/work -w /work \
  python:3.11 bash
```

- `--cpus=4`는 CPU 사용 시간만 4코어 분량으로 제한한다. 컨테이너 안에서는 여전히 전체 코어가
  보이므로 라이브러리가 스레드를 코어 수만큼 띄운다. 코어 수 자체를 4개로 묶으려면
  `--cpuset-cpus=0-3`을 쓴다.
- `--memory-swap`을 `--memory`와 같게 두어야 스왑으로 넘치지 않는다.

### 방법 2: WSL2 (Windows)

`C:\Users\<사용자>\.wslconfig`에 적는다.

```ini
[wsl2]
memory=8GB
processors=4
swap=0
```

PowerShell에서 `wsl --shutdown` 후 WSL을 다시 열어야 적용된다.
Docker Desktop이 WSL2 기반이면 Docker에도 같은 제한이 걸린다.

### 방법 3: macOS

Docker Desktop > Settings > Resources에서 CPU 4, Memory 8GB, Swap 0으로 두고 방법 1로 실행한다.

### 적용 확인

제한한 환경 안에서 실행한다.

```bash
nproc      # 4
free -h    # total 약 8G, Swap 0
```

### 스레드 수 맞추기

코어를 4개로 묶어도 라이브러리가 스레드를 더 띄울 수 있으므로 측정할 때는 스레드 수도 4로 맞춘다.

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

- PyTorch: `torch.set_num_threads(4)`
- llama.cpp: `-t 4`

## 측정 기록

| 날짜 | 기기 | 제한 방법 | nproc | 메모리 | 비고 |
| 2026-10-04 | 박서연 노트북 | WSL2 (.wslconfig: processors=4, memory=8GB, swap=0) | 4 | 7.8Gi, Swap 0B | 하이브리드 CPU라 어떤 코어 4개가 잡히는지는 지정 안 됨 |
