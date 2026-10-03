# 현장 기기 기준 환경

현장 기기에서 동작하는지 측정하기 위한 기준 환경을 정한다. 팀원 기기 사양을 모으고,
모든 측정은 아래 기준으로 제한한 환경에서 한다. GPU 결과는 참고용으로만 남긴다.

담당: 박서연 / 상태: 작성 중

## 기준

| 항목 | 기준 |
|---|---|
| CPU | 4코어 |
| 메모리 | 8GB (스왑 없음) |
| GPU | 사용 안 함 |

## 팀원 기기

| 이름 | 기기 | CPU 모델 | 코어(물리/논리) | 메모리 | GPU | GPU 메모리 | OS |
|---|---|---|---|---|---|---|---|
| 이예진 | 노트북 | | | | | | |
| 이예진 | 폰 | | | | - | - | |
| 박서연 | 노트북 | Intel Core Ultra 7 255U | 12 / 14 | 32GB | Intel Graphics (내장), NPU: Intel AI Boost | 전용 없음, 공유 최대 17.9GB | Windows |
| 박서연 | 폰 | Apple A18 (iPhone 16) | 6 / 6 | 8GB | 내장 | - | iOS |
| 이수진 | 노트북 | | | | | | |
| 이수진 | 폰 | | | | - | - | |

- 박서연 노트북: 하이브리드 CPU(성능 2 + 효율 2 + 저전력 효율 8 구조 계열). 4코어 제한 시 어떤 코어가 잡히는지에 따라 측정값이 달라질 수 있다.

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
|---|---|---|---|---|---|
| | | | | | |
