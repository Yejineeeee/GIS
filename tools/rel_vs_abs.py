# -*- coding: utf-8 -*-
"""절대 위치 오차와 상대 거리 오차를 나란히 잰다.

"절대적인 위치 추정이 어렵다면 객체 간 상대 거리를 제공하자"는 제안이 실제로
유리한지 확인한다. 같은 프레임 안 두 점은 드론 GPS 오차를 똑같이 나눠 쓰므로
두 점 사이 거리를 낼 때 그 오차가 상쇄될 것으로 본다. 그 짐작이 맞는지, 얼마나
유리한지 숫자로 낸다.

재는 방법
    카메라 값을 하나 고정하고 화면에서 픽셀 두 개를 고른다. 그 카메라로 두
    픽셀이 가리키는 지상 점을 구해 정답으로 둔다. 그다음 카메라 값에 실제
    오차를 넣고 같은 픽셀로 다시 구한다.

        절대 오차   정답 점과 다시 구한 점 사이 거리
        상대 오차   두 점 사이 거리가 정답 거리와 얼마나 다른가

    오차는 두 갈래로 나눠 넣는다. 카메라 오차는 두 점이 함께 겪고, 픽셀 오차는
    점마다 따로 생긴다. 상쇄되는 것은 앞쪽뿐이므로 따로 봐야 한다.

    python rel_vs_abs.py 저장소경로 [시행횟수]
"""
import math
import random
import statistics
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

sys.path.insert(0, str(REPO / "src" / "geo"))
import transform as tf  # noqa: E402

# sim_0002.json 의 카메라 값. 고도는 지면 기준이다
CAM = dict(w=1024, h=768, alt=120.0, heading=45.0, pitch=-18.9, hfov=60.0)

# 넣을 오차. 1시그마
SIG_GPS = 5.0        # 드론 GNSS 위치 (transform.GNSS_ERROR_M 과 같은 값)
SIG_ALT = 2.0        # 고도
SIG_HEAD = 2.0       # 기수 방향(도)
SIG_PITCH = 1.0      # 촬영 각도(도)
SIG_PIX = 3.0        # 탐지 픽셀. 실제 사진 190장에서 라벨과 마스크의 밑동 차이 중앙값

SEPS = [10, 30, 50, 100]      # 두 점을 이만큼 떼어 놓고 본다
random.seed(42)


def gpoint(px, py, cam):
    """픽셀이 가리키는 지상 점. 드론 바로 아래를 원점으로 한 (동, 북) m."""
    return tf.ground_offset(px, py, cam["w"], cam["h"], cam["alt"],
                            cam["heading"], cam["pitch"], cam["hfov"])


def find_pair(sep, axis):
    """화면 가운데에서 지상 거리가 sep 가 되는 픽셀 둘을 찾는다.

    axis 가 "가로" 면 화면 좌우로, "세로" 면 위아래로 뗀다. 비스듬히 내려다보는
    화면에서는 세로 1픽셀이 가로 1픽셀보다 훨씬 먼 거리를 뜻한다. 그래서 두 점을
    어느 쪽으로 떼느냐에 따라 픽셀 오차가 다르게 실린다.
    """
    cx, cy = CAM["w"] / 2, CAM["h"] / 2
    a = gpoint(cx, cy, CAM)
    lo, hi = 0.5, (CAM["w"] / 2 - 1 if axis == "가로" else CAM["h"] / 2 - 1)
    for _ in range(60):
        mid = (lo + hi) / 2
        b = gpoint(cx + mid, cy, CAM) if axis == "가로" else gpoint(cx, cy - mid, CAM)
        if b is None or math.dist(a, b) < sep:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return (cx, cy), ((cx + d, cy) if axis == "가로" else (cx, cy - d))


def jitter_cam():
    """카메라 값에 오차를 넣는다. 두 점이 함께 겪는 오차다."""
    c = dict(CAM)
    c["alt"] = CAM["alt"] + random.gauss(0, SIG_ALT)
    c["heading"] = CAM["heading"] + random.gauss(0, SIG_HEAD)
    c["pitch"] = CAM["pitch"] + random.gauss(0, SIG_PITCH)
    # GPS 오차는 드론이 통째로 옮겨 앉은 것이라 두 점에 같은 양으로 실린다.
    # 두 점의 차를 낼 때 그대로 빠지므로 상대 거리에는 영향이 없다.
    return c, (random.gauss(0, SIG_GPS), random.gauss(0, SIG_GPS))


def run(sep, use_cam, use_pix, axis):
    p1, p2 = find_pair(sep, axis)
    a0, b0 = gpoint(*p1, CAM), gpoint(*p2, CAM)
    true_d = math.dist(a0, b0)

    abs_e, rel_e = [], []
    for _ in range(N):
        cam, shift = (jitter_cam() if use_cam else (dict(CAM), (0.0, 0.0)))
        q1 = p1
        q2 = p2
        if use_pix:
            q1 = (p1[0] + random.gauss(0, SIG_PIX), p1[1] + random.gauss(0, SIG_PIX))
            q2 = (p2[0] + random.gauss(0, SIG_PIX), p2[1] + random.gauss(0, SIG_PIX))
        a, b = gpoint(*q1, cam), gpoint(*q2, cam)
        if a is None or b is None:
            continue
        a = (a[0] + shift[0], a[1] + shift[1])
        b = (b[0] + shift[0], b[1] + shift[1])
        abs_e.append(math.dist(a, a0))
        abs_e.append(math.dist(b, b0))
        rel_e.append(abs(math.dist(a, b) - true_d))
    return true_d, abs_e, rel_e


def p90(v):
    s = sorted(v)
    return s[int(0.9 * (len(s) - 1))]


print(f"카메라  고도 {CAM['alt']:g}m(지면 기준), 기수 {CAM['heading']:g}도, "
      f"촬영각 {CAM['pitch']:g}도, 화각 {CAM['hfov']:g}도")
print(f"오차    GPS {SIG_GPS:g}m, 고도 {SIG_ALT:g}m, 기수 {SIG_HEAD:g}도, "
      f"촬영각 {SIG_PITCH:g}도, 픽셀 {SIG_PIX:g}px  (각각 1시그마)")
print(f"시행    조건마다 {N}회\n")

for axis in ("가로", "세로"):
    print(f"########## 두 점을 화면 {axis}로 떼어 놓은 경우 ##########")
    for label, uc, up in [("카메라 오차만", True, False),
                          ("픽셀 오차만", False, True),
                          ("둘 다", True, True)]:
        print(f"--- {label}")
        print(f"{'두 점 거리':>10}{'절대 오차 중앙':>14}{'절대 90%':>10}"
              f"{'상대 오차 중앙':>14}{'상대 90%':>10}{'중앙값 비율':>12}")
        for sep in SEPS:
            d, ae, re = run(sep, uc, up, axis)
            ma, mr = statistics.median(ae), statistics.median(re)
            ratio = f"{mr / ma:.2f}배" if ma > 0 else "-"
            print(f"{d:>9.1f}m{ma:>13.1f}m{p90(ae):>9.1f}m"
                  f"{mr:>13.1f}m{p90(re):>9.1f}m{ratio:>12}")
        print()
