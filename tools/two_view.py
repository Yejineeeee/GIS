# -*- coding: utf-8 -*-
"""한 장으로 풀 때와 두 장을 교차할 때의 위치 오차를 나란히 잰다.

지금 transform.py 는 프레임 한 장으로 푼다. 픽셀 하나에서 그은 선 하나로는 점을
정할 수 없으므로, "연기는 땅에 있다" 고 보고 그 선이 지면과 만나는 자리를 답으로
쓴다. 그런데 연기는 공중에 떠 있어서 이 가정이 깨진다. 8주차에 찾은 565m 오차가
그것이었고, 밑동을 찍는 것으로 149.8m 까지 줄였다.

드론이 움직이면서 같은 연기를 두 번 찍으면 서로 다른 자리에서 그은 선 두 개가
생기고, 두 선이 만나는 점이 답이 된다. 이때는 땅에 있다는 가정이 필요 없다.
높이까지 계산에서 나온다. 그 이점이 실제로 얼마인지 잰다.

좌표
    x 동쪽, y 북쪽, z 위. 발화점 지면을 원점에 둔다.
    관측 대상은 연기 밑동이고 지면에서 b 만큼 떠 있다.
    드론은 발화점에서 수평으로 D 떨어진 곳을 지나며 고도 H 로 난다.
    두 장은 그 경로에서 B 만큼 떨어진 두 지점이다.

넣는 오차
    드론 위치   프레임마다 따로 (시간이 다르므로 상쇄되지 않는다)
    자세각     기수와 촬영각
    픽셀       탐지 픽셀 오차를 각도로 바꿔 넣는다

    python two_view.py [시행횟수]
"""
import math
import random
import statistics
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

H = 120.0            # 지면 기준 비행 고도 (팀 결정값)
D = 350.0            # 발화점까지 수평 거리. sim_0002 와 같은 조건
HFOV, WPIX = 60.0, 1024

SIG_GPS_H = 5.0      # 드론 수평 위치 (제안서 일반 GNSS 약 5m)
SIG_GPS_V = 2.0      # 드론 고도
SIG_HEAD = 2.0       # 기수 방향(도)
SIG_PITCH = 1.0      # 촬영각(도)
SIG_PIX = 3.0        # 탐지 픽셀. 실제 사진 190장 밑동 차이 중앙값

# 픽셀 오차를 각도로. 화각 60도가 가로 1024픽셀에 걸린다
PIX_DEG = SIG_PIX * HFOV / WPIX
SIG_AZ = math.hypot(SIG_HEAD, PIX_DEG)
SIG_EL = math.hypot(SIG_PITCH, PIX_DEG)

BASES = [0, 5, 20, 40]          # 연기 밑동이 떠 있는 높이(m)
BASELINES = [25, 50, 100, 200]  # 두 촬영 지점 사이 거리(m)
random.seed(7)


def look(drone, target):
    """드론에서 대상을 보는 방향을 방위각과 올려본각으로."""
    dx, dy, dz = (target[i] - drone[i] for i in range(3))
    return math.atan2(dx, dy), math.asin(dz / math.dist(drone, target))


def ray(az, el):
    c = math.cos(el)
    return (c * math.sin(az), c * math.cos(az), math.sin(el))


def observe(drone, target):
    """오차를 넣은 드론 위치와 시선 방향을 돌려준다."""
    az, el = look(drone, target)
    az += math.radians(random.gauss(0, SIG_AZ))
    el += math.radians(random.gauss(0, SIG_EL))
    d = (drone[0] + random.gauss(0, SIG_GPS_H),
         drone[1] + random.gauss(0, SIG_GPS_H),
         drone[2] + random.gauss(0, SIG_GPS_V))
    return d, ray(az, el)


def to_ground(d, u):
    """선이 평평한 지면(z=0)과 만나는 자리. transform.py 가 하는 일이다."""
    if u[2] >= -1e-9:
        return None
    t = -d[2] / u[2]
    return (d[0] + u[0] * t, d[1] + u[1] * t, 0.0)


def cross(d1, u1, d2, u2):
    """두 선이 가장 가까워지는 자리의 가운데 점. 선이 꼬여 있어도 답이 나온다."""
    w = [d1[i] - d2[i] for i in range(3)]
    a = sum(x * x for x in u1)
    b = sum(u1[i] * u2[i] for i in range(3))
    c = sum(x * x for x in u2)
    dd = sum(u1[i] * w[i] for i in range(3))
    e = sum(u2[i] * w[i] for i in range(3))
    den = a * c - b * b
    if abs(den) < 1e-12:            # 두 선이 거의 나란하면 못 푼다
        return None
    s, t = (b * e - c * dd) / den, (a * e - b * dd) / den
    p1 = [d1[i] + s * u1[i] for i in range(3)]
    p2 = [d2[i] + t * u2[i] for i in range(3)]
    return [(p1[i] + p2[i]) / 2 for i in range(3)]


def p90(v):
    s = sorted(v)
    return s[int(0.9 * (len(s) - 1))]


print(f"드론 고도 {H:g}m, 발화점까지 수평 거리 {D:g}m")
print(f"오차  위치 {SIG_GPS_H:g}m, 고도 {SIG_GPS_V:g}m, 기수 {SIG_HEAD:g}도, "
      f"촬영각 {SIG_PITCH:g}도, 픽셀 {SIG_PIX:g}px(={PIX_DEG:.3f}도)")
print(f"드론 위치 오차는 프레임마다 따로 넣는다. 시행 {N}회\n")

print("한 장으로 풀 때 (지금 방식). 연기가 뜬 높이만 바꾼다")
print(f"{'밑동 높이':>9}{'위치 오차 중앙':>14}{'90%':>9}")
single = {}
for b in BASES:
    tgt = (0.0, 0.0, float(b))
    es = []
    for _ in range(N):
        d, u = observe((0.0, -D, H), tgt)
        g = to_ground(d, u)
        if g:
            es.append(math.hypot(g[0], g[1]))
    single[b] = es
    print(f"{b:>8}m{statistics.median(es):>13.1f}m{p90(es):>8.1f}m")

print("\n두 장을 교차할 때")
print(f"{'밑동 높이':>9}{'두 지점 간격':>12}{'위치 오차 중앙':>14}{'90%':>9}"
      f"{'높이 오차 중앙':>14}{'한 장 대비':>11}")
for b in BASES:
    tgt = (0.0, 0.0, float(b))
    for B in BASELINES:
        es, hs, fail = [], [], 0
        for _ in range(N):
            d1, u1 = observe((-B / 2, -D, H), tgt)
            d2, u2 = observe((+B / 2, -D, H), tgt)
            m = cross(d1, u1, d2, u2)
            if m is None:
                fail += 1
                continue
            es.append(math.hypot(m[0], m[1]))
            hs.append(abs(m[2] - b))
        med = statistics.median(es)
        ratio = med / statistics.median(single[b])
        print(f"{b:>8}m{B:>11}m{med:>13.1f}m{p90(es):>8.1f}m"
              f"{statistics.median(hs):>13.1f}m{ratio:>10.2f}배")
    print()
