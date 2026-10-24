# -*- coding: utf-8 -*-
"""오차 영역 안의 위치 후보가 건물에 얼마나 가까운지 점수로 낸다.

기존 규칙은 "오차 반경 안에 건물이 한 채라도 있으면 굴뚝"이었다. 반경이 커지면
산림 한가운데도 건물이 몇 채씩 들어와 전부 굴뚝으로 판정된다. 반경 100m 에서
정확도가 56% 였고 찍기 기준선이 50% 였다.

대신 오차 영역에서 위치 후보를 고르게 뽑아, 그중 건물 가까이에 있는 후보의 비율을
낸다. 반경이 커지면 비율이 중간값으로 몰려 "가릴 수 없다"가 숫자로 드러난다.
단정하지 않는 것이 목적이고 정확도를 올리는 것이 목적이 아니다.

    S = (건물에서 THRESH 미터 안에 있는 후보 수) / (전체 후보 수)

이 점수는 위치 후보가 건물에 가까울 가능성이지 굴뚝에서 났을 확률이 아니다. 건물이
많은 산림 경계에서는 산불도 높은 점수를 받는다.

실제 오차 분포를 모르므로 원 안에서 고르게 뽑는다. 방향을 아는 근거가 없어 타원을
쓰지 않는다. 이것도 검증되지 않은 가정이므로 한계로 남긴다.

    python building_score.py 건물요약csv 지점csv [반경...]
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_landuse import to_5179

N_CAND = 1000          # 후보 수. 1000이면 비율이 0.1%p 단위로 안정된다
THRESH = (5, 15, 30)   # 건물까지 몇 미터 안이면 가깝다고 볼지
RADII = (10, 30, 50, 100)
SEED = 42


def load_buildings(path):
    pts = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                pts.append((float(r["x_5179"]), float(r["y_5179"])))
            except (TypeError, ValueError):
                continue
    return pts


def grid_of(pts, cell):
    g = {}
    for x, y in pts:
        g.setdefault((int(x // cell), int(y // cell)), []).append((x, y))
    return g


def near(g, cell, x, y, r):
    """(x, y) 에서 r 미터 안에 건물이 있는지 본다."""
    span = int(r // cell) + 1
    gx, gy = int(x // cell), int(y // cell)
    rr = r * r
    for i in range(gx - span, gx + span + 1):
        for j in range(gy - span, gy + span + 1):
            for bx, by in g.get((i, j), ()):
                if (bx - x) ** 2 + (by - y) ** 2 <= rr:
                    return True
    return False


def candidates(x, y, radius, n, rng):
    """원 안에서 고르게 뽑는다. 반지름에 제곱근을 씌워야 중심에 몰리지 않는다."""
    out = []
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        d = radius * math.sqrt(rng.random())
        out.append((x + d * math.cos(a), y + d * math.sin(a)))
    return out


def score(g, cell, x, y, radius, thresh, n=N_CAND, rng=None):
    rng = rng or random.Random(SEED)
    cs = candidates(x, y, radius, n, rng)
    return sum(1 for cx, cy in cs if near(g, cell, cx, cy, thresh)) / n


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pts = load_buildings(sys.argv[1])
    radii = [int(v) for v in sys.argv[3:]] or list(RADII)
    cell = 100
    g = grid_of(pts, cell)
    print(f"건물 {len(pts)}채, 후보 {N_CAND}개\n")

    with open(sys.argv[2], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    head = f"  {'지점':6}{'구간':6}"
    for r in radii:
        for t in THRESH:
            head += f"{f'{r}m/{t}m':>10}"
    print(head)
    rng = random.Random(SEED)
    for row in rows:
        if row.get("lat") and row.get("lon"):
            x, y = to_5179(float(row["lon"]), float(row["lat"]))
        else:
            x, y = float(row["x_5179"]), float(row["y_5179"])
        line = f"  {row.get('pos_id', ''):6}{row.get('band', ''):6}"
        for r in radii:
            for t in THRESH:
                line += f"{score(g, cell, x, y, r, t, rng=rng):>10.2f}"
        print(line)


if __name__ == "__main__":
    main()
