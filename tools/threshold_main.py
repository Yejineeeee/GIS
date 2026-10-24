# -*- coding: utf-8 -*-
"""판정 임계값 모의 실험 (WBS 3.5 본 실험).

정답 용도를 아는 지점 1,000곳에 오차 반경(10, 30, 50, 100m)만큼 위치 오차를 주고,
틀린 위치를 중심으로 반경 안 용도 비율을 조회해 확정 판정이 얼마나 맞는지 잰다.
확정 판정의 정확도가 목표(90, 95, 99%)를 넘는 가장 낮은 임계값을 고른다.

    python3 tools/threshold_main.py <landuse_5179.geojson> [출력 폴더] [--points 1000] [--seed 42]

판정 규칙 두 가지를 같은 조건에서 비교한다 (제안서 6항).
    최다비율: 1순위 비율 >= 임계값이면 확정(1순위 용도)
    격차:     1순위 비율 - 2순위 비율 >= 임계값이면 확정(1순위 용도)
확정이 아니면, 1·2순위 합이 0.9 이상이면 모호(후보 두 개), 아니면 미확정으로 둔다.

다른 지역 검증: 범위를 서쪽 절반(임계값 정하기)과 동쪽 절반(검증)으로 나눠,
서쪽에서 고른 임계값을 동쪽에 그대로 적용한 정확도도 함께 낸다.

계산을 빠르게 하려고 용도지역을 5m 격자로 바꿔 조회한다(경계에서 최대 수 m 차이).
외부 패키지 없이 동작한다.

출력
    threshold_points.csv    지점별 정답, 조회 비율 1·2순위
    threshold_summary.csv   반경 x 규칙 x 목표별 임계값, 확정 비율, 정확도
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

RADII = [10, 30, 50, 100]
TARGETS = [0.90, 0.95, 0.99]
CELL = 5.0
MARGIN = 200.0          # 범위 가장자리에서 띄우는 거리 (원이 범위를 벗어나지 않게)
AMBIG_SUM = 0.9         # 1·2순위 합이 이 이상이면 모호
MIN_CONFIRMED = 20      # 확정 지점이 이보다 적으면 그 임계값은 고르지 않음


def polys(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


class Raster:
    """용도지역 폴리곤을 CELL m 격자로 바꾼다. 칸 값은 용도 번호(없으면 -1)."""

    def __init__(self, path):
        feats = json.load(open(path, encoding="utf-8"))["features"]
        items = []
        xs, ys = [], []
        for f in feats:
            name = (f.get("properties") or {}).get("uname") or "(이름 없음)"
            for rings in polys(f["geometry"]):
                items.append((name, rings))
                for r in rings:
                    for p in r:
                        xs.append(p[0])
                        ys.append(p[1])
        self.x0, self.y0 = min(xs), min(ys)
        self.nx = int((max(xs) - self.x0) / CELL) + 1
        self.ny = int((max(ys) - self.y0) / CELL) + 1
        self.names = []
        index = {}
        self.grid = [[-1] * self.nx for _ in range(self.ny)]
        for name, rings in items:
            k = index.setdefault(name, len(index))
            if k == len(self.names):
                self.names.append(name)
            self._fill(rings, k)

    def _fill(self, rings, k):
        rows = {}
        for ring in rings:
            for i in range(len(ring) - 1):
                (x1, y1), (x2, y2) = ring[i][:2], ring[i + 1][:2]
                if y1 == y2:
                    continue
                lo, hi = min(y1, y2), max(y1, y2)
                r0 = max(int(math.ceil((lo - self.y0) / CELL - 0.5)), 0)
                r1 = min(int(math.floor((hi - self.y0) / CELL - 0.5)), self.ny - 1)
                for r in range(r0, r1 + 1):
                    yc = self.y0 + (r + 0.5) * CELL
                    if lo <= yc < hi:
                        rows.setdefault(r, []).append(x1 + (yc - y1) * (x2 - x1) / (y2 - y1))
        for r, cx in rows.items():
            cx.sort()
            row = self.grid[r]
            for a, b in zip(cx[0::2], cx[1::2]):
                c0 = max(int(math.ceil((a - self.x0) / CELL - 0.5)), 0)
                c1 = min(int(math.floor((b - self.x0) / CELL - 0.5)), self.nx - 1)
                for c in range(c0, c1 + 1):
                    if row[c] < 0:          # 먼저 들어온 폴리곤을 남긴다
                        row[c] = k

    def at(self, x, y):
        c = int((x - self.x0) / CELL)
        r = int((y - self.y0) / CELL)
        if 0 <= r < self.ny and 0 <= c < self.nx:
            return self.grid[r][c]
        return -1

    def ratios(self, x, y, offsets):
        c0 = int((x - self.x0) / CELL)
        r0 = int((y - self.y0) / CELL)
        cnt = {}
        for dr, dc in offsets:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < self.ny and 0 <= c < self.nx:
                k = self.grid[r][c]
                if k >= 0:
                    cnt[k] = cnt.get(k, 0) + 1
        tot = sum(cnt.values())
        if not tot:
            return []
        return sorted(((k, v / tot) for k, v in cnt.items()), key=lambda kv: -kv[1])


def disk(radius):
    k = int(radius / CELL) + 1
    return [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1)
            if (i * CELL) ** 2 + (j * CELL) ** 2 <= radius ** 2]


def score(rule, rat):
    if not rat:
        return -1.0
    if rule == "최다비율":
        return rat[0][1]
    second = rat[1][1] if len(rat) > 1 else 0.0
    return rat[0][1] - second


def pick(rows, rule, target):
    """정확도가 target 이상인 가장 낮은 임계값. (임계값, 확정 비율, 정확도, 확정 수)"""
    for i in range(0, 101):
        t = i / 100
        conf = [r for r in rows if score(rule, r["rat"]) >= t]
        if len(conf) < MIN_CONFIRMED:
            break
        acc = sum(1 for r in conf if r["rat"][0][0] == r["truth"]) / len(conf)
        if acc >= target:
            return t, len(conf) / len(rows), acc, len(conf)
    return None, 0.0, None, 0


def apply(rows, rule, t):
    conf = [r for r in rows if score(rule, r["rat"]) >= t]
    if not conf:
        return 0.0, None
    acc = sum(1 for r in conf if r["rat"][0][0] == r["truth"]) / len(conf)
    return len(conf) / len(rows), acc


def main():
    ap = argparse.ArgumentParser(description="판정 임계값 모의 실험 (WBS 3.5)")
    ap.add_argument("landuse")
    ap.add_argument("out", nargs="?", default=".")
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ras = Raster(a.landuse)
    rng = random.Random(a.seed)
    xmin, ymin = ras.x0 + MARGIN, ras.y0 + MARGIN
    xmax, ymax = ras.x0 + ras.nx * CELL - MARGIN, ras.y0 + ras.ny * CELL - MARGIN
    xmid = (xmin + xmax) / 2

    truths = []
    while len(truths) < a.points:
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        k = ras.at(x, y)
        if k >= 0:
            truths.append((x, y, k))
    print(f"용도지역 {len(ras.names)}종, 격자 {ras.nx}x{ras.ny} ({CELL:g}m), 정답 지점 {len(truths)}곳")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pts_rows, summary = [], []
    for R in RADII:
        offs = disk(R)
        rows = []
        for i, (x, y, k) in enumerate(truths):
            ang = rng.uniform(0, 2 * math.pi)
            d = R * math.sqrt(rng.random())         # 반경 안에 고르게 틀린 위치
            ex, ey = x + d * math.cos(ang), y + d * math.sin(ang)
            rat = ras.ratios(ex, ey, offs)
            rows.append({"id": i, "truth": k, "rat": rat, "west": x < xmid})
            top = rat + [(-1, 0.0), (-1, 0.0)]
            pts_rows.append([R, i, round(x, 1), round(y, 1), ras.names[k], round(d, 1),
                             ras.names[top[0][0]] if top[0][0] >= 0 else "",
                             round(top[0][1], 4),
                             ras.names[top[1][0]] if top[1][0] >= 0 else "",
                             round(top[1][1], 4)])

        west = [r for r in rows if r["west"]]
        east = [r for r in rows if not r["west"]]
        print(f"\n반경 {R}m")
        print(f"  {'규칙':6s} {'목표':>5s} {'임계값':>6s} {'확정 비율':>8s} {'정확도':>6s}  | 서쪽에서 정해 동쪽 적용: 임계값 확정 비율 정확도")
        for rule in ("최다비율", "격차"):
            for tg in TARGETS:
                t, cov, acc, n = pick(rows, rule, tg)
                tw, _, _, _ = pick(west, rule, tg)
                ecov, eacc = apply(east, rule, tw) if tw is not None else (0.0, None)
                # 확정이 아닌 것 중 모호/미확정, 모호일 때 정답이 후보 두 개 안에 있는 비율
                amb = [r for r in rows if t is None or score(rule, r["rat"]) < t]
                amb2 = [r for r in amb if len(r["rat"]) >= 2 and r["rat"][0][1] + r["rat"][1][1] >= AMBIG_SUM]
                amb_hit = (sum(1 for r in amb2 if r["truth"] in (r["rat"][0][0], r["rat"][1][0])) / len(amb2)) if amb2 else None
                f = lambda v: "-" if v is None else f"{v:.2f}"
                p = lambda v: "-" if v is None else f"{v * 100:.1f}%"
                print(f"  {rule:6s} {tg * 100:4.0f}% {f(t):>6s} {cov * 100:7.1f}% {p(acc):>6s}  |"
                      f" {f(tw):>6s} {ecov * 100:7.1f}% {p(eacc):>6s}")
                summary.append([R, rule, tg, t, round(cov, 4), None if acc is None else round(acc, 4), n,
                                len(amb2) / len(rows), None if amb_hit is None else round(amb_hit, 4),
                                (len(amb) - len(amb2)) / len(rows),
                                tw, round(ecov, 4), None if eacc is None else round(eacc, 4)])

    with open(out / "threshold_points.csv", "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["radius_m", "id", "x", "y", "truth", "offset_m", "top1", "top1_ratio", "top2", "top2_ratio"])
        w.writerows(pts_rows)
    with open(out / "threshold_summary.csv", "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["radius_m", "rule", "target", "threshold", "confirmed_share", "confirmed_acc", "confirmed_n",
                    "ambiguous_share", "ambiguous_top2_hit", "undetermined_share",
                    "west_threshold", "east_confirmed_share", "east_confirmed_acc"])
        w.writerows(summary)
    print(f"\n저장: {out / 'threshold_summary.csv'}, {out / 'threshold_points.csv'}")


if __name__ == "__main__":
    main()
