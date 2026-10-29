# -*- coding: utf-8 -*-
"""가상 배치 위치 36곳을 3.2 확정 용도지역으로 다시 재서 확인한다 (WBS 3.7).

`make_demo_positions.py` 로 36곳을 고를 때 쓴 것은 `fetch_zoning.py` 가 받은
`LT_C_UQ111`(도시지역) 하나다. 3.2 확정본에는 관리지역(`LT_C_UQ112`)과
농림지역(`LT_C_UQ113`)까지 들어 있어, 고르는 결과가 달라지는지 확인해야 했다.

세 조건으로 돌려 `docs/demo_positions.csv` 와 비교한다.

    A  도시지역만                  고를 때 쓴 조건을 확정본에서 재현
    B  확정본 전체                 레이어만 늘어난 경우
    C  확정본 전체, 농림지역과 관리지역을 산림 계열로 본 경우

함께 두 가지를 더 센다. 기존 36곳의 좌표를 그대로 두고 확정본으로 다시 쟀을 때
구간이 바뀌는지, 그리고 농림 계열과 주거 계열이 한 반경 안에서 섞이는 칸이
있는지다. 결과는 `docs/demo_area.md` 의 "3.2 확정본으로 재확인" 에 적어 두었다.

    python recheck_demo_positions.py 확정본geojson [출력폴더]

확정본은 `fetch_landuse.py` 가 낸 `landuse_4326.geojson` 이다. 출력폴더를 주면
조건마다 고른 위치를 CSV 로 남긴다.
"""
import csv
import json
import sys
from pathlib import Path

from shapely.geometry import Point

import make_demo_positions as mdp

LANDUSE = Path(sys.argv[1] if len(sys.argv) > 1 else "landuse_4326.geojson")
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OLD_CSV = Path(__file__).resolve().parent.parent / "docs" / "demo_positions.csv"

# 산림 계열에 넣어 볼 용도. 농림지역과 관리지역은 도시지역 레이어에 없어서
# 36곳을 고를 때는 아예 후보에 없었다.
ADD_FOREST = ("농림지역", "보전관리지역", "생산관리지역")
# 위 셋에 계획관리지역까지 넣은 것. 섞이는 칸을 셀 때만 쓴다.
AGRI = ADD_FOREST + ("계획관리지역",)
MIX_MIN = 0.05                     # mixes_forest_resid 가 쓰는 하한과 같다


def urban_only(path, outdir):
    """확정본에서 도시지역 레이어만 남긴 파일을 만든다."""
    gj = json.load(open(path, encoding="utf-8"))
    gj["features"] = [f for f in gj["features"]
                      if f["properties"].get("kind") == "도시지역"]
    out = (outdir or path.parent) / "_urban_only.geojson"
    json.dump(gj, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    return out, len(gj["features"])


def select(zones):
    """make_demo_positions.main 에서 고르는 부분만 떼어 온 것이다."""
    cx, cy = mdp.fwd.transform(mdp.SITE[1], mdp.SITE[0])
    site = Point(cx, cy)

    found = []
    for dx in range(-mdp.SPAN, mdp.SPAN + 1, mdp.STEP):
        for dy in range(-mdp.SPAN, mdp.SPAN + 1, mdp.STEP):
            pt = Point(cx + dx, cy + dy)
            parts, cov = mdp.measure(zones, pt, mdp.PICK_R)
            if cov < mdp.MIN_COV or not parts:
                continue
            band = next((b for b, lo, hi in mdp.BANDS
                         if lo <= parts[0][1] < hi), None)
            if band is None:
                continue
            found.append({"pt": pt, "band": band,
                          "mix": mdp.mixes_forest_resid(parts),
                          "dist": site.distance(pt)})

    picked, short = [], {}
    for b, _, _ in mdp.BANDS:
        pool = sorted((f for f in found if f["band"] == b),
                      key=lambda f: (not f["mix"], f["dist"]))
        chosen = []
        for f in pool:
            if any(f["pt"].distance(c["pt"]) < mdp.MIN_GAP for c in chosen):
                continue
            chosen.append(f)
            if len(chosen) == mdp.PER_BAND:
                break
        if len(chosen) < mdp.PER_BAND:
            short[b] = len(chosen)
        picked.extend(chosen)

    rows = []
    for i, f in enumerate(sorted(picked, key=lambda f: (f["band"], f["dist"])), 1):
        lon, lat = mdp.inv.transform(f["pt"].x, f["pt"].y)
        row = {"pos_id": f"P{i:03d}", "lat": round(lat, 6), "lon": round(lon, 6),
               "band": f["band"], "forest_resid_mix": int(f["mix"]),
               "dist_from_site_m": round(f["dist"])}
        for r in mdp.RADII:
            parts, cov = mdp.measure(zones, f["pt"], r)
            row[f"top_{r}"] = parts[0][0] if parts else ""
            row[f"ratio_{r}"] = round(parts[0][1], 3) if parts else 0
            row[f"second_{r}"] = parts[1][0] if len(parts) > 1 else ""
            row[f"cov_{r}"] = round(cov, 3)
        rows.append(row)
    return found, rows, short


def compare(old, rows):
    """기존 CSV 와 좌표가 같은 위치 수, 값이 다른 칸 수를 돌려준다."""
    same = sum(1 for o, n in zip(old, rows)
               if o["lat"] == str(n["lat"]) and o["lon"] == str(n["lon"]))
    diff = sum(1 for o, n in zip(old, rows) for k in o if o[k] != str(n[k]))
    return same, diff


def remeasure(zones, old):
    """기존 36곳의 좌표를 그대로 두고 다시 잰다. 구간이 바뀌는 곳을 찾는다."""
    moved, uncovered = [], []
    for r in old:
        pt = Point(*mdp.fwd.transform(float(r["lon"]), float(r["lat"])))
        parts, _ = mdp.measure(zones, pt, mdp.PICK_R)
        band = next((b for b, lo, hi in mdp.BANDS
                     if parts and lo <= parts[0][1] < hi), "구간밖")
        if band != r["band"]:
            moved.append((r["pos_id"], r["band"], band))
        _, cov_far = mdp.measure(zones, pt, max(mdp.RADII))
        if cov_far < mdp.MIN_COV:
            uncovered.append((r["pos_id"], round(cov_far, 3)))
    return moved, uncovered


def count_mix(zones):
    """농림 계열과 주거 계열이 한 반경 안에서 섞이는 칸을 센다."""
    cx, cy = mdp.fwd.transform(mdp.SITE[1], mdp.SITE[0])
    # 여기서 세는 칸은 덮인 비율만 넘긴 칸이다. 위에서 쓰는 "쓸 수 있는 칸" 은
    # 구간까지 맞아야 하므로 수가 더 적다.
    n = {"덮인 비율을 넘긴 칸": 0, "농림과 관리 계열": 0, "주거 계열": 0,
         "농림관리와 주거가 함께": 0, "녹지와 주거가 함께": 0}
    for dx in range(-mdp.SPAN, mdp.SPAN + 1, mdp.STEP):
        for dy in range(-mdp.SPAN, mdp.SPAN + 1, mdp.STEP):
            parts, cov = mdp.measure(zones, Point(cx + dx, cy + dy), mdp.PICK_R)
            if cov < mdp.MIN_COV or not parts:
                continue
            a = sum(r for k, r in parts if k in AGRI)
            d = sum(r for k, r in parts if k in mdp.RESID)
            g = sum(r for k, r in parts if k in mdp.FOREST)
            n["덮인 비율을 넘긴 칸"] += 1
            n["농림과 관리 계열"] += a >= MIX_MIN
            n["주거 계열"] += d >= MIX_MIN
            n["농림관리와 주거가 함께"] += a >= MIX_MIN and d >= MIX_MIN
            n["녹지와 주거가 함께"] += g >= MIX_MIN and d >= MIX_MIN
    return n


def write(rows, path):
    cols = (["pos_id", "lat", "lon", "band", "forest_resid_mix",
             "dist_from_site_m"] +
            [f"{p}_{r}" for r in mdp.RADII
             for p in ("top", "ratio", "second", "cov")])
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    if OUTDIR:
        OUTDIR.mkdir(parents=True, exist_ok=True)
    old = list(csv.DictReader(open(OLD_CSV, encoding="utf-8-sig")))
    n_cells = (2 * mdp.SPAN // mdp.STEP + 1) ** 2
    base_forest = mdp.FOREST

    urban, n_urban = urban_only(LANDUSE, OUTDIR)
    cases = [("A 도시지역만", urban, base_forest),
             ("B 확정본 전체", LANDUSE, base_forest),
             ("C 농림과 관리를 산림으로", LANDUSE, base_forest + ADD_FOREST)]

    print(f"기존 위치 {len(old)}곳   훑는 칸 {n_cells}개 "
          f"(한 변 {2 * mdp.SPAN}m, 격자 {mdp.STEP}m)")
    print(f"도시지역만 남긴 폴리곤 {n_urban}건\n")

    for label, path, forest in cases:
        mdp.FOREST = forest
        zones, n_poly = mdp.load_zones(str(path))
        found, rows, short = select(zones)
        same, diff = compare(old, rows)

        print(f"--- {label}")
        print(f"    폴리곤 {n_poly}건, 용도 {len(zones)}종, "
              f"쓸 수 있는 칸 {len(found)}개 ({len(found) / n_cells:.0%})")
        for b, _, _ in mdp.BANDS:
            c = [f for f in found if f["band"] == b]
            print(f"    1순위 {b:3} {len(c):5}칸   산림과 주거가 섞인 칸 "
                  f"{sum(1 for f in c if f['mix']):4}개")
        if short:
            print(f"    모자란 구간 {short}")
        print(f"    기존과 좌표가 같은 위치 {same}곳, 값이 다른 칸 {diff}개")

        if label.startswith("B"):
            moved, uncovered = remeasure(zones, old)
            print(f"    기존 36곳을 다시 재면 구간이 바뀌는 곳 {len(moved)}개 {moved}")
            print(f"    반경 {max(mdp.RADII)}m 덮인 비율이 {mdp.MIN_COV} 미만인 "
                  f"위치 {len(uncovered)}개 {uncovered}")
            for k, v in count_mix(zones).items():
                print(f"    {k:22} {v:5}칸")
        print()

        if OUTDIR:
            write(rows, OUTDIR / f"positions_{label.split()[0]}.csv")

    mdp.FOREST = base_forest
    if OUTDIR:
        print("저장 ->", OUTDIR)
    else:
        # 출력폴더를 안 주면 도시지역만 남긴 파일이 확정본 옆에 생긴다. 팀 공용
        # 데이터 폴더에 찌꺼기를 두지 않도록 지운다.
        urban.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
