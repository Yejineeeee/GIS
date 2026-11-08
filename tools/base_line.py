# -*- coding: utf-8 -*-
"""연기 밑변 전체를 지면에 투사해 불이 걸친 구간을 판정한다 (번진 산불 대응 시험).

지금 파이프라인은 탐지 하나에서 밑동 한 점만 골라 좌표 하나와 오차 반경을 낸다
(src/geo/transform.py target_pixel). 산불이 번져 숲과 주거지에 걸치면, 그 한 점이
숲 쪽이면 "자연녹지 확정"이라고 말해 버린다. 여기서는 마스크 아래 경계선을 따라 점을
여러 개 뽑아 각각 지면에 투사하고(좌표 변환은 to_position 그대로), 점마다 판정한 뒤
구간 전체로 묶어 판정한다.

    python tools/base_line.py sim/frames_line runs/smoke_1260_960/weights/best.pt \
        --geo-dir data/geo_jagok --sim-terrain sim/terrain_jagok3d.json

밑변 점 뽑기
    탐지마다 마스크를 그리고, 열마다 가장 아래 픽셀을 밑변으로 본다. 바람에 날린 연기의
    아래쪽 가장자리는 밑변이 아니므로, 그 탐지 마스크 높이의 BAND 안(가장 낮은 곳에서)에
    드는 열만 쓴다. 남은 열에서 STEP 픽셀마다 한 점을 뽑는다.

구간 판정
    점마다 기존 규칙(verdict.judge)으로 판정한다.
    - 점들이 모두 같은 용도로 확정    -> 확정 (그 용도)
    - 확정된 점들의 용도가 둘 이상    -> 걸침 (용도들)   새 판정
    - 그 밖                          -> 점들의 반경을 합친 영역의 용도 비율로 기존 규칙
    한 점 방식(지금 파이프라인)은 conf 1등 탐지의 밑동 한 점으로 판정한다.

정답
    프레임 메타 truth.line 의 연기 자리마다 용도를 읽어, 걸친 용도 집합을 정답으로 본다.
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/geo", "src/pipeline"):
    sys.path.insert(0, str(ROOT / p))

from query import GeoData, circle_points, to_5179, query  # noqa: E402
from transform import to_position  # noqa: E402
from verdict import judge, threshold_for, AMBIGUOUS_SUM, MIN_COVERAGE  # noqa: E402

BAND = 0.15      # 탐지 마스크 높이의 아래 15% 안에 드는 열만 밑변으로 본다
STEP = 12        # 밑변에서 몇 픽셀마다 점을 뽑나
N_POINTS = 300   # 점 하나의 반경 안 조회 지점 수


def base_pixels(det, W, H):
    """탐지 하나의 밑변 점들 [(x, y)]."""
    import cv2
    poly = det.get("polygon")
    if not poly or len(poly) < 3:
        x, y, w, h = det["bbox"]
        return [(x + w / 2, y + h)]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    cols = np.where(m.any(0))[0]
    if len(cols) == 0:
        return []
    bottom = np.array([np.where(m[:, c])[0].max() for c in cols])
    top = min(p[1] for p in poly)
    keep = bottom >= bottom.max() - BAND * (bottom.max() - top)
    cols, bottom = cols[keep], bottom[keep]
    out = []
    for i in range(0, len(cols), STEP):
        out.append((float(cols[i]), float(bottom[i])))
    if len(cols) and (len(cols) - 1) % STEP:
        out.append((float(cols[-1]), float(bottom[-1])))
    return out


def project(frame, px, py, dem):
    det = {"polygon": None, "bbox": [px - 0.5, py - 1, 1, 1]}
    pos, info = to_position(frame, det, dem=dem)
    return pos, info


def area_ratios(geo, pts):
    """여러 점의 반경을 합친 영역의 용도 비율과 덮인 비율."""
    seen, names = set(), Counter()
    for p in pts:
        cx, cy = to_5179(p["lon"], p["lat"])
        for x, y in circle_points(cx, cy, p["error_radius_m"], N_POINTS):
            k = (round(x / 3), round(y / 3))          # 3m 칸으로 겹치는 지점을 한 번만 센다
            if k in seen:
                continue
            seen.add(k)
            names[geo.landuse_at(x, y)] += 1
    total = sum(names.values())
    cov = 1 - names.get(None, 0) / total if total else 0.0
    lu = [{"name": n, "ratio": round(c / total, 4)} for n, c in names.most_common() if n]
    return lu, cov


def span_verdict(geo, pts):
    """점마다 판정 -> 구간 판정. (판정, 용도들, 자세히)."""
    per = []
    for p in pts:
        lu, _, qi = query(geo, p["lat"], p["lon"], p["error_radius_m"])
        v, _ = judge(lu, p["error_radius_m"], qi.get("landuse_coverage", 1.0), p["shallow"])
        per.append((v, lu[0]["name"] if lu else None))
    sure = {n for v, n in per if v == "확정"}
    if per and all(v == "확정" for v, _ in per) and len(sure) == 1:
        return "확정", sorted(sure), per
    if len(sure) >= 2:
        return "걸침", sorted(sure), per
    lu, cov = area_ratios(geo, pts)
    r = max(p["error_radius_m"] for p in pts)
    v, _ = judge(lu, r, cov, any(p["shallow"] for p in pts))
    names = [u["name"] for u in lu[:2]] if v == "모호" else [lu[0]["name"]] if (v == "확정" and lu) else []
    return v, names, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("weights")
    ap.add_argument("--geo-dir", default="data/geo_jagok")
    ap.add_argument("--sim-terrain", required=True)
    ap.add_argument("--out", default="runs/base_line")
    a = ap.parse_args()

    from detect import Detector
    from sim_terrain import SimTerrain
    geo = GeoData(str(Path(a.geo_dir) / "landuse_5179.geojson"))
    dem = SimTerrain(a.sim_terrain)
    detector = Detector(a.weights)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, png in enumerate(sorted(glob.glob(str(Path(a.frames) / "sim_*.png")))):
        meta = json.load(open(png.replace(".png", ".json"), encoding="utf-8"))
        frame, _ = detector.run(Path(png), i)
        frame["camera"] = meta["camera"]
        W, H = meta["image_size"]["width"], meta["image_size"]["height"]
        truth = sorted({geo.landuse_at(*to_5179(p["lon"], p["lat"])) for p in meta["truth"]["line"]} - {None})
        dets = frame.get("detections") or []
        rec = {"image": Path(png).name, "D": meta["sweep"]["D"], "bearing": meta["sweep"]["bearing"],
               "truth": truth, "n_det": len(dets)}
        if not dets:
            rows.append(rec)
            print(rec["image"], rec["D"], rec["bearing"], "탐지 없음")
            continue
        # 한 점 방식: conf 1등의 밑동 한 점 (지금 파이프라인)
        top = max(dets, key=lambda d: d["score"])
        pos, info = to_position(frame, top, dem=dem)
        lu, _, qi = query(geo, pos["lat"], pos["lon"], pos["error_radius_m"])
        v1, _ = judge(lu, pos["error_radius_m"], qi.get("landuse_coverage", 1.0), info["shallow"])
        rec["one"] = {"verdict": v1, "names": [u["name"] for u in lu[:2]], "radius": pos["error_radius_m"]}
        # 밑변 방식: 모든 탐지의 밑변 점
        pts = []
        for d in dets:
            for px, py in base_pixels(d, W, H):
                try:
                    p, inf = project(frame, px, py, dem)
                except ValueError:
                    continue
                pts.append(dict(p, shallow=inf["shallow"], pixel=[px, py]))
        v2, names2, per = span_verdict(geo, pts)
        rec["span"] = {"verdict": v2, "names": names2, "n_points": len(pts),
                       "points": [[round(p["lat"], 6), round(p["lon"], 6), p["error_radius_m"]] for p in pts],
                       "per_point": per}
        rows.append(rec)
        print(rec["image"], rec["D"], rec["bearing"], "정답", truth, "| 한 점", v1, rec["one"]["names"][:1],
              "| 밑변", v2, names2, f"(점 {len(pts)})", flush=True)
    with open(out / "base_line.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
