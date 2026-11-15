# -*- coding: utf-8 -*-
"""지형 가림 판단(src/geo/occlusion.py)의 효과를 시뮬레이터 프레임으로 잰다.

같은 탐지(conf 1등)로 좌표를 두 번 낸다.
    지금     transform.to_position 그대로
    가림 반영 hidden_span 이 구간을 찾으면 위치를 교차점과 구간 끝의 가운데로 옮기고
              반경을 extra_radius_m 만큼 키운 뒤 다시 판정한다
정답과의 거리, 정답이 반경 안에 드는지, 판정을 비교한다.

    python tools/occlusion_eval.py sim/frames_occl --sim-terrain sim/terrain_jagok3d.json   # 밑동이 가려진 장면
    python tools/occlusion_eval.py sim/frames_jagok3d --sim-terrain sim/terrain_jagok3d.json # 밑동이 보이는 장면(헛판정 확인)
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/geo", "src/pipeline"):
    sys.path.insert(0, str(ROOT / p))

from occlusion import adjust  # noqa: E402
from query import GeoData, query, to_5179  # noqa: E402
from transform import to_position  # noqa: E402
from verdict import judge  # noqa: E402


def dist_m(la1, lo1, la2, lo2):
    x1, y1 = to_5179(lo1, la1)
    x2, y2 = to_5179(lo2, la2)
    return math.hypot(x1 - x2, y1 - y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--weights", default=str(ROOT / "runs/smoke_1260_960/weights/best.pt"))
    ap.add_argument("--geo-dir", default=str(ROOT / "data/geo_jagok"))
    ap.add_argument("--sim-terrain", required=True)
    ap.add_argument("--out", help="장면별 결과를 저장할 json")
    a = ap.parse_args()

    from detect import Detector
    from sim_terrain import SimTerrain
    dem = SimTerrain(a.sim_terrain)
    geo = GeoData(str(Path(a.geo_dir) / "landuse_5179.geojson"))
    det = Detector(a.weights)
    n = flagged = 0
    rows = []
    for i, png in enumerate(sorted(glob.glob(str(Path(a.frames) / "sim_*.png")))):
        meta = json.load(open(png.replace(".png", ".json"), encoding="utf-8"))
        fr, _ = det.run(Path(png), i)
        fr["camera"] = meta["camera"]
        dets = fr.get("detections") or []
        t = meta["truth"]
        if not dets:
            print(Path(png).name, "탐지 없음")
            continue
        n += 1
        top = max(dets, key=lambda d: d["score"])
        pos, info = to_position(fr, top, dem=dem)
        e0 = dist_m(pos["lat"], pos["lon"], t["fire_lat"], t["fire_lon"])
        lu, _, qi = query(geo, pos["lat"], pos["lon"], pos["error_radius_m"])
        v0, _ = judge(lu, pos["error_radius_m"], qi.get("landuse_coverage", 1.0), info["shallow"])
        fr["image_size"] = meta["image_size"]
        p2, span = adjust(fr, info, pos, dem)
        if span:
            flagged += 1
            lu2, _, qi2 = query(geo, p2["lat"], p2["lon"], p2["error_radius_m"])
            v1, _ = judge(lu2, p2["error_radius_m"], qi2.get("landuse_coverage", 1.0), info["shallow"])
        else:
            v1 = v0
        lat, lon, rad = p2["lat"], p2["lon"], p2["error_radius_m"]
        e1 = dist_m(lat, lon, t["fire_lat"], t["fire_lon"])
        row = {"image": Path(png).name, "D": meta["sweep"]["D"], "bearing": meta["sweep"]["bearing"],
               "now": {"err": round(e0, 1), "radius": pos["error_radius_m"], "in": e0 <= pos["error_radius_m"], "verdict": v0},
               "occl": {"err": round(e1, 1), "radius": round(rad, 1), "in": e1 <= rad, "verdict": v1, "span": span}}
        rows.append(row)
        print(row["image"], row["D"], row["bearing"],
              f"| 지금 오차 {e0:.0f}m 반경 {pos['error_radius_m']:.0f}m {'안' if row['now']['in'] else '밖'} {v0}",
              f"| 가림 {'있음' if span else '없음'}" + (f" 구간 {span['length_m']:.0f}m" if span else ""),
              f"| 반영 오차 {e1:.0f}m 반경 {rad:.0f}m {'안' if row['occl']['in'] else '밖'} {v1}", flush=True)
    if rows:
        k = lambda key: sum(r[key]["in"] for r in rows)
        print(f"\n탐지된 프레임 {n}, 가림 표시 {flagged}")
        print(f"정답이 반경 안: 지금 {k('now')}/{n}, 가림 반영 {k('occl')}/{n}")
        med = lambda key: sorted(r[key]["err"] for r in rows)[len(rows) // 2]
        print(f"정답과 거리 중앙값: 지금 {med('now'):.0f}m, 가림 반영 {med('occl'):.0f}m")
        for key, name in (("now", "지금"), ("occl", "가림 반영")):
            cnt = {v: sum(r[key]["verdict"] == v for r in rows) for v in ("확정", "모호", "미확정")}
            print(f"판정({name}): " + ", ".join(f"{v} {c}" for v, c in cnt.items()))
        if a.out:
            Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
