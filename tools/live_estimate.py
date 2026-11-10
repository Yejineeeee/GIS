# -*- coding: utf-8 -*-
"""시뮬레이터 프레임 폴더를 지켜보며 새 장마다 위치를 추정해 알람을 쓴다 (시험용).

파이프라인의 프레임 폴더 지켜보기 모드(WBS 5.6)가 생기기 전까지, 시뮬레이터의
실시간 따라가기(sim/vworld_sim.js 의 vwPursue)를 시험하려고 만든 임시 도구다.
run.py 와 같은 함수(탐지, 좌표 변환, 지리 조회, 판정)를 쓰지만 설명(VLM), 같은 사건
묶기, 대시보드 지연 기록은 하지 않는다. 5.6 이 들어오면 run.py 로 바꾼다.

    python tools/live_estimate.py sim/frames_live results/live --geo-dir data/geo_jagok \\
        --sim-terrain sim/terrain_jagok3d.json

알람은 out/alerts/a-0001.json 하나를 장마다 덮어쓴다(같은 사건이 갱신되는 모양).
장별 기록은 out/frames/ 에 run.py --no-merge 결과와 같은 모양(alerts/, detections/)으로 남아
tools/drone_overlay.py, tools/make_demo_video.py 로 영상을 만들 수 있다.
판정은 hold 장 연속 같을 때만 바꾼다(tools/replay_alerts.py 의 held 와 같은 규칙).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/geo", "src/pipeline"):
    sys.path.insert(0, str(ROOT / p))

from query import GeoData, query  # noqa: E402
from transform import to_position  # noqa: E402
from verdict import judge  # noqa: E402

try:                                   # 밑동 가림 판단(occlusion.py)이 있으면 같이 쓴다
    from occlusion import adjust as occlusion_adjust  # noqa: E402
except ImportError:
    occlusion_adjust = None


def write_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    for i in range(20):                   # 윈도우에서는 서버가 읽는 중이면 바꾸기가 막힌다. 잠깐 기다렸다 다시
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="시뮬레이터가 프레임을 저장하는 폴더")
    ap.add_argument("out", help="알람을 쓸 폴더 (sim_server.py 세 번째 인자로 준다)")
    ap.add_argument("--weights", default=str(ROOT / "runs/smoke_1260_960/weights/best.pt"))
    ap.add_argument("--geo-dir", default=str(ROOT / "data/geo_jagok"))
    ap.add_argument("--sim-terrain", required=True)
    ap.add_argument("--hold", type=int, default=2, help="판정이 이만큼 연속으로 같아야 바꾼다")
    ap.add_argument("--idle", type=float, default=120, help="새 프레임이 이 초 동안 없으면 끝낸다")
    a = ap.parse_args()

    from detect import Detector
    from sim_terrain import SimTerrain
    dem = SimTerrain(a.sim_terrain)
    geo = GeoData(str(Path(a.geo_dir) / "landuse_5179.geojson"))
    det = Detector(a.weights)
    frames, out = Path(a.frames), Path(a.out)
    rec = out / "frames"                       # 장별 기록
    for d in (out / "alerts", out / "detections", rec / "alerts", rec / "detections"):
        d.mkdir(parents=True, exist_ok=True)

    done, shown, run, last_new = set(), None, [], time.time()
    print(f"{frames} 를 지켜본다. 알람은 {out / 'alerts'}", flush=True)
    while time.time() - last_new < a.idle:
        new = [p for p in sorted(frames.glob("sim_*.png"))
               if p.name not in done and p.with_suffix(".json").is_file()]
        if not new:
            time.sleep(0.3)
            continue
        for png in new:
            done.add(png.name)
            last_new = time.time()
            meta = json.load(open(png.with_suffix(".json"), encoding="utf-8"))
            fr, _ = det.run(png, len(done) - 1)
            write_json(rec / "detections" / (png.stem + ".json"), fr)
            write_json(out / "detections" / (png.stem + ".json"), fr)   # 대시보드가 탐지 사진을 찾는 곳
            dets = fr.get("detections") or []
            if not dets:
                print(png.name, "탐지 없음", flush=True)
                continue
            top = max(dets, key=lambda d: d["score"])
            try:
                pos, info = to_position(fr, top, dem=dem)
            except ValueError as e:
                print(png.name, "좌표 변환 실패", e, flush=True)
                continue
            if occlusion_adjust:
                pos, _ = occlusion_adjust(fr, info, pos, dem)
            lu, ev, qi = query(geo, pos["lat"], pos["lon"], pos["error_radius_m"])
            v, _ = judge(lu, pos["error_radius_m"], qi.get("landuse_coverage", 1.0), info.get("shallow", False))
            alert = {"schema_version": fr["schema_version"], "alert_id": "a-0001", "time": fr["time"],
                     "detection_id": top["id"], "position": pos, "location_verdict": v,
                     "land_use": lu, "evidence": ev, "frame": png.name}
            write_json(rec / "alerts" / (png.stem + ".json"), dict(alert, alert_id=f"a-{png.stem[4:]}"))
            # 판정은 hold 장 연속 같을 때만 바꾼다. 위치는 매번 최신으로 쓴다(드론이 따라간다)
            run = run + [v] if run and run[-1] == v else [v]
            if shown is None or (v != shown and len(run) >= a.hold):
                shown = v
            write_json(out / "alerts" / "a-0001.json", dict(alert, location_verdict=shown))
            print(png.name, f"{v}(보임 {shown}) 반경 {pos['error_radius_m']:.0f}m",
                  f"{pos['lat']:.6f}, {pos['lon']:.6f}", flush=True)
    print("새 프레임이 없어 끝낸다", flush=True)


if __name__ == "__main__":
    main()
