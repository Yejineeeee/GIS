# -*- coding: utf-8 -*-
"""전체 파이프라인 첫 연결 (WBS 5.2).

탐지 결과 JSON → 좌표 변환(3.3) → 지리 조회(3.4) → 임시 판정 → 알람 발행
→ VLM 사실 추출 → 템플릿 문장 → 같은 알람 번호로 설명을 덧붙여 다시 발행

    # 1) 탐지 결과 만들기 (둘 중 하나)
    python tools/predict_seg.py 가중치.pt 이미지폴더 results/detections
    python tools/replay.py --images 이미지폴더 --out results/replay --no-sleep

    # 2) 파이프라인 실행
    python src/pipeline/run.py --detections results/detections --frames 이미지폴더 \
        --geo-dir data/geo_moraegogae --vlm fake

입력
    --detections  탐지 결과 JSON 폴더 (docs/schema.md 탐지 형식)
    --frames      사진 폴더. 탐지 JSON의 image 가 파일 이름만 있을 때 여기서 찾는다
    --geo-dir     landuse_5179.geojson, buildings_5179.geojson, *.img(DEM) 가 있는 폴더
                  (따로 줄 때는 --landuse, --buildings, --dem)
    --vlm         none(설명 없음) / fake(고정값, 연결 확인용) / qwen(실제 모델)

출력 (--out, 기본 results/pipeline)
    alerts/<alert_id>.json   알람의 최신 상태. 대시보드가 읽는 곳
    alerts.jsonl             발행 기록. 설명 전 알람과 설명 후 알람이 차례로 쌓인다
    debug.jsonl              형식에 없는 중간 값(대표 픽셀, 가정값, 조회 지점 수, 규칙 위반 등)
    끝나면 tools/check_schema.py 로 alerts/*.json 을 검사한다

이번 연결에서 정한 것
    - 판정은 임시 규칙이다(verdict.py). 3.6이 끝나면 judge()만 바꾼다
    - 같은 사건이 여러 프레임에 잡히면 알람을 하나만 낸다. 새 탐지 위치가 이미 낸
      알람의 오차 원과 겹치면 같은 사건으로 본다. 알람 갱신·해제는 향후 과제
    - 알람은 설명 없이 먼저 내고, 설명은 같은 알람 번호로 다시 낸다(5.4의 바탕).
      지금은 한 줄로 차례대로 돌므로 VLM이 도는 동안 다음 프레임은 기다린다
"""
import argparse
import glob
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "vlm"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from transform import to_position          # noqa: E402  3.3
from query import GeoData, query            # noqa: E402  3.4
from verdict import judge, RULE             # noqa: E402  임시 판정
from vlm_client import VLM                  # noqa: E402
import templates                            # noqa: E402  4.5
import verify                               # noqa: E402

SCHEMA_VERSION = "0.3"
KST = timezone(timedelta(hours=9))


def now_iso():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def ms(t0):
    return round((time.perf_counter() - t0) * 1000)


def dist_m(lat1, lon1, lat2, lon2):
    """두 위경도 사이 거리(m). 수 km 이내라 평면 근사로 충분하다."""
    dy = math.radians(lat2 - lat1) * 6371000
    dx = math.radians(lon2 - lon1) * 6371000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def load_frames(folder):
    files = sorted(glob.glob(str(Path(folder) / "*.json")))
    frames = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        if "detections" in d:
            frames.append(d)
    frames.sort(key=lambda d: (d.get("video_id", ""), d.get("frame_id", 0)))
    return frames


def find_image(frame, frames_dir):
    p = Path(frame.get("image", ""))
    if p.is_file():
        return p
    if frames_dir:
        q = Path(frames_dir) / p.name
        if q.is_file():
            return q
    return None


class Publisher:
    """알람을 파일로 낸다. 대시보드(5.5)는 alerts/ 폴더를 읽으면 된다."""

    def __init__(self, out):
        self.out = Path(out)
        (self.out / "alerts").mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "alerts.jsonl", "w", encoding="utf-8")
        self.debug = open(self.out / "debug.jsonl", "w", encoding="utf-8")

    def send(self, alert):
        with open(self.out / "alerts" / f"{alert['alert_id']}.json", "w", encoding="utf-8") as f:
            json.dump(alert, f, ensure_ascii=False, indent=2)
        self.log.write(json.dumps(alert, ensure_ascii=False) + "\n")
        self.log.flush()

    def note(self, **kw):
        self.debug.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.debug.flush()


def main():
    ap = argparse.ArgumentParser(description="전체 파이프라인 첫 연결 (WBS 5.2)")
    ap.add_argument("--detections", required=True, help="탐지 결과 JSON 폴더")
    ap.add_argument("--frames", help="사진 폴더 (VLM에 넣을 사진을 찾을 때)")
    ap.add_argument("--geo-dir", default="data/geo_moraegogae")
    ap.add_argument("--landuse", help="landuse_5179.geojson (기본: geo-dir 안)")
    ap.add_argument("--buildings", help="buildings_5179.geojson (기본: geo-dir 안, 없으면 건너뜀)")
    ap.add_argument("--dem", nargs="*", help="DEM IMG 파일들 (기본: geo-dir 안 *.img)")
    ap.add_argument("--vlm", default="none", choices=["none", "fake", "qwen"])
    ap.add_argument("--interval", type=float, default=0.0,
                    help="프레임 사이 대기(초). 실제 속도처럼 보려면 1~2")
    ap.add_argument("--no-merge", action="store_true", help="같은 사건 묶기를 끈다")
    ap.add_argument("--out", default="results/pipeline")
    a = ap.parse_args()

    # ---------- 준비 ----------
    geo_dir = Path(a.geo_dir)
    landuse = a.landuse or geo_dir / "landuse_5179.geojson"
    if not Path(landuse).is_file():
        sys.exit(f"용도지역 파일이 없다: {landuse}  (--geo-dir 나 --landuse 확인)")
    buildings = a.buildings or geo_dir / "buildings_5179.geojson"
    buildings = buildings if Path(buildings).is_file() else None
    dem = a.dem if a.dem is not None else sorted(glob.glob(str(geo_dir / "*.img")))

    t = time.perf_counter()
    geo = GeoData(str(landuse), str(buildings) if buildings else None, dem or None)
    print(f"[준비] 지리 자료 {ms(t)}ms: 용도지역 {len(geo.landuse)}개, "
          f"건물 {len(geo.buildings) if buildings else '없음'}, DEM {len(dem) if dem else '없음'}")

    t = time.perf_counter()
    vlm = VLM(a.vlm)
    print(f"[준비] VLM {a.vlm}" + (f" ({vlm.device}) {ms(t)}ms" if a.vlm == "qwen" else ""))
    if a.vlm == "fake":
        print("[주의] VLM fake: 고정된 사실을 넣는다. 연결 확인용이며 결과로 쓰지 않는다")

    frames = load_frames(a.detections)
    if not frames:
        sys.exit(f"탐지 결과 JSON이 없다: {a.detections}")
    pub = Publisher(a.out)
    print(f"[시작] 프레임 {len(frames)}개 → {a.out}\n")

    events = []          # 이미 낸 알람: (lat, lon, radius, alert_id)
    stats = {"frames": 0, "detections": 0, "alerts": 0, "merged": 0, "fail": 0,
             "verdict": {"확정": 0, "모호": 0, "미확정": 0}, "lat": []}

    # ---------- 프레임마다 ----------
    for frame in frames:
        stats["frames"] += 1
        t_frame = time.perf_counter()
        det_ms = (frame.get("latency_ms") or {}).get("detect") or 0
        for det in frame.get("detections") or []:
            stats["detections"] += 1

            # 3.3 좌표 변환
            t = time.perf_counter()
            try:
                position, tinfo = to_position(frame, det, dem=geo.dem)
            except ValueError as e:
                stats["fail"] += 1
                pub.note(detection_id=det.get("id"), stage="좌표 변환", error=str(e))
                print(f"  {det.get('id')}: 좌표 변환 실패 - {e}")
                continue
            t_tr = ms(t)

            # 같은 사건이면 알람을 새로 내지 않는다
            if not a.no_merge:
                same = next((ev for ev in events
                             if dist_m(ev[0], ev[1], position["lat"], position["lon"])
                             <= max(ev[2], position["error_radius_m"])), None)
                if same:
                    stats["merged"] += 1
                    pub.note(detection_id=det.get("id"), stage="묶음", alert_id=same[3])
                    continue

            # 3.4 지리 조회
            t = time.perf_counter()
            land_use, evidence, qinfo = query(geo, position["lat"], position["lon"],
                                              position["error_radius_m"])
            t_q = ms(t)

            # 판정 (임시)
            t = time.perf_counter()
            verdict, thr = judge(land_use, position["error_radius_m"],
                                 qinfo.get("landuse_coverage", 1.0), tinfo.get("shallow", False))
            t_v = ms(t)

            stats["alerts"] += 1
            alert_id = f"a-{stats['alerts']:04d}"
            alert = {
                "schema_version": SCHEMA_VERSION,
                "alert_id": alert_id,
                "time": now_iso(),
                "detection_id": det.get("id"),
                "position": position,
                "location_verdict": verdict,
                "land_use": land_use,
                "verdict_rule": {"rule": RULE, "threshold": thr, "samples": qinfo["n_points"]},
                "evidence": evidence,
                "observed": None,
                "description": None,
                "description_method": None,
                "latency_ms": {
                    "detect": det_ms, "geo_transform": t_tr, "geo_query": t_q,
                    "verdict": t_v, "vlm": None,
                    "detect_to_alert": det_ms + ms(t_frame),
                    "alert_to_description": None,
                },
            }
            pub.send(alert)
            t_alert = time.perf_counter()
            events.append((position["lat"], position["lon"], position["error_radius_m"], alert_id))
            stats["verdict"][verdict] += 1
            print(f"  {alert_id} [{verdict}] {det.get('id')}  반경 {position['error_radius_m']}m  "
                  f"1순위 {land_use[0]['name'] + ' ' + str(land_use[0]['ratio']) if land_use else '-'}  "
                  f"알람까지 {alert['latency_ms']['detect_to_alert']}ms")

            # VLM → 템플릿 → 같은 알람으로 다시 발행
            note = {"alert_id": alert_id, "transform": tinfo, "query": qinfo}
            if a.vlm != "none":
                img = find_image(frame, a.frames)
                if img is None and a.vlm == "qwen":
                    note["vlm"] = f"사진을 못 찾음: {frame.get('image')} (--frames 확인)"
                    print(f"     설명 건너뜀: {note['vlm']}")
                else:
                    observed, vlm_ms, raw = vlm.observe(str(img) if img else None)
                    if observed is None:
                        note["vlm"] = f"스키마 실패: {raw[:200]}"
                        print("     설명 실패: VLM 출력이 스키마와 다름")
                    else:
                        desc = templates.build(observed, alert)
                        alert["observed"] = observed
                        alert["description"] = desc
                        alert["description_method"] = "구조화템플릿"
                        alert["latency_ms"]["vlm"] = vlm_ms
                        alert["latency_ms"]["alert_to_description"] = ms(t_alert)
                        alert["time"] = now_iso()
                        pub.send(alert)
                        note["violations"] = verify.check(desc, alert, observed)
                        print(f"     설명 {alert['latency_ms']['alert_to_description']}ms: {desc}")
                        if note["violations"]:
                            print(f"     규칙 위반: {note['violations']}")
            stats["lat"].append(alert["latency_ms"])
            pub.note(**note)

        if a.interval > 0:
            time.sleep(a.interval)

    # ---------- 요약 ----------
    print("\n[요약]")
    print(f"  프레임 {stats['frames']}, 탐지 {stats['detections']}, 알람 {stats['alerts']} "
          f"(같은 사건으로 묶음 {stats['merged']}, 좌표 변환 실패 {stats['fail']})")
    print(f"  판정: " + ", ".join(f"{k} {v}" for k, v in stats["verdict"].items()))
    if stats["lat"]:
        for k in ("geo_transform", "geo_query", "verdict", "detect_to_alert", "alert_to_description"):
            vals = sorted(x[k] for x in stats["lat"] if x[k] is not None)
            if vals:
                print(f"  {k}: 중앙값 {vals[len(vals) // 2]}ms (최대 {vals[-1]}ms)")
    print(f"  결과: {a.out}/alerts/, {a.out}/alerts.jsonl, {a.out}/debug.jsonl")

    files = sorted(glob.glob(str(Path(a.out) / "alerts" / "*.json")))
    if files:
        print("\n[형식 검사]")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_schema.py"), *files],
                           capture_output=True, text=True, encoding="utf-8")
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr)
        if r.returncode:
            print(r.stdout)


if __name__ == "__main__":
    main()
