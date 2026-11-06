# -*- coding: utf-8 -*-
"""전체 파이프라인 (WBS 5.2 첫 연결, 5.4 알람과 해설 분리).

사진 → 탐지 → 좌표 변환(3.3) → 지리 조회(3.4) → 판정 → 알람 발행(설명 없음)
                                                      └→ VLM 사실 추출 → 템플릿 문장
                                                         → 같은 alert_id로 설명을 붙여 다시 발행
설명은 따로 도는 스레드에서 만들기 때문에, VLM이 도는 동안에도 다음 프레임의 탐지와
알람은 멈추지 않는다(5.4). --sync 를 주면 예전처럼 차례대로 돈다.

    # 방법 1) 사진을 넣고 탐지부터 한 번에 (가중치 필요)
    python src/pipeline/run.py --frames 사진폴더 --weights best.pt \
        --geo-dir data/geo_moraegogae --vlm fake

    # 방법 2) 미리 만든 탐지 결과 JSON을 넣기
    python tools/replay.py --images 사진폴더 --out results/replay --no-sleep \
        --lat 37.73679 --lon 128.906239
    python src/pipeline/run.py --detections results/replay --frames 사진폴더 \
        --geo-dir data/geo_moraegogae --vlm fake

    # 대시보드(5.5)를 같이 켜 두면 알람이 지도에 바로 뜬다
    python src/dashboard/server.py --out results/pipeline

입력
    --weights     분할 모델 가중치. 주면 --frames 의 사진을 직접 탐지한다(detect.py).
                  사진 옆에 같은 이름의 JSON(시뮬레이터 프레임)이 있으면 그 카메라 값을 쓰고,
                  truth(정답 좌표)가 있으면 알람 위치가 정답에서 몇 m 떨어졌는지 함께 낸다
    --detections  탐지 결과 JSON 폴더 (docs/schema.md 탐지 형식). --weights 대신 쓴다
    --frames      사진 폴더
    --geo-dir     landuse_5179.geojson, buildings_5179.geojson, *.img(DEM) 가 있는 폴더
                  (따로 줄 때는 --landuse, --buildings, --dem)
    --vlm         none(설명 없음) / fake(고정값, 연결 확인용) / qwen(실제 모델)
    --no-verify   후보 검증기를 끈다 (아래)

출력 (--out, 기본 results/pipeline)
    detections/              프레임별 탐지 결과 JSON (대시보드가 드론 위치와 사진을 찾는 곳)
    alerts/<alert_id>.json   알람의 최신 상태. 대시보드가 읽는 곳
    alerts.jsonl             발행 기록. 설명 전 알람과 설명 후 알람이 차례로 쌓인다
    debug.jsonl              형식에 없는 중간 값(대표 픽셀, 가정값, 조회 지점 수, 규칙 위반 등)
    끝나면 tools/check_schema.py 로 alerts/*.json 을 검사한다

정한 것
    - 판정은 임시 규칙이다(verdict.py). 3.6이 끝나면 judge()만 바꾼다
    - 후보 검증기(4.7, docs/vlm_base_pick.md): VLM 이 있으면 알람을 YOLO conf 1등 탐지로
      먼저 낸 뒤, 따로 도는 스레드에서 탐지마다 조각을 잘라 연기인지 묻는다. P(예) 0.7
      이상인 탐지 중 conf 1등이 처음 것과 다르면 같은 alert_id 로 위치와 판정을 고쳐
      다시 낸다. 다른 자리에서 통과한 탐지는 새 알람이 된다. 설명은 검증이 끝난 뒤에
      만든다. 알람이 뜨는 시각은 검증기가 없을 때와 같고, 밭이나 비닐하우스 같은 헛탐지
      자리에 찍힌 알람은 몇 초 뒤 연기 자리로 옮겨진다
    - 같은 사건이 여러 프레임에 잡히면 알람을 하나만 낸다. 새 탐지 위치가 이미 낸
      알람의 오차 원과 겹치면 같은 사건으로 본다. 알람 갱신·해제는 향후 과제
"""
import argparse
import glob
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "vlm"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from transform import to_position          # noqa: E402  3.3
from query import GeoData, query            # noqa: E402  3.4
from verdict import judge, RULE             # noqa: E402  임시 판정
from vlm_client import VLM                  # noqa: E402
from alert import (Publisher, Describer,    # noqa: E402  5.4
                   make_alert, ms, now_iso)

VERIFY_TAU = 0.7     # 후보 검증기 문턱 (docs/vlm_base_pick.md, 시뮬레이터에서 정하고 실제 사진으로 확인)


class Worker:
    """일을 따로 도는 스레드에서 차례로 한다. sync=True 면 그 자리에서 한다."""

    def __init__(self, fn, sync=False):
        self.fn, self.sync, self.q = fn, sync, queue.Queue()
        if not sync:
            threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job):
        if self.sync:
            self.fn(job)
        else:
            self.q.put(job)

    def pending(self):
        return self.q.qsize()

    def busy(self):
        """대기 중이거나 하고 있는 일이 있는가."""
        return self.q.unfinished_tasks > 0

    def wait(self):
        if not self.sync:
            self.q.join()

    def _loop(self):
        while True:
            job = self.q.get()
            try:
                self.fn(job)
            except Exception as e:      # 검증이 실패해도 알람은 이미 나가 있다
                print(f"  검증 오류: {e!r}")
            finally:
                self.q.task_done()


def dist_m(lat1, lon1, lat2, lon2):
    """두 위경도 사이 거리(m). 수 km 이내라 평면 근사로 충분하다."""
    dy = math.radians(lat2 - lat1) * 6371000
    dx = math.radians(lon2 - lon1) * 6371000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def load_frames(folder):
    out = []
    for f in sorted(glob.glob(str(Path(folder) / "*.json"))):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        if "detections" in d:
            out.append((d, Path(f).stem))
    out.sort(key=lambda x: (x[0].get("video_id", ""), x[0].get("frame_id", 0)))
    return out


def find_image(frame, frames_dir):
    p = Path(frame.get("image", ""))
    if p.is_file():
        return p
    if frames_dir:
        q = Path(frames_dir) / p.name
        if q.is_file():
            return q
    return None


def main():
    ap = argparse.ArgumentParser(description="전체 파이프라인 (WBS 5.2, 5.4)")
    ap.add_argument("--detections", help="탐지 결과 JSON 폴더 (--weights 대신)")
    ap.add_argument("--weights", help="분할 모델 가중치(best.pt). 주면 --frames 사진을 직접 탐지")
    ap.add_argument("--frames", help="사진 폴더")
    ap.add_argument("--geo-dir", default="data/geo_moraegogae")
    ap.add_argument("--landuse", help="landuse_5179.geojson (기본: geo-dir 안)")
    ap.add_argument("--buildings", help="buildings_5179.geojson (기본: geo-dir 안, 없으면 건너뜀)")
    ap.add_argument("--dem", nargs="*", help="DEM IMG 파일들 (기본: geo-dir 안 *.img)")
    ap.add_argument("--vlm", default="none", choices=["none", "fake", "qwen"])
    ap.add_argument("--fake-sec", type=float, default=0.0,
                    help="fake VLM이 걸리는 시간(초). 분리 출력을 시험할 때 (예: 5)")
    ap.add_argument("--sync", action="store_true", help="설명을 기다렸다가 다음 프레임 (5.2 방식)")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="프레임 사이 대기(초). 실제 속도처럼 보려면 1~2")
    ap.add_argument("--no-merge", action="store_true", help="같은 사건 묶기를 끈다")
    ap.add_argument("--no-verify", action="store_true", help="후보 검증기를 끈다 (VLM 이 있을 때만 켜짐)")
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
    vlm = VLM(a.vlm, fake_sec=a.fake_sec)
    print(f"[준비] VLM {a.vlm}" + (f" ({vlm.device}) {ms(t)}ms" if a.vlm == "qwen" else "")
          + (f", 설명 {'차례대로' if a.sync else '따로 도는 스레드에서'}" if a.vlm != "none" else ""))
    if a.vlm == "fake":
        print("[주의] VLM fake: 고정된 사실을 넣는다. 연결 확인용이며 결과로 쓰지 않는다")

    pub = Publisher(a.out)
    describer = Describer(vlm, pub, sync=a.sync) if a.vlm != "none" else None

    if a.weights:
        if not a.frames:
            sys.exit("--weights 를 쓰면 --frames 사진 폴더가 필요하다")
        from detect import Detector
        t = time.perf_counter()
        detector = Detector(a.weights)
        print(f"[준비] 탐지 모델 {ms(t)}ms (imgsz {detector.imgsz}, conf {detector.conf})")
        images = sorted(p for p in Path(a.frames).iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not images:
            sys.exit(f"사진이 없다: {a.frames}")

        def source():
            for i, img in enumerate(images):
                frame, truth = detector.run(img, i)
                yield frame, truth, img.stem
        n_frames = len(images)
    elif a.detections:
        frames = load_frames(a.detections)
        if not frames:
            sys.exit(f"탐지 결과 JSON이 없다: {a.detections}")

        def source():
            for fr, stem in frames:
                yield fr, None, stem
        n_frames = len(frames)
    else:
        sys.exit("--weights(+ --frames) 나 --detections 중 하나가 필요하다")
    print(f"[시작] 프레임 {n_frames}개 → {a.out}\n")

    events = []          # 이미 낸 알람: [lat, lon, radius, alert_id]
    verify = a.vlm != "none" and not a.no_verify
    if verify:
        print(f"[준비] 후보 검증기 켬: 알람은 conf 1등으로 먼저 내고, P(예) {VERIFY_TAU} 이상인 "
              f"탐지로 위치를 고쳐 다시 낸다" + (" (차례대로)" if a.sync else " (따로 도는 스레드에서)"))
    stats = {"frames": 0, "detections": 0, "alerts": 0, "merged": 0, "fail": 0, "verify_skip": 0,
             "relocated": 0, "verify_saved": 0, "verdict": {"확정": 0, "모호": 0, "미확정": 0}, "truth_err": {}}
    lock = threading.Lock()      # events, stats 를 검증 스레드와 같이 쓴다
    t_run = time.perf_counter()

    def transform(frame, det):
        """(position, tinfo, 걸린 ms) 또는 실패하면 None."""
        t = time.perf_counter()
        try:
            position, tinfo = to_position(frame, det, dem=geo.dem)
        except ValueError as e:
            with lock:
                stats["fail"] += 1
            pub.note(detection_id=det.get("id"), stage="좌표 변환", error=str(e))
            print(f"  {det.get('id')}: 좌표 변환 실패 - {e}")
            return None
        return position, tinfo, ms(t)

    def truth_error(position, truth):
        if not truth:
            return None
        return round(dist_m(position["lat"], position["lon"], truth["fire_lat"], truth["fire_lon"]), 1)

    def lookup(position, tinfo):
        """3.4 지리 조회와 판정."""
        t = time.perf_counter()
        land_use, evidence, qinfo = query(geo, position["lat"], position["lon"], position["error_radius_m"])
        t_q = ms(t)
        t = time.perf_counter()
        verdict, thr = judge(land_use, position["error_radius_m"],
                             qinfo.get("landuse_coverage", 1.0), tinfo.get("shallow", False))
        return land_use, evidence, qinfo, verdict, thr, t_q, ms(t)

    def new_alert(frame, det, stem, truth, t_frame, det_ms, image, describe=True):
        """탐지 하나로 알람을 낸다. 같은 사건이면 None, 아니면 (alert, t_alert)."""
        tr = transform(frame, det)
        if tr is None:
            return None
        position, tinfo, t_tr = tr
        err = truth_error(position, truth)
        with lock:              # 묶음 확인과 자리 잡기를 한 번에 해야 두 스레드가 겹쳐 내지 않는다
            same = None if a.no_merge else next(
                (ev for ev in events if dist_m(ev[0], ev[1], position["lat"], position["lon"])
                 <= max(ev[2], position["error_radius_m"])), None)
            if same:
                stats["merged"] += 1
            else:
                stats["alerts"] += 1
                alert_id = f"a-{stats['alerts']:04d}"
                events.append([position["lat"], position["lon"], position["error_radius_m"], alert_id])
        if same:
            pub.note(detection_id=det.get("id"), stage="묶음", alert_id=same[3], truth_error_m=err)
            print(f"  {det.get('id')}: {same[3]}과 같은 사건으로 묶음"
                  + (f" (정답과 {err}m)" if err is not None else ""))
            return None

        land_use, evidence, qinfo, verdict, thr, t_q, t_v = lookup(position, tinfo)
        alert = make_alert(
            alert_id, det, position, verdict, land_use,
            {"rule": RULE, "threshold": thr, "samples": qinfo["n_points"]}, evidence,
            {"detect": det_ms, "geo_transform": t_tr, "geo_query": t_q, "verdict": t_v,
             "detect_to_alert": det_ms + ms(t_frame)})
        pub.send(alert)
        t_alert = time.perf_counter()
        with lock:
            stats["verdict"][verdict] += 1
            if err is not None:
                stats["truth_err"][alert_id] = (err, position["error_radius_m"])
        print(f"  {alert_id} [{verdict}] {det.get('id')}  반경 {position['error_radius_m']}m  "
              f"1순위 {land_use[0]['name'] + ' ' + str(land_use[0]['ratio']) if land_use else '-'}  "
              f"알람까지 {alert['latency_ms']['detect_to_alert']}ms"
              + (f"  정답과 {err}m" if err is not None else ""))
        pub.note(alert_id=alert_id, stage="알람", transform=tinfo, query=qinfo,
                 truth_error_m=err, score=det.get("score"), frame=stem,
                 p_smoke=None if det.get("_p_smoke") is None else round(det["_p_smoke"], 4),
                 p_smoke_ms=det.get("_p_ms"))
        if describe and describer:            # 설명은 따로 (5.4)
            describer.submit(alert, image, t_alert)
        return alert, t_alert

    def relocate(alert, frame, det, truth, t_alert):
        """검증기가 고른 탐지로 같은 alert_id 의 위치와 판정을 고쳐 다시 낸다."""
        tr = transform(frame, det)
        if tr is None:
            return
        position, tinfo, t_tr = tr
        err = truth_error(position, truth)
        land_use, evidence, qinfo, verdict, thr, t_q, t_v = lookup(position, tinfo)
        aid, old_det, old_verdict = alert["alert_id"], alert["detection_id"], alert["location_verdict"]
        alert.update({"time": now_iso(), "detection_id": det.get("id"), "position": position,
                      "location_verdict": verdict, "land_use": land_use, "evidence": evidence,
                      "verdict_rule": {"rule": RULE, "threshold": thr, "samples": qinfo["n_points"]}})
        alert["latency_ms"].update({"geo_transform": t_tr, "geo_query": t_q, "verdict": t_v})
        with lock:
            for ev in events:
                if ev[3] == aid:
                    ev[:3] = [position["lat"], position["lon"], position["error_radius_m"]]
            stats["verdict"][old_verdict] -= 1
            stats["verdict"][verdict] += 1
            stats["relocated"] += 1
            if err is not None:
                stats["truth_err"][aid] = (err, position["error_radius_m"])
        pub.send(alert)
        pub.note(alert_id=aid, stage="검증기 위치 수정", from_detection=old_det, to_detection=det.get("id"),
                 p_smoke=round(det["_p_smoke"], 4), transform=tinfo, query=qinfo, truth_error_m=err,
                 alert_to_relocate_ms=ms(t_alert))
        print(f"  {aid} 위치 수정 [{old_verdict}→{verdict}] {old_det} → {det.get('id')}  "
              f"반경 {position['error_radius_m']}m  알람 후 {ms(t_alert)}ms"
              + (f"  정답과 {err}m" if err is not None else ""))

    def check(job):
        """후보 검증기 (4.7). 알람을 낸 뒤에 돈다."""
        frame, dets, top, first, image, stem, truth, t_frame, det_ms = job
        # 이미 낸 알람과 같은 사건으로 묶일 탐지는 알람을 바꾸지 못하므로 묻지 않는다.
        # 큰 연기를 여러 조각으로 쪼갠 것이 대부분 여기서 빠진다
        cands = [top] if first else []
        for d in dets:
            if d is top:
                continue
            tr = transform(frame, d)
            if tr is None:
                continue
            with lock:
                same = None if a.no_merge else next(
                    (ev for ev in events if dist_m(ev[0], ev[1], tr[0]["lat"], tr[0]["lon"])
                     <= max(ev[2], tr[0]["error_radius_m"])), None)
                if same:
                    stats["verify_saved"] += 1
            if same:
                pub.note(detection_id=d.get("id"), stage="묶음, 검증 생략", alert_id=same[3])
            else:
                cands.append(d)
        for d in cands:
            d["_p_smoke"], d["_p_ms"] = vlm.p_smoke(image, d["bbox"])
        passed = sorted((d for d in cands if d["_p_smoke"] >= VERIFY_TAU), key=lambda d: -d["score"])
        for d in cands:
            if d not in passed:
                with lock:
                    stats["verify_skip"] += 1
                pub.note(detection_id=d.get("id"), stage="검증기 미통과", p_smoke=round(d["_p_smoke"], 4),
                         p_smoke_ms=d.get("_p_ms"), score=d.get("score"))
                print(f"  {d.get('id')}: 검증기 미통과 (P(예) {d['_p_smoke']:.2f}, conf {d['score']:.2f})")
        rest = [d for d in passed if d is not top]
        if first:
            alert, t_alert = first
            if passed and passed[0] is not top:          # conf 1등이 헛탐지로 걸러졌다
                relocate(alert, frame, passed[0], truth, t_alert)
                rest = passed[1:]
            if describer:
                describer.submit(alert, image, t_alert)
        for d in rest:                                   # 다른 자리에서 통과한 연기
            new_alert(frame, d, stem, truth, t_frame, det_ms, image)

    checker = Worker(check, sync=a.sync) if verify else None
    if checker and describer:
        describer.yield_to = checker.busy    # 같은 GPU 를 쓰므로 위치를 고치는 검증을 먼저 한다

    # ---------- 프레임마다 ----------
    for frame, truth, stem in source():
        stats["frames"] += 1
        t_frame = time.perf_counter()
        pub.frame(frame, stem)
        det_ms = (frame.get("latency_ms") or {}).get("detect") or 0
        if not frame.get("detections"):
            print(f"  프레임 {frame.get('frame_id')}: 탐지 없음 ({det_ms}ms)")
        dets = list(frame.get("detections") or [])
        stats["detections"] += len(dets)
        image = find_image(frame, a.frames) if dets else None
        if checker and dets and image:
            # 알람은 YOLO conf 1등으로 바로 내고, 검증기는 뒤에서 돌린다
            top = max(dets, key=lambda d: d["score"])
            first = new_alert(frame, top, stem, truth, t_frame, det_ms, image, describe=False)
            checker.submit((frame, dets, top, first, image, stem, truth, t_frame, det_ms))
        else:
            for det in dets:
                new_alert(frame, det, stem, truth, t_frame, det_ms, image)

        if a.interval > 0:
            time.sleep(a.interval)

    t_alerts_done = ms(t_run)
    if checker and not a.sync and checker.pending():
        print(f"\n[대기] 남은 검증 {checker.pending()}건을 기다린다")
    if checker:
        checker.wait()
    if describer and not a.sync and describer.pending():
        print(f"\n[대기] 남은 설명 {describer.pending()}건을 기다린다")
    if describer:
        describer.wait()

    # ---------- 요약 ----------
    alerts = []
    for f in sorted(glob.glob(str(Path(a.out) / "alerts" / "*.json"))):
        with open(f, encoding="utf-8") as fp:
            alerts.append(json.load(fp))
    print("\n[요약]")
    print(f"  프레임 {stats['frames']}, 탐지 {stats['detections']}, 알람 {stats['alerts']} "
          f"(같은 사건으로 묶음 {stats['merged']}, 좌표 변환 실패 {stats['fail']}, "
          f"검증기 미통과 {stats['verify_skip']}, 검증기로 위치 수정 {stats['relocated']}, "
          f"묶여서 검증 생략 {stats['verify_saved']})")
    print("  판정: " + ", ".join(f"{k} {v}" for k, v in stats["verdict"].items()))
    print(f"  마지막 알람까지 {t_alerts_done}ms, 설명까지 포함 {ms(t_run)}ms")
    for k in ("geo_transform", "geo_query", "verdict", "detect_to_alert", "vlm", "alert_to_description"):
        vals = sorted(x["latency_ms"][k] for x in alerts if x["latency_ms"].get(k) is not None)
        if vals:
            print(f"  {k}: 중앙값 {vals[len(vals) // 2]}ms (최대 {vals[-1]}ms)")
    n_desc = sum(1 for x in alerts if x.get("description"))
    if describer:
        print(f"  설명 붙은 알람 {n_desc}/{len(alerts)}")
    for aid, (err, rad) in sorted(stats["truth_err"].items()):
        print(f"  {aid}: 정답과 {err}m, 오차 반경 {rad}m → {'반경 안' if err <= rad else '반경 밖'}")
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
