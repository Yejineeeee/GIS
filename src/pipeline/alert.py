# -*- coding: utf-8 -*-
"""알람 JSON 만들기, 발행, 설명 붙이기 (WBS 5.4).

알람과 해설을 나눠서 낸다.
    1. 탐지 → 좌표 → 지리 조회 → 판정이 끝나면 설명 없이(description null) 바로 알람을 낸다
    2. 설명은 Describer 가 따로 도는 스레드에서 만들고, 끝나면 같은 alert_id 로 다시 낸다
VLM이 수십 초 걸려도 다음 프레임의 탐지와 알람은 기다리지 않는다.

형식은 docs/schema.md 알람(v0.4)을 따른다. 형식에 없는 값은 debug.jsonl 에 따로 남긴다.

v0.4 (11/19 회의)
    risk      위험도 {level, reason}. 알람을 낼 때마다 src/vlm/risk.py 로 다시 계산한다
    verifier  검증기(4.7)가 탐지 조각에서 본 결과 {p_smoke, passed}. 검증 전이나 검증기를 끄면 null
    evidence.nearest_building_m  가장 가까운 건물까지 거리 (지리 조회)
"""
import copy
import json
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import risk
import templates
import verify

SCHEMA_VERSION = "0.4"
KST = timezone(timedelta(hours=9))


def now_iso():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def ms(t0):
    return round((time.perf_counter() - t0) * 1000)


def make_alert(alert_id, det, position, verdict, land_use, rule, evidence, latency):
    """설명이 아직 없는 알람. latency 는 detect, geo_transform, geo_query, verdict,
    detect_to_alert 를 담은 dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "alert_id": alert_id,
        "time": now_iso(),
        "detection_id": det.get("id"),
        "position": position,
        "location_verdict": verdict,
        "land_use": land_use,
        "verdict_rule": rule,
        "evidence": evidence,
        "risk": None,
        "verifier": None,
        "observed": None,
        "description": None,
        "description_method": None,
        "latency_ms": {
            "detect": latency["detect"], "geo_transform": latency["geo_transform"],
            "geo_query": latency["geo_query"], "verdict": latency["verdict"], "vlm": None,
            "detect_to_alert": latency["detect_to_alert"], "alert_to_description": None,
        },
    }


class Publisher:
    """알람을 파일로 낸다. 대시보드(5.5)는 alerts/ 폴더를 읽는다. 스레드에서 같이 써도 된다."""

    def __init__(self, out):
        self.out = Path(out)
        for d in ("alerts", "detections"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        for old in (self.out / "alerts").glob("*.json"):   # 지난 실행의 알람이 섞이지 않게
            old.unlink()
        self.log = open(self.out / "alerts.jsonl", "w", encoding="utf-8")
        self.debug = open(self.out / "debug.jsonl", "w", encoding="utf-8")
        self.lock = threading.Lock()

    def send(self, alert):
        # 위험도는 판정, 반경 안 건물, 검증기, 불꽃(설명)에 따라 바뀐다. 낼 때마다 다시 계산한다
        if "risk" in alert:
            r = risk.assess(alert)
            alert["risk"] = {"level": r["level"], "reason": r["reason"]}
        with self.lock:
            path = self.out / "alerts" / f"{alert['alert_id']}.json"
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(alert, f, ensure_ascii=False, indent=2)
            tmp.replace(path)          # 대시보드가 반쯤 쓴 파일을 읽지 않게 한 번에 바꾼다
            self.log.write(json.dumps(alert, ensure_ascii=False) + "\n")
            self.log.flush()

    def frame(self, frame, name):
        """탐지 결과 프레임을 남긴다. 대시보드가 드론 위치와 사진을 찾을 때 쓴다."""
        with self.lock:
            with open(self.out / "detections" / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(frame, f, ensure_ascii=False, indent=2)

    def note(self, **kw):
        with self.lock:
            self.debug.write(json.dumps(kw, ensure_ascii=False) + "\n")
            self.debug.flush()


class Describer:
    """VLM 설명을 따로 도는 스레드에서 만든다.

    sync=True 면 스레드 없이 그 자리에서 만든다 (5.2 첫 연결 때 방식, 비교용).
    """

    def __init__(self, vlm, pub, sync=False):
        self.vlm, self.pub, self.sync = vlm, pub, sync
        self.done = []                 # (alert_id, 걸린 ms, 문장 또는 실패 사유)
        self.yield_to = None           # 참을 돌려주는 동안 설명을 미룬다 (같은 GPU 의 검증기 먼저)
        self.current = None            # alert_id -> 지금 알람(dict). 그 사이 갱신됐으면 옛 설명을 내지 않는다
        self.bbox_of = None            # detection_id -> bbox. 설명에 탐지 조각을 넣을 때 쓴다 (11/19 회의)
        self.q = queue.Queue()
        if not sync:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    def submit(self, alert, image_path, t_alert):
        job = (copy.deepcopy(alert), image_path, t_alert)
        if self.sync:
            self._work(*job)
        else:
            self.q.put(job)

    def pending(self):
        return self.q.qsize()

    def wait(self):
        if not self.sync:
            self.q.join()

    def _stale(self, alert):
        """이 설명 작업이 만들어진 뒤 같은 alert_id 가 다른 탐지로 갱신됐는가."""
        if not self.current:
            return False
        now = (self.current(alert["alert_id"]) or {}).get("detection_id")
        if now is not None and now != alert["detection_id"]:
            self.pub.note(alert_id=alert["alert_id"], stage="설명", skipped="알람이 갱신되어 옛 설명을 내지 않음",
                          detection_id=alert["detection_id"], current=now)
            return True
        return False

    def _loop(self):
        while True:
            job = self.q.get()
            try:
                self._work(*job)
            except Exception as e:     # 설명이 실패해도 알람은 이미 나가 있다
                self.pub.note(alert_id=job[0]["alert_id"], stage="설명", error=repr(e))
                print(f"     {job[0]['alert_id']} 설명 오류: {e!r}")
            finally:
                self.q.task_done()

    def _work(self, alert, image_path, t_alert):
        aid = alert["alert_id"]
        if image_path is None and self.vlm.mode == "qwen":
            self.pub.note(alert_id=aid, stage="설명", error="사진을 못 찾음 (--frames 확인)")
            print(f"     {aid} 설명 건너뜀: 사진을 못 찾음")
            return
        while self.yield_to and not self.sync and self.yield_to():
            time.sleep(0.1)
        if self._stale(alert):
            return
        bbox = self.bbox_of(alert["detection_id"]) if self.bbox_of else None
        observed, vlm_ms, raw = self.vlm.observe(str(image_path) if image_path else None, bbox)
        if observed is None:
            self.pub.note(alert_id=aid, stage="설명", error=f"스키마 실패: {raw[:200]}")
            print(f"     {aid} 설명 실패: VLM 출력이 스키마와 다름")
            return
        # 불꽃은 구조화 질문의 답을 쓰지 않고 같은 조각에 따로 묻는다 (docs/vlm_flame_obs.md)
        p_flame = None
        if bbox and image_path and hasattr(self.vlm, "p_flame"):
            from vlm_client import FLAME_TAU
            p_flame, flame_ms = self.vlm.p_flame(str(image_path), bbox)
            if p_flame is not None:
                observed["flame_visible"] = p_flame >= FLAME_TAU
                vlm_ms = (vlm_ms or 0) + (flame_ms or 0)
        cur = self.current(aid) if self.current else None
        if cur is not None:            # 설명을 기다리는 동안 검증기가 낸 결과를 문장과 알람에 반영한다
            alert["verifier"] = copy.deepcopy(cur.get("verifier"))
        desc = templates.build(observed, alert)
        alert["observed"] = observed
        alert["description"] = desc
        alert["description_method"] = "구조화템플릿"
        alert["latency_ms"]["vlm"] = vlm_ms
        alert["latency_ms"]["alert_to_description"] = ms(t_alert)
        alert["time"] = now_iso()
        if self._stale(alert):     # 설명을 만드는 동안 알람이 갱신됐다. 새 위치의 설명이 뒤따른다
            return
        self.pub.send(alert)
        if cur is not None:            # 파이프라인이 들고 있는 알람에도 설명을 써 둔다 (다음에 다시 낼 때 지워지지 않게)
            for k in ("observed", "description", "description_method", "time", "risk"):
                cur[k] = copy.deepcopy(alert[k])
            cur["latency_ms"]["vlm"] = alert["latency_ms"]["vlm"]
            cur["latency_ms"]["alert_to_description"] = alert["latency_ms"]["alert_to_description"]
        bad = verify.check(desc, alert, observed)
        self.pub.note(alert_id=aid, stage="설명", violations=bad, raw=raw,
                      p_flame=None if p_flame is None else round(p_flame, 3))
        self.done.append((aid, alert["latency_ms"]["alert_to_description"], desc))
        print(f"     {aid} 설명 (알람 후 {alert['latency_ms']['alert_to_description']}ms): {desc}")
        if not observed.get("smoke_present"):
            print(f"     {aid} VLM은 연기가 보이지 않는다고 답함 (헛탐지 가능성)")
        if bad:
            print(f"     {aid} 규칙 위반: {bad}")
