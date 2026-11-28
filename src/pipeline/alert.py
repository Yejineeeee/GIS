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
import contextlib
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
        self.chimney = {}      # alert_id -> VLM 이 굴뚝·설비 연기로 봤는지 (알람 형식에는 칸이 없어 여기 둔다)

    def send(self, alert):
        # 위험도는 판정, 반경 안 건물, 검증기, 불꽃(설명)에 따라 바뀐다. 낼 때마다 다시 계산한다
        if "risk" in alert:
            r = risk.assess(alert, chimney=self.chimney.get(alert["alert_id"], False))
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
        self.lock = None               # 파이프라인의 알람 잠금. 지금 알람을 읽고 낼 때 갱신과 엇갈리지 않게
        self.obs = {}                  # alert_id -> VLM 이 본 것. 같은 사건의 갱신에 다시 쓴다 (7.4)
        self.t_pub = {}                # detection_id -> 그 판을 낸 시각 (알람 → 설명 시간)
        self.q = queue.Queue()
        if not sync:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    def submit(self, alert, image_path, t_alert):
        self.t_pub[alert["detection_id"]] = t_alert
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
        """설명 하나를 만든다.

        11/28 (7.4): 연기 색, 퍼짐, 불꽃처럼 VLM 이 본 것은 같은 사건이면 갱신돼도 그대로다. 바뀌는 것은
        위치와 지도 계산 결과뿐이고, 문장은 템플릿이 그 둘을 합쳐 바로 만든다. 그래서
          - 사건마다 VLM 은 한 번만 부르고(self.obs), 갱신된 판은 그 관찰로 문장만 다시 만든다
          - VLM 을 부르는 동안 알람이 갱신됐으면 버리지 않고 지금 알람에 문장을 만들어 낸다
        전에는 갱신마다 VLM(CPU 에서 검증 11초 + 설명 10초)을 다시 불렀고, 끝날 즈음 또 갱신되면 버려
        드론이 다가가는 동안 설명이 1분 넘게 붙지 않았다 (docs/latency.md).
        """
        aid = alert["alert_id"]
        observed, vlm_ms, raw, reused = self.obs.get(aid), 0, None, aid in self.obs
        p_flame = flame_ms = chimney = chimney_probs = chimney_ms = None
        if reused:
            if self._stale(alert):     # 더 새 판의 작업이 뒤따른다. 그쪽에서 만든다
                return
        else:
            if image_path is None and self.vlm.mode == "qwen":
                self.pub.note(alert_id=aid, stage="설명", error="사진을 못 찾음 (--frames 확인)")
                print(f"     {aid} 설명 건너뜀: 사진을 못 찾음")
                return
            while self.yield_to and not self.sync and self.yield_to():
                time.sleep(0.1)
            if aid in self.obs:        # 기다리는 동안 앞선 작업이 관찰을 받아 왔다
                observed, reused = self.obs[aid], True
                if self._stale(alert):
                    return
            else:
                bbox = self.bbox_of(alert["detection_id"]) if self.bbox_of else None
                # 불꽃은 구조화 질문의 답을 쓰지 않고 같은 조각에 따로 묻는다 (docs/vlm_flame_obs.md).
                # 검증기와 같은 조각이라, 설명보다 먼저 물어야 llama-server 가 방금 본 조각을 다시 쓴다.
                # 불꽃과 굴뚝도 연기 색처럼 사건마다 한 번만 묻고 갱신된 판에는 다시 쓴다
                if bbox and image_path and hasattr(self.vlm, "p_flame"):
                    p_flame, flame_ms = self.vlm.p_flame(str(image_path), bbox)
                # 굴뚝은 사진 전체로 묻는다. 알람 형식(observed)에는 칸이 없어 위험도와 그 근거에만 반영한다
                if image_path and hasattr(self.vlm, "is_chimney"):
                    chimney, chimney_probs, chimney_ms = self.vlm.is_chimney(str(image_path))
                    if chimney is not None:
                        self.pub.chimney[aid] = chimney
                observed, vlm_ms, raw = self.vlm.observe(str(image_path) if image_path else None, bbox)
                if observed is None:
                    self.pub.note(alert_id=aid, stage="설명", error=f"스키마 실패: {raw[:200]}")
                    print(f"     {aid} 설명 실패: VLM 출력이 스키마와 다름")
                    return
                if p_flame is not None:
                    from vlm_client import FLAME_TAU
                    observed["flame_visible"] = p_flame >= FLAME_TAU
                vlm_ms = (vlm_ms or 0) + (flame_ms or 0) + (chimney_ms or 0)
                self.obs[aid] = observed
        # 지금 알람(그 사이 갱신됐으면 새 위치)에 문장을 만들어 낸다
        lock = self.lock or contextlib.nullcontext()
        with lock:
            cur = self.current(aid) if self.current else None
            target = copy.deepcopy(cur) if cur is not None else alert
            if target.get("description"):      # 이미 붙었다 (앞선 작업)
                return
            t_ref = self.t_pub.get(target["detection_id"], t_alert)
            desc = templates.build(dict(observed, chimney=bool(self.pub.chimney.get(aid))), target)
            target["observed"] = observed
            target["description"] = desc
            target["description_method"] = "구조화템플릿"
            target["latency_ms"]["vlm"] = vlm_ms
            target["latency_ms"]["alert_to_description"] = ms(t_ref)
            target["time"] = now_iso()
            self.pub.send(target)
            if cur is not None:        # 파이프라인이 들고 있는 알람에도 써 둔다 (다음에 다시 낼 때 지워지지 않게)
                for k in ("observed", "description", "description_method", "time", "risk"):
                    cur[k] = copy.deepcopy(target[k])
                cur["latency_ms"]["vlm"] = vlm_ms
                cur["latency_ms"]["alert_to_description"] = target["latency_ms"]["alert_to_description"]
        bad = verify.check(desc, target, observed)
        self.pub.note(alert_id=aid, stage="설명", violations=bad, raw=raw, reused_observation=reused,
                      described_detection=target["detection_id"], requested_detection=alert["detection_id"],
                      p_flame=None if p_flame is None else round(p_flame, 3), flame_ms=flame_ms,
                      chimney=chimney, situation=chimney_probs, chimney_ms=chimney_ms)
        self.done.append((aid, target["latency_ms"]["alert_to_description"], desc))
        print(f"     {aid} 설명 (알람 후 {target['latency_ms']['alert_to_description']}ms"
              + (", 관찰 재사용" if reused else "") + f"): {desc}")
        if not observed.get("smoke_present"):
            print(f"     {aid} VLM은 연기가 보이지 않는다고 답함 (헛탐지 가능성)")
        if bad:
            print(f"     {aid} 규칙 위반: {bad}")
