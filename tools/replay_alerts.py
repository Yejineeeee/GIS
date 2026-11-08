# -*- coding: utf-8 -*-
"""파이프라인 결과를 한 사건의 알람이 갱신되는 것처럼 한 단계씩 다시 쓴다 (시연 영상용).

지금 파이프라인은 같은 사건으로 묶이면 첫 알람을 그대로 둔다. 시연에서는 드론이
다가갈수록 같은 알람의 반경과 판정이 바뀌는 장면을 보여 주고 싶다. 그래서
--no-merge 로 돌린 결과에서 프레임마다 conf 1등 탐지의 알람을 골라, 같은 alert_id
(a-0001)로 덮어쓴다. 대시보드(src/dashboard/server.py)는 이 폴더를 1초마다 다시 읽는다.

알람 갱신을 파이프라인에 정식으로 넣기 전까지 쓰는 재현용이다.

    python tools/replay_alerts.py results/pipeline_approach3 results/demo_replay --step 12
    python tools/replay_alerts.py results/pipeline_approach3 results/demo_replay --list
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "vlm"))
from templates import build  # noqa: E402


def steps(src):
    """프레임 번호 순서로 (프레임 번호, 알람, 탐지 프레임 JSON 경로)."""
    det_dir = src / "detections"
    by_det = {}
    for f in det_dir.glob("*.json"):
        fr = json.load(open(f, encoding="utf-8"))
        for d in fr.get("detections") or []:
            by_det[d["id"]] = (fr, f, d)
    best = {}
    for f in (src / "alerts").glob("*.json"):
        a = json.load(open(f, encoding="utf-8"))
        fr, path, d = by_det.get(a["detection_id"], (None, None, None))
        if fr is None:
            continue
        k = int(fr.get("frame_id"))
        if k not in best or d["score"] > best[k][2]["score"]:
            best[k] = (a, path, d)
    return [(k, best[k][0], best[k][1]) for k in sorted(best)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="run.py --no-merge 결과 폴더")
    ap.add_argument("dst", help="대시보드가 읽을 폴더")
    ap.add_argument("--step", type=int, help="이 순번(0부터)의 상태를 쓴다")
    ap.add_argument("--list", action="store_true", help="순번별 프레임, 판정, 반경을 보여 준다")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    S = steps(src)
    if a.list:
        for i, (k, al, _) in enumerate(S):
            print(i, k, al["location_verdict"], al["position"]["error_radius_m"])
        return
    (dst / "alerts").mkdir(parents=True, exist_ok=True)
    (dst / "detections").mkdir(parents=True, exist_ok=True)
    k, al, det_path = S[a.step]
    shutil.copy(det_path, dst / "detections" / det_path.name)
    al = dict(al, alert_id="a-0001")
    # VLM 없이 돌린 결과라 설명이 비어 있다. 같은 템플릿으로 문장을 만든다(VLM 사실은 쓰지 않음).
    # 설명까지 걸린 시간은 지어내지 않고 비워 둔다
    if not al.get("description"):
        al["description"] = build(al.get("observed") or {}, al)
        al["description_method"] = "구조화템플릿"
    tmp = dst / "alerts" / "a-0001.json.tmp"
    tmp.write_text(json.dumps(al, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dst / "alerts" / "a-0001.json")
    print(k, al["location_verdict"], al["position"]["error_radius_m"])


if __name__ == "__main__":
    main()
