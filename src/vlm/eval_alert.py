# -*- coding: utf-8 -*-
"""평가셋 100장(6.1)을 생성 방식 비교(4.6)의 입력으로 바꾼다.

알람은 사실 목록(results/eval_facts.jsonl)에서 그대로 가져온다. 탐지 결과는
100장 모두 같은 값으로 고정한다.

탐지 값을 고정하는 이유는 정답이 입력으로 새지 않게 하려는 것이다. 라벨의 면적
비율을 그대로 넣으면 오탐 사진 30장 중 28장은 비율이 0 이라, 사진을 보지 않고도
연기가 없다고 답할 수 있다. 오탐 사진을 넣은 목적(연기 아님이라고 말하는지 보는
것)이 사라진다. 세 방식에 같은 탐지 값이 들어가므로 방식끼리 비교도 된다.

탐지 성능은 6.4 에서 따로 잰다. 여기서 재는 것은 같은 입력에 대해 세 생성 방식이
내는 설명이다.

위치는 평가셋이 가상으로 붙인 것이다(docs/eval_set.md). 사진 속 풍경과 관계가
없으므로 설명의 지리 부분은 사실 목록 기준으로만 채점한다.

    tools/run_pilot.py        방식 (i)
    tools/run_verify_retry.py 방식 (ii)
    tools/run_structured.py   방식 (iii)
"""
import json
import os
import unicodedata

# 100장에 똑같이 넣는 탐지 값
DET_SCORE = 0.90
DET_AREA_RATIO = 0.05
ALT_M = 120.0


def nfc(s):
    # 맥에서 만든 zip 은 한글 이름을 자모로 쪼개 두기도 해서(NFD) 비교 전에 맞춘다
    return unicodedata.normalize("NFC", s)


def has_smoke(fact):
    """연기 사진인지. 채점기(tools/score_descriptions.py)와 같은 기준이다."""
    return "연기" in fact.get("class", "")


def detection(name, i, size=None):
    """평가셋용 탐지 결과. 값은 100장 모두 같다."""
    w, h = size or (3840, 2160)
    return {"schema_version": "0.2", "video_id": "E000", "frame_id": i,
            "image": name, "image_size": {"width": w, "height": h},
            "detections": [{"id": f"E000-{i}-0", "class": "연기",
                            "score": DET_SCORE,
                            "bbox": None, "polygon": None, "area_px": None,
                            "area_ratio": DET_AREA_RATIO}],
            "camera": {"lat": None, "lon": None, "alt_m": ALT_M, "alt_ref": "AGL",
                       "heading_deg": None, "pitch_deg": None, "hfov_deg": None,
                       "source": "virtual"}}


def alert(fact, i):
    """사실 목록 한 줄을 알람 모양으로 바꾼다. 값은 그대로 쓴다."""
    return {"alert_id": f"e-{i:04d}",
            "detection_id": f"E000-{i}-0",
            "location_verdict": fact["verdict"],
            "position": {"lat": fact["lat"], "lon": fact["lon"],
                         "error_radius_m": fact["error_radius_m"]},
            "land_use": fact["land_use"],
            "evidence": {"buildings_in_radius": fact["buildings_in_radius"],
                         "slope_deg": fact["terrain_slope_deg"],
                         "risk_targets": fact["risk_targets"]},
            "description": None,
            "latency_ms": {}}


def load(images_dir, facts_path, with_size=True):
    """[(사진 경로, 파일 이름, 사실, 탐지, 알람), ...] 를 사실 목록 순서로 돌려준다."""
    on_disk = {nfc(n): n for n in os.listdir(images_dir)}
    items, missing = [], []
    with open(facts_path, encoding="utf-8") as f:
        facts = [json.loads(line) for line in f if line.strip()]

    for i, fact in enumerate(facts):
        real = on_disk.get(nfc(fact["image"]))
        if real is None:
            missing.append(fact["image"])
            continue
        path = os.path.join(images_dir, real)
        size = None
        if with_size:
            from PIL import Image
            with Image.open(path) as im:
                size = im.size
        items.append((path, fact["image"], fact,
                      detection(fact["image"], i, size), alert(fact, i)))

    if missing:
        raise SystemExit(f"{images_dir} 에 사진 {len(missing)}장이 없습니다: "
                         + ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else ""))
    return items


if __name__ == "__main__":
    import sys
    items = load(sys.argv[1], sys.argv[2])
    print(f"{len(items)}장, 연기 {sum(has_smoke(f) for _, _, f, _, _ in items)}장")
    for path, name, fact, det, alt in items[:2]:
        print(" ", name)
        print("  탐지", json.dumps(det["detections"][0], ensure_ascii=False))
        print("  알람", json.dumps(alt, ensure_ascii=False))
