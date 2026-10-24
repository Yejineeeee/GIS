# -*- coding: utf-8 -*-
"""AI Hub 71330 라벨 폴더에서 학습 계획에 필요한 숫자를 센다.

촬영 장비별 건수, GPS와 고도가 채워진 비율, 클래스별 객체 수와 이미지 수를
전체와 드론 촬영분으로 나눠 출력한다.

학습 목록(2.2)은 같은 영상의 비슷한 프레임이 겹치지 않도록 영상 하나에서 1장만
쓴다. 그래서 학습에 쓸 수 있는 장수의 상한은 사진 수가 아니라 영상 수다. 분류별로
영상이 몇 개인지도 함께 센다.

    python count_labels.py "라벨 폴더 경로"
"""
import sys
import os
import json
from collections import Counter, defaultdict

# 시연과 평가에 쓸 지역은 학습에서 빼므로 상한에서도 뺀 값을 함께 낸다
EXCLUDE_PLACES = {"강원도강릉시모래고개입구", "강원도강릉시장현동"}

if len(sys.argv) < 2:
    raise SystemExit('사용법: python count_labels.py "라벨 폴더 경로"')

ROOT = sys.argv[1]
files = [os.path.join(dirpath, name)
         for dirpath, _, names in os.walk(ROOT)
         for name in names if name.lower().endswith(".json")]

print(f"라벨 파일: {len(files)}개")
if not files:
    raise SystemExit("json 파일을 찾지 못했습니다. 경로를 확인하세요.")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def env_of(label):
    """환경 정보가 객체로 올 때와 배열로 올 때를 모두 처리한다."""
    env = label.get("environment", {})
    if isinstance(env, list):
        env = env[0] if env else {}
    return env if isinstance(env, dict) else {}


def stat(items, title):
    device = Counter()
    place = Counter()
    per_class_obj = Counter()
    per_class_img = Counter()
    gps = elevation = 0

    for _, label in items:
        env = env_of(label)
        device[str(env.get("device", "")).strip() or "(없음)"] += 1
        place[str(env.get("address", "")).strip() or "(없음)"] += 1
        if str(env.get("gps", "")).strip():
            gps += 1
        if str(env.get("elevation", "")).strip():
            elevation += 1

        names = {c.get("id"): c.get("name") for c in label.get("categories", [])}
        seen = set()
        for ann in label.get("annotations", []):
            name = names.get(ann.get("category_id"), str(ann.get("category_id")))
            per_class_obj[name] += 1
            seen.add(name)
        for name in seen:
            per_class_img[name] += 1

    total = max(1, len(items))
    print(f"\n── {title} ({len(items)}개)")
    print("  촬영 장비:", dict(device.most_common()))
    print(f"  GPS 채워진 비율: {gps / total:.1%}   고도 채워진 비율: {elevation / total:.1%}")
    print("  클래스별 객체 수:", dict(per_class_obj.most_common()))
    print("  클래스별 이미지 수:", dict(per_class_img.most_common()))
    print("  촬영 장소 상위 10곳:")
    for name, count in place.most_common(10):
        print(f"    {name}: {count}")


loaded = []
for path in files:
    label = load(path)
    if label is not None:
        loaded.append((path, label))

def parts_of(path):
    """파일 이름에서 분류와 영상 번호를 꺼낸다.

    이름은 DB_클래스_장소_카메라_영상_발화원_프레임.json 꼴이다. 형식이 다르면
    None 을 돌려주고 세지 않는다.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    p = name.split("_")
    if len(p) < 5:
        return None
    return {"db": p[0], "cls": p[1], "place": p[2], "vid": (p[3], p[4])}


def video_stat(items, title):
    """분류별 영상 수를 센다. 영상당 1장만 쓰므로 이 값이 쓸 수 있는 장수의 상한이다."""
    vids = defaultdict(set)
    vids_keep = defaultdict(set)
    frames = Counter()
    bad = 0
    for path, _ in items:
        p = parts_of(path)
        if p is None:
            bad += 1
            continue
        key = (p["db"], p["cls"])
        vids[key].add(p["vid"])
        frames[key] += 1
        if p["place"] not in EXCLUDE_PLACES:
            vids_keep[key].add(p["vid"])

    all_v = set()
    for s in vids.values():
        all_v |= s
    print(f"\n── {title} 영상 수")
    if bad:
        print(f"  이름 형식이 달라 세지 못한 파일 {bad}개")
    print(f"  전체 영상 {len(all_v)}개, 사진 {sum(frames.values())}장 "
          f"(영상당 평균 {sum(frames.values()) / max(1, len(all_v)):.1f}장)")
    print(f"  {'분류':34}{'영상':>7}{'시연 지역 뺀 영상':>18}{'사진':>8}")
    for key in sorted(vids, key=lambda k: -len(vids[k])):
        label = f"{key[0]} {key[1]}"[:32]
        print(f"  {label:34}{len(vids[key]):>7}{len(vids_keep[key]):>18}"
              f"{frames[key]:>8}")
    print("  영상 하나에서 1장만 쓰므로, 시연 지역을 뺀 영상 수가 학습에 쓸 수 있는")
    print("  장수의 상한이다.")


stat(loaded, "전체")

drone = []
for path, label in loaded:
    device = str(env_of(label).get("device", ""))
    if "드론" in device or "drone" in device.lower():
        drone.append((path, label))

stat(drone, "드론 촬영분만")
video_stat(drone, "드론 촬영분만")
