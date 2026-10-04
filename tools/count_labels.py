# -*- coding: utf-8 -*-
"""AI Hub 71330 라벨 폴더에서 학습 계획에 필요한 숫자를 센다.

촬영 장비별 건수, GPS와 고도가 채워진 비율, 클래스별 객체 수와 이미지 수를
전체와 드론 촬영분으로 나눠 출력한다.

    python count_labels.py "라벨 폴더 경로"
"""
import sys
import os
import json
from collections import Counter

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

stat(loaded, "전체")

drone = []
for path, label in loaded:
    device = str(env_of(label).get("device", ""))
    if "드론" in device or "drone" in device.lower():
        drone.append((path, label))

stat(drone, "드론 촬영분만")
