# -*- coding: utf-8 -*-
"""VLM 상황 판단(2단계)을 다시 잴 사진 목록을 71330 Validation 라벨에서 고른다.

평가셋 100장은 산불 사진이 모두 모래고개입구, 굴뚝연기와 구름이 모두 장현동이라 VLM 이
연기가 아니라 배경으로 가렸을 가능성이 남는다(docs/vlm_situation.md 한계). 장소를 섞은
목록을 따로 만든다. 평가셋(6.1)은 그대로 둔다.

    python tools/select_situation.py <VL.zip> [출력 csv]

기준
- 드론 촬영분만
- 영상(camera_id + video_id)당 1장
- 모래고개입구, 장현동은 뺀다 (평가셋에 이미 있다)
- 클래스마다 장소를 돌아가며 뽑아 한 장소에 몰리지 않게 한다
- 탐지 모델 학습과 검증에 쓴 영상도 넣되 seen 칸에 1로 적는다. 안 쓴 영상만 고르면 장소가
  클래스마다 한두 곳밖에 남지 않는다. VLM 은 이 사진으로 학습한 적이 없어 2단계 답에는
  영향이 없지만, 탐지와 검증기 통과 여부는 seen 0 인 사진으로만 봐야 한다.
  쓴 영상은 data/ 아래 사진 파일 이름에서 영상 번호를 읽어 안다. 장소 안에서는 안 쓴 영상을 먼저 고른다
- 난수 시드 42

출력은 docs/eval_set.csv 와 같은 칸이라 tools/extract_71330.py 로 그대로 꺼낼 수 있다.
"""
import csv
import json
import random
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXCLUDE_PLACES = ("모래고개입구", "장현동")
PLAN = [  # (DB, 클래스, 장수)
    ("PositiveRealDB", "흑색연기", 30),
    ("PositiveRealDB", "백색회색연기", 30),
    ("NegativeDB", "굴뚝연기", 30),
    ("NegativeDB", "구름", 30),
    ("NegativeDB", "안개연무", 30),
]
USED_DIRS = ["data/yolo_1260", "data/val_252", "data/large_smoke", "data/large_val",
             "data/yolo_mix", "data/yolo_mix240", "data/eval_100"]
VIDEO = re.compile(r"_([A-Z]{2}\d{6})_(\d{3})_")


def real_name(info):
    """zip 안 한글 이름을 되돌린다. 맥에서 묶은 것은 자모가 쪼개져 있어 합친다."""
    s = info.filename
    if not info.flag_bits & 0x800:
        for enc in ("utf-8", "cp949"):
            try:
                s = s.encode("cp437").decode(enc)
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
    return unicodedata.normalize("NFC", s)


def used_videos():
    used = set()
    for d in USED_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.suffix.lower() in (".jpg", ".png", ".json", ".txt"):
                m = VIDEO.search(p.name)
                if m:
                    used.add(m.groups())
    return used


def main(vl_zip, out):
    rng = random.Random(SEED)
    used = used_videos()
    print(f"이미 쓴 영상 {len(used)}개")

    rows = []
    devices = Counter()
    with zipfile.ZipFile(vl_zip) as zf:
        for info in zf.infolist():
            name = real_name(info)
            if not name.endswith(".json") or "__MACOSX" in name:
                continue
            base = name.rsplit("/", 1)[-1]
            parts = base.split("_")              # DB_클래스_장소_카메라_영상_발화원_번호.json
            if len(parts) < 7:
                continue
            db, cls = parts[0], parts[1]
            if (db, cls) not in {(d, c) for d, c, _ in PLAN}:
                continue
            lab = json.loads(zf.read(info))
            env = lab.get("environment", {})
            devices[env.get("device")] += 1
            if env.get("device") != "drone":
                continue
            rows.append({
                "file_name": base, "class": cls, "place": env.get("address", parts[2]),
                "split": "eval", "db": db, "smoke": int(db.startswith("Positive")),
                "camera_id": lab["info"]["camera_id"], "video_id": lab["info"]["video_id"],
                "image_file": lab["images"][0]["file_name"], "gps": env.get("gps", ""),
            })
    print("촬영 장비:", dict(devices))

    picked = []
    taken = set()
    for db, cls, n in PLAN:
        by_place = defaultdict(lambda: defaultdict(list))      # 장소 -> 영상 -> 사진
        for r in rows:
            v = (r["camera_id"], r["video_id"])
            if r["db"] == db and r["class"] == cls and v not in taken \
                    and not any(x in r["place"] for x in EXCLUDE_PLACES):
                by_place[r["place"]][v].append(r)
        places = sorted(by_place)
        rng.shuffle(places)
        got = []
        while len(got) < n and any(by_place[p] for p in places):
            for p in places:                    # 장소를 돌아가며 영상 하나씩
                if len(got) >= n or not by_place[p]:
                    continue
                fresh = sorted(v for v in by_place[p] if v not in used)
                v = rng.choice(fresh or sorted(by_place[p]))       # 안 쓴 영상을 먼저
                r = dict(rng.choice(sorted(by_place[p][v], key=lambda r: r["file_name"])))
                r["seen"] = int(v in used)
                got.append(r)
                taken.add(v)
                del by_place[p][v]
        print(f"{cls}: {len(got)}/{n}장, 장소 {len({r['place'] for r in got})}곳, "
              f"탐지 학습에 쓴 영상 {sum(r['seen'] for r in got)}장")
        picked += got

    assert len({(r["camera_id"], r["video_id"]) for r in picked}) == len(picked)
    with open(out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(picked[0].keys()))
        w.writeheader()
        w.writerows(picked)
    print(f"-> {out} ({len(picked)}장)")
    for cls in [c for _, c, _ in PLAN]:
        c = Counter(r["place"] for r in picked if r["class"] == cls)
        print(f"  {cls}: " + ", ".join(f"{p} {k}" for p, k in c.most_common()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "docs" / "situation_set.csv"))
