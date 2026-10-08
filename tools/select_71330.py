# -*- coding: utf-8 -*-
"""71330 드론 라벨에서 분할 모델 학습용 300장을 고른다.

사용법:
    python3 tools/select_71330.py <drone_env.csv> [출력 csv]

기준 (docs/data_select.md 참고)
- 드론 촬영분만, 실제 발화(PositiveRealDB)의 흑색연기·백색회색연기 270장 + 오탐 유발(NegativeDB) 30장
- 같은 영상의 프레임이 섞이지 않도록 영상(camera_id + video_id)당 1장만 고른다
- 시연·평가용으로 남겨둘 지역(모래고개입구, 장현동)은 제외한다
- 영상 단위로 학습(train) 80% / 검증(val) 20% 로 나눈다
- 같은 결과가 나오도록 난수 시드를 고정한다
"""
import csv, random, sys
from collections import defaultdict

SEED = 42
EXCLUDE_PLACES = {"강원도강릉시모래고개입구", "강원도강릉시장현동"}
PLAN = [  # (DB, 클래스, 장수)
    ("PositiveRealDB", "흑색연기", 135),
    ("PositiveRealDB", "백색회색연기", 135),
    ("NegativeDB", "구름", 10),
    ("NegativeDB", "안개연무", 10),
    ("NegativeDB", "굴뚝연기", 10),
]
VAL_RATIO = 0.2


def main(src, out):
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    picked = []
    for db, cls, n in PLAN:
        pool = [r for r in rows
                if r["file"].split("_")[0] == db and r["file"].split("_")[1] == cls
                and r["address"] not in EXCLUDE_PLACES]
        # 장소 -> 영상 -> 프레임
        by_place = defaultdict(lambda: defaultdict(list))
        for r in pool:
            by_place[r["address"]][(r["camera_id"], r["video_id"])].append(r)
        places = sorted(by_place)
        for p in places:
            vids = sorted(by_place[p])
            rng.shuffle(vids)
            by_place[p] = {v: by_place[p][v] for v in vids}
        # 장소를 번갈아 돌며 영상 1개씩 뽑아 장소가 한쪽으로 쏠리지 않게 한다
        chosen, iters = [], {p: iter(by_place[p].items()) for p in places}
        while len(chosen) < n and iters:
            for p in list(iters):
                try:
                    _, frames = next(iters[p])
                except StopIteration:
                    del iters[p]
                    continue
                frames = sorted(frames, key=lambda r: r["file"])
                chosen.append(rng.choice(frames))
                if len(chosen) == n:
                    break
        if len(chosen) < n:
            sys.exit(f"{db} {cls}: {n}장이 필요한데 {len(chosen)}장만 가능")
        rng.shuffle(chosen)
        n_val = round(n * VAL_RATIO)
        for i, r in enumerate(chosen):
            picked.append((r, "val" if i < n_val else "train", db, cls))

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label_file", "image_file", "split", "db", "class", "smoke",
                    "place", "camera_id", "video_id", "gps"])
        for r, split, db, cls in sorted(picked, key=lambda t: (t[1], t[3], t[0]["file"])):
            w.writerow([r["file"], r["file"][:-5] + ".jpg", split, db, cls,
                        1 if db.startswith("Positive") else 0,
                        r["address"], r["camera_id"], r["video_id"], r["gps"]])
    print(f"{len(picked)}장 저장 -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/select_71330.csv")
