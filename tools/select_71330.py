# -*- coding: utf-8 -*-
"""71330 드론 라벨에서 분할 모델 학습용 300장을 고른다.

사용법:
    python3 tools/select_71330.py <drone_env.csv> [출력 csv]

기준 (docs/data_select.md 참고)
- 드론 촬영분만. PositiveRealDB의 흑색연기·백색회색연기 270장 + NegativeDB(오탐 유발) 30장
- 영상(camera_id + video_id)당 1장. 뽑은 영상 집합을 PLAN 바깥에서 하나로 관리해
  클래스가 달라도 같은 영상이 두 번 뽑히지 않게 한다
- 시연·평가 후보 지역(모래고개입구, 장현동)은 제외한다
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


def vid(r):
    return (r["camera_id"], r["video_id"])


def main(src, out):
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    used = set()  # 전역으로 뽑은 영상 집합
    picked = []
    for db, cls, n in PLAN:
        pool = [r for r in rows
                if r["file"].split("_")[0] == db and r["file"].split("_")[1] == cls
                and r["address"] not in EXCLUDE_PLACES and vid(r) not in used]
        # 장소 -> 영상 -> 프레임
        by_place = defaultdict(lambda: defaultdict(list))
        for r in pool:
            by_place[r["address"]][vid(r)].append(r)
        places = sorted(by_place)
        iters = {}
        for p in places:
            vids = sorted(by_place[p])
            rng.shuffle(vids)
            iters[p] = iter([(v, by_place[p][v]) for v in vids])
        # 장소를 번갈아 돌며 영상 1개씩 뽑아 장소가 한쪽으로 쏠리지 않게 한다
        chosen = []
        while len(chosen) < n and iters:
            for p in list(iters):
                try:
                    v, frames = next(iters[p])
                except StopIteration:
                    del iters[p]
                    continue
                if v in used:
                    continue
                used.add(v)
                chosen.append(rng.choice(sorted(frames, key=lambda r: r["file"])))
                if len(chosen) == n:
                    break
        if len(chosen) < n:
            sys.exit(f"{db} {cls}: {n}장이 필요한데 {len(chosen)}장만 가능")
        rng.shuffle(chosen)
        n_val = round(n * VAL_RATIO)
        for i, r in enumerate(chosen):
            picked.append((r, "val" if i < n_val else "train", db, cls))

    files = [r["file"] for r, *_ in picked]
    assert len(files) == len(set(files)), "중복 사진"
    assert len({vid(r) for r, *_ in picked}) == len(picked), "같은 영상이 두 번 뽑힘"
    tr = {vid(r) for r, s, *_ in picked if s == "train"}
    va = {vid(r) for r, s, *_ in picked if s == "val"}
    assert not (tr & va), "train과 val에 같은 영상"

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "class", "place", "split", "db", "smoke",
                    "camera_id", "video_id", "image_file", "gps"])
        for r, split, db, cls in sorted(picked, key=lambda t: (t[1], t[3], t[0]["file"])):
            w.writerow([r["file"], cls, r["address"], split, db,
                        1 if db.startswith("Positive") else 0,
                        r["camera_id"], r["video_id"], r["file"][:-5] + ".jpg", r["gps"]])
    print(f"{len(picked)}장 저장 -> {out} (영상 {len(tr | va)}개)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "docs/select_71330.csv")
