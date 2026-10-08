# -*- coding: utf-8 -*-
"""71330 드론 라벨에서 분할 모델 학습용 300장을 고른다.

사용법:
    python3 tools/select_71330.py <drone_env.csv> [출력 csv]

기준 (docs/data_select.md 참고)
- 시연 지역(장현동)과 촬영 1위 지역(솔향수목원) 위주로, 흑색연기와 백색회색연기를 섞는다
- 솔향수목원: 실제 발화 흑색연기 150장, 백색회색연기 120장. 영상(camera_id + video_id)당 1장
- 장현동: 백색회색연기 30장. 영상이 2개뿐이라 영상 안에서 간격을 두고 여러 장을 고른다
- 학습(train)과 검증(val)은 영상 단위로 나눠 같은 영상이 양쪽에 섞이지 않게 한다
- 같은 결과가 나오도록 난수 시드를 고정한다
"""
import csv, random, sys
from collections import defaultdict

SEED = 42
SOLHYANG = "강원도강릉시솔향수목원"
JANGHYEON = "강원도강릉시장현동"
VAL_RATIO = 0.2
# 장현동 영상 2개: 하나는 train, 하나는 val
JANGHYEON_PLAN = {("BD011025", "003"): ("train", 20), ("BD011003", "002"): ("val", 10)}


def cls_of(r):
    return r["file"].split("_")[1]


def db_of(r):
    return r["file"].split("_")[0]


def one_per_video(rows, n, rng):
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[(r["camera_id"], r["video_id"])].append(r)
    vids = sorted(by_vid)
    rng.shuffle(vids)
    if len(vids) < n:
        sys.exit(f"영상 {len(vids)}개로 {n}장을 고를 수 없음")
    chosen = [rng.choice(sorted(by_vid[v], key=lambda r: r["file"])) for v in vids[:n]]
    n_val = round(n * VAL_RATIO)
    return [(r, "val" if i < n_val else "train") for i, r in enumerate(chosen)]


def spaced(frames, n):
    frames = sorted(frames, key=lambda r: r["file"])
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def main(src, out):
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    picked = []

    for cls, n in [("흑색연기", 150), ("백색회색연기", 120)]:
        pool = [r for r in rows if r["address"] == SOLHYANG and cls_of(r) == cls
                and db_of(r) == "PositiveRealDB"]
        picked += one_per_video(pool, n, rng)

    jh = [r for r in rows if r["address"] == JANGHYEON and cls_of(r) == "백색회색연기"]
    for vid, (split, n) in JANGHYEON_PLAN.items():
        frames = [r for r in jh if (r["camera_id"], r["video_id"]) == vid]
        picked += [(r, split) for r in spaced(frames, n)]

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "class", "place", "split", "db", "camera_id", "video_id",
                    "image_file", "gps"])
        for r, split in sorted(picked, key=lambda t: (t[1], t[0]["address"], cls_of(t[0]), t[0]["file"])):
            w.writerow([r["file"], cls_of(r), r["address"], split, db_of(r),
                        r["camera_id"], r["video_id"], r["file"][:-5] + ".jpg", r["gps"]])
    print(f"{len(picked)}장 저장 -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "docs/select_71330.csv")
