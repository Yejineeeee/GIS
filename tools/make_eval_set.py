# -*- coding: utf-8 -*-
"""평가셋 100장과 사진별 사실 목록을 자동으로 만든다 (WBS 6.1).

    python3 tools/make_eval_set.py <VL.zip> <landuse_5179.geojson> <buildings_5179.geojson> \
        <37808.img> <37804.img> [--positions docs/demo_positions.csv]

고르는 법 (학습 1,260장과 겹치지 않게 시연 지역에서만 뽑는다)
    - 연기 70장: 모래고개입구 PositiveRealDB 흑색연기 30, 백색회색연기 40
    - 오탐 유발 30장: 장현동 NegativeDB 구름 15, 굴뚝연기 15
    - 드론 촬영분만, 영상(camera_id + video_id)당 1장, 난수 시드 42
    - 모자라면 같은 묶음(연기/오탐)의 다른 클래스로 채운다

사진마다 붙이는 것
    - 가상 위치: docs/demo_positions.csv 36곳을 돌아가며 붙인다 (높음·중간·낮음 고르게)
    - 오차 반경: 10, 30, 50, 100m를 돌아가며 붙인다 (3.5와 같은 4단계)
    - 라벨에서: 클래스, 화염 여부, 연기 면적 비율
    - 지리 조회(3.4 query.py)에서: 반경 안 용도 비율, 건물 수, 지형 경사
    - 판정: 3.5 본 실험 최다비율 규칙, 정확도 95% 임계값 (docs/threshold.md)

출력
    results/eval_facts.jsonl   채점기(tools/score_descriptions.py)가 읽는 정답 파일
    docs/eval_set.csv          평가셋 목록. tools/extract_71330.py 로 사진을 꺼낼 때 쓴다
    docs/eval_set.md           평가셋 설명 문서 (실행 결과 숫자가 들어감)

채점기와 맞추려고 바꾼 이름 (원래 이름은 label_class 칸에 남긴다)
    - 백색회색연기 → 백색및회색연기 (채점기의 연기 색 기준 이름)
    - 굴뚝연기 → 굴뚝 (채점기는 이름에 '연기'가 있으면 연기 사진으로 보기 때문)
"""
import argparse
import csv
import json
import random
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "geo"))
from query import GeoData, query  # noqa: E402

SEED = 42
PLAN = [  # (묶음, 장소, DB, 클래스, 장수)
    ("연기", "모래고개입구", "PositiveRealDB", "흑색연기", 30),
    ("연기", "모래고개입구", "PositiveRealDB", "백색회색연기", 40),
    ("오탐", "장현동", "NegativeDB", "구름", 15),
    ("오탐", "장현동", "NegativeDB", "굴뚝연기", 15),
]
RADII = [10, 30, 50, 100]
THRESHOLD = {10: 0.00, 30: 0.57, 50: 0.72, 100: 0.80}  # 3.5 최다비율, 정확도 95%
AMBIG_SUM = 0.9                                          # 1·2순위 합이 이 이상이면 모호
SCORER_CLASS = {"백색회색연기": "백색및회색연기", "굴뚝연기": "굴뚝"}
SMOKE_CATS = {"흑색연기", "백색/회색연기"}
FLAME_CATS = {"화염"}
RISK = [("주거", "주거지"), ("녹지", "산림"), ("농림", "산림"), ("자연환경보전", "산림"),
        ("보전관리", "산림"), ("생산관리", "농경지"), ("계획관리", "농경지"), ("상업", "시설")]


def nfc(s):
    return unicodedata.normalize("NFC", s)


def real_name(info):
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def verdict_of(land_use, radius):
    if not land_use:
        return "미확정"
    top = land_use[0]["ratio"]
    second = land_use[1]["ratio"] if len(land_use) > 1 else 0.0
    if top >= THRESHOLD[radius]:
        return "확정"
    if top + second >= AMBIG_SUM:
        return "모호"
    return "미확정"


def risk_targets(land_use, buildings):
    out = []
    for u in land_use:
        for key, word in RISK:
            if key in u["name"] and word not in out:
                out.append(word)
    if buildings:
        out.append("건물")
    return out


def replace_dropped(picked, drop, cands):
    """검토에서 교체로 표시한 사진을 같은 장소·DB·클래스의 다른 영상 사진으로 바꾼다.
    자리(가상 위치, 반경)는 그대로 둔다. 영상마다 고르는 사진은 늘 같게 정해진다."""
    used = {(p[1], p[4]) for p in picked}
    pools, out, n = {}, [], 0
    for item in picked:
        group, place, db, cls, vid, name, lab = item
        if name[:-5] + ".jpg" not in drop:
            out.append(item)
            continue
        key = (db, cls, place)
        if key not in pools:
            vids = sorted(cands.get(key, {}).items())
            random.Random(f"{SEED}-{db}-{cls}-{place}").shuffle(vids)
            pools[key] = vids
        for v, files in pools[key]:
            if (place, v) in used:
                continue
            ok = [f for f in sorted(files, key=lambda t: t[0]) if f[0][:-5] + ".jpg" not in drop]
            if not ok:
                continue
            used.add((place, v))
            nm, lb = random.Random(f"{SEED}-{place}-{v[0]}-{v[1]}").choice(ok)
            out.append((group, place, db, cls, v, nm, lb))
            n += 1
            break
        else:
            print(f"  주의: {name} 를 바꿀 사진이 없어 그대로 둠")
            out.append(item)
    return out, n


def main():
    ap = argparse.ArgumentParser(description="평가셋 100장과 사실 목록 만들기 (WBS 6.1)")
    ap.add_argument("vl_zip")
    ap.add_argument("landuse")
    ap.add_argument("buildings")
    ap.add_argument("dem", nargs="+")
    ap.add_argument("--positions", default=str(ROOT / "docs" / "demo_positions.csv"))
    ap.add_argument("--review", help="6.2 검토표(docs/eval_review.csv). 교체·수정을 반영한다")
    a = ap.parse_args()

    # 1. 라벨에서 후보 모으기 (시연 지역 두 곳의 드론 촬영분)
    places = {p for _, p, _, _, _ in PLAN}
    cands = {}   # (db, 클래스) -> {영상: [(파일 이름, 라벨)]}
    with zipfile.ZipFile(a.vl_zip) as zf:
        for info in zf.infolist():
            name = nfc(real_name(info).split("/")[-1])
            if not name.endswith(".json"):
                continue
            parts = name.split("_")
            if len(parts) < 7 or not any(parts[2].endswith(p) for p in places):
                continue
            lab = json.loads(zf.read(info).decode("utf-8"))
            if (lab.get("environment") or {}).get("device") != "drone":
                continue
            place = next(p for p in places if parts[2].endswith(p))
            key = (parts[0], parts[1], place)
            vid = (parts[3], parts[4])
            cands.setdefault(key, {}).setdefault(vid, []).append((name, lab))

    # 2. 영상당 1장씩 뽑기
    rng = random.Random(SEED)
    picked, used = [], set()
    short = {}
    for group, place, db, cls, n in PLAN:
        vids = sorted(cands.get((db, cls, place), {}).items())
        rng.shuffle(vids)
        got = 0
        for vid, files in vids:
            if got == n:
                break
            if (place, vid) in used:
                continue
            used.add((place, vid))
            name, lab = rng.choice(sorted(files, key=lambda t: t[0]))
            picked.append((group, place, db, cls, vid, name, lab))
            got += 1
        print(f"  {place} {db} {cls}: 영상 {len(vids)}개 중 {got}장")
        if got < n:
            short[group] = short.get(group, 0) + (n - got)
    for group, need in short.items():   # 모자라면 같은 묶음의 다른 클래스로 채운다
        for g, place, db, cls, n in PLAN:
            if g != group or need == 0:
                continue
            for vid, files in sorted(cands.get((db, cls, place), {}).items()):
                if need == 0:
                    break
                if (place, vid) in used:
                    continue
                used.add((place, vid))
                name, lab = sorted(files, key=lambda t: t[0])[0]
                picked.append((g, place, db, cls, vid, name, lab))
                need -= 1
        if need:
            print(f"  주의: {group} 묶음이 {need}장 모자람")

    # 3. 가상 위치와 반경 붙이기
    with open(a.positions, encoding="utf-8-sig") as f:
        pos = list(csv.DictReader(f))
    bands = {}
    for p in pos:
        bands.setdefault(p["band"], []).append(p)
    order = [p for trio in zip(*[bands[b] for b in sorted(bands)]) for p in trio]  # 구간을 번갈아
    rng.shuffle(picked)
    review = {}
    if a.review:   # 6.2 사람 검토 반영
        with open(a.review, encoding="utf-8-sig") as f:
            review = {r["image"]: r for r in csv.DictReader(f)}
        drop = {k for k, r in review.items() if r["판정"].strip() == "교체"}
        picked, n_rep = replace_dropped(picked, drop, cands)
        print(f"  검토 반영: 교체 {n_rep}장")
    geo = GeoData(a.landuse, a.buildings, a.dem)

    facts, rows = [], []
    for i, (group, place, db, cls, vid, name, lab) in enumerate(picked):
        p = order[i % len(order)]
        radius = RADII[(i + i // len(order)) % len(RADII)]   # 같은 위치에 같은 반경만 몰리지 않게
        lat, lon = float(p["lat"]), float(p["lon"])
        land_use, ev, info = query(geo, lat, lon, radius)

        img = lab["images"][0]
        cat = {c["id"]: c["name"] for c in lab["categories"]}
        anns = lab.get("annotations") or []
        smoke_area = sum(x.get("area", 0) for x in anns if cat.get(x["category_id"]) in SMOKE_CATS)
        flame = any(cat.get(x["category_id"]) in FLAME_CATS for x in anns)
        n_smoke = sum(1 for x in anns if cat.get(x["category_id"]) in SMOKE_CATS)
        image_file = name[:-5] + ".jpg"
        verdict = verdict_of(land_use, radius)
        r = review.get(image_file)
        state = "검토 전" if review else "-"
        if r:
            state = r["판정"].strip() or "통과"
            if state == "수정":
                if r["고친_클래스"].strip():
                    cls = r["고친_클래스"].strip()
                if r["고친_화염"].strip():
                    flame = r["고친_화염"].strip() in ("True", "true", "있음", "O", "o", "1")

        facts.append({
            "image": image_file,
            "class": SCORER_CLASS.get(cls, cls),
            "label_class": cls,
            "flame": flame,
            "area_ratio": round(smoke_area / (img["width"] * img["height"]), 5),
            "smoke_count": n_smoke,
            "land_use": land_use,
            "verdict": verdict,
            "error_radius_m": radius,
            "risk_targets": risk_targets(land_use, ev.get("buildings_in_radius")),
            "buildings_in_radius": ev.get("buildings_in_radius"),
            "terrain_slope_deg": ev.get("terrain_slope_deg"),
            "dem_resolution_m": ev.get("dem_resolution_m"),
            "pos_id": p["pos_id"], "band": p["band"], "lat": lat, "lon": lon,
            "virtual": True,
            "location_level": "가상 배치 (실제 촬영 위치 아님)",
            "source": "AI Hub 71330 Validation 라벨",
            "place": place, "camera_id": vid[0], "video_id": vid[1],
            "group": group,
            "review": state,
        })
        rows.append({"file_name": name, "class": cls, "place": place, "split": "eval",
                     "db": db, "smoke": int(group == "연기"), "camera_id": vid[0],
                     "video_id": vid[1], "image_file": image_file,
                     "gps": (lab.get("environment") or {}).get("gps", "")})

    assert len({r["image_file"] for r in rows}) == len(rows), "중복 사진"
    assert len({(r["place"], r["camera_id"], r["video_id"]) for r in rows}) == len(rows), "영상 중복"

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "eval_facts.jsonl", "w", encoding="utf-8") as f:
        for x in facts:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(ROOT / "docs" / "eval_set.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4. 요약
    print(f"\n평가셋 {len(facts)}장 (영상 {len(rows)}개, 위치 {len({x['pos_id'] for x in facts})}곳)")
    count = {}
    for x in facts:
        count[x["label_class"]] = count.get(x["label_class"], 0) + 1
    print("  클래스:", ", ".join(f"{k} {v}" for k, v in count.items()))
    print(f"  {'반경':>5} {'확정':>4} {'모호':>4} {'미확정':>5}")
    for r in RADII:
        xs = [x for x in facts if x["error_radius_m"] == r]
        c = {v: sum(1 for x in xs if x["verdict"] == v) for v in ("확정", "모호", "미확정")}
        print(f"  {r:>4}m {c['확정']:>4} {c['모호']:>4} {c['미확정']:>5}")
    write_doc(facts, bool(review))
    print("\n저장: results/eval_facts.jsonl, docs/eval_set.csv, docs/eval_set.md")


def write_doc(facts, reviewed=False):
    """실행 결과 숫자를 넣어 docs/eval_set.md 를 만든다."""
    def cnt(xs, key):
        out = {}
        for x in xs:
            out[x[key]] = out.get(x[key], 0) + 1
        return out

    cls = cnt(facts, "label_class")
    band = cnt(facts, "band")
    L = []
    L.append("# 평가셋 100장 (WBS 6.1)\n")
    L.append("설명 품질(6.2)과 오차 반경별 판정(6.5)을 잴 때 쓰는 평가셋과, 사진마다 정답이 되는 사실 목록이다.\n")
    if reviewed:
        st = {}
        for x in facts:
            st[x["review"]] = st.get(x["review"], 0) + 1
        L.append("담당: 이수진 / 상태: 6.2 사람 검토 반영 (확정본). "
                 + ", ".join(f"{k} {v}" for k, v in sorted(st.items())) + "\n")
    else:
        L.append("담당: 이수진 / 상태: 100장과 사실 목록 생성 완료, 채점기 이름 맞춤은 예진 확인 전\n")
    L.append("## 만드는 법\n")
    L.append("```\npython3 tools/make_eval_set.py <VL.zip> <landuse_5179.geojson> <buildings_5179.geojson> \\\n"
             "    <37808.img> <37804.img>\n```\n")
    L.append("- 같은 명령을 다시 돌리면 같은 100장이 나온다 (난수 시드 42).")
    L.append("- 사진 꺼내기: `python3 tools/extract_71330.py docs/eval_set.csv <VS.zip> <VL.zip> data/eval_100`\n")
    L.append("## 고른 기준\n")
    L.append("- AI Hub 71330 Validation에서 시연 지역 두 곳(모래고개입구, 장현동)만 썼다. "
             "학습용 1,260장에는 이 두 곳 사진이 0장이라 학습과 겹치지 않는다.")
    L.append("- 드론 촬영분만, 영상 하나당 1장만 골랐다 (비슷한 장면이 몰리지 않게).")
    L.append("- 연기 70장 + 오탐 유발 30장(구름, 굴뚝연기). 오탐 사진은 \"연기 아님\"이라고 말하는지 보려고 넣었다.\n")
    L.append("| 클래스 | 장수 |\n|---|---|")
    for k, v in cls.items():
        L.append(f"| {k} | {v} |")
    L.append(f"| 합계 | {len(facts)} |\n")
    L.append("## 위치와 오차 반경\n")
    L.append("- 71330 사진에는 실제 연기 위치가 없어서, 예진이 만든 `docs/demo_positions.csv` 36곳을 "
             "돌아가며 **가상으로** 붙였다. 사실 목록에 `virtual: true`로 표시했다.")
    L.append("- 위치 구간: " + ", ".join(f"{k} {v}장" for k, v in sorted(band.items())))
    L.append("- 오차 반경은 10, 30, 50, 100m를 돌아가며 붙였다 (3.5와 같은 4단계). 반경별 결과를 6.5에서 따로 볼 수 있다.")
    L.append("- 판정은 3.5 본 실험 기준(최다비율 규칙, 정확도 95% 임계값)을 그대로 썼다. "
             "서연 파이프라인(5.2)과 같은 기준이다.\n")
    L.append("| 반경 | 임계값 | 확정 | 모호 | 미확정 |\n|---|---|---|---|---|")
    for r in RADII:
        xs = [x for x in facts if x["error_radius_m"] == r]
        c = cnt(xs, "verdict")
        L.append(f"| {r}m | {THRESHOLD[r]:.2f} | {c.get('확정', 0)} | {c.get('모호', 0)} | {c.get('미확정', 0)} |")
    L.append("")
    L.append("## 사실 목록 (`results/eval_facts.jsonl`)\n")
    L.append("한 줄에 사진 하나. 채점기(`tools/score_descriptions.py`)가 그대로 읽는다.\n")
    L.append("| 칸 | 뜻 |\n|---|---|")
    for k, v in [("image", "사진 파일 이름"), ("class", "채점기용 클래스 이름"),
                 ("label_class", "라벨 원래 클래스 이름"), ("flame", "화염 라벨이 있는지"),
                 ("area_ratio", "연기 라벨 면적 / 사진 전체 면적"), ("land_use", "반경 안 용도지역 비율"),
                 ("verdict", "확정 / 모호 / 미확정"), ("error_radius_m", "오차 반경(m)"),
                 ("risk_targets", "반경 안 위험 대상 (주거지, 산림, 농경지, 시설, 건물)"),
                 ("buildings_in_radius", "반경 안 전체 건물 수 (건물 자료에 용도 정보 없음)"),
                 ("terrain_slope_deg", "DEM 90m 주변 3x3칸(270m) 경사"),
                 ("pos_id, band, lat, lon", "붙인 가상 위치"), ("virtual", "가상 위치인지 (항상 true)")]:
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## 채점기와 이름 맞춤\n")
    L.append("- 라벨의 `백색회색연기`가 채점기 COLOR_OK의 `백색및회색연기`와 달라서 `class`는 `백색및회색연기`로 적었다.")
    L.append("- 채점기는 이름에 \"연기\"가 들어가면 연기 사진으로 보기 때문에 `굴뚝연기`는 `class`를 `굴뚝`으로 적었다.")
    L.append("- 원래 이름은 `label_class`에 남겼다. 채점기를 고치게 되면 `class`를 원래 이름으로 돌리면 된다.\n")
    L.append("## 한계\n")
    L.append("- 위치가 가상이라 사진 속 풍경과 위치의 용도지역은 서로 관계가 없다. 설명의 지리 부분은 사실 목록 기준으로만 채점한다.")
    L.append("- 원본 사진과 라벨은 저장소에 올리지 않는다 (`data/`는 gitignore).")
    with open(ROOT / "docs" / "eval_set.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
