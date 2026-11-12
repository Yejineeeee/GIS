# -*- coding: utf-8 -*-
"""평가셋 사람 검토용 화면과 검토표를 만든다 (WBS 6.2).

    python3 tools/make_review_sheet.py data/eval_100

    data/eval_100/review.html   사진 100장에 라벨 테두리와 사실 목록을 같이 띄운 화면 (브라우저로 연다)
    docs/eval_review.csv        검토표. '판정' 칸에 통과 / 수정 / 교체 를 적는다

검토표가 이미 있으면 적어 둔 내용은 그대로 두고, 새로 들어온 사진만 아래에 더한다
(교체 후 다시 돌렸을 때 새 사진만 검토하면 되도록).
"""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"흑색연기": "#ff3b30", "백색/회색연기": "#00c8ff", "화염": "#ff9500",
          "구름": "#34c759", "안개/연무": "#8e8e93", "굴뚝연기": "#bf5af2"}
FIELDS = ["번호", "image", "group", "label_class", "flame", "area_ratio",
          "판정", "고친_클래스", "고친_화염", "메모"]
CHECK = """
<h2>검토 기준</h2>
<ul>
<li><b>연기 사진</b>: 산불 연기가 눈으로 보이는가 (너무 작거나 흐려서 사람도 못 찾으면 교체)</li>
<li><b>연기 사진</b>: 라벨 색(흑색 / 백색회색)이 사진과 맞는가, 불꽃(화염) 여부가 맞는가 (틀리면 수정)</li>
<li><b>오탐 사진</b>: 정말 산불 연기가 없는가 (구름, 굴뚝 연기만 있는가). 산불 연기가 섞여 있으면 교체</li>
<li><b>공통</b>: 사진이 깨지거나 거의 안 보이지 않는가, 바로 앞 사진과 사실상 같은 장면이 아닌가 (교체)</li>
</ul>
<p>검토표(docs/eval_review.csv)의 '판정' 칸: <b>통과</b> (그대로), <b>수정</b> (고친_클래스·고친_화염 칸에 바른 값),
<b>교체</b> (다른 사진으로 바꿈). 이유는 메모 칸에 짧게.</p>
<p>테두리 색: """ + " ".join(f'<span style="color:{c}">■ {n}</span>' for n, c in COLORS.items()) + "</p>"


def load_label(data_dir, image):
    name = image[:-4] + ".json"
    for p in data_dir.rglob(name):
        return json.load(open(p, encoding="utf-8"))
    return None


def find_image(data_dir, image):
    for p in data_dir.rglob(image):
        return p
    return None


def overlay(lab):
    """라벨 다각형(없으면 bbox)을 SVG로."""
    img = lab["images"][0]
    cat = {c["id"]: c["name"] for c in lab["categories"]}
    shapes = []
    for a in lab.get("annotations") or []:
        color = COLORS.get(cat.get(a["category_id"]), "#ffffff")
        polys = [s for s in (a.get("segmentation") or []) if isinstance(s, list) and len(s) >= 6]
        if polys:
            for s in polys:
                pts = " ".join(f"{s[i]},{s[i + 1]}" for i in range(0, len(s) - 1, 2))
                shapes.append(f'<polygon points="{pts}" fill="none" stroke="{color}" stroke-width="12"/>')
        elif a.get("bbox"):
            x, y, w, h = a["bbox"]
            shapes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
                          f'stroke="{color}" stroke-width="12"/>')
    return (f'<svg viewBox="0 0 {img["width"]} {img["height"]}" preserveAspectRatio="none">'
            + "".join(shapes) + "</svg>")


def main():
    ap = argparse.ArgumentParser(description="평가셋 사람 검토 화면과 검토표 만들기 (WBS 6.2)")
    ap.add_argument("data_dir", help="extract_71330.py로 꺼낸 폴더 (예: data/eval_100)")
    ap.add_argument("--facts", default=str(ROOT / "results" / "eval_facts.jsonl"))
    ap.add_argument("--csv", default=str(ROOT / "docs" / "eval_review.csv"))
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    facts = [json.loads(line) for line in open(a.facts, encoding="utf-8") if line.strip()]

    old = {}
    if Path(a.csv).exists():
        with open(a.csv, encoding="utf-8-sig") as f:
            old = {r["image"]: r for r in csv.DictReader(f)}

    cards, rows, missing, new = [], [], 0, 0
    for i, x in enumerate(facts, 1):
        img_path = find_image(data_dir, x["image"])
        lab = load_label(data_dir, x["image"])
        if img_path is None:
            missing += 1
        r = old.get(x["image"])
        if r is None:
            new += 1
            r = {"판정": "통과", "고친_클래스": "", "고친_화염": "", "메모": ""}
        rows.append({"번호": i, "image": x["image"], "group": x["group"], "label_class": x["label_class"],
                     "flame": x["flame"], "area_ratio": x["area_ratio"], "판정": r["판정"],
                     "고친_클래스": r["고친_클래스"], "고친_화염": r["고친_화염"], "메모": r["메모"]})
        src = Path(img_path).resolve().relative_to(data_dir.resolve()).as_posix() if img_path else ""
        tag = "" if x["image"] in old else '<span class="new">새로 검토</span>'
        cards.append(f"""
<div class="card" id="n{i}">
  <div class="pic"><img loading="lazy" src="{html.escape(src)}" alt="">{overlay(lab) if lab else ""}</div>
  <div class="meta"><b>{i}.</b> {tag} [{html.escape(x['group'])}] <b>{html.escape(x['label_class'])}</b>
   · 화염 {'있음' if x['flame'] else '없음'} · 연기 면적 {x['area_ratio'] * 100:.2f}% · 연기 라벨 {x['smoke_count']}개
   <div class="file">{html.escape(x['image'])}</div></div>
</div>""")

    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>평가셋 검토</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:16px;background:#f5f5f7;color:#1d1d1f}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px}}
.card{{background:#fff;border-radius:10px;padding:8px;box-shadow:0 1px 3px #0002}}
.pic{{position:relative}} .pic img{{width:100%;display:block;border-radius:6px;background:#ddd;aspect-ratio:16/9}}
.pic svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.meta{{font-size:14px;margin-top:6px;line-height:1.5}} .file{{font-size:11px;color:#888;word-break:break-all}}
.new{{background:#ff9500;color:#fff;border-radius:4px;padding:0 5px;font-size:12px}}
</style></head><body>
<h1>평가셋 {len(facts)}장 사람 검토 (WBS 6.2)</h1>{CHECK}
<div class="grid">{''.join(cards)}</div></body></html>"""
    # 교체된 사진의 기록은 지우지 않고 아래에 남긴다 (지우면 다음에 다시 뽑혀 들어온다)
    now = {x["image"] for x in facts}
    for k, r in old.items():
        if k not in now:
            r = dict(r)
            r["번호"] = "-"
            rows.append({f: r.get(f, "") for f in FIELDS})
    out = data_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    with open(a.csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"화면: {out}  (브라우저로 열기)")
    print(f"검토표: {a.csv}  (새로 들어온 사진 {new}장)")
    if missing:
        print(f"주의: 사진 파일을 못 찾은 것이 {missing}장 (extract_71330.py로 꺼냈는지 확인)")


if __name__ == "__main__":
    sys.exit(main())
