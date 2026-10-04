# -*- coding: utf-8 -*-
"""파일럿용 이미지를 위키미디어 공용에서 받는다.

자유 검색은 소방차나 그림 같은 엉뚱한 것이 섞인다. 그래서 분류에서 가져오고,
받은 뒤에 흑백이나 너무 작은 것을 걸러낸다. 본 실험은 AI Hub 71330 으로 하고
이 이미지는 파일럿에만 쓴다. 출처와 라이선스를 같이 적어 둔다.

    python fetch_pilot_images.py [출력폴더]
"""
import sys, os, csv, json, time, io, urllib.parse, urllib.request
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/pilot"
API = "https://commons.wikimedia.org/w/api.php"
UA = "edgeeye-capstone/1.0 (university project)"

CATS = {
    "연기": ["Aerial photographs of forest fires in Canada",
             "Aerial photographs of forest fires in Alberta",
             "Aerial photographs of forest fires",
             "Aerial photographs of wildfires",
             "Smoke from wildfires"],
    "무연기": ["Aerial photographs of forests",
               "Cloud forests",
               "Aerial photographs of forests in Finland"],
}
WANT = {"연기": 15, "무연기": 5}
BAD = ("map", "diagram", "poster", "plaque", "painting", "drawing", "chart",
       "logo", "seal", "graph", "sign", "cover", "page", "document", "icon",
       "screenshot", "church", "chapel", "building", "monument", "portrait",
       "satellite", "modis", "suomi", "viirs")
# 제목에 주제어가 없으면 분류가 맞아도 엉뚱한 사진이 섞인다.
MUST = {"연기": ("fire", "smoke", "wildfire", "bushfire", "burn"),
        "무연기": ("forest", "aerial", "wood", "tree", "canopy")}


def members(cat, limit=60):
    p = {"action": "query", "format": "json", "generator": "categorymembers",
         "gcmtitle": f"Category:{cat}", "gcmtype": "file", "gcmlimit": str(limit),
         "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
         "iiurlwidth": "1280"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(p),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=50) as r:
        return json.loads(r.read().decode("utf-8")).get("query", {}).get("pages", {})


def meta(info, key):
    v = info.get("extmetadata", {}).get(key, {}).get("value", "")
    return v.replace("<br>", " ").replace("&amp;", "&").strip()[:120]


def usable(data):
    """흑백이거나 너무 작으면 거른다. 항공 산불 사진은 색이 있어야 쓸모가 있다."""
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None, "열 수 없음"
    w, h = im.size
    if min(w, h) < 400:
        return None, f"너무 작음 {w}x{h}"
    if not (0.5 < w / h < 2.4):
        return None, f"비율 이상 {w}x{h}"
    small = im.resize((80, 60))
    px = list(small.getdata())
    sat = sum(max(p) - min(p) for p in px) / len(px)
    # 연기는 원래 회색이다. 채도로 거르면 항공 산불 사진이 먼저 날아간다.
    # 흑백 스캔본만 걸러내도록 기준을 낮춘다.
    if sat < 5:
        return None, f"흑백 스캔 (채도 {sat:.0f})"
    return im, None


os.makedirs(OUT, exist_ok=True)
rows, taken, seen = [], {"연기": 0, "무연기": 0}, set()

for kind, cats in CATS.items():
    for cat in cats:
        if taken[kind] >= WANT[kind]:
            break
        try:
            pages = members(cat)
        except Exception as e:
            print(f"  분류 실패 {cat}: {e}")
            continue
        for page in pages.values():
            if taken[kind] >= WANT[kind]:
                break
            info = (page.get("imageinfo") or [{}])[0]
            title = page.get("title", "")
            url = info.get("thumburl") or info.get("url")
            low = title.lower()
            if not url or title in seen or any(b in low for b in BAD):
                continue
            if not any(m in low for m in MUST[kind]):
                continue
            if "image/" not in info.get("mime", ""):
                continue
            lic = meta(info, "LicenseShortName")
            if not lic or "fair use" in lic.lower():
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
            except Exception as e:
                print(f"  내려받기 실패 {title[:40]}: {e}")
                continue
            im, why = usable(data)
            if im is None:
                print(f"  거름 {title[:45]} — {why}")
                continue
            seen.add(title)
            taken[kind] += 1
            name = f"{'smoke' if kind == '연기' else 'nosmoke'}_{taken[kind]:02d}.jpg"
            im.save(os.path.join(OUT, name), quality=90)
            rows.append([name, kind, title.replace("File:", ""), lic,
                         meta(info, "Artist"), info.get("descriptionurl", ""), cat])
            print(f"  {name}  {im.size[0]}x{im.size[1]}  {lic}")
            time.sleep(0.3)

with open(os.path.join(OUT, "출처.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["파일", "구분", "원본 제목", "라이선스", "저작자", "원본 주소", "분류"])
    w.writerows(rows)
print(f"\n연기 {taken['연기']}장, 연기 없음 {taken['무연기']}장 → {OUT}/출처.csv")
