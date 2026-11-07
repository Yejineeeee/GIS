# -*- coding: utf-8 -*-
"""모호 판정을 VLM 으로 풀어 본다: GIS 가 후보 두 용도를 주고, VLM 이 사진으로 고른다.

반경 안이 "녹지 또는 주거"처럼 모호할 때, 연기 밑동 주변을 잘라 VLM 에게 두 가지를 묻는다.
    숲   "이 사진 가운데는 나무가 우거진 숲인가"           P(예)
    건물 "이 사진 가운데에 건물이나 집이 있는가"            P(예)
점수 = P(숲) - P(건물). 후보 두 용도가 녹지 하나, 시가지(주거, 상업, 공업) 하나일 때만 쓴다.
점수가 +문턱 이상이면 녹지 쪽, -문턱 이하면 시가지 쪽으로 확정하고, 그 사이면 모호로 둔다.

정답은 불 위치 한 점의 용도지역이다(시뮬레이터 truth). 모호 알람만이 아니라 확정 알람에도
점수를 매겨, VLM 이 얼마나 맞히는지(AUC)와 문턱별로 몇 개를 풀고 몇 개를 틀리는지 본다.

    python tools/vlm_landcover.py results/pipeline_boundary --geo-dir data/geo_jagok
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/geo", "src/pipeline"):
    sys.path.insert(0, str(ROOT / p))

from query import GeoData, query  # noqa: E402
from transform import target_pixel  # noqa: E402

Q_FOREST = "이 사진 가운데는 나무가 우거진 숲인가. 예 또는 아니오로만 답하라."
Q_BUILT = "이 사진 가운데에 건물이나 집이 있는가. 예 또는 아니오로만 답하라."
PX = 448


def kind(name):
    """용도지역 이름 → 녹지 / 시가지 / 기타."""
    if not name:
        return "기타"
    if "녹지" in name or "보전" in name or "농림" in name:
        return "녹지"
    if "주거" in name or "상업" in name or "공업" in name:
        return "시가지"
    return "기타"


def crop_base(img, px, py, side):
    """밑동 픽셀을 가운데 두고 정사각형으로 잘라 448px 로 키운다. 화면 밖은 회색."""
    from PIL import Image
    W, H = img.size
    half = side / 2
    b = (int(px - half), int(py - half), int(px + half), int(py + half))
    cv = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), (128, 128, 128))
    cv.paste(img.crop((max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H))),
             (max(-b[0], 0), max(-b[1], 0)))
    return cv.resize((PX, PX), Image.LANCZOS)


def ask(vlm, crop, q):
    import torch
    from vlm_client import VERIFY_YES, VERIFY_NO
    ip = vlm.processor.image_processor
    ip.max_pixels, ip.min_pixels = PX * PX, 256 * 256
    msg = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": q}]}]
    inp = vlm.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(vlm.model.device)
    with torch.inference_mode():
        pr = torch.softmax(vlm.model(**inp).logits[0, -1].float(), -1)
    y, n = float(pr[VERIFY_YES]), float(pr[VERIFY_NO])
    return y / (y + n)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="run.py --no-merge 결과 폴더")
    ap.add_argument("--geo-dir", default=str(ROOT / "data/geo_jagok"))
    ap.add_argument("--side", type=float, default=0.25, help="자르는 크기 (화면 너비 대비)")
    ap.add_argument("--out", help="점수 jsonl (기본: src/landcover.jsonl)")
    a = ap.parse_args()
    src = Path(a.src)
    geo = GeoData(str(Path(a.geo_dir) / "landuse_5179.geojson"))

    from PIL import Image
    from vlm_client import VLM
    vlm = VLM("qwen")

    dets = {}
    for f in (src / "detections").glob("*.json"):
        fr = json.load(open(f, encoding="utf-8"))
        for d in fr.get("detections") or []:
            dets[d["id"]] = (fr, d)

    rows = []
    for f in sorted((src / "alerts").glob("*.json")):
        al = json.load(open(f, encoding="utf-8"))
        fr, d = dets[al["detection_id"]]
        img_path = Path(fr["image"])
        meta = json.load(open(img_path.with_suffix(".json"), encoding="utf-8"))
        t = meta["truth"]
        truth_lu, _, _ = query(geo, t["fire_lat"], t["fire_lon"], 0)
        truth = kind(truth_lu[0]["name"]) if truth_lu else "기타"
        lu = al.get("land_use") or []
        k1 = kind(lu[0]["name"]) if lu else "기타"
        k2 = kind(lu[1]["name"]) if len(lu) > 1 else "기타"
        p = al["position"]
        err = math.hypot((p["lat"] - t["fire_lat"]) * 111320,
                         (p["lon"] - t["fire_lon"]) * 111320 * math.cos(math.radians(p["lat"])))
        img = Image.open(img_path).convert("RGB")
        px, py = target_pixel(d)
        crop = crop_base(img, px, py, img.size[0] * a.side)
        pf, pb = ask(vlm, crop, Q_FOREST), ask(vlm, crop, Q_BUILT)
        row = {"alert": al["alert_id"], "image": img_path.name, "score": d["score"],
               "verdict": al["location_verdict"], "top1": lu[0]["name"] if lu else None,
               "k1": k1, "k2": k2, "r1": lu[0]["ratio"] if lu else 0, "r2": lu[1]["ratio"] if len(lu) > 1 else 0,
               "radius": p["error_radius_m"], "err_m": round(err, 1), "truth": truth,
               "truth_name": truth_lu[0]["name"] if truth_lu else None,
               "p_forest": round(pf, 4), "p_built": round(pb, 4), "vlm": round(pf - pb, 4)}
        rows.append(row)
        print(row["alert"], row["image"], row["verdict"], f"{k1}/{k2}", "정답", truth,
              f"오차 {err:.0f}m", f"숲 {pf:.2f} 건물 {pb:.2f}", flush=True)

    out = Path(a.out) if a.out else src / "landcover.jsonl"
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    # VLM 이 정답 쪽을 얼마나 가리키나 (정답이 녹지/시가지인 알람 전부)
    g = [r["vlm"] for r in rows if r["truth"] == "녹지"]
    b = [r["vlm"] for r in rows if r["truth"] == "시가지"]
    print(f"\n알람 {len(rows)}개. 정답 녹지 {len(g)}, 시가지 {len(b)}. 점수 AUC {auc(g, b):.3f}")
    mix = [r for r in rows if r["verdict"] == "모호" and {r["k1"], r["k2"]} == {"녹지", "시가지"}]
    print(f"모호이면서 녹지/시가지가 1, 2순위인 알람 {len(mix)}개 (모호 전체 {sum(r['verdict'] == '모호' for r in rows)})")
    for tau in (0.2, 0.4, 0.6):
        solved = right = 0
        for r in mix:
            pick = "녹지" if r["vlm"] >= tau else "시가지" if r["vlm"] <= -tau else None
            if pick:
                solved += 1
                right += pick == r["truth"]
        print(f"  문턱 {tau}: 확정으로 바꿈 {solved}, 그중 맞음 {right}, 틀림 {solved - right}")


if __name__ == "__main__":
    main()
