# -*- coding: utf-8 -*-
"""불꽃을 VLM 이 알아보는지 잰다 (WBS 4.7 후속).

알람 문장은 VLM 이 flame_visible=true 라고 하면 "불꽃도 함께 보인다"를 붙인다
(src/vlm/templates.py). 평가셋 100장에는 화염 라벨이 한 장도 없어 이 문장을 잰 적이
없다. 학습용 1,260장 가운데 화염 라벨이 있는 사진으로 잰다.

    python tools/flame_eval.py data/yolo_1260 data/yolo_1260_raw_labels

대상
    불꽃 있음  화염 라벨이 있는 PositiveRealDB 사진 전부
    불꽃 없음  화염 라벨이 없는 PositiveRealDB 사진. 불꽃 있는 사진과 같은 카메라·영상에서
               먼저 고르고, 모자라면 같은 카메라, 그다음 나머지에서 고정 시드로 채운다
               (불꽃 있는 사진의 2배)

방식
    전체-JSON  지금 파이프라인과 같다. 사진 전체에 구조화 질문을 하고 flame_visible 을 본다
    전체-P     사진 전체에 "불꽃이 보이는가"를 묻고 첫 토큰 P(예)/(P(예)+P(아))
    조각-P     연기 라벨을 모두 감싼 박스를 탐지 대신 쓰고, 후보 검증기와 같이 잘라 묻는다
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from PIL import Image  # noqa: E402

from vlm_client import VLM, VERIFY_PX, VERIFY_YES, VERIFY_NO, crop_det  # noqa: E402

PROMPT = "이 사진에 불꽃이 보이는가. 예 또는 아니오로만 답하라."
SMOKE = {"흑색연기", "백색/회색연기"}


def load(raw_dir):
    rows = []
    for f in sorted(Path(raw_dir).rglob("*.json")):
        if not f.name.startswith("PositiveRealDB"):
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        cat = {c["id"]: c["name"] for c in j["categories"]}
        img = j["images"][0]
        smoke = [a["bbox"] for a in j["annotations"] if cat[a["category_id"]] in SMOKE]
        flame = [a for a in j["annotations"] if cat[a["category_id"]] == "화염"]
        if not smoke:
            continue
        x0 = min(b[0] for b in smoke); y0 = min(b[1] for b in smoke)
        x1 = max(b[0] + b[2] for b in smoke); y1 = max(b[1] + b[3] for b in smoke)
        rows.append({
            "name": f.stem, "split": f.parent.name, "camera": j["info"]["camera_id"],
            "video": j["info"]["camera_id"] + "_" + j["info"]["video_id"],
            "flame": bool(flame),
            "flame_ratio": sum(a.get("area", 0) for a in flame) / (img["width"] * img["height"]),
            "flame_boxes": [a["bbox"] for a in flame],
            "smoke_box": [x0, y0, x1 - x0, y1 - y0]})
    return rows


def pick(rows, seed=0):
    pos = [r for r in rows if r["flame"]]
    rest = [r for r in rows if not r["flame"]]
    rnd = random.Random(seed)
    rnd.shuffle(rest)
    want = 2 * len(pos)
    vids = {r["video"] for r in pos}
    cams = {r["camera"] for r in pos}
    neg = [r for r in rest if r["video"] in vids][:want]
    neg += [r for r in rest if r["camera"] in cams and r not in neg][:want - len(neg)]
    neg += [r for r in rest if r not in neg][:want - len(neg)]
    return pos, neg


def p_yes(vlm, image):
    import torch
    ip = vlm.processor.image_processor
    msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": PROMPT}]}]
    inp = vlm.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(vlm.model.device)
    with torch.inference_mode():
        pr = torch.softmax(vlm.model(**inp).logits[0, -1].float(), -1)
    y, n = float(pr[VERIFY_YES]), float(pr[VERIFY_NO])
    return y / (y + n)


def auc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def inside(box, crop_box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return crop_box[0] <= cx <= crop_box[2] and crop_box[1] <= cy <= crop_box[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("raw_labels")
    ap.add_argument("--out", default=str(ROOT / "runs" / "flame_eval.jsonl"))
    a = ap.parse_args()

    pos, neg = pick(load(a.raw_labels))
    print(f"불꽃 있음 {len(pos)}장, 없음 {len(neg)}장")
    vlm = VLM("qwen")
    ip = vlm.processor.image_processor
    out = []
    for r in pos + neg:
        path = Path(a.images) / "images" / r["split"] / (r["name"] + ".jpg")
        img = Image.open(path).convert("RGB")
        ob, ms, _ = vlm.observe(str(path))
        r["json"] = bool(ob and ob.get("flame_visible"))
        r["json_ms"] = ms
        full = img.copy(); full.thumbnail((1024, 1024))
        ip.max_pixels, ip.min_pixels = 768 * 768, 256 * 256
        r["p_full"] = p_yes(vlm, full)
        W = img.size[0]
        crop = crop_det(img, r["smoke_box"], min_side=max(160, W // 6))
        x, y, w, h = r["smoke_box"]
        side = max(max(w, h) * 1.5, max(160, W // 6))
        cb = (x + w / 2 - side / 2, y + h / 2 - side / 2, x + w / 2 + side / 2, y + h / 2 + side / 2)
        r["flame_in_crop"] = all(inside(b, cb) for b in r["flame_boxes"]) if r["flame"] else None
        ip.max_pixels, ip.min_pixels = VERIFY_PX * VERIFY_PX, 256 * 256
        r["p_crop"] = p_yes(vlm, crop)
        print(f"  {'불꽃' if r['flame'] else '    '} JSON {int(r['json'])}  전체 {r['p_full']:.2f}  "
              f"조각 {r['p_crop']:.2f}  {r['name'][:60]}", flush=True)
        out.append(r)

    Path(a.out).parent.mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    P = [r for r in out if r["flame"]]; N = [r for r in out if not r["flame"]]
    print(f"\n전체-JSON  불꽃 있음 중 true {sum(r['json'] for r in P)}/{len(P)}, "
          f"없음 중 true {sum(r['json'] for r in N)}/{len(N)}")
    for k, lab in (("p_full", "전체-P"), ("p_crop", "조각-P")):
        print(f"{lab:8s}  AUC {auc([r[k] for r in P], [r[k] for r in N]):.3f}  "
              f"0.5 기준 있음 {sum(r[k] >= 0.5 for r in P)}/{len(P)}, 없음 {sum(r[k] >= 0.5 for r in N)}/{len(N)}")
    print(f"불꽃이 조각 안에 든 사진 {sum(bool(r['flame_in_crop']) for r in P)}/{len(P)}")
    rs = sorted(r["flame_ratio"] for r in P)
    print(f"불꽃 면적 비율 중앙값 {rs[len(rs) // 2]:.5f}, 최대 {rs[-1]:.5f}")


if __name__ == "__main__":
    main()
