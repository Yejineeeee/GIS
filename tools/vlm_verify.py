# -*- coding: utf-8 -*-
"""VLM 으로 YOLO 탐지마다 "연기가 있는가" 를 물어 헛탐지를 거른다 (WBS 4.7, 후보 검증기).

4.7 ① 에서 VLM 은 여러 후보 중 밑동을 고르지 못했다(docs/vlm_base_pick.md). 대신
같은 그림에서 "연기가 있는가" 는 잘 갈랐다(AUC 0.93). 그래서 역할을 나눈다.

    VLM   무엇인가: 탐지 조각마다 연기인지 아닌지(비닐하우스, 밭, 도로) 판정
    기하  어디인가: 연기로 판정된 조각 중에서 밑동을 고른다

지금 파이프라인은 conf 가 가장 높은 탐지 하나를 쓴다. conf 1등이 헛탐지면(400m 에서
비닐하우스, 276m) 크게 틀린다. 검증기는 이 경우를 막는 것이 목적이다.

    python tools/vlm_verify.py sim/frames runs/smoke_1260_960/weights/best.pt
    python tools/vlm_verify.py sim/frames best.pt --out runs/verify --tau 0.5 0.6 0.7

탐지 박스를 사방 25% 넓혀(짧은 변 160px 이상) 정사각형으로 자르고 448px 로 키워
묻는다. 첫 토큰의 P(예)/(P(예)+P(아)) 를 쓴다. 계산 dtype 은 bf16 (vlm_pick.py 와 같음).

고르는 방식
    V0 conf 최고 (지금 방식)
    V1 P(예) 최고
    V2 P(예) >= tau 인 것 중 conf 최고 (없으면 V0)
    V3 P(예) >= tau 인 것 중 밑동이 가장 아래 (없으면 V0)
    V4 conf x P(예) 최고
"""
import argparse
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "geo"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from transform import target_pixel, to_position  # noqa: E402

PROMPT = "이 사진에 연기가 있는가. 예 또는 아니오로만 답하라."
YES, NO = 127027, 52959          # "예", "아"(아니오의 첫 토큰), Qwen2.5-VL 토크나이저
OUT_PX = 448


def meters(la1, lo1, la2, lo2):
    R = 6378137.0
    dy = math.radians(la2 - la1) * R
    dx = math.radians(lo2 - lo1) * R * math.cos(math.radians(la1))
    return math.hypot(dx, dy)


def crop_det(img, bbox, margin=0.25, min_side=160):
    from PIL import Image
    W, H = img.size
    x, y, w, h = bbox
    side = max(w, h) * (1 + 2 * margin)
    side = max(side, min_side)
    cx, cy = x + w / 2, y + h / 2
    half = side / 2
    b = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    cv = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), (128, 128, 128))
    cv.paste(img.crop((max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H))),
             (max(-b[0], 0), max(-b[1], 0)))
    return cv.resize((OUT_PX, OUT_PX), Image.LANCZOS)


class Verifier:
    def __init__(self):
        from vlm_pick import Picker
        self.p = Picker()
        self.p.processor.image_processor.max_pixels = OUT_PX * OUT_PX

    def p_yes(self, image):
        import torch
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": PROMPT}]}]
        inp = self.p.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.p.model.device)
        with torch.inference_mode():
            pr = torch.softmax(self.p.model(**inp).logits[0, -1].float(), -1)
        y, n = float(pr[YES]), float(pr[NO])
        return y / (y + n)


def choose(dets, rule, tau=None):
    """dets: [{score, p_yes, y, ...}] -> 고른 탐지."""
    if rule == "V0":
        return max(dets, key=lambda d: d["score"])
    if rule == "V1":
        return max(dets, key=lambda d: d["p_yes"])
    if rule == "V4":
        return max(dets, key=lambda d: d["score"] * d["p_yes"])
    ok = [d for d in dets if d["p_yes"] >= tau]
    if not ok:
        return max(dets, key=lambda d: d["score"])
    if rule == "V2":
        return max(ok, key=lambda d: d["score"])
    if rule == "V3":
        return max(ok, key=lambda d: d["y"])
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser(description="VLM 후보 검증기 (4.7)")
    ap.add_argument("frames")
    ap.add_argument("weights")
    ap.add_argument("--out", default="runs/verify")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    a = ap.parse_args()

    from PIL import Image
    from detect import Detector
    ter = None
    if (ROOT / "sim" / "terrain.json").is_file():
        from sim_terrain import SimTerrain
        ter = SimTerrain(str(ROOT / "sim" / "terrain.json"))
    det = Detector(a.weights)
    ver = Verifier()
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    frames = []
    for i, png in enumerate(sorted(glob.glob(str(Path(a.frames) / "*.png")))):
        frame, _ = det.run(png, i)
        if not frame["detections"]:
            continue
        meta = json.load(open(Path(png).with_suffix(".json"), encoding="utf-8"))
        t = meta.get("truth")
        img = Image.open(png).convert("RGB")
        dets = []
        for j, d in enumerate(frame["detections"]):
            c = crop_det(img, d["bbox"])
            c.save(out / "crops" / f"{Path(png).stem}_{j}.png")
            x, y = target_pixel(d)
            err = None
            if t:
                try:
                    pos, _ = to_position(frame, d, dem=ter)
                    err = meters(pos["lat"], pos["lon"], t["fire_lat"], t["fire_lon"])
                except ValueError:
                    err = float("inf")
            dets.append({"j": j, "score": d["score"], "p_yes": round(ver.p_yes(c), 4),
                         "x": round(x, 1), "y": round(y, 1), "bbox": d["bbox"],
                         "err_m": None if err is None else round(err, 1)})
        sw = meta.get("sweep") or {}
        frames.append({"image": Path(png).name, "D": sw.get("D"), "bearing": sw.get("bearing"),
                       "dets": dets})
        print(Path(png).name, [(d["j"], d["score"], d["p_yes"], d["err_m"]) for d in dets], flush=True)

    with open(out / "verify.jsonl", "w", encoding="utf-8") as fo:
        for f in frames:
            fo.write(json.dumps(f, ensure_ascii=False) + "\n")

    rules = [("V0 conf 최고 (지금)", "V0", None), ("V1 P(예) 최고", "V1", None),
             ("V4 conf x P(예)", "V4", None)]
    for tau in a.tau:
        rules += [(f"V2 통과({tau}) 중 conf 최고", "V2", tau), (f"V3 통과({tau}) 중 가장 아래", "V3", tau)]
    scored = [f for f in frames if all(d["err_m"] is not None for d in f["dets"])]
    best = [min(d["err_m"] for d in f["dets"]) for f in scored]
    print(f"\n{len(scored)}장   상한(후보 중 최선) 평균 {st.mean(best):.1f}m 중앙 {st.median(best):.1f}m")
    print(f"{'방식':<26}{'평균':>8}{'중앙':>8}{'100m 넘음':>10}{'상한 적중':>9}")
    for name, r, tau in rules:
        e = [choose(f["dets"], r, tau)["err_m"] for f in scored]
        hit = sum(abs(x - b) < 0.5 for x, b in zip(e, best))
        print(f"{name:<26}{st.mean(e):>7.1f}m{st.median(e):>7.1f}m{sum(x > 100 for x in e):>8}장{hit:>7}장")
    print(f"\n결과: {out / 'verify.jsonl'}   잘린 그림: {out / 'crops'}")


if __name__ == "__main__":
    main()
