# -*- coding: utf-8 -*-
"""VLM 상황 판단(2단계)을 71330 실제 사진(평가셋 100장)으로 시험한다.

11/12 유원지리정보시스템 미팅에서 받은 조언: VLM 을 (1) 탐지된 것이 연기가 맞는지
(2) 어떤 상황인지 (3) 위험도, 세 단계로 쓴다. (1)은 tools/vlm_verify_real.py 로 쟀고
파이프라인에 들어가 있다. 이 도구는 (2)를 잰다.

    python tools/vlm_situation.py data/eval_100

묻는 것: 사진에 보이는 것을 넷 중 하나로 고른다.
    1 산이나 들에서 난 불의 연기      정답: 흑색연기, 백색회색연기 (70장)
    2 굴뚝, 공장, 건물 설비의 연기     정답: 굴뚝연기 (15장)
    3 구름이나 안개                    정답: 구름 (15장)
    4 연기가 없다

답은 생성하지 않고 다음 토큰 확률에서 "1"~"4" 만 비교한다(검증기의 P(예) 와 같은 방식).

공간정보를 같이 넣으면 달라지는지
    71330 사진에는 사진별 촬영 위치가 없어 실제 조회 결과를 붙일 수 없다. 그래서 문맥을
    두 가지로 지어 모든 사진에 똑같이 넣고, 정답과 맞는 문맥일 때와 어긋나는 문맥일 때를
    나눠 본다.
        none    사진만
        forest  "반경 안에 건물이 없고 보전녹지지역이다"
        built   "반경 안에 건물이 12채 있고 일반공업지역이다"
    맞는 문맥: 산불 사진에 forest, 굴뚝 사진에 built. 어긋나는 문맥은 그 반대.
    맞는 문맥의 정확도는 위치와 조회가 다 맞았을 때의 위쪽 한계이고, 어긋나는 문맥의
    정확도는 위치가 틀렸을 때 VLM 이 문맥에 끌려가는 정도다.

보는 화면
    full  사진 전체 (긴 변 1024)
    wide  conf 1등 탐지를 가운데 두고 박스의 3배 넓이로 자른 것. 탐지가 없는 사진은 뺀다.
          탐지는 runs/verify_real/verify_real.jsonl 에서 읽는다 (vlm_verify_real.py 결과)
"""
import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

CHOICES = ["산불", "굴뚝", "구름", "없음"]
QUESTION = ("드론으로 찍은 사진이다. 사진에 보이는 것을 아래에서 하나 골라 번호만 답하라.\n"
            "1 산이나 들에서 난 불의 연기\n"
            "2 굴뚝이나 공장, 건물 설비에서 나오는 연기나 수증기\n"
            "3 구름이나 안개\n"
            "4 연기가 없다")
# v1 은 "연기가 보이는 지점" 이라고 적어 연기가 있다고 전제했다. 어떤 문맥을 넣어도 "없음" 답이
# 줄어 문맥 내용의 효과와 섞였다. v2 는 연기를 말하지 않고, 내용 없는 참고 문장(ctrl)을 대조로 둔다
CONTEXTS = {
    "none": {"none": ""},
    "v1": {
        "none": "",
        "forest": "참고: 연기가 보이는 지점 반경 50m 안에는 건물이 없고 용도지역은 보전녹지지역이다.\n",
        "built": "참고: 연기가 보이는 지점 반경 50m 안에는 건물이 12채 있고 용도지역은 일반공업지역이다.\n",
    },
    "v2": {
        "none": "",
        "ctrl": "참고: 이 사진은 드론이 고도 120m 에서 찍었다.\n",
        "forest": "참고: 사진 가운데 지역은 건물이 없는 보전녹지지역이다.\n",
        "built": "참고: 사진 가운데 지역은 건물이 12채 있는 일반공업지역이다.\n",
    },
}
MATCH = {"산불": "forest", "굴뚝": "built"}          # 정답과 맞는 문맥
TRUTH = {"흑색연기": "산불", "백색회색연기": "산불", "굴뚝연기": "굴뚝", "구름": "구름", "안개연무": "구름"}


def label_class(name):
    """파일 이름 두 번째 칸이 71330 클래스다.

    zip 에서 꺼낼 때 한글 이름이 깨진 폴더도 있다(UTF-8 바이트를 cp437 로 읽은 꼴). 되돌려 읽는다.
    """
    cls = name.split("_")[1]
    if cls not in TRUTH:
        cls = name.encode("cp437").decode("utf-8").split("_")[1]
    return TRUTH[cls]


def wide_crop(img, bbox, scale=3.0, min_side=640):
    W, H = img.size
    x, y, w, h = bbox
    side = max(max(w, h) * scale, min_side)
    cx, cy = x + w / 2, y + h / 2
    half = side / 2
    box = (max(int(cx - half), 0), max(int(cy - half), 0), min(int(cx + half), W), min(int(cy + half), H))
    return img.crop(box)


class Judge:
    def __init__(self):
        from vlm_pick import Picker
        self.p = Picker()
        tok = self.p.processor.tokenizer
        self.ids = []
        for n in "1234":
            t = tok.encode(n, add_special_tokens=False)
            assert len(t) == 1, (n, t)
            self.ids.append(t[0])

    def probs(self, image, context):
        import torch
        msg = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": context + QUESTION}]}]
        inp = self.p.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.p.model.device)
        with torch.inference_mode():
            pr = torch.softmax(self.p.model(**inp).logits[0, -1].float(), -1)
        v = [float(pr[i]) for i in self.ids]
        s = sum(v)
        return [round(x / s, 4) for x in v]


def table(recs, key):
    """정답별로 고른 답을 센다. key: 기록에서 답을 꺼내는 함수(없으면 None)."""
    rows = {}
    for r in recs:
        a = key(r)
        if a is not None:
            rows.setdefault(r["truth"], Counter())[a] += 1
    return rows


def show(title, rows):
    print(f"\n{title}")
    print(f"{'정답':>6}" + "".join(f"{c:>7}" for c in CHOICES) + f"{'맞음':>10}")
    ok = n = 0
    for t in ["산불", "굴뚝", "구름"]:
        c = rows.get(t)
        if not c:
            continue
        tot = sum(c.values())
        ok += c[t]
        n += tot
        print(f"{t:>6}" + "".join(f"{c[x]:>7}" for x in CHOICES) + f"{c[t]:>6}/{tot:<3}")
    if n:
        print(f"{'합계':>6}{'':>28}{ok:>6}/{n:<3} ({ok / n:.0%})")


def main():
    ap = argparse.ArgumentParser(description="VLM 상황 판단 시험 (2단계)")
    ap.add_argument("eval_dir", help="images/eval 이 있는 폴더")
    ap.add_argument("--dets", default="runs/verify_real/verify_real.jsonl")
    ap.add_argument("--out", default="runs/situation")
    ap.add_argument("--limit", type=int, help="앞에서 이만큼만 (시험용). 클래스가 섞이게 고르게 뽑는다")
    ap.add_argument("--set", default="v1", choices=list(CONTEXTS), help="문맥 문장 묶음")
    ap.add_argument("--reuse", help="사진만(none) 답을 이 situation.jsonl 에서 가져온다")
    ap.add_argument("--tau", type=float, default=0.7, help="검증기 문턱 (1단계 통과 여부)")
    a = ap.parse_args()

    from PIL import Image
    CONTEXT = CONTEXTS[a.set]
    reuse = {}
    if a.reuse:
        for line in open(a.reuse, encoding="utf-8"):
            r = json.loads(line)
            reuse[r["image"]] = r["probs"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dets = {}
    if Path(a.dets).is_file():
        for line in open(a.dets, encoding="utf-8"):
            r = json.loads(line)
            dets[r["image"]] = r["dets"]
    imgs = sorted(glob.glob(str(Path(a.eval_dir) / "images" / "eval" / "*.jpg")))
    if a.limit:
        step = max(1, len(imgs) // a.limit)
        imgs = imgs[::step][:a.limit]

    # 장마다 바로 적는다. 중간에 끊기면 같은 명령으로 이어서 돌린다
    recs = []
    path = out / "situation.jsonl"
    if path.is_file():
        recs = [json.loads(x) for x in open(path, encoding="utf-8")]
    done = {r["image"] for r in recs}
    judge = Judge()
    fo = open(path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    n_new = 0
    for i, jpg in enumerate(imgs):
        name = Path(jpg).name
        if name in done:
            continue
        img = Image.open(jpg).convert("RGB")
        full = img.copy()
        full.thumbnail((1024, 1024))
        views = {"full": full}
        ds = dets.get(name) or []
        if ds:
            top = max(ds, key=lambda d: d["score"])
            views["wide"] = wide_crop(img, top["bbox"])
        rec = {"image": name, "truth": label_class(name), "n_dets": len(ds),
               "passed": any(d["p_yes"] >= a.tau for d in ds), "probs": {}}
        for v, im in views.items():
            for c, text in CONTEXT.items():
                k = f"{v}/{c}"
                old = reuse.get(name, {}).get(k) if c == "none" else None
                rec["probs"][k] = old or judge.probs(im, text)
                n_new += old is None
        recs.append(rec)
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        pick = {k: CHOICES[max(range(4), key=lambda j: p[j])] for k, p in rec["probs"].items()}
        print(i, rec["truth"], pick, flush=True)
    sec = time.perf_counter() - t0

    fo.close()

    def ans(r, k):
        p = r["probs"].get(k)
        return None if p is None else CHOICES[max(range(4), key=lambda j: p[j])]

    n_calls = sum(len(r["probs"]) for r in recs)
    print(f"\n사진 {len(recs)}장, 질문 {n_calls}번, 한 번에 {sec / max(n_new, 1):.2f}초")
    for v in ("full", "wide"):
        if not any(f"{v}/none" in r["probs"] for r in recs):
            continue
        show(f"[{v}] 사진만", table(recs, lambda r: ans(r, f"{v}/none")))
        if any(f"{v}/ctrl" in r["probs"] for r in recs):
            show(f"[{v}] 내용 없는 참고 문장 (대조)", table(recs, lambda r: ans(r, f"{v}/ctrl")))
        show(f"[{v}] 맞는 문맥 (산불 사진에 산림, 굴뚝 사진에 건물)",
             table(recs, lambda r: ans(r, f"{v}/{MATCH[r['truth']]}") if r["truth"] in MATCH else None))
        show(f"[{v}] 어긋나는 문맥 (산불 사진에 건물, 굴뚝 사진에 산림)",
             table(recs, lambda r: ans(r, f"{v}/{'built' if MATCH[r['truth']] == 'forest' else 'forest'}")
                   if r["truth"] in MATCH else None))
        show(f"[{v}] 구름 사진에 산림 문맥", table([r for r in recs if r["truth"] == "구름"],
                                              lambda r: ans(r, f"{v}/forest")))
        show(f"[{v}] 구름 사진에 건물 문맥", table([r for r in recs if r["truth"] == "구름"],
                                              lambda r: ans(r, f"{v}/built")))

    # 1단계(검증기)를 통과해 알람이 남는 사진만. 파이프라인에서 2단계가 실제로 받는 것
    passed = [r for r in recs if r["passed"]]
    if passed:
        v = "wide"
        print(f"\n검증기 문턱 {a.tau} 를 통과한 사진 {len(passed)}장 "
              f"(산불 {sum(r['truth'] == '산불' for r in passed)}, "
              f"굴뚝 {sum(r['truth'] == '굴뚝' for r in passed)}, 구름 {sum(r['truth'] == '구름' for r in passed)})")
        show(f"[통과분, {v}] 사진만", table(passed, lambda r: ans(r, f"{v}/none")))
        show(f"[통과분, {v}] 맞는 문맥",
             table(passed, lambda r: ans(r, f"{v}/{MATCH[r['truth']]}") if r["truth"] in MATCH else None))
    print(f"\n결과: {out / 'situation.jsonl'}")


if __name__ == "__main__":
    main()
