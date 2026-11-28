# -*- coding: utf-8 -*-
"""hard negative 로 VLM 검증기를 보강할 수 있는지 잰다 (linear probe).

VLM 은 학습 없이 "이 사진에 연기가 있는가"에 답한다. 굴뚝연기, 안개연무처럼 연기처럼 보이는 것은
가르지 못했다(docs/verifier_ablation.md). 4GB GPU 로는 VLM 을 LoRA 로 학습하기 어려워서,
VLM 은 그대로 두고 답하기 직전의 내부 값(마지막 층, 마지막 위치)을 뽑아 그 위에 작은 분류기
(로지스틱 회귀)를 학습한다. 학습 데이터는 탐지기 학습에 쓴 AI Hub 71330 사진(NegativeDB 포함)이고,
평가셋의 장소와 장소를 섞은 150장의 영상은 뺀다(150장은 학습 장소를 모두 포함해서 영상 단위로만 뺀다).

    python tools/hardneg_probe.py prep runs/smoke_1260_960/weights/best.pt --out runs/hardneg
    python tools/hardneg_probe.py feats --out runs/hardneg
    python tools/hardneg_probe.py eval --out runs/hardneg
    python tools/hardneg_probe.py thr --out runs/hardneg
    python tools/hardneg_probe.py final --out runs/hardneg
    python tools/hardneg_probe.py lopo --out runs/hardneg

prep   학습 사진에 탐지기를 돌려 조각과 정답(연기 라벨과 겹치면 연기)을 만든다. VLM 은 쓰지 않는다
feats  학습 조각과 두 평가 세트의 기존 조각에서 VLM 특징을 뽑는다(GPU)
eval   분류기를 학습하고 지금 검증기 P(예)와 견준다(문턱 0.7)
thr    학습 조각 교차검증으로 문턱을 정해 본다
final  문턱 0.5, 지금 검증기와 둘 다 통과해야 남기는 경우
lopo   장소를 하나씩 빼고 학습해 뺀 장소에서 잰다(처음 보는 장소)
"""
import argparse
import glob
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

TESTS = {  # 이름: (탐지 결과, 조각 폴더, 지금 검증기 점수 파일)
    "eval": ("runs/verify_real_v2/verify_real.jsonl", "runs/verify_real_v2/crops", "results/verify_margin_eval.jsonl"),
    "mix150": ("runs/verify_situation/verify_real.jsonl", "runs/verify_situation/crops", "results/verify_margin_mix150.jsonl"),
}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def place(name):
    parts = nfc(Path(name).name).split("_")
    return parts[2] if len(parts) > 2 else ""


def kind(name):
    parts = nfc(Path(name).name).split("_")
    return parts[1] if len(parts) > 1 else ""


def video(name):
    parts = nfc(Path(name).name).split("_")
    return "_".join(parts[3:5]) if len(parts) > 4 else name


def test_keys():
    """평가셋(학습에 없던 장소)은 장소를 통째로 빼고, 150장(장소를 섞은 세트)은 같은 영상만 뺀다.
    150장은 원래 학습 장소를 섞어 만든 세트라 장소로 빼면 학습 사진이 남지 않는다."""
    places, videos = set(), set()
    for line in open(ROOT / TESTS["eval"][0], encoding="utf-8"):
        places.add(place(json.loads(line)["image"]))
    for line in open(ROOT / TESTS["mix150"][0], encoding="utf-8"):
        videos.add(video(json.loads(line)["image"]))
    return places, videos


def yolo_boxes(txt, W, H):
    boxes = []
    if not Path(txt).is_file():
        return boxes
    for line in open(txt, encoding="utf-8"):
        v = line.split()
        if len(v) < 7:
            continue
        xs = [float(x) * W for x in v[1::2]]
        ys = [float(y) * H for y in v[2::2]]
        boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
    return boxes


def cmd_prep(a):
    from PIL import Image
    from detect import Detector
    from vlm_verify import crop_det
    from vlm_verify_real import iou, inside
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    skip_places, skip_videos = test_keys()
    print("평가셋 장소:", sorted(skip_places))
    det = Detector(a.weights, conf=a.conf)
    det_neg = Detector(a.weights, conf=a.conf_neg)   # 오탐 사진은 탐지기가 학습해 거의 안 잡으므로 문턱을 낮춘다
    rows, n_img, n_skip = [], 0, 0
    for split in ("train", "val"):
        for jpg in sorted(glob.glob(str(ROOT / "data/yolo_1260/images" / split / "*.jpg"))):
            name = nfc(Path(jpg).name)
            if place(name) in skip_places or video(name) in skip_videos:
                n_skip += 1
                continue
            n_img += 1
            img = Image.open(jpg).convert("RGB")
            W, H = img.size
            positive = name.startswith("Positive")
            gt = yolo_boxes(ROOT / "data/yolo_1260/labels" / split / (Path(jpg).stem + ".txt"), W, H) if positive else []
            frame, _ = (det if positive else det_neg).run(jpg, n_img)
            for j, d in enumerate(frame["detections"][: a.max_dets]):
                b = d["bbox"]
                c = (b[0] + b[2] / 2, b[1] + b[3] / 2)
                smoke = positive and any(iou(b, s) >= 0.1 or inside(c, s) for s in gt)
                cid = f"{n_img:04d}_{j}"
                crop_det(img, b, min_side=max(160, W // 6)).save(out / "crops" / f"{cid}.png")
                rows.append({"id": cid, "image": name, "kind": kind(name), "place": place(name),
                             "score": d["score"], "smoke": smoke})
            if n_img % 50 == 0:
                print(n_img, len(rows), flush=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(r["smoke"] for r in rows)
    print(f"사진 {n_img}장(평가 장소라 뺀 것 {n_skip}장), 조각 {len(rows)}개: 연기 {pos}, 헛탐지 {len(rows) - pos}")
    from collections import Counter
    print("헛탐지 종류:", Counter(r["kind"] for r in rows if not r["smoke"]))


def feat(ver, image):
    import torch
    from vlm_verify import PROMPT, YES, NO
    msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inp = ver.p.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt").to(ver.p.model.device)
    with torch.inference_mode():
        o = ver.p.model(**inp, output_hidden_states=True)
    pr = torch.softmax(o.logits[0, -1].float(), -1)
    y, n = float(pr[YES]), float(pr[NO])
    hs = o.hidden_states
    mid = len(hs) // 2
    return y / (y + n), hs[-1][0, -1].float().cpu().numpy(), hs[mid][0, -1].float().cpu().numpy()


def cmd_feats(a):
    import numpy as np
    from PIL import Image
    from vlm_verify import Verifier
    out = Path(a.out)
    ver = Verifier()
    sets = {"train": [(r["id"], out / "crops" / f"{r['id']}.png")
                      for r in map(json.loads, open(out / "train.jsonl", encoding="utf-8"))]}
    for nm, (dets, crops, _) in TESTS.items():
        items = []
        for line in open(ROOT / dets, encoding="utf-8"):
            r = json.loads(line)
            for d in r["dets"]:
                items.append((f"{r['i']:03d}_{d['j']}", ROOT / crops / f"{r['i']:03d}_{d['j']}.png"))
        sets[nm] = items
    for nm, items in sets.items():
        f = out / f"feat_{nm}.npz"
        if f.is_file():
            print(nm, "이미 있음")
            continue
        P, L, M = [], [], []
        for k, (cid, png) in enumerate(items):
            p, last, mid = feat(ver, Image.open(png).convert("RGB"))
            P.append(p); L.append(last); M.append(mid)
            if k % 50 == 0:
                print(nm, k, len(items), round(p, 3), flush=True)
        np.savez_compressed(f, ids=np.array([i for i, _ in items]), p=np.array(P),
                            last=np.array(L, dtype=np.float32), mid=np.array(M, dtype=np.float32))
        print(nm, "저장", len(items))


def test_rows(nm):
    dets, _, _ = TESTS[nm]
    rows = []
    for line in open(ROOT / dets, encoding="utf-8"):
        r = json.loads(line)
        for d in r["dets"]:
            rows.append({"id": f"{r['i']:03d}_{d['j']}", "image": nfc(r["image"]), "kind": kind(r["image"]),
                         "smoke": d["smoke"], "p_old": d["p_yes"]})
    return rows


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def cmd_eval(a):
    import numpy as np
    out = Path(a.out)
    tr = [json.loads(l) for l in open(out / "train.jsonl", encoding="utf-8")]
    F = {nm: np.load(out / f"feat_{nm}.npz") for nm in ("train", "eval", "mix150")}
    y = np.array([r["smoke"] for r in tr], dtype=int)
    assert list(F["train"]["ids"]) == [r["id"] for r in tr]
    report = {}
    for layer in ("last", "mid"):
        for C in a.C:
            clf = Logit(C).fit(F["train"][layer], y)
            for nm in ("eval", "mix150"):
                rows = test_rows(nm)
                assert list(F[nm]["ids"]) == [r["id"] for r in rows]
                q = clf.proba(F[nm][layer])
                key = f"{layer} C={C} {nm}"
                report[key] = summarize(rows, q)
    for nm in ("eval", "mix150"):
        rows = test_rows(nm)
        report[f"지금 검증기 {nm}"] = summarize(rows, np.array([r["p_old"] for r in rows]))
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_thr(a):
    """문턱을 시험셋을 보지 않고 정한다.
    학습 조각을 장소로 묶어 5겹으로 나눠 out-of-fold 점수를 만들고,
    지금 검증기(문턱 0.7)가 학습 조각에서 연기를 남기는 비율과 같아지는 문턱을 고른다."""
    import numpy as np
    out = Path(a.out)
    tr = [json.loads(l) for l in open(out / "train.jsonl", encoding="utf-8")]
    F = {nm: np.load(out / f"feat_{nm}.npz") for nm in ("train", "eval", "mix150")}
    y = np.array([r["smoke"] for r in tr], dtype=int)
    places = sorted({r["place"] for r in tr})
    rng = np.random.default_rng(0)
    fold_of = {pl: i % 5 for i, pl in enumerate(rng.permutation(places))}
    fold = np.array([fold_of[r["place"]] for r in tr])
    keep_old = float((F["train"]["p"][y == 1] >= 0.7).mean())
    report = {"지금 검증기 학습 조각 연기 남김 비율": round(keep_old, 3)}
    for layer, C in a.cfg:
        X = F["train"][layer]
        oof = np.zeros(len(y))
        for k in range(5):
            m = fold == k
            oof[m] = Logit(C).fit(X[~m], y[~m]).proba(X[m])
        tau = float(np.quantile(oof[y == 1], 1 - keep_old))
        clf = Logit(C).fit(X, y)
        for nm in ("eval", "mix150"):
            rows = test_rows(nm)
            q = clf.proba(F[nm][layer])
            report[f"{layer} C={C} 문턱 {tau:.3f} {nm}"] = summarize(rows, q, tau)
            mix = (q + np.array([r["p_old"] for r in rows])) / 2
            report[f"{layer} C={C} 평균(지금+프로브) 0.7 {nm}"] = summarize(rows, mix, 0.7)
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    json.dump(report, open(out / "report_thr.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_final(a):
    """고른 설정(층, C)을 기본 문턱 0.5로 보고, 지금 검증기(0.7)와 둘 다 통과해야 남기는 경우도 본다."""
    import numpy as np
    out = Path(a.out)
    tr = [json.loads(l) for l in open(out / "train.jsonl", encoding="utf-8")]
    y = np.array([r["smoke"] for r in tr], dtype=int)
    F = {nm: np.load(out / f"feat_{nm}.npz") for nm in ("train", "eval", "mix150")}
    report = {}
    for layer in ("last", "mid"):
        clf = Logit(a.C).fit(F["train"][layer], y)
        for nm in ("eval", "mix150"):
            rows = test_rows(nm)
            q = clf.proba(F[nm][layer])
            report[f"{layer} C={a.C} 문턱 0.5 {nm}"] = summarize(rows, q, 0.5)
            if layer == "last":
                old = np.array([r["p_old"] for r in rows])
                both = ((q >= 0.5) & (old >= 0.7)).astype(float)
                report[f"둘 다 통과 {nm}"] = summarize(rows, both, 0.5)
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    json.dump(report, open(out / "report_tau05.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_lopo(a):
    """장소를 하나씩 빼고 학습해, 뺀 장소의 조각을 처음 보는 장소로 시험한다 (leave-one-place-out).
    150장 세트의 안개연무는 학습과 장소가 겹쳐서, 학습 조각 안에서 장소 단위로 다시 잰다.
    같은 조각에 지금 검증기(P(예) 0.7)를 쓴 결과도 함께 적는다."""
    import numpy as np
    from collections import defaultdict
    out = Path(a.out)
    tr = [json.loads(l) for l in open(out / "train.jsonl", encoding="utf-8")]
    F = np.load(out / "feat_train.npz")
    X, p_old = F[a.layer], F["p"]
    y = np.array([r["smoke"] for r in tr], dtype=int)
    place = np.array([r["place"] for r in tr])
    kind = np.array(["연기" if r["smoke"] else r["kind"] for r in tr])
    q = np.zeros(len(y))
    for pl in sorted(set(place)):
        m = place == pl
        q[m] = Logit(a.C).fit(X[~m], y[~m]).proba(X[m])
    report = {}
    for k in ("연기", "안개연무", "굴뚝연기", "구름", "백색회색연기", "흑색연기"):
        rows = defaultdict(lambda: [0, 0, 0])      # 장소 -> [수, 분류기 맞힘, 지금 검증기 맞힘]
        for i in np.where(kind == k)[0]:
            good_new = q[i] >= 0.5 if k == "연기" else q[i] < 0.5
            good_old = p_old[i] >= 0.7 if k == "연기" else p_old[i] < 0.7
            r = rows[place[i]]
            r[0] += 1; r[1] += int(good_new); r[2] += int(good_old)
        if not rows:
            continue
        tot = [sum(v[j] for v in rows.values()) for j in range(3)]
        what = "남김" if k == "연기" else "거름"
        report[k] = {"기준": what, "전체": {"수": tot[0], "분류기": tot[1], "지금 검증기": tot[2]},
                     "장소별": {pl: {"수": v[0], "분류기": v[1], "지금 검증기": v[2]} for pl, v in sorted(rows.items())}}
        print(f"{k} ({what}): 분류기 {tot[1]}/{tot[0]}, 지금 검증기 {tot[2]}/{tot[0]}")
        for pl, v in sorted(rows.items()):
            print(f"    {pl}: {v[1]}/{v[0]} (지금 {v[2]}/{v[0]})")
    # 연기를 지금 검증기와 같은 수만큼 남기는 문턱에서 견준다 (문턱이 달라 생기는 차이를 없앤다)
    k_old = int((p_old[y == 1] >= 0.7).sum())
    tau_m = float(np.sort(q[y == 1])[::-1][k_old - 1])
    same = {"문턱": round(tau_m, 3), "연기 남김": f"{int((q[y == 1] >= tau_m).sum())}/{int(y.sum())}"}
    for k in ("안개연무", "굴뚝연기", "구름", "백색회색연기", "흑색연기"):
        m = kind == k
        same[k] = {"분류기": f"{int((q[m] < tau_m).sum())}/{int(m.sum())}",
                   "지금 검증기": f"{int((p_old[m] < 0.7).sum())}/{int(m.sum())}"}
    report["같은 연기 남김에서"] = same
    print("같은 연기 남김에서", json.dumps(same, ensure_ascii=False))
    pos, neg = q[y == 1], q[y == 0]
    report["AUC"] = {"분류기": round(auc(list(pos), list(neg)), 3),
                     "지금 검증기": round(auc(list(p_old[y == 1]), list(p_old[y == 0])), 3)}
    report["AUC"] = {k: float(v) for k, v in report["AUC"].items()}
    print("AUC", report["AUC"])
    json.dump(report, open(out / "report_lopo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


class Logit:
    """표준화 + L2 로지스틱 회귀 (클래스 균형 가중치). sklearn 없이 torch LBFGS 로 푼다.
    C 는 sklearn 과 같은 뜻(작을수록 규제가 세다)."""

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        import numpy as np
        import torch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        n, n1 = len(y), max(int(y.sum()), 1)
        w = torch.where(yt > 0, n / (2 * n1), n / (2 * max(n - n1, 1)))
        self.w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ self.w + self.b
            loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).sum()
            loss = loss + 0.5 / self.C * (self.w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def proba(self, X):
        import torch
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        with torch.no_grad():
            return torch.sigmoid(Xt @ self.w + self.b).numpy()


def summarize(rows, q, tau=0.7):
    pos = [s for r, s in zip(rows, q) if r["smoke"]]
    neg = [s for r, s in zip(rows, q) if not r["smoke"]]
    res = {"auc": round(auc(pos, neg), 3),
           "연기 남김": f"{sum(s >= tau for s in pos)}/{len(pos)}",
           "헛탐지 거름": f"{sum(s < tau for s in neg)}/{len(neg)}"}
    for k in ("굴뚝연기", "안개연무"):
        ks = [s for r, s in zip(rows, q) if not r["smoke"] and r["kind"] == k]
        if ks:
            res[k] = f"{sum(s < tau for s in ks)}/{len(ks)}"
    ks = [s for r, s in zip(rows, q) if not r["smoke"] and r["kind"] not in ("굴뚝연기", "안개연무", "구름")]
    res["연기 사진 속 다른 곳"] = f"{sum(s < tau for s in ks)}/{len(ks)}"
    return res


def main():
    ap = argparse.ArgumentParser(description="hard negative linear probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("weights"); p.add_argument("--out", default="runs/hardneg")
    p.add_argument("--conf", type=float, default=0.05); p.add_argument("--conf-neg", type=float, default=0.001); p.add_argument("--max-dets", type=int, default=5)
    p = sub.add_parser("feats"); p.add_argument("--out", default="runs/hardneg")
    p = sub.add_parser("eval"); p.add_argument("--out", default="runs/hardneg")
    p.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p = sub.add_parser("thr"); p.add_argument("--out", default="runs/hardneg")
    p.add_argument("--cfg", nargs=2, action="append", metavar=("LAYER", "C"),
                   type=lambda v: float(v) if v[0].isdigit() else v)
    p = sub.add_parser("final"); p.add_argument("--out", default="runs/hardneg"); p.add_argument("--C", type=float, default=0.01)
    p = sub.add_parser("lopo"); p.add_argument("--out", default="runs/hardneg")
    p.add_argument("--C", type=float, default=0.01); p.add_argument("--layer", default="last")
    a = ap.parse_args()
    if a.cmd == "thr" and not a.cfg:
        a.cfg = [("last", 0.01), ("mid", 0.01)]
    {"prep": cmd_prep, "feats": cmd_feats, "eval": cmd_eval, "thr": cmd_thr, "final": cmd_final, "lopo": cmd_lopo}[a.cmd](a)


if __name__ == "__main__":
    main()
