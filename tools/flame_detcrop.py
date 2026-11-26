# -*- coding: utf-8 -*-
"""불꽃 질문을 파이프라인과 같은 조각(실제 탐지 박스)에서 다시 잰다.

tools/flame_eval.py 는 연기 라벨을 감싼 박스로 조각을 만들었다. 파이프라인에서는 탐지기가 낸 박스 중
검증기를 통과한 conf 1등(없으면 conf 1등)이 알람의 탐지가 되고, 불꽃은 그 조각에 따로 묻는다
(src/pipeline/vlm_client.py VLM.p_flame). 사진은 flame_eval.py 와 같은 57장(불꽃 19, 없음 38)이다.
문턱은 미리 정한 0.7(FLAME_TAU, 검증기와 같은 값)로 보고, 다른 문턱은 참고로만 적는다.

    python tools/flame_detcrop.py <yolo_1260 폴더> <yolo_1260_raw_labels 폴더> <탐지 가중치> --out runs/flame_detcrop.jsonl

현장 구성(llama-server, Qwen2.5-VL-3B Q4_K_M)으로 같은 조각을 다시 물을 때는 앞 결과의 박스를 그대로 쓴다.
사진마다 이미지가 달라 프롬프트 캐시는 맞지 않는다(정확도만 본다).

    python tools/flame_detcrop.py <yolo_1260> <raw_labels> - --vlm gguf --boxes runs/flame_detcrop.jsonl --out runs/flame_detcrop_gguf.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from flame_eval import load, pick, auc, inside  # noqa: E402
from vlm_client import VLM, FLAME_TAU, FLAME_PROMPT  # noqa: E402
from detect import Detector  # noqa: E402

VERIFY_TAU = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("raw_labels")
    ap.add_argument("weights")
    ap.add_argument("--out", default=str(ROOT / "runs" / "flame_detcrop.jsonl"))
    ap.add_argument("--vlm", default="qwen", choices=["qwen", "gguf"])
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8099")
    ap.add_argument("--boxes", help="앞 결과 jsonl. 주면 탐지와 검증을 다시 하지 않고 그 박스에 불꽃만 묻는다")
    a = ap.parse_args()

    pos, neg = pick(load(a.raw_labels))
    print(f"불꽃 있음 {len(pos)}장, 없음 {len(neg)}장", flush=True)
    vlm = VLM(a.vlm, url=a.vlm_url)
    out = []
    if a.boxes:
        prev = [json.loads(l) for l in open(a.boxes, encoding="utf-8")]
        for r in prev:
            path = Path(a.images) / "images" / r["split"] / (r["name"] + ".jpg")
            if r.get("det_bbox"):
                # 파이프라인은 gguf 에서 불꽃을 묻지 않으므로(FLAME_MODES), 잴 때는 같은 질문을 직접 한다
                r["p_flame"], r["flame_ms"] = vlm.p_yes(str(path), r["det_bbox"], FLAME_PROMPT)
                print(f"  {'불꽃' if r['flame'] else '    '} 조각 {r['p_flame']:.2f}  {r['flame_ms']}ms", flush=True)
            out.append(r)
        pos, neg = [], []
    else:
        det = Detector(a.weights)
    for r in pos + neg:
        path = Path(a.images) / "images" / r["split"] / (r["name"] + ".jpg")
        frame, _ = det.run(path, 0)
        dets = frame["detections"]
        r["n_det"] = len(dets)
        if not dets:
            r["p_flame"] = None
            out.append(r)
            print(f"  {'불꽃' if r['flame'] else '    '} 탐지 없음", flush=True)
            continue
        for d in dets:
            d["p_smoke"], _ = vlm.p_smoke(str(path), d["bbox"])
        ok = [d for d in dets if d["p_smoke"] >= VERIFY_TAU]
        top = max(ok or dets, key=lambda d: d["score"])
        r["det_bbox"], r["det_p_smoke"] = top["bbox"], round(top["p_smoke"], 3)
        x, y, w, h = top["bbox"]
        side = max(max(w, h) * 1.5, max(160, frame["image_size"]["width"] // 6))
        cb = (x + w / 2 - side / 2, y + h / 2 - side / 2, x + w / 2 + side / 2, y + h / 2 + side / 2)
        r["flame_in_crop"] = all(inside(b, cb) for b in r["flame_boxes"]) if r["flame"] else None
        r["p_flame"], r["flame_ms"] = vlm.p_flame(str(path), top["bbox"])
        print(f"  {'불꽃' if r['flame'] else '    '} 조각 {r['p_flame']:.2f}  안에 {r['flame_in_crop']}  "
              f"{r['flame_ms']}ms", flush=True)
        out.append(r)

    Path(a.out).parent.mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    P = [r for r in out if r["flame"] and r["p_flame"] is not None]
    N = [r for r in out if not r["flame"] and r["p_flame"] is not None]
    print(f"\n탐지 없음: 불꽃 {sum(r['flame'] and r['p_flame'] is None for r in out)}, "
          f"없음 {sum((not r['flame']) and r['p_flame'] is None for r in out)}")
    print(f"AUC {auc([r['p_flame'] for r in P], [r['p_flame'] for r in N]):.3f}")
    for t in (FLAME_TAU, 0.5, 0.9):
        tp = sum(r["p_flame"] >= t for r in P)
        fp = sum(r["p_flame"] >= t for r in N)
        print(f"문턱 {t}{' (정한 값)' if t == FLAME_TAU else ''}: 불꽃 {tp}/{len(P)}, 헛말 {fp}/{len(N)}")
    print(f"불꽃이 조각 안에 든 사진 {sum(bool(r['flame_in_crop']) for r in P)}/{len(P)}")


if __name__ == "__main__":
    main()
