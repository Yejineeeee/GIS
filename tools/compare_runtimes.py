# -*- coding: utf-8 -*-
"""탐지 모델(YOLO11n-seg, 960)을 실행 방식과 칩을 바꿔 돌려 속도와 결과를 견준다 (WBS 7.6).

같은 가중치를 ONNX 와 OpenVINO 로 내보내 두고, 같은 사진을 같은 크기(960x960, 레터박스)로
만들어 모델 계산 시간만 잰다. 결과가 같은지는 Ultralytics 로 끝까지 예측해 견준다. PyTorch(.pt)는 Ultralytics 가 사진 비율에 맞춰
여백을 줄인 입력(예: 960x544)을 쓰고, 내보낸 모델은 960x960 고정 입력을 쓴다. 그래서 같은 고정 입력을
쓰는 TorchScript 를 기준으로 견주고, .pt 와의 차이는 따로 적는다.

    yolo export model=runs/compat/yolo11n_seg_960.pt format=openvino imgsz=960
    yolo export model=runs/compat/yolo11n_seg_960.pt format=onnx imgsz=960
    yolo export model=runs/compat/yolo11n_seg_960.pt format=torchscript imgsz=960
    python tools/compare_runtimes.py --images data/eval_100/images/eval --out runs/compat

조건
    PyTorch      CPU(스레드 4), NVIDIA GPU
    ONNX Runtime CPU(스레드 4)
    OpenVINO     CPU(스레드 4), 인텔 내장 GPU, 인텔 NPU, NVIDIA GPU(OpenVINO 가 잡으면)

시간은 사진마다 한 번씩, 앞의 5장은 데우기로 버리고 중앙값을 쓴다.
결과는 conf 0.10 에서 연기 사진 중 탐지가 있는 수와 오탐 유발 사진 중 탐지가 있는 수(헛경보)를 세고,
TorchScript 와 알람 일치, 탐지 수 일치, 마스크를 합친 영역의 IoU(원본 사진 크기)를 견준다.
"""
import argparse
import json
import statistics
import time
import unicodedata
from pathlib import Path

import numpy as np

THREADS = 4
SIZE = 960
WARM = 5
CONF = 0.10


def fixed(name):
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unicodedata.normalize("NFC", name)


def load_inputs(paths):
    import cv2
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(SIZE, SIZE), auto=False)
    arrs = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        x = lb(image=img)[:, :, ::-1].transpose(2, 0, 1)
        arrs.append(np.ascontiguousarray(x[None], dtype=np.float32) / 255.0)
    return arrs


def timed(fn, arrs):
    ms = []
    for i, x in enumerate(arrs):
        t0 = time.perf_counter()
        fn(x)
        if i >= WARM:
            ms.append((time.perf_counter() - t0) * 1000)
    return ms


def raw_times(a, arrs):
    out = {}
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(THREADS)
    net = YOLO(a.pt).model.float().eval()
    with torch.inference_mode():
        out["PyTorch / CPU"] = timed(lambda x: net(torch.from_numpy(x)), arrs)
        if torch.cuda.is_available():
            g = YOLO(a.pt).model.float().eval().cuda()

            def run(x):
                g(torch.from_numpy(x).cuda())
                torch.cuda.synchronize()
            out["PyTorch / NVIDIA GPU"] = timed(run, arrs)
    print("PyTorch 끝", flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    out["ONNX Runtime / CPU"] = timed(lambda x: s.run(None, {name: x}), arrs)
    print("ONNX Runtime 끝", flush=True)

    import openvino as ov
    core = ov.Core()
    model = core.read_model(a.ov_xml)
    devs = {"CPU": "OpenVINO / CPU", "GPU.0": "OpenVINO / 인텔 내장 GPU", "NPU": "OpenVINO / 인텔 NPU",
            "GPU.1": "OpenVINO / NVIDIA GPU"}
    for d, label in devs.items():
        if d not in core.available_devices:
            continue
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if d == "CPU":
            cfg["INFERENCE_NUM_THREADS"] = THREADS
        try:
            t0 = time.perf_counter()
            cm = core.compile_model(model, d, cfg)
            comp = time.perf_counter() - t0
            req = cm.create_infer_request()
            out[label] = timed(lambda x: req.infer({0: x}), arrs)
            out[label + " 준비(초)"] = comp
            print(label, "끝", f"(준비 {comp:.1f}초)", flush=True)
        except Exception as e:                          # 칩이 이 모델을 못 돌리면 이유를 남긴다
            out[label] = f"실패: {type(e).__name__}: {str(e)[:200]}"
            print(label, out[label], flush=True)
    return out


def union_mask(r):
    if r.masks is None or len(r.masks) == 0:
        return None
    m = r.masks.data.cpu().numpy() > 0.5
    return m.any(0)


def agreement(a, paths):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(THREADS)
    runs = {"PyTorch .pt / CPU": (a.pt, "cpu"), "TorchScript / CPU": (a.ts, "cpu"),
            "OpenVINO / CPU": (a.ov_dir, "intel:cpu"),
            "OpenVINO / 인텔 내장 GPU": (a.ov_dir, "intel:gpu"), "OpenVINO / 인텔 NPU": (a.ov_dir, "intel:npu"),
            "ONNX Runtime / CPU": (a.onnx, "cpu")}
    preds = {}
    for label, (w, dev) in runs.items():
        try:
            m = YOLO(w, task="segment")
            res = []
            for p in paths:
                r = m.predict(str(p), imgsz=SIZE, conf=CONF, device=dev, verbose=False, retina_masks=True)[0]
                res.append((len(r.boxes), union_mask(r), "PositiveRealDB" in p.name))
            preds[label] = res
            print("예측", label, "끝", flush=True)
        except Exception as e:
            preds[label] = f"실패: {type(e).__name__}: {str(e)[:200]}"
            print("예측", label, preds[label], flush=True)
    ref = preds["TorchScript / CPU"]
    pt = preds["PyTorch .pt / CPU"]
    summ = {}
    for label, res in preds.items():
        if isinstance(res, str):
            summ[label] = res
            continue
        alarm = sum((n > 0) == (rn > 0) for (n, _, _), (rn, _, _) in zip(res, ref))
        alarm_pt = sum((n > 0) == (rn > 0) for (n, _, _), (rn, _, _) in zip(res, pt))
        count = sum(n == rn for (n, _, _), (rn, _, _) in zip(res, ref))
        hit = sum(n > 0 for n, _, pos in res if pos)
        fa = sum(n > 0 for n, _, pos in res if not pos)
        npos = sum(pos for _, _, pos in res)
        ious = []
        for (_, m, _), (_, rm, _) in zip(res, ref):
            if m is None and rm is None:
                continue
            if m is None or rm is None or m.shape != rm.shape:
                ious.append(0.0 if (m is None) != (rm is None) else float("nan"))
                continue
            u = (m | rm).sum()
            ious.append(float((m & rm).sum() / u) if u else 1.0)
        ious = [x for x in ious if x == x]
        summ[label] = {"연기 사진 잡음": f"{hit}/{npos}", "헛경보": f"{fa}/{len(res) - npos}",
                       "TorchScript 와 알람 일치": f"{alarm}/{len(ref)}",
                       "TorchScript 와 탐지 수 일치": f"{count}/{len(ref)}",
                       "TorchScript 와 마스크 IoU 평균": round(statistics.mean(ious), 3) if ious else None,
                       ".pt 와 알람 일치": f"{alarm_pt}/{len(pt)}"}
    return summ


def main():
    ap = argparse.ArgumentParser(description="탐지 모델 실행 방식과 칩 비교")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="runs/compat")
    ap.add_argument("--pt", default="runs/compat/yolo11n_seg_960.pt")
    ap.add_argument("--onnx", default="runs/compat/yolo11n_seg_960.onnx")
    ap.add_argument("--ov-dir", default="runs/compat/yolo11n_seg_960_openvino_model")
    ap.add_argument("--ts", default="runs/compat/yolo11n_seg_960.torchscript")
    a = ap.parse_args()
    a.ov_xml = str(Path(a.ov_dir) / "yolo11n_seg_960.xml")
    paths = sorted(Path(a.images).glob("*.jpg"), key=lambda p: fixed(p.name))
    arrs = load_inputs(paths)
    print(f"사진 {len(arrs)}장, 입력 {arrs[0].shape}", flush=True)

    raw = raw_times(a, arrs)
    agree = agreement(a, paths)

    print(f"\n[모델 계산 시간] 입력 1x3x{SIZE}x{SIZE}, 앞 {WARM}장 빼고 {len(arrs) - WARM}장, CPU 는 스레드 {THREADS}")
    rows = {}
    for k, v in raw.items():
        if k.endswith("준비(초)"):
            continue
        if isinstance(v, str):
            print(f"  {k:<28} {v}")
            rows[k] = v
            continue
        med = statistics.median(v)
        prep = raw.get(k + " 준비(초)")
        rows[k] = {"중앙값_ms": round(med, 1), "최소_ms": round(min(v), 1), "최대_ms": round(max(v), 1),
                   "준비_초": None if prep is None else round(prep, 1)}
        print(f"  {k:<28} 중앙값 {med:7.1f}ms  (최소 {min(v):.1f} ~ 최대 {max(v):.1f})"
              + ("" if prep is None else f"  모델 준비 {prep:.1f}초"))
    print("\n[결과] conf 0.10. 견주는 기준은 같은 960x960 입력을 쓰는 TorchScript")
    for k, v in agree.items():
        print(f"  {k:<28} {v}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "compare_runtimes.json", "w", encoding="utf-8") as fo:
        json.dump({"times": rows, "agreement": agree}, fo, ensure_ascii=False, indent=1)
    print(f"\n-> {Path(a.out) / 'compare_runtimes.json'}")


if __name__ == "__main__":
    main()
