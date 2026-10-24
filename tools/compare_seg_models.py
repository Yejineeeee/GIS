# -*- coding: utf-8 -*-
"""분할 모델 후보를 같은 조건에서 재서 비교표를 만든다.

발표할 처리 시간은 현장 기기 기준인 CPU 4코어에서 잰 값이다. GPU 값은 개발 중
반복 실행 속도를 보기 위한 참고다.

측정을 오염시키지 않으려고 이렇게 한다.
- 가중치를 모두 먼저 내려받은 뒤에 잰다
- 이미지를 미리 메모리에 올려 디스크 읽는 시간을 뺀다
- ultralytics 가 보고하는 추론 시간만 쓴다. 전처리와 후처리는 따로 적는다
- 장치마다 모델을 새로 만든다

    python compare_seg_models.py [이미지경로]
"""
import sys, os, csv, statistics

CANDIDATES = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
              "yolov8n-seg.pt", "yolov8s-seg.pt"]
THREADS = 4          # 현장 기기 기준
WARMUP, RUNS = 3, 9
IMGSZ = 640


def measure(name, dev, img, torch):
    from ultralytics import YOLO
    model = YOLO(name)
    model.to(dev)
    for _ in range(WARMUP):
        model.predict(img, imgsz=IMGSZ, device=dev, verbose=False)
    inf, pre, post = [], [], []
    for _ in range(RUNS):
        r = model.predict(img, imgsz=IMGSZ, device=dev, verbose=False)[0]
        inf.append(r.speed["inference"])
        pre.append(r.speed["preprocess"])
        post.append(r.speed["postprocess"])
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return statistics.median(inf), statistics.median(pre), statistics.median(post)


def main():
    import torch, cv2
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    IMAGE = sys.argv[1] if len(sys.argv) > 1 else "data/pilot/synthetic_smoke.jpg"
    if not os.path.exists(IMAGE):
        raise SystemExit(f"{IMAGE} 가 없습니다.")

    torch.set_num_threads(THREADS)
    img = cv2.imread(IMAGE)
    print(f"이미지 {IMAGE} {img.shape[1]}x{img.shape[0]}   입력 {IMGSZ}   "
          f"CPU 스레드 {torch.get_num_threads()}")

    print("\n가중치 준비")
    specs = {}
    for name in CANDIDATES:
        m = YOLO(name)
        specs[name] = (sum(x.numel() for x in m.model.parameters()),
                       get_flops(m.model, IMGSZ),
                       os.path.getsize(name) / 1024**2 if os.path.exists(name) else 0)
        del m
    print("  완료")

    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    result = {name: {} for name in CANDIDATES}
    for dev in devices:                      # 장치별로 몰아서 잰다
        print(f"\n{dev} 측정")
        for name in CANDIDATES:
            result[name][dev] = measure(name, dev, img, torch)
            print(f"  {name:<16} 추론 {result[name][dev][0]:>7.1f}ms")

    rows = []
    print(f"\n{'모델':<16}{'파라미터':>9}{'연산량':>9}{'파일':>8}{'CPU추론':>10}{'GPU추론':>9}")
    for name in CANDIDATES:
        p, g, mb = specs[name]
        cpu = result[name]["cpu"]
        gpu = result[name].get("cuda:0", (0, 0, 0))
        print(f"{name.replace('.pt',''):<16}{p/1e6:>8.2f}M{g:>8.1f}G{mb:>7.1f}MB"
              f"{cpu[0]:>9.0f}ms{gpu[0]:>8.0f}ms")
        rows.append([name.replace(".pt", ""), round(p/1e6, 2), round(g, 1), round(mb, 1),
                     round(cpu[0]), round(cpu[1] + cpu[2]), round(gpu[0]),
                     round(gpu[1] + gpu[2])])

    with open("results/seg_model_compare.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["모델", "파라미터_M", "연산량_GFLOPs", "파일_MB",
                    f"CPU{THREADS}스레드_추론ms", "CPU_전후처리ms",
                    "GPU_추론ms", "GPU_전후처리ms"])
        w.writerows(rows)
    print("\n저장: results/seg_model_compare.csv")


if __name__ == "__main__":
    main()
