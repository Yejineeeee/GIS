# -*- coding: utf-8 -*-
"""연기 분할 모델을 학습한다.

    python train_seg.py dataset.yaml [에폭] [이미지크기] [배치] [워커]

VRAM 4GB 기준으로 기본값을 잡았다. 메모리가 모자라면 배치를 먼저 줄이고,
그래도 모자라면 이미지 크기를 640에서 512로 내린다.

워커는 사진을 읽어 들이는 프로세스 수다. 0으로 두면 메인 스레드가 혼자 읽는데,
mosaic 증강이 샘플 하나에 사진 4장을 읽으므로 GPU 가 데이터를 기다리며 놀게 된다.
워커 0으로 재 보니 에폭당 47초로 CPU 학습과 차이가 거의 없었다. 그래서 기본값을
4로 둔다.

Windows 주의. 아래 main 가드가 없으면 데이터 로더가 프로세스를 새로 띄울 때
이 파일을 처음부터 다시 실행해서 학습이 무한히 겹친다. 가드를 지우지 말 것.
워커를 1 이상으로 쓸 수 있는 것도 이 가드가 있기 때문이다.
"""
import sys, os, time


def main():
    import torch
    from ultralytics import YOLO

    DATA = sys.argv[1] if len(sys.argv) > 1 else "data/yolo/dataset.yaml"
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    MODEL = os.environ.get("SEG_MODEL", "yolo11n-seg.pt")

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"데이터 {DATA}\n모델 {MODEL}  장치 {dev}")
    print(f"에폭 {EPOCHS}  이미지 {IMGSZ}  배치 {BATCH}  워커 {WORKERS}")
    if dev.startswith("cuda"):
        print(f"여유 VRAM {torch.cuda.mem_get_info()[0]/1024**3:.2f}GB")

    t0 = time.time()
    # ultralytics 에는 전역 runs_dir 설정이 있어서 상대 경로를 주면 저장소 밖으로
    # 나간다. 절대 경로로 못박는다.
    project = os.path.abspath("runs")
    model = YOLO(MODEL)
    model.train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=dev,
                project=project, name="smoke", exist_ok=True,
                workers=WORKERS, plots=False, verbose=False)
    save_dir = model.trainer.save_dir
    train_s = time.time() - t0

    m = model.val(data=DATA, imgsz=IMGSZ, batch=BATCH, device=dev,
                  project=project, name="smoke_val", exist_ok=True,
                  workers=WORKERS, plots=False, verbose=False)
    print(f"\n학습 {train_s/60:.1f}분")
    if dev.startswith("cuda"):
        print(f"최대 VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
    print(f"박스 mAP50 {m.box.map50:.3f}   분할 mAP50 {m.seg.map50:.3f}")
    print(f"가중치 {os.path.join(save_dir, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
