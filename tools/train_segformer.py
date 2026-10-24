# -*- coding: utf-8 -*-
"""SegFormer-B0 로 연기 의미 분할을 학습한다.

YOLO-seg 와 같은 데이터로 돌려 개체 분할과 의미 분할을 비교하려는 것이다.
YOLO 쪽 설정은 docs/train_results.md 에 있다.

    python train_segformer.py 데이터폴더 [에폭] [크기] [배치]

데이터 폴더는 make_masks.py 가 만든 구조여야 한다.
    images/train/*.jpg, masks/train/*.png (0 배경, 1 연기)

Windows 주의. 데이터 로더 워커를 쓰므로 아래 main 가드를 지우면 안 된다.
"""
import glob
import os
import sys
import time

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/seg_1260"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 960
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 4
OUT = "runs/segformer"
MODEL = os.environ.get("SEGFORMER_MODEL", "nvidia/mit-b0")


class SmokeSet:
    """사진과 마스크를 짝지어 낸다. 좌우 뒤집기만 증강으로 쓴다."""

    def __init__(self, root, split, size, train):
        self.items = []
        for ip in sorted(glob.glob(os.path.join(root, "images", split, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(root, "masks", split, stem + ".png")
            if os.path.exists(mp):
                self.items.append((ip, mp))
        self.size, self.train = size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import numpy as np
        import torch
        from PIL import Image
        ip, mp = self.items[i]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp)
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        # ImageNet 평균과 표준편차. 사전학습 가중치가 이 기준으로 학습됐다
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        y = torch.from_numpy(np.array(mask)).long()
        return x, y


def evaluate(model, loader, dev, torch):
    """연기 픽셀의 IoU 를 낸다. 배경이 대부분이라 정확도는 의미가 없다."""
    model.eval()
    inter = union = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(pixel_values=x).logits
            logits = torch.nn.functional.interpolate(
                logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(1)
            inter += ((pred == 1) & (y == 1)).sum().item()
            union += ((pred == 1) | (y == 1)).sum().item()
    model.train()
    return inter / union if union else 0.0


def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import SegformerForSemanticSegmentation

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"데이터 {DATA}\n모델 {MODEL}  장치 {dev}")
    print(f"에폭 {EPOCHS}  크기 {SIZE}  배치 {BATCH}")
    if dev.startswith("cuda"):
        print(f"여유 VRAM {torch.cuda.mem_get_info()[0] / 1024**3:.2f}GB")

    tr = SmokeSet(DATA, "train", SIZE, True)
    va = SmokeSet(DATA, "val", SIZE, False)
    print(f"학습 {len(tr)}장, 검증 {len(va)}장")

    trl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4,
                     drop_last=True, persistent_workers=True)
    val = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2,
                     persistent_workers=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=6e-5, total_steps=EPOCHS * len(trl), pct_start=0.1)
    scaler = torch.amp.GradScaler(dev) if dev.startswith("cuda") else None

    # 연기 픽셀이 전체의 몇 퍼센트뿐이라 가중치를 준다
    w = torch.tensor([1.0, 8.0]).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "results.csv"), "w", encoding="utf-8")
    log.write("epoch,loss,val_iou,sec\n")
    best = -1.0
    t_all = time.time()
    model.train()
    for ep in range(1, EPOCHS + 1):
        t0, tot = time.time(), 0.0
        for x, y in trl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=scaler is not None):
                logits = model(pixel_values=x).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = lossf(logits, y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            tot += loss.item()
        iou = evaluate(model, val, dev, torch)
        sec = time.time() - t0
        log.write(f"{ep},{tot / len(trl):.4f},{iou:.4f},{sec:.1f}\n")
        log.flush()
        if iou > best:
            best = iou
            model.save_pretrained(os.path.join(OUT, "best"))
        print(f"  에폭 {ep:3}/{EPOCHS}  손실 {tot / len(trl):.4f}  "
              f"연기 IoU {iou:.4f}  {sec:.1f}초", flush=True)

    log.close()
    print(f"\n학습 {(time.time() - t_all) / 60:.1f}분")
    if dev.startswith("cuda"):
        print(f"최대 VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
    print(f"가장 좋은 연기 IoU {best:.4f}")
    print(f"가중치 {os.path.join(OUT, 'best')}")


if __name__ == "__main__":
    main()
