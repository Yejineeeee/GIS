"""크롭 CNN 기준선: VLM 검증기 대신 작은 CNN 분류기를 같은 조각으로 학습해 비교한다.

hard negative 실험(tools/hardneg_probe.py)이 만든 학습 조각 2,025개와 같은 시험 조각을 쓴다.
백본은 YOLO11n 의 앞 11층(Conv ~ C2PSA)이고, 전역 평균 풀링과 선형층을 붙여 통째로 미세조정한다.
사전학습 가중치는 새로 받지 않고 저장소에 있는 것만 쓴다.
  coco  : yolo11n-seg.pt (COCO 사전학습)
  smoke : runs/smoke_1260_960/weights/best.pt (우리 탐지기, 같은 학습 사진으로 학습)

학습 조각의 장소 20% 를 검증용으로 떼어 그 AUC 로 에폭을 고른다. 시험셋은 고르는 데 쓰지 않는다.
문턱은 클래스 균형 손실의 기본값 0.5.

    python tools/crop_cnn.py --init coco smoke --seeds 0 1 2 --out runs/crop_cnn
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hardneg_probe as H  # noqa: E402

INITS = {"coco": "yolo11n-seg.pt", "smoke": "runs/smoke_1260_960/weights/best.pt"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(paths, size):
    return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
                                      for p in paths])).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,S,S


class Net(nn.Module):
    def __init__(self, init):
        super().__init__()
        from ultralytics import YOLO
        layers = YOLO(str(ROOT / INITS[init])).model.model[:11]
        self.body = nn.Sequential(*layers)
        for p in self.body.parameters():
            p.requires_grad = True
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3))).squeeze(1)


def prep(x, dev):
    return ((x.to(dev).float() / 255) - MEAN.to(dev)) / STD.to(dev)


def augment(x):
    """좌우 뒤집기, 밝기/대비 흔들기, 작은 이동. x: float N,3,S,S (0~1 전)"""
    if np.random.rand() < 0.5:
        x = x.flip(3)
    b = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.4
    c = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.4
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    dx, dy = np.random.randint(-16, 17, 2)
    return torch.roll(x, (int(dy), int(dx)), (2, 3))


@torch.no_grad()
def predict(net, X, dev, bs=64):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(net(prep(X[i:i + bs], dev))).cpu())
    return torch.cat(out).numpy()


def train_one(init, seed, Xtr, ytr, Xva, yva, dev, epochs, size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(init).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n1 = ytr.sum()
    pw = torch.tensor((len(ytr) - n1) / n1, device=dev)  # 연기:비연기 균형
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(ytr, dtype=torch.float32)
    best = (-1, None, -1)
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]
            x = augment(Xtr[b].to(dev).float() / 255)
            x = (x - MEAN.to(dev)) / STD.to(dev)
            loss = lossf(net(x), yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        q = predict(net, Xva, dev)
        a = H.auc(list(q[yva == 1]), list(q[yva == 0]))
        if a > best[0]:
            best = (a, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    return net, best[0], best[2]


def cpu_time(net, size, n=20):
    net = net.cpu().eval()
    torch.set_num_threads(4)
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(3):
            net(x)
        t = time.perf_counter()
        for _ in range(n):
            net(x)
    return (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", nargs="+", default=["coco", "smoke"], choices=list(INITS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--hn", default="runs/hardneg")
    ap.add_argument("--out", default="runs/crop_cnn")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = [json.loads(l) for l in open(ROOT / a.hn / "train.jsonl", encoding="utf-8")]
    y = np.array([r["smoke"] for r in tr], dtype=int)
    places = sorted({r["place"] for r in tr})
    rng = np.random.default_rng(0)
    va_places = set(rng.permutation(places)[: max(1, len(places) // 5)])
    va = np.array([r["place"] in va_places for r in tr])
    X = load([ROOT / a.hn / "crops" / f"{r['id']}.png" for r in tr], a.size)
    print("학습", int((~va).sum()), "검증", int(va.sum()), "장소", len(places), flush=True)

    tests = {}
    for nm, (dets, crops, _) in H.TESTS.items():
        rows = H.test_rows(nm)
        tests[nm] = (rows, load([ROOT / crops / f"{r['id']}.png" for r in rows], a.size))

    report = {}
    for init in a.init:
        qs = {nm: [] for nm in tests}
        for seed in a.seeds:
            net, va_auc, ep = train_one(init, seed, X[~va], y[~va], X[va], y[va], dev, a.epochs, a.size)
            for nm, (rows, Xt) in tests.items():
                q = predict(net, Xt, dev)
                qs[nm].append(q)
                key = f"{init} seed={seed} {nm}"
                report[key] = H.summarize(rows, q, 0.5)
                print(key, f"(검증 AUC {va_auc:.3f}, 에폭 {ep})", json.dumps(report[key], ensure_ascii=False), flush=True)
        for nm, (rows, _) in tests.items():
            q = np.mean(qs[nm], 0)
            report[f"{init} 시드 평균 점수 {nm}"] = H.summarize(rows, q, 0.5)
            aucs = [H.summarize(rows, qq, 0.5)["auc"] for qq in qs[nm]]
            report[f"{init} AUC 범위 {nm}"] = [min(aucs), max(aucs)]
        report[f"{init} CPU 1회(초, 4스레드, {a.size}px)"] = round(cpu_time(net, a.size), 4)
        net.to(dev)
    for nm, (rows, _) in tests.items():
        report[f"지금 검증기 {nm}"] = H.summarize(rows, np.array([r["p_old"] for r in rows]), 0.7)
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
