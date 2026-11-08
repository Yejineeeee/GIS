# -*- coding: utf-8 -*-
"""최종 보고서용 그림을 results/report_figs/ 에 만든다.

    python tools/report_figs.py            # 전부
    python tools/report_figs.py pick verify curves

그림마다 쓰는 자료
    pick     runs/pick_all (후보 밑동에 번호를 붙여 VLM 이 고르게 한 실험, docs/vlm_base_pick.md ①)
    verify   runs/verify_real (후보 검증기, 실제 사진 100장, docs/vlm_base_pick.md ③)
    train    runs/smoke_1260_960/weights/best.pt 안의 학습 기록 (운영 모델)
    curves   runs/smoke_*/results.csv (큰 연기 재학습, docs/train_results.md)
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report_figs"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def fig_pick():
    rows = [json.loads(l) for l in open(ROOT / "runs/pick_all/vlm_pick.jsonl", encoding="utf-8")]
    by = {r["image"]: r for r in rows}
    show = ["sim_0011.png", "sim_0002.png"]
    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], hspace=0.28, wspace=0.05)
    for k, name in enumerate(show):
        r = by[name]
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(Image.open(ROOT / "runs/pick_all" / r["image"].replace(".png", "_pick.png")))
        ax.axis("off")
        ax.set_title(f"{r['distance_m']}m, 후보 {r['n_candidates']}개\n"
                     f"VLM 선택 {r['vlm_no']} ({r['vlm_err_m']:.0f}m 틀림) · "
                     f"conf 1등 {r['top_conf_no']} ({r['now_err_m']:.0f}m) · "
                     f"가장 가까운 후보 {r['best_no']} ({r['best_err_m']:.0f}m)", fontsize=11)
    ax = fig.add_subplot(gs[1, :])
    names = ["지금 방식\n(conf 1등)", "VLM 이 고름", "상한\n(후보 중 최선)"]
    vals = [st.mean(r["now_err_m"] for r in rows), st.mean(r["vlm_err_m"] for r in rows),
            st.mean(r["best_err_m"] for r in rows)]
    bars = ax.barh(names, vals, color=["#888780", "#D85A30", "#1D9E75"])
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}m", va="center", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(f"평균 위치 오차 (시뮬레이터 {len(rows)}장)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("① 후보 밑동에 글자를 붙여 VLM 에게 진짜 밑동을 고르게 한 실험", fontsize=14)
    fig.text(0.5, 0.01, "VLM 은 순서와 상관없이 A 를 고르는 쏠림이 있었고(28장 중 20장), 평균 오차가 지금 방식보다 컸다",
             ha="center", fontsize=10, color="#5F5E5A")
    fig.savefig(OUT / "vlm_pick.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_verify():
    rows = [json.loads(l) for l in open(ROOT / "runs/verify_real/verify_real.jsonl", encoding="utf-8")]
    dets = [(r["i"], d) for r in rows for d in r["dets"]]
    crop = lambda i, d: ROOT / "runs/verify_real/crops" / f"{i:03d}_{d['j']}.png"  # noqa: E731
    dets = [(i, d) for i, d in dets if crop(i, d).is_file()]
    scene = {r["i"]: r["image"].rsplit("_", 1)[0] for r in rows}   # 같은 영상의 이웃 프레임은 한 장만

    def pick(items, n=4):
        out, seen = [], set()
        for i, d in items:
            if scene[i] not in seen:
                out.append((i, d))
                seen.add(scene[i])
            if len(out) == n:
                break
        return out

    smoke = sorted([x for x in dets if x[1]["smoke"]], key=lambda x: -x[1]["p_yes"])
    false = sorted([x for x in dets if not x[1]["smoke"]], key=lambda x: x[1]["p_yes"])
    groups = [("연기 → 남김 (P(예) 높음)", pick(smoke), "#0F6E56"),
              ("헛탐지 → 걸러냄 (P(예) 낮음)", pick(false), "#0F6E56"),
              ("놓친 연기 (가는 초기 연기)", pick(smoke[::-1]), "#A32D2D"),
              ("못 거른 헛탐지 (굴뚝 연기)", pick(false[::-1]), "#A32D2D")]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.5))
    for row, (title, items, color) in enumerate(groups):
        for col in range(4):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(items):
                continue
            i, d = items[col]
            ax.imshow(Image.open(crop(i, d)))
            ax.set_title(f"P(예) {d['p_yes']:.2f} · conf {d['score']:.2f}", fontsize=10, color=color)
        axes[row][0].text(-0.08, 0.5, title, transform=axes[row][0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=11, color=color)
    n_s, n_f = len(smoke), len(false)
    keep = sum(d["p_yes"] >= 0.7 for _, d in smoke)
    cut = sum(d["p_yes"] < 0.7 for _, d in false)
    fig.suptitle("③ 후보 검증기: 탐지 조각마다 VLM 에게 \"연기인가\" 를 묻는다 (실제 사진 100장)", fontsize=14)
    fig.text(0.5, 0.005, f"문턱 0.7 에서 연기 {n_s}개 중 {keep}개를 남기고 헛탐지 {n_f}개 중 {cut}개를 걸렀다. "
             "아래 두 줄은 실패한 예", ha="center", fontsize=10, color="#5F5E5A")
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.97))
    fig.savefig(OUT / "vlm_verify.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_curves():
    import csv
    runs = [("smoke_mix_960", "처음부터 +481장"), ("smoke_mix240_960", "처음부터 +240장"),
            ("smoke_ft", "미세조정 lr 3e-4"), ("smoke_ft2", "미세조정 lr 1e-4")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    keys = [("train/seg_loss", "학습 분할 손실"), ("val/seg_loss", "검증 분할 손실"),
            ("metrics/mAP50(M)", "검증 분할 mAP50\n(처음부터: val_252+큰 연기 / 미세조정: val_252)")]
    for run, label in runs:
        f = ROOT / "runs" / run / "results.csv"
        if not f.is_file():
            continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        ep = [int(float(r["epoch"])) for r in rows]
        for ax, (k, t) in zip(axes, keys):
            if k in rows[0]:
                ax.plot(ep, [float(r[k]) for r in rows], label=label)
                ax.set_title(t)
    for ax in axes:
        ax.set_xlabel("에폭")
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].legend(fontsize=9)
    fig.suptitle("큰 연기를 섞은 재학습 곡선 (채택 안 함, docs/train_results.md)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "retrain_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _op_row(path, conf=0.10):
    J = json.load(open(ROOT / path, encoding="utf-8"))
    return J, min(J["rows"], key=lambda r: abs(r["conf"] - conf))


def fig_confusion():
    """운영 문턱 conf 0.10 에서 사진 단위 혼동행렬 (val_252, eval_100)."""
    sets = [("runs/eval_detect_val/eval_detect.json", "검증셋 val_252 (학습과 같은 장소)"),
            ("runs/eval_detect/eval_detect.json", "평가셋 eval_100 (처음 보는 장소)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (path, title) in zip(axes, sets):
        _, r = _op_row(path)
        tp = round(r["recall_img"] * r["n_pos"])
        fn = r["n_pos"] - tp
        fp = round(r["alarm_neg"] * r["n_neg"])
        tn = r["n_neg"] - fp
        m = [[tp, fn], [fp, tn]]
        ax.imshow([[tp / r["n_pos"], fn / r["n_pos"]], [fp / r["n_neg"], tn / r["n_neg"]]],
                  cmap="Blues", vmin=0, vmax=1)
        for y in range(2):
            for x in range(2):
                tot = r["n_pos"] if y == 0 else r["n_neg"]
                v = m[y][x]
                ax.text(x, y, f"{v}장\n({v / tot:.0%})", ha="center", va="center", fontsize=13,
                        color="white" if v / tot > 0.5 else "#2C2C2A")
        ax.set_xticks([0, 1], ["알람", "알람 없음"])
        ax.set_yticks([0, 1], [f"연기 사진\n{r['n_pos']}장", f"연기 없는 사진\n{r['n_neg']}장"])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("모델 판단")
    fig.suptitle("연기 탐지 성능, 운영 문턱 conf 0.10 (사진 단위)", fontsize=14)
    fig.text(0.5, 0.0, "평가셋의 헛경보 5장은 모두 굴뚝 연기(진짜 연기)이고, 구름 사진 15장에서는 알람 0장 (docs/detect_eval.md)",
             ha="center", fontsize=10, color="#5F5E5A")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "detect_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_examples():
    """평가셋에서 맞힘 / 놓침 / 헛경보 예시를 탐지 마스크와 정답 윤곽으로 그린다."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "src" / "pipeline"))
    from detect import Detector
    J, _ = _op_row("runs/eval_detect/eval_detect.json")
    per = J["per_image"]
    hit = [p for p in per if p["positive"] and any(d["score"] >= 0.10 and d["smoke_overlap"] > 0 for d in p["dets"])]
    hit.sort(key=lambda p: p["area_pct"])
    hits = [hit[len(hit) // 5], hit[len(hit) // 2], hit[-3]]          # 작은, 보통, 큰 연기
    miss = [p for p in per if p["image"] in set(J["missed_at_op"]) and p["area_pct"] > 0.1][:3]
    fa = [p for p in per if not p["positive"] and any(d["score"] >= 0.10 for d in p["dets"])][:3]
    det = Detector(str(ROOT / "runs/smoke_1260_960/weights/best.pt"), conf=0.10)
    groups = [("맞힘", hits, "#0F6E56"), ("놓침", miss, "#A32D2D"), ("헛경보", fa, "#854F0B")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    for row, (title, items, color) in enumerate(groups):
        for col in range(3):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(items):
                continue
            p = items[col]
            jpg = ROOT / "data/eval_100/images/eval" / p["image"]
            lab = json.load(open(ROOT / "data/eval_100/labels/eval" / (jpg.stem + ".json"), encoding="utf-8"))
            img = Image.open(jpg).convert("RGB")
            ax.imshow(img)
            if p["positive"]:
                for an in lab["annotations"]:
                    if an["category_id"] in (1, 2, 3):   # 흑색, 백색/회색 연기, 화염
                        for poly in an["segmentation"]:
                            xy = np.array(poly).reshape(-1, 2)
                            ax.plot(*np.vstack([xy, xy[:1]]).T, color="#FAC775", lw=1.2, ls="--")
            frame, _ = det.run(str(jpg), 0)
            best = max((d["score"] for d in frame["detections"]), default=0)
            for d in frame["detections"]:
                if d.get("polygon"):
                    xy = np.array(d["polygon"])
                    ax.fill(xy[:, 0], xy[:, 1], color="#E24B4A", alpha=0.35)
                    ax.plot(*np.vstack([xy, xy[:1]]).T, color="#E24B4A", lw=1.2)
            top = max(p["dets"], key=lambda d: d["score"], default={})
            sub = f"연기 넓이 {p['area_pct']:.1f}%" if p["positive"] else f"잡은 곳: {top.get('on') or '라벨 없는 곳'}"
            ax.set_title(f"{title} · {sub} · 최고 conf {best:.2f}", fontsize=10, color=color)
    fig.suptitle("평가셋 eval_100 탐지 예시 (빨강: 모델 탐지, 노란 점선: 정답 라벨)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "detect_examples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_scenario():
    """신고 시나리오 세 장면: 드론 화면과 대시보드를 나란히 (docs/simulator.md 자곡동 V-World 3D)."""
    frames = ROOT / "sim/frames_scenario"
    shots = ROOT / "results/demo_scenario/shots"
    stages = [(2, "수색 중 첫 탐지: 밑동이 아파트에 가려 추정이 크게 틀림 → 미확정 (반경 184m)"),
              (23, "추정 위치로 접근: 자연녹지 확정 (반경 79m)"),
              (37, "가까이 다가감: 반경 29m 로 줄며 확정 유지")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5), gridspec_kw={"width_ratios": [1, 1.25]})
    for row, (k, text) in enumerate(stages):
        meta = json.load(open(frames / f"sim_{k:04d}.json", encoding="utf-8"))
        axes[row][0].imshow(Image.open(frames / f"sim_{k:04d}.png"))
        axes[row][0].set_title(f"드론 화면 · 발화점까지 {meta['sweep']['D']:.0f}m", fontsize=11)
        axes[row][1].imshow(Image.open(shots / f"dash_{k:04d}.png"))
        axes[row][1].set_title(text, fontsize=11)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("신고 시나리오 (자곡동 V-World 3D 시뮬레이터)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "scenario.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_train():
    """운영 모델 학습 곡선. results.csv 는 초기화로 잃었지만 best.pt 안에 100에폭 기록이 있다."""
    import torch
    ck = torch.load(ROOT / "runs/smoke_1260_960/weights/best.pt", map_location="cpu", weights_only=False)
    r = ck["train_results"]
    ep = r["epoch"]
    seg = r["metrics/mAP50(M)"]
    best = max(range(len(seg)), key=lambda i: seg[i])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for k, lab in [("train/seg_loss", "학습"), ("val/seg_loss", "검증")]:
        axes[0].plot(ep, r[k], label=lab)
    axes[0].set_title("분할 손실")
    for k, lab in [("train/box_loss", "학습"), ("val/box_loss", "검증")]:
        axes[1].plot(ep, r[k], label=lab)
    axes[1].set_title("박스 손실")
    axes[2].plot(ep, seg, label="분할 mAP50")
    axes[2].plot(ep, r["metrics/mAP50(B)"], label="박스 mAP50")
    axes[2].axvline(ep[best], color="#888780", ls="--", lw=1)
    axes[2].annotate(f"최고 {ep[best]}에폭\n분할 {seg[best]:.3f}", (ep[best], seg[best]),
                     xytext=(-30, -110), textcoords="offset points", fontsize=10, ha="right",
                     bbox=dict(boxstyle="round", fc="white", ec="#B4B2A9"),
                     arrowprops=dict(arrowstyle="->", color="#5F5E5A"))
    axes[2].set_title("검증 mAP50 (val 252장)")
    for ax in axes:
        ax.set_xlabel("에폭")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="upper right" if ax is not axes[2] else "lower right")
    fig.suptitle("운영 모델 학습 곡선 (yolo11n-seg, 1,260장, 960px, 100에폭)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "train_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_size():
    """학습 사진 수에 따른 성능 (tools/data_size_curve.py, 100% 는 운영 모델)."""
    import csv
    pcts = [25, 50, 75, 100]
    n = [252, 504, 756, 1008]
    seg, v_rec, v_fa, e_rec, e_fa = [], [], [], [], []
    for p in pcts:
        name = "smoke_1260_960" if p == 100 else f"size_{p}"
        cmp = "cmp_smoke_1260_960" if p == 100 else f"size_{p}"
        if p == 100:
            seg.append(0.559)
        else:
            rows = list(csv.DictReader(open(ROOT / "runs" / name / "results.csv")))
            k = [c for c in rows[0] if "mAP50(M)" in c][0]
            seg.append(max(float(r[k]) for r in rows))
        v = _op_row(ROOT / "runs" / f"{cmp}_val_252" / "eval_detect.json")[1]
        e = _op_row(ROOT / "runs" / f"{cmp}_eval_100" / "eval_detect.json")[1]
        v_rec.append(100 * v["recall_img"]); v_fa.append(100 * v["bg_alarm"][0] / v["bg_alarm"][1])
        e_rec.append(100 * e["recall_img"]); e_fa.append(100 * e["bg_alarm"][0] / e["bg_alarm"][1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(n, seg, "o-", color="#2C2C2A")
    axes[0].set_title("검증 분할 mAP50 (val 252장)")
    axes[1].plot(n, v_rec, "o-", color="#2C2C2A", label="val 252장")
    axes[1].plot(n, e_rec, "s--", color="#888780", label="평가 100장")
    axes[1].set_title("연기 사진을 찾은 비율 (%)")
    axes[2].plot(n, v_fa, "o-", color="#2C2C2A", label="val 252장")
    axes[2].plot(n, e_fa, "s--", color="#888780", label="평가 100장")
    axes[2].set_title("연기 없는 사진 헛경보 (%)")
    for ax, ys in zip(axes, [[seg], [v_rec, e_rec], [v_fa, e_fa]]):
        for i, y in enumerate(ys):
            for x, val in zip(n, y):
                ax.annotate(f"{val:.3f}" if ax is axes[0] else f"{val:.0f}", (x, val),
                            xytext=(0, 7 if i == 0 else -14), textcoords="offset points",
                            ha="center", fontsize=9)
        ax.set_xticks(n)
        ax.set_xlabel("학습 사진 수")
        ax.spines[["top", "right"]].set_visible(False)
        if ax is not axes[0]:
            ax.legend(fontsize=9)
    fig.suptitle("학습 사진 수에 따른 성능 (같은 설정, conf 0.10)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "data_size_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_existing():
    """이미 있는 그림을 같은 폴더로 모은다."""
    import shutil
    for src in ["results/chart_risk_coverage.png", "results/chart_rule_compare.png",
                "results/chimney_compare.png", "results/demo_area_candidates.png",
                "results/demo_scenario/shots/dash_0037.png"]:
        p = ROOT / src
        if p.is_file():
            name = "dashboard.png" if "dash_" in src else p.name
            shutil.copy(p, OUT / name)


FIGS = {"pick": fig_pick, "verify": fig_verify, "train": fig_train, "size": fig_size, "curves": fig_curves,
        "confusion": fig_confusion, "examples": fig_examples, "scenario": fig_scenario,
        "copy": copy_existing}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
        print("만듦", name)
