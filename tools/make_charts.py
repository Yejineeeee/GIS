# -*- coding: utf-8 -*-
"""모의 실험 결과 표를 그래프 두 장으로 그린다.

입력은 threshold_sim.py 가 만든 threshold_sim_result.csv 이고, 오차 반경별
곡선과 판정 규칙 비교 그림을 저장한다.
"""
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

for cand in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = cand
        break
rcParams["axes.unicode_minus"] = False

INK = "#2B2B2B"
MUTED = "#6E6E6E"
GRID = "#E3E3E3"
SURFACE = "#FCFCFB"
RADIUS_RAMP = {10: "#BCD2EA", 30: "#7FA9D6", 50: "#3B7DD8", 100: "#16456F"}  # 반경은 크기 개념이라 한 색 계열
RULE_COLOR = {"최다비율": "#2E6FB7", "1-2순위격차": "#D9480F"}

rows = []
with open("threshold_sim_result.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["확정정확도"] == "":
            continue
        rows.append({
            "규칙": r["규칙"], "반경": int(r["반경m"]), "방식": r["방식"],
            "임계값": float(r["임계값"]), "확정비율": float(r["확정비율"]),
            "정확도": float(r["확정정확도"]),
        })


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
    if title:
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold", pad=10, loc="left")


# ── 1. 반경별 위험-커버리지 곡선 (최다 비율 규칙, 면적 비율 방식)
fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200, facecolor="white")
for rad in [10, 30, 50, 100]:
    pts = sorted([r for r in rows if r["규칙"] == "최다비율" and r["방식"] == "면적비율" and r["반경"] == rad],
                 key=lambda x: x["확정비율"])
    xs = [p["확정비율"] * 100 for p in pts]
    ys = [p["정확도"] * 100 for p in pts]
    ax.plot(xs, ys, color=RADIUS_RAMP[rad], linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2, label=f"오차 반경 {rad}m", zorder=3)
    ax.annotate(f"{rad}m", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -2),
                color=RADIUS_RAMP[rad], fontsize=10, fontweight="bold")
ax.axhline(95, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
ax.annotate("목표 정확도 95%", (2, 95.4), color=MUTED, fontsize=10)
style(ax, "확정 판정 비율 (%)", "확정 판정 정확도 (%)",
      "오차 반경이 커질수록 같은 정확도를 지키려면 확정을 줄여야 한다")
ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="lower left")
ax.set_ylim(86, 101)
fig.text(0.012, 0.015, "정답 위치 300곳, 용도지역 폴리곤 기준 모의 실험 (최다 비율 규칙, 면적 비율 방식)",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("chart_risk_coverage.png", facecolor="white")
print("saved chart_risk_coverage.png")

# ── 2. 판정 규칙 비교 (반경 50m, 100m)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=200, facecolor="white", sharey=True)
for ax, rad in zip(axes, [50, 100]):
    for rule in ["최다비율", "1-2순위격차"]:
        pts = sorted([r for r in rows if r["규칙"] == rule and r["방식"] == "면적비율" and r["반경"] == rad],
                     key=lambda x: x["확정비율"])
        xs = [p["확정비율"] * 100 for p in pts]
        ys = [p["정확도"] * 100 for p in pts]
        label = "최다 비율 기준" if rule == "최다비율" else "1, 2순위 격차 기준"
        ax.plot(xs, ys, color=RULE_COLOR[rule], linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
    ax.axhline(99, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("목표 99%", (2, 99.2), color=MUTED, fontsize=9)
    style(ax, "확정 판정 비율 (%)", "확정 판정 정확도 (%)" if rad == 50 else "", f"오차 반경 {rad}m")
axes[0].legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="lower left")
axes[0].set_ylim(88, 101)
fig.suptitle("판정 규칙 비교: 최다 비율 기준이 높은 정확도 구간까지 버틴다",
             color=INK, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.015, "정답 위치 300곳, 면적 비율 방식. 격차 기준은 반경 50m와 100m에서 목표 99%를 달성하지 못한다",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig("chart_rule_compare.png", facecolor="white")
print("saved chart_rule_compare.png")
