"""
挑模型方式對照：平均最後 K 個 epoch 的權重 (avg) vs 驗證 MAE 最低的 epoch (best)。
每組每個 seed 一對點 (空心 = best，實心 = avg)，灰線連起同一個 seed。
三格：TS MAE (越小越好)、斜率 (越接近 1 越好)、低分 10–25 的平均偏差 (越接近 0 越好)。
數值與 analyze_runs.py (compare_v6_seeds.md) 相同。

    python plot_avg_vs_best.py --out runs/compare_plots_v6/avg_vs_best.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import plot_runs  # noqa: F401  (字型與外觀)
import matplotlib.pyplot as plt
from analyze_runs import load_folds, analyze
from dataset import load_kimore
from plot_runs import INK, INK2, MUTED, AXIS, SURFACE, SERIES, L

METRICS = [  # (key, 標題, 參考線, 方向說明)
    ("fold_mae_mean", L("TS MAE（越小越好）", "TS MAE (lower better)"), None),
    ("slope", L("斜率（越接近 1 越好）", "Slope (closer to 1 better)"), 1.0),
    ("bias_lo", L("10–25 分平均偏差（越接近 0 越好）", "Bias on TS 10–25 (closer to 0)"), 0.0),
]


def collect(names, samples, key):
    out = {}
    for sel in ("avg", "best"):
        rows = []
        for n in names:
            r = analyze(load_folds(n, samples=samples, sel=sel), pred_key=key)
            r["bias_lo"] = r["bins"]["10-25"]["bias"]
            rows.append(r)
        out[sel] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/kimore/processed/kimore_full_v3.pkl")
    ap.add_argument("--base", default="v6_base_s42,v6_base_s43,v6_base_s44")
    ap.add_argument("--mt", default="v6_mt_s42,v6_mt_s43,v6_mt_s44")
    ap.add_argument("--out", default="runs/compare_plots_v6/avg_vs_best.png")
    a = ap.parse_args()
    samples, _ = load_kimore(a.pkl)
    base, mt = a.base.split(","), a.mt.split(",")
    groups = [("baseline", base, "ts_pred", SERIES[0]),
              (L("多任務：TS head", "MT: TS head"), mt, "ts_pred", SERIES[1]),
              (L("多任務：PO+CF 相加", "MT: PO+CF sum"), mt, "ts_sum", SERIES[1])]
    data = [(g, collect(ns, samples, k), c) for g, ns, k, c in groups]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"wspace": 0.12}, sharey=True)
    n = len(base)
    offs = np.linspace(-0.22, 0.22, n)
    for ax, (key, title, ref) in zip(axes, METRICS):
        if ref is not None:
            ax.axvline(ref, color=AXIS, lw=1.5)
        for gi, (g, d, c) in enumerate(data):
            y = len(data) - 1 - gi
            av = [r[key] for r in d["avg"]]
            bv = [r[key] for r in d["best"]]
            for o, x0, x1 in zip(offs, bv, av):
                ax.plot([x0, x1], [y + o, y + o], color=AXIS, lw=1.5, zorder=1)
            ax.scatter(bv, y + offs, s=60, facecolors=SURFACE, edgecolors=c, linewidths=2, zorder=3)
            ax.scatter(av, y + offs, s=60, color=c, edgecolors=SURFACE, linewidths=1, zorder=3)
            fmt = "{:+.2f}" if key == "bias_lo" else ("{:.3f}" if key == "slope" else "{:.2f}")
            ax.text(1.0, y + 0.36, L("avg ", "avg ") + fmt.format(np.mean(av)) + L("｜best ", " | best ")
                    + fmt.format(np.mean(bv)), transform=ax.get_yaxis_transform(), ha="right",
                    fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        lo, hi = ax.get_xlim()
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(range(len(data)), [g for g, _, _ in data][::-1])
    axes[0].set_ylim(-0.6, len(data) - 0.35)
    # 圖例：選法用填色表示
    axes[0].scatter([], [], s=60, color=INK2, label=L("avg（平均最後 10 個 epoch）", "avg (last-10 mean)"))
    axes[0].scatter([], [], s=60, facecolors=SURFACE, edgecolors=INK2, linewidths=2,
                    label=L("best（驗證 MAE 最低）", "best (lowest val MAE)"))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    better = sum(np.mean([r["fold_mae_mean"] for r in d["avg"]]) < np.mean([r["fold_mae_mean"] for r in d["best"]])
                 for _, d, _ in data)
    fig.suptitle(L(f"挑模型方式：avg 在 {better}/{len(data)} 組 TS MAE 較低，且斜率更接近 1、低分高估較少",
                   f"Model selection: avg has lower MAE in {better}/{len(data)} groups"),
                 fontweight="bold", color=INK, y=1.1)
    fig.text(0.01, -0.06, L("每個點 = 一個 seed（42/43/44），灰線連起同一個 seed 的 best 與 avg。v6 runs、v3 資料。",
                            "One dot per seed; grey line links best and avg of the same seed."),
             color=MUTED, fontsize=8.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    for g, d, _ in data:
        print(g, {k: (round(np.mean([r[k] for r in d["avg"]]), 3), round(np.mean([r[k] for r in d["best"]]), 3))
                  for k, _, _ in METRICS})
    print("已輸出", a.out)


if __name__ == "__main__":
    main()
