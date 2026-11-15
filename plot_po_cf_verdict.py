"""
一張圖回答「多加 PO/CF 有沒有用」(v6 runs、v3 pkl、avg 模型)：
  左：TS MAE —— 盲猜 / baseline / 多任務 TS head / 多任務 PO+CF 相加 (每個點 = 一個 seed)
  右：「哪一類比較嚴重」猜對率 —— 一律猜多數 / 依動作種類猜 / 多任務模型

    python plot_po_cf_verdict.py --out runs/compare_plots_v6/po_cf_verdict.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import plot_runs  # noqa: F401  (套用同一套字型與外觀設定)
import matplotlib.pyplot as plt
from analyze_runs import load_folds, direction_stats
from dataset import load_kimore, mean_baseline
from plot_runs import INK, INK2, MUTED, BASELINE_GRAY, SERIES, L

BLUE, ORANGE = SERIES[0], SERIES[1]


def run_mae(folds, key):
    """與 analyze_runs 相同：五折各自的 MAE 再平均。"""
    return float(np.mean([np.mean(np.abs(f[key] - f["ts_true"])) for f in folds]))


def dot_row(ax, y, vals, color, hollow=False):
    """一列：每個 seed 一個點 + 平均的短直線 + 數值標註。"""
    m = float(np.mean(vals))
    ax.scatter(vals, [y] * len(vals), s=70, zorder=3, linewidths=2,
               facecolors="none" if hollow else color, edgecolors=color)
    ax.plot([m, m], [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/kimore/processed/kimore_full_v3.pkl")
    ap.add_argument("--base", default="v6_base_s42,v6_base_s43,v6_base_s44")
    ap.add_argument("--mt", default="v6_mt_s42,v6_mt_s43,v6_mt_s44")
    ap.add_argument("--out", default="runs/compare_plots_v6/po_cf_verdict.png")
    a = ap.parse_args()

    samples, _ = load_kimore(a.pkl)
    base = [load_folds(n, samples=samples) for n in a.base.split(",")]
    mt = [load_folds(n, samples=samples) for n in a.mt.split(",")]
    guess = float(mean_baseline(samples, n_folds=5, target="ts")["per_fold"][:, 1].mean())
    b_ts = [run_mae(f, "ts_pred") for f in base]
    m_head = [run_mae(f, "ts_pred") for f in mt]
    m_sum = [run_mae(f, "ts_sum") for f in mt]
    ds = [direction_stats(f, samples) for f in mt]
    acc = [100 * d["acc"] for d in ds]
    acc_ex, acc_maj = 100 * ds[0]["acc_blind"], 100 * ds[0]["majority"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.55})

    # ---- 左：TS MAE
    rows = [(L("多任務：PO+CF 相加", "MT: PO+CF sum"), m_sum, ORANGE, True),
            (L("多任務：TS head", "MT: TS head"), m_head, ORANGE, False),
            ("baseline（只學 TS）", b_ts, BLUE, False)]
    for y, (name, vals, c, hollow) in enumerate(rows):
        m = dot_row(ax1, y, vals, c, hollow)
        ax1.text(max(vals) + 0.12, y, f"{m:.2f} ± {np.std(vals, ddof=1):.2f}", va="center", color=INK2, fontsize=9)
    ax1.axvline(guess, color=BASELINE_GRAY, ls="--", lw=2)
    ax1.text(guess - 0.06, len(rows) - 0.3, L(f"盲猜（依動作平均）\n{guess:.2f}", f"guess\n{guess:.2f}"),
             ha="right", va="center", color=MUTED, fontsize=9)
    ax1.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax1.set_ylim(-0.6, len(rows) - 0.1)
    ax1.set_xlim(5.4, max(guess, max(m_sum + m_head)) + 0.5)
    ax1.set_xlabel(L("TS 測試 MAE（分，越小越好）", "TS test MAE (lower is better)"))
    d = np.mean(m_head) - np.mean(b_ts)
    ax1.set_title(L(f"加入 PO/CF 後 TS 沒有變準（{d:+.2f}）", f"Adding PO/CF did not improve TS ({d:+.2f})"),
                  loc="left")
    ax1.grid(axis="y", visible=False)

    # ---- 右：哪一類比較嚴重
    ax2.axvline(acc_maj, color=BASELINE_GRAY, ls=":", lw=2)
    ax2.text(acc_maj - 0.5, 1.75, L(f"一律猜多數\n{acc_maj:.1f}%", f"majority\n{acc_maj:.1f}%"),
             ha="right", va="center", color=MUTED, fontsize=9)
    ax2.axvline(acc_ex, color=BASELINE_GRAY, ls="--", lw=2)
    ax2.text(acc_ex + 0.5, 1.75, L(f"依動作種類猜\n{acc_ex:.1f}%", f"by exercise\n{acc_ex:.1f}%"),
             ha="left", va="center", color=MUTED, fontsize=9)
    m = dot_row(ax2, 1, acc, ORANGE)
    ax2.text(min(acc) - 0.8, 1, f"{m:.1f}%", ha="right", va="center", color=INK2, fontsize=9)
    ax2.set_yticks([1], [L("多任務模型", "MT model")])
    ax2.set_ylim(0.4, 2.1)
    ax2.set_xlim(50, 85)
    ax2.set_xlabel(L(f"猜對率（%，越大越好；n = {ds[0]['n']}）", f"Accuracy (%, n = {ds[0]['n']})"))
    won = sum(x > acc_ex for x in acc)
    ax2.set_title(L(f"判斷 PO/CF 哪一類較差：{won}/{len(acc)} 個 run 贏過依動作猜",
                    f"Which of PO/CF is worse: {won}/{len(acc)} runs beat by-exercise guess"), loc="left")
    ax2.grid(axis="y", visible=False)

    fig.text(0.01, -0.06, L("每個點 = 一個 seed（42/43/44），黑色短線 = 平均。v6 runs、v3 資料、平均最後 10 個 epoch 的模型。"
                            "右圖只計 |PO/15 − CF/35| > 0.20 的樣本。",
                            "Each dot = one seed; black tick = mean."), color=MUTED, fontsize=8.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("已輸出", a.out)


if __name__ == "__main__":
    main()
