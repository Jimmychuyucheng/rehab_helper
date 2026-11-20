"""
兩張說明用的圖 (baseline，單任務 TS)：

  story_1_seed_variance.png  單跑不可靠：上週兩次單跑 vs v5 同設定重跑 3 次 (seed 42/43/44，分組相同)
  story_2_ensemble.png       MAE 一路下降：v5 單一模型 (best) → 平均權重 → 資料修正 (v6) → 3 個模型取平均

    python plot_story.py --out runs/v6/compare_plots_v6
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import plot_runs  # noqa: F401  (字型與外觀)
import matplotlib.pyplot as plt
from analyze_runs import load_folds
from dataset import load_kimore
from plot_runs import INK, INK2, MUTED, AXIS, SURFACE, SERIES, BASELINE_GRAY, L

PKL_OLD = "data/kimore/processed/kimore_full.pkl"
PKL_V2 = "data/kimore/processed/kimore_full_pofix.pkl"
PKL_V3 = "data/kimore/processed/kimore_full_v3.pkl"
SEEDS = (42, 43, 44)


def fold_maes(folds, pred=None):
    return [float(np.abs((f["ts_pred"] if pred is None else pred[i]) - f["ts_true"]).mean())
            for i, f in enumerate(folds)]


def load_group(prefix, pkl, sel):
    s, _ = load_kimore(pkl)
    F = [load_folds(f"{prefix}_s{x}", samples=s, sel=sel) for x in SEEDS]
    for k in range(5):                     # 確認 3 次的測試集逐筆相同 (分組相同)
        for f in F[1:]:
            assert np.array_equal(f[k]["index"], F[0][k]["index"]), "分組不同"
    return F


def ensemble_mae(F):
    pred = [np.mean([fs[k]["ts_pred"] for fs in F], 0) for k in range(5)]
    return float(np.mean(fold_maes(F[0], pred)))


def story1(out):
    s_old, _ = load_kimore(PKL_OLD)
    s_v2, _ = load_kimore(PKL_V2)
    base_old = load_folds("baseline", samples=s_old, clip=True)      # 上週 6.23 (事後 clip)
    v4 = load_folds("v4_ts", samples=s_v2)
    F = load_group("v5_base", PKL_V2, "best")                          # 單跑也是用 best，同一種挑法
    single = {"baseline": np.mean(fold_maes(base_old)), "v4_ts": np.mean(fold_maes(v4))}
    per_seed = np.array([fold_maes(f) for f in F])                     # (3, 5)
    means = per_seed.mean(1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.3})
    # 左：單跑 vs 3 次
    a1.scatter([single["baseline"]], [1], s=80, facecolors=SURFACE, edgecolors=BASELINE_GRAY, linewidths=2, zorder=3)
    a1.scatter([single["v4_ts"]], [1], s=80, color=BASELINE_GRAY, zorder=3)
    a1.text(single["baseline"], 1.18, L(f"baseline\n{single['baseline']:.2f}", "baseline"), ha="center",
            va="bottom", fontsize=8.5, color=INK2)
    a1.text(single["v4_ts"], 1.18, L(f"v4_ts\n{single['v4_ts']:.2f}", "v4_ts"), ha="center", va="bottom",
            fontsize=8.5, color=INK2)
    m, sd = means.mean(), means.std(ddof=1)
    a1.axvspan(m - sd, m + sd, ymin=0.05, ymax=0.45, color=SERIES[0], alpha=0.12, lw=0)
    a1.scatter(means, [0] * 3, s=80, color=SERIES[0], edgecolors=SURFACE, lw=1, zorder=3)
    a1.plot([m, m], [-0.3, 0.3], color=INK, lw=2)
    for x, sd_ in zip(means, SEEDS):
        a1.text(x, -0.38, f"s{sd_}", ha="center", va="top", fontsize=8, color=MUTED)
    a1.text(m, 0.38, L(f"平均 {m:.2f} ± {sd:.2f}", f"mean {m:.2f} ± {sd:.2f}"), ha="center", va="bottom",
            fontsize=9, color=INK)
    a1.set_yticks([0, 1], [L("v5：同設定重跑 3 次", "v5: 3 seeds"), L("上週：單跑", "last week: single")])
    a1.set_ylim(-0.8, 1.8)
    a1.set_xlim(6.0, 7.0)
    a1.set_xlabel(L("TS 測試 MAE（五折平均，越小越好）", "TS test MAE (5-fold mean)"))
    a1.set_title(L("單跑結果落在 6.23～6.80，重跑 3 次平均 6.43", "Single runs 6.23–6.80; 3-seed mean 6.43"), loc="left")
    a1.grid(axis="y", visible=False)

    # 右：同一折、只換 seed 的差距
    ks = np.arange(5)
    lo, hi = per_seed.min(0), per_seed.max(0)
    a2.vlines(ks, lo, hi, color=AXIS, lw=6, zorder=1)
    for i, sd_ in enumerate(SEEDS):
        a2.scatter(ks + (i - 1) * 0.08, per_seed[i], s=46, color=SERIES[0], alpha=[1, 0.7, 0.45][i],
                   edgecolors=SURFACE, lw=0.8, zorder=3, label=f"seed {sd_}")
    for k in ks:
        a2.text(k + 0.2, (lo[k] + hi[k]) / 2, f"{hi[k] - lo[k]:.2f}", va="center", fontsize=8.5, color=INK2)
    a2.set_xticks(ks, [f"fold {k}" for k in ks])
    a2.set_xlim(-0.4, 4.6)
    a2.set_ylabel(L("測試 MAE（分）", "Test MAE"))
    a2.set_title(L(f"同一折、同一批受試者，只換 seed：最多差 {np.max(hi - lo):.2f} 分",
                   f"Same fold, only seed differs: up to {np.max(hi - lo):.2f}"), loc="left")
    a2.set_ylim(per_seed.min() - 0.3, per_seed.max() + 0.45)
    a2.legend(loc="upper left", ncol=3, fontsize=8)
    a2.grid(axis="x", visible=False)

    fig.text(0.01, -0.1, L(
        "3 次 5-fold 的分組完全相同 (同一人每次都在同一折；已逐折確認測試集相同)，只有訓練 seed 不同 (初始權重、資料增強、資料順序)。"
        "v5 與單跑都用驗證 MAE 最低的 epoch (best)。\n"
        "baseline (6.23)：舊 pkl (3 筆標籤與 v5 不同)、模型內無 clip、事後 clip。"
        "v4_ts (6.80)：與 v5 完全相同的設定與資料、seed 42，但 v5 的 seed 42 是 6.34 —— MPS 本身也不完全可重現。",
        "Same folds; only training seed differs."), color=MUTED, fontsize=8)
    path = os.path.join(out, "story_1_seed_variance.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path, single, means


def story2(out):
    v5b = load_group("v5_base", PKL_V2, "best")
    v5a = load_group("v5_base", PKL_V2, "avg")
    v6a = load_group("v6_base", PKL_V3, "avg")
    steps = [
        (L("v5 單一模型\n(best epoch)", "v5 single\n(best)"), [np.mean(fold_maes(f)) for f in v5b], None),
        (L("+ 平均最後 10 個\nepoch 的權重", "+ last-10\nweight avg"), [np.mean(fold_maes(f)) for f in v5a], None),
        (L("+ 資料修正\n(v6，v3 資料)", "+ data fix\n(v6)"), [np.mean(fold_maes(f)) for f in v6a], None),
        (L("+ 3 個模型\n預測取平均", "+ 3-model\nensemble"), None, ensemble_mae(v6a)),
    ]
    ys = [np.mean(v) if v is not None else e for _, v, e in steps]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, ys, color=SERIES[0], lw=2, zorder=2)
    for i, (name, vals, e) in enumerate(steps):
        if vals is not None:
            ax.scatter([i] * len(vals), vals, s=28, color=SERIES[0], alpha=0.35, zorder=2)
        ax.scatter([i], [ys[i]], s=110, color=SERIES[0] if i < len(steps) - 1 else SERIES[1],
                   edgecolors=SURFACE, lw=1.5, zorder=3)
        ax.text(i - 0.1, ys[i], f"{ys[i]:.2f}", ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
        if i:
            d = ys[i] - ys[i - 1]
            ax.text(i - 0.5, (ys[i] + ys[i - 1]) / 2 + 0.06, f"{d:+.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
    total = (ys[-1] - ys[0]) / ys[0]
    ax.annotate("", xy=(len(steps) - 1 + 0.32, ys[-1]), xytext=(len(steps) - 1 + 0.32, ys[0]),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(len(steps) - 1 + 0.38, (ys[0] + ys[-1]) / 2, L(f"共 {total:+.0%}", f"total {total:+.0%}"),
            va="center", fontsize=10, color=INK2)
    ax.set_xticks(x, [n for n, _, _ in steps])
    ax.set_xlim(-0.4, len(steps) - 0.2)
    ax.set_ylim(5.2, 6.9)
    ax.set_ylabel(L("TS 測試 MAE（五折平均，越小越好）", "TS test MAE (5-fold mean)"))
    ax.set_title(L(f"baseline 的 MAE：{ys[0]:.2f} → {ys[-1]:.2f}", f"Baseline MAE: {ys[0]:.2f} → {ys[-1]:.2f}"), loc="left")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.09, L(
        "淡點 = 單一 seed，大點 = 3 個 seed 的平均；最後一步是把 3 個模型對每個樣本的預測取平均後再算 MAE (訓練與推論成本 ×3)。\n"
        "第 3 步換了資料版本 (v3：選檔、多人幀、標籤修正，測試集 376 → 380 筆)，前後的 MAE 不是在同一份考卷上比的；"
        "只看沒變動的 360 筆，v5 → v6 是 5.82 → 5.73。",
        "Faint dots = single seeds."), color=MUTED, fontsize=8)
    path = os.path.join(out, "story_2_ensemble.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v6/compare_plots_v6")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p1, single, means = story1(a.out)
    print(p1, {k: round(v, 2) for k, v in single.items()}, np.round(means, 2))
    p2, ys = story2(a.out)
    print(p2, np.round(ys, 2))


if __name__ == "__main__":
    main()
