"""
================================================================================
檔案名稱: plot_mac_vs_server.py
功能說明:
    同一設定 (v6 baseline，seed 42/43/44) 在 Mac (mps) 與 server (cuda) 各跑一次的總結對照圖：
      (a) 五折平均 MAE：各 seed、三次平均 ± 標準差、三模型預測取平均 (ensemble)，avg 與 best 兩種挑模型
      (b) 各折 MAE：三次平均 (線) 與最小～最大 (帶)
      (c) 重症偏差：真實 TS 10–25 分的平均偏差 (預測 − 真實)

    python plot_mac_vs_server.py --pkl data/kimore/processed/kimore_full_v3.pkl \
        --out runs/server_v6/plots_v6_mac_vs_server/summary_mac_vs_server.png
================================================================================
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import analyze_runs as A
from dataset import load_kimore
from plot_runs import INK, INK2, MUTED, AXIS, SERIES, L

GROUPS = {
    "Mac v6": ["v6_base_s42", "v6_base_s43", "v6_base_s44"],
    "server v6": ["server_v6_base_s42", "server_v6_base_s43", "server_v6_base_s44"],
}
SEEDS = ["s42", "s43", "s44"]
COLORS = {g: SERIES[i] for i, g in enumerate(GROUPS)}


def stats(names, samples, sel):
    runs = [A.load_folds(n, samples=samples, sel=sel) for n in names]
    fold = np.array([[np.abs(fd["ts_pred"] - fd["ts_true"]).mean() for fd in r] for r in runs])  # seed × fold
    bias = []
    for r in runs:
        t = np.concatenate([fd["ts_true"] for fd in r])
        p = np.concatenate([fd["ts_pred"] for fd in r])
        m = (t >= 10) & (t < 25)
        bias.append((p[m] - t[m]).mean())
    ens = np.mean([np.abs(np.mean([r[k]["ts_pred"] for r in runs], 0) - runs[0][k]["ts_true"]).mean()
                   for k in range(len(runs[0]))])
    return {"fold": fold, "seed_mae": fold.mean(1), "bias": np.array(bias), "ens": ens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="runs/server_v6/plots_v6_mac_vs_server/summary_mac_vs_server.png")
    args = ap.parse_args()
    samples, _ = load_kimore(args.pkl)
    S = {(g, sel): stats(n, samples, sel) for g, n in GROUPS.items() for sel in ("avg", "best")}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.15, 1.2, 0.9]})
    names = list(GROUPS)
    rng = np.random.default_rng(0)

    # (a) 五折平均 MAE
    ax = axes[0]
    for si, sel in enumerate(("avg", "best")):
        for gi, g in enumerate(names):
            s = S[(g, sel)]
            x = si * 2.6 + gi
            m, sd = s["seed_mae"].mean(), s["seed_mae"].std(ddof=1)
            ax.hlines(m, x - 0.3, x + 0.3, color=COLORS[g], lw=3, label=g if si == 0 else None)
            ax.errorbar(x, m, sd, color=INK2, capsize=4, lw=1.2)
            ax.scatter(x + rng.uniform(-0.15, 0.15, 3), s["seed_mae"], s=30, color=COLORS[g],
                       edgecolor="white", lw=1, zorder=3)
            ax.scatter(x, s["ens"], marker="D", s=40, color=INK, zorder=4,
                       label=L("三模型取平均", "3-model ensemble") if si == 0 and gi == 0 else None)
            ax.text(x, m + sd + 0.08, f"{m:.2f}±{sd:.2f}", ha="center", fontsize=8, color=INK2)
            ax.text(x + 0.38, s["ens"], f"{s['ens']:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_xticks([0.5, 3.1], [L("avg（平均最後 K 個 epoch）", "avg (last-K)"), L("best（驗證最低）", "best (val)")])
    ax.set_ylim(5.0, 6.9)
    ax.set_ylabel(L("五折平均 MAE（分，越低越好）", "5-fold mean MAE (lower is better)"))
    ax.set_title(L("(a) 整體 MAE：點 = 各 seed，橫線 = 平均 ± 標準差", "(a) Overall MAE"), fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)

    # (b) 各折
    ax = axes[1]
    k = np.arange(S[(names[0], "avg")]["fold"].shape[1])
    means = {g: S[(g, "avg")]["fold"].mean(0) for g in names}
    for g in names:
        f = S[(g, "avg")]["fold"]
        ax.fill_between(k, f.min(0), f.max(0), color=COLORS[g], alpha=0.15, lw=0)
        ax.plot(k, means[g], marker="o", ms=7, color=COLORS[g], label=g)
    for x in k:  # 較高的標在上方、較低的標在下方，避免重疊
        hi, lo = sorted(names, key=lambda g: -means[g][x])
        ax.text(x, means[hi][x] + 0.08, f"{means[hi][x]:.2f}", ha="center", va="bottom", fontsize=7.5, color=COLORS[hi])
        ax.text(x, means[lo][x] - 0.08, f"{means[lo][x]:.2f}", ha="center", va="top", fontsize=7.5, color=COLORS[lo])
    ax.set_xticks(k, [f"fold {i}" for i in k])
    ax.set_ylabel("MAE")
    ax.set_title(L("(b) 各折 MAE（avg；線 = 三次平均，帶 = 最小～最大）", "(b) Per-fold MAE (avg)"), fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)

    # (c) 重症偏差
    ax = axes[2]
    for gi, g in enumerate(names):
        b = S[(g, "avg")]["bias"]
        m, sd = b.mean(), b.std(ddof=1)
        ax.bar(gi, m, 0.6, color=COLORS[g], alpha=0.25)
        ax.errorbar(gi, m, sd, color=INK2, capsize=4, lw=1.2)
        ax.scatter(gi + rng.uniform(-0.12, 0.12, 3), b, s=30, color=COLORS[g], edgecolor="white", lw=1, zorder=3)
        ax.text(gi, m + sd + 0.2, f"+{m:.2f}±{sd:.2f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 8.5)
    ax.set_ylabel(L("平均偏差（預測 − 真實，分）", "Mean bias (pred − true)"))
    ax.set_title(L("(c) 重症偏差：真實 10–25 分（avg）", "(c) Bias on true TS 10–25 (avg)"), fontsize=10)
    ax.grid(axis="x", visible=False)

    fig.suptitle(L("v6 baseline：Mac（mps）vs server（cuda），seed 42/43/44",
                   "v6 baseline: Mac (mps) vs server (cuda), seeds 42/43/44"), fontweight="bold", color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("已輸出", args.out)
    for (g, sel), s in S.items():
        print(f"{g:10s} {sel:4s} seeds {' / '.join(f'{v:.2f}' for v in s['seed_mae'])} | "
              f"{s['seed_mae'].mean():.2f} ± {s['seed_mae'].std(ddof=1):.2f} | ens {s['ens']:.2f} | "
              f"bias10-25 {s['bias'].mean():+.2f} ± {s['bias'].std(ddof=1):.2f} | folds "
              f"{' '.join(f'{v:.2f}' for v in s['fold'].mean(0))}")


if __name__ == "__main__":
    main()
