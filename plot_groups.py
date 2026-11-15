"""
把同一設定、不同 seed 的多個 run 合成「一組一張」的圖 (plot_runs.py 的 01–07 單 run 圖的分組版)。

    python plot_groups.py --pkl data/kimore/processed/kimore_full_v3.pkl --out runs/plots_v6_groups \
        --group base=v6_base_s42,v6_base_s43,v6_base_s44 --group mt=v6_mt_s42,v6_mt_s43,v6_mt_s44

輸出 <out>/<組名>_01_learning_curves.png … ；07 (PO/CF) 只有多任務組才有。

彙整方式 (每組 n 個 run)：
  - 數值 (MAE、偏差、斜率、ρ) = 各 run 各自算出後取平均；誤差線 / 色帶 = n 個 run 的最小～最大。
    不是把預測先平均再算 (那是 ensemble，MAE 會偏低)。
  - 散布圖的點 = 每個樣本在 n 個 run 的平均預測，淡灰直線 = 該樣本 n 個預測的最小～最大。
  - 學習曲線：各 seed 提早停止的 epoch 不同，所以每個 seed 各畫一條細線，不取平均。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import plot_runs as P
from plot_runs import (INK, INK2, MUTED, AXIS, SURFACE, SERIES, BASELINE_GRAY, SUBGROUP_ORDER,
                       BINS, BIN_NAMES, RANGES, L, mae, pearson, slope, exercise_baseline, save, note)
import matplotlib.pyplot as plt

ERR = dict(ecolor=INK2, capsize=3, elinewidth=1)


def minmax_err(a):
    """a: (n_runs, k) → (平均, [下, 上] 誤差線長度)。"""
    a = np.asarray(a, float)
    m = np.nanmean(a, 0)
    return m, [m - np.nanmin(a, 0), np.nanmax(a, 0) - m]


def range_txt(v):
    return f"{np.mean(v):.2f}（{np.min(v):.2f}～{np.max(v):.2f}）"


# ------------------------------------------------------------------ 01
def g_learning_curves(name, runs, out, made):
    nf = len(runs[0]["folds"])
    cols = 3
    rows_n = int(np.ceil(nf / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.2 * rows_n), squeeze=False)
    ymax = max(max(float(r["train_MAE"]), float(r["val_MAE"]))
               for run in runs for fd in run["folds"] for r in fd["log"])
    for i, ax in enumerate(axes.flat):
        if i >= nf:
            ax.axis("off")
            continue
        bests = []
        for run in runs:
            fd = run["folds"][i]
            ep = [int(r["epoch"]) for r in fd["log"]]
            tr = [float(r["train_MAE"]) for r in fd["log"]]
            va = [float(r["val_MAE"]) for r in fd["log"]]
            ax.plot(ep, tr, color=SERIES[0], lw=1.2, alpha=0.55)
            ax.plot(ep, va, color=SERIES[1], lw=1.2, alpha=0.55)
            b = int(np.argmin(va))
            bests.append((ep[b], va[b]))
            ax.scatter([ep[b]], [va[b]], s=30, color=SERIES[1], edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.text(0.98, 0.97, L("最佳 epoch " + "/".join(str(e) for e, _ in bests) +
                              f"\n最佳驗證 MAE {np.mean([v for _, v in bests]):.2f}",
                              "best epoch " + "/".join(str(e) for e, _ in bests)),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_title(f"fold {runs[0]['folds'][i]['k']}")
        ax.set_xlabel("epoch")
        if i % cols == 0:
            ax.set_ylabel(L("MAE（分）", "MAE (points)"))
    axes.flat[0].plot([], [], color=SERIES[0], label=L("訓練 MAE", "Train MAE"))
    axes.flat[0].plot([], [], color=SERIES[1], label=L("驗證 MAE", "Validation MAE"))
    axes.flat[0].legend(loc="upper right", bbox_to_anchor=(1, 0.8))
    fig.suptitle(L(f"{name}：訓練與驗證 MAE（每條細線 = 一個 seed，共 {len(runs)} 個）",
                   f"{name}: train / val MAE (one thin line per seed)"), fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, os.path.join(out, f"{name}_01_learning_curves.png"), made)


# ------------------------------------------------------------------ 02
def g_fold_bars(name, runs, out, made, samples):
    folds0 = runs[0]["folds"]
    ks = [f"fold {fd['k']}" for fd in folds0] + [L("平均", "mean")]
    base = []
    for fd in folds0:
        em = exercise_baseline(samples, fd["k"])
        base.append(mae([r["t"] for r in fd["rows"]], [em[r["exercise"]] for r in fd["rows"]]))
    base.append(np.mean(base))
    per_run = []
    for run in runs:
        v = [mae([r["t"] for r in fd["rows"]], [r["p"] for r in fd["rows"]]) for fd in run["folds"]]
        per_run.append(v + [np.mean(v)])
    m, err = minmax_err(per_run)
    x = np.arange(len(ks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w / 2 - 0.01, base, w, color=BASELINE_GRAY, label=L("盲猜（依動作平均）", "Mean baseline"))
    b2 = ax.bar(x + w / 2 + 0.01, m, w, color=SERIES[0], yerr=err, error_kw=ERR,
                label=L(f"模型（{len(runs)} seed 平均，線 = 最小～最大）", "Model (seed mean, min–max)"))
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    for b, hi in zip(b2, err[1]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + hi + 0.1, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    wins = [sum(v < bb for v, bb in zip(r[:-1], base[:-1])) for r in per_run]
    ax.set_xticks(x, ks)
    ax.set_ylabel(L("測試集 MAE（分，越低越好）", "Test MAE (lower is better)"))
    ax.set_title(L(f"{name}：各折模型 vs 盲猜（各 seed 勝出折數 " + "/".join(map(str, wins)) + f"，共 {len(folds0)} 折）",
                   f"{name}: model vs baseline per fold"))
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(0, max(max(base), max(m + err[1])) * 1.25)
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(out, f"{name}_02_fold_mae_vs_baseline.png"), made)


# ------------------------------------------------------------------ 03 / 07 散布圖
def per_sample(runs, key_t, key_p, clip=None):
    """依樣本 index 合併：回傳 [(t, 平均預測, 最小, 最大, subgroup)] 與各 run 的 (t, p) 陣列。"""
    by = {}
    runs_tp = []
    for run in runs:
        t, p, f = [], [], []
        for r in run["rows"]:
            tv, pv = float(r[key_t]), float(r[key_p])
            if clip:
                pv = float(np.clip(pv, *clip))
            t.append(tv)
            p.append(pv)
            f.append(r["fold"])
            by.setdefault(int(r["index"]), {"t": tv, "g": r["subgroup"], "p": []})["p"].append(pv)
        runs_tp.append((np.array(t), np.array(p), np.array(f)))
    pts = [(d["t"], np.mean(d["p"]), np.min(d["p"]), np.max(d["p"]), d["g"]) for d in by.values()]
    return pts, runs_tp


def g_scatter(ax, pts, runs_tp, rng, title):
    lo, hi = rng
    for t, m, a, b, g in pts:
        ax.plot([t, t], [a, b], color=AXIS, lw=0.8, alpha=0.7, zorder=1)
    for i, g in enumerate(SUBGROUP_ORDER):
        s = [p for p in pts if p[4] == g]
        if s:
            ax.scatter([p[0] for p in s], [p[1] for p in s], s=22, alpha=0.85, color=SERIES[i],
                       edgecolor=SURFACE, lw=0.6, label=g, zorder=2)
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
    fits = [np.polyfit(t, p, 1) for t, p, _ in runs_tp]
    a, b = np.mean(fits, 0)
    ax.plot([lo, hi], [a * lo + b, a * hi + b], color=INK, lw=1.5)
    ax.set_xlim(lo - 1, hi + 1)
    ax.set_ylim(lo - 1, hi + 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    # MAE 與 analyze_runs 相同：各折 MAE 再平均 (不是全部樣本混在一起)
    maes = [np.mean([mae(t[f == k], p[f == k]) for k in np.unique(f)]) for t, p, f in runs_tp]
    rhos = [pearson(t, p) for t, p, _ in runs_tp]
    note(ax, f"MAE {range_txt(maes)}\nρ {np.mean(rhos):.2f}\n" + L(f"斜率 {a:.2f}", f"slope {a:.2f}"))


def g_pred_vs_true(name, runs, out, made):
    for run in runs:
        for r in run["rows"]:
            r["_t"], r["_p"] = r["t"], r["p"]
    pts, tp = per_sample(runs, "_t", "_p")
    fig, ax = plt.subplots(figsize=(6, 6))
    g_scatter(ax, pts, tp, RANGES["ts"], L(f"{name}：臨床 TS vs 模型預測 TS", f"{name}: clinical vs predicted TS"))
    ax.set_xlabel(L("臨床 TS（治療師評分）", "Clinical TS"))
    ax.set_ylabel(L(f"預測 TS（{len(runs)} seed 平均）", "Predicted TS (seed mean)"))
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.30, L("點：各 seed 平均預測\n灰線：各 seed 最小～最大\n虛線：y = x；實線：趨勢\nMAE：各 seed 平均（範圍）",
                          "dot: seed mean; grey: min–max"),
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save(fig, os.path.join(out, f"{name}_03_pred_vs_true_ts.png"), made)


# ------------------------------------------------------------------ 04
def g_bins(name, runs, out, made):
    M, B = [], []
    n = None
    for run in runs:
        rows = run["rows"]
        m, b, nn = [], [], []
        for lo, hi in BINS:
            s = [r for r in rows if lo <= r["t"] < hi]
            nn.append(len(s))
            m.append(mae([r["t"] for r in s], [r["p"] for r in s]) if s else np.nan)
            b.append(float(np.mean([r["p"] - r["t"] for r in s])) if s else np.nan)
        M.append(m)
        B.append(b)
        n = nn
    x = np.arange(len(BINS))
    labels = [f"{b}\n(n={k})" for b, k in zip(BIN_NAMES, n)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    m, err = minmax_err(M)
    bars = a1.bar(x, m, 0.6, color=SERIES[0], yerr=err, error_kw=ERR)
    for bb, hi in zip(bars, err[1]):
        a1.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + hi + 0.15, f"{bb.get_height():.2f}",
                ha="center", fontsize=8, color=INK2)
    bal = np.mean(np.nanmean(M, 1))
    a1.axhline(bal, color=MUTED, ls="--", lw=1)
    a1.text(0.98, 0.98, L(f"虛線：平衡 MAE {bal:.2f}", f"dashed: balanced MAE {bal:.2f}"),
            transform=a1.transAxes, va="top", ha="right", fontsize=8, color=MUTED)
    a1.set_ylim(0, np.nanmax(m + err[1]) * 1.3)
    a1.set_xticks(x, labels)
    a1.set_xlabel(L("真實 TS 區間", "True TS range"))
    a1.set_ylabel(L("MAE（分）", "MAE (points)"))
    a1.set_title(L("各分數區間錯多少", "Error by score range"))
    a1.grid(axis="x", visible=False)
    bm, berr = minmax_err(B)
    colors = [SERIES[1] if v > 0 else SERIES[0] for v in bm]
    bars = a2.bar(x, bm, 0.6, color=colors, yerr=berr, error_kw=ERR)
    for bb, v, lo_, hi_ in zip(bars, bm, berr[0], berr[1]):
        a2.text(bb.get_x() + bb.get_width() / 2, v + (hi_ + 0.3 if v >= 0 else -lo_ - 0.3), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    a2.axhline(0, color=AXIS, lw=1)
    a2.set_xticks(x, labels)
    a2.set_xlabel(L("真實 TS 區間", "True TS range"))
    a2.set_ylabel(L("平均偏差（預測 − 真實）", "Mean bias (pred − true)"))
    a2.set_title(L("往哪邊錯（正 = 給太高）", "Direction of error (+ = too high)"))
    lim = np.nanmax(np.abs(np.r_[bm + berr[1], bm - berr[0]])) * 1.35
    a2.set_ylim(-lim, lim)
    a2.grid(axis="x", visible=False)
    fig.suptitle(L(f"{name}：分數區間分析（{len(runs)} seed 平均，線 = 最小～最大）",
                   f"{name}: score-range analysis (seed mean, min–max)"), fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, os.path.join(out, f"{name}_04_score_bins_mae_bias.png"), made)


# ------------------------------------------------------------------ 05
def g_per_exercise(name, runs, out, made, samples):
    exs = sorted({r["exercise"] for r in runs[0]["rows"]})
    base = np.full((len(runs[0]["folds"]), len(exs)), np.nan)
    for i, fd in enumerate(runs[0]["folds"]):
        em = exercise_baseline(samples, fd["k"])
        for j, e in enumerate(exs):
            s = [r for r in fd["rows"] if r["exercise"] == e]
            base[i, j] = mae([r["t"] for r in s], [em[e]] * len(s))
    model = []                                   # 每個 run：各動作的五折平均 MAE
    for run in runs:
        v = np.full((len(run["folds"]), len(exs)), np.nan)
        for i, fd in enumerate(run["folds"]):
            for j, e in enumerate(exs):
                s = [r for r in fd["rows"] if r["exercise"] == e]
                v[i, j] = mae([r["t"] for r in s], [r["p"] for r in s])
        model.append(np.nanmean(v, 0))
    m, err = minmax_err(model)
    bm = np.nanmean(base, 0)
    x = np.arange(len(exs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    bb = ax.bar(x - w / 2 - 0.01, bm, w, color=BASELINE_GRAY, label=L("盲猜（依動作平均）", "Mean baseline"))
    bars = ax.bar(x + w / 2 + 0.01, m, w, color=SERIES[0], yerr=err, error_kw=ERR,
                  label=L(f"模型（{len(runs)} seed 平均，線 = 最小～最大）", "Model (seed mean, min–max)"))
    for b in bb:
        ax.text(b.get_x() + b.get_width() / 2, 0.2, f"{b.get_height():.2f}", ha="center", fontsize=8, color=INK)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, 0.2, f"{b.get_height():.2f}", ha="center", fontsize=8, color="#ffffff")
    ax.set_xticks(x, exs)
    ax.set_ylabel(L("測試集 MAE（分，五折平均）", "Test MAE (5-fold mean)"))
    ax.set_title(L(f"{name}：各動作 MAE vs 盲猜", f"{name}: MAE per exercise vs baseline"))
    ax.set_ylim(0, max(bm.max(), (m + err[1]).max()) * 1.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(out, f"{name}_05_per_exercise_mae.png"), made)


# ------------------------------------------------------------------ 06
def g_subgroups(name, runs, out, made):
    rows0 = runs[0]["rows"]
    gs = [g for g in SUBGROUP_ORDER if any(r["subgroup"] == g for r in rows0)]
    ns = [sum(r["subgroup"] == g for r in rows0) for g in gs]
    V = [[mae([r["t"] for r in run["rows"] if r["subgroup"] == g],
              [r["p"] for r in run["rows"] if r["subgroup"] == g]) for g in gs] for run in runs]
    m, err = minmax_err(V)
    order = np.argsort(m)
    y = np.arange(len(gs))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(y, m[order], 0.6, color=SERIES[0], xerr=[err[0][order], err[1][order]], error_kw=ERR)
    for yi, i in zip(y, order):
        ax.text(m[i] + err[1][i] + 0.1, yi, f"{m[i]:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f"{gs[i]} (n={ns[i]})" for i in order])
    ax.set_xlim(0, (m + err[1]).max() * 1.2)
    ax.set_xlabel(L(f"測試集 MAE（分，{len(runs)} seed 平均，線 = 最小～最大）", "Test MAE (seed mean, min–max)"))
    ax.set_title(L(f"{name}：各組別 MAE", f"{name}: MAE by subgroup"))
    ax.grid(axis="y", visible=False)
    save(fig, os.path.join(out, f"{name}_06_subgroup_mae.png"), made)


# ------------------------------------------------------------------ 07
def g_po_cf(name, runs, out, made):
    for run in runs:
        for r in run["rows"]:
            for k in ("po", "cf"):
                r[f"_t{k}"] = float(r[f"y_true_{k}"])
                r[f"_p{k}"] = float(np.clip(float(r[f"y_pred_{k}"]), *RANGES[k]))
            r["_psum"] = float(np.clip(r["_ppo"] + r["_pcf"], *RANGES["ts"]))
            r["_t"] = r["t"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (kt, kp, rng, title) in zip(axes, [
            ("_tpo", "_ppo", RANGES["po"], L("PO：臨床 vs 預測", "PO")),
            ("_tcf", "_pcf", RANGES["cf"], L("CF：臨床 vs 預測", "CF")),
            ("_t", "_psum", RANGES["ts"], L("TS：臨床 vs（預測 PO + 預測 CF）", "TS: PO + CF"))]):
        pts, tp = per_sample(runs, kt, kp)
        g_scatter(ax, pts, tp, rng, title)
    for ax, n in zip(axes, ["PO", "CF", "TS"]):
        ax.set_xlabel(L(f"臨床 {n}", f"Clinical {n}"))
        ax.set_ylabel(L(f"預測 {n}（{len(runs)} seed 平均）", f"Predicted {n}"))
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle(L(f"{name}：PO / CF 多任務輸出（點 = seed 平均，灰線 = 最小～最大）",
                   f"{name}: PO / CF outputs"), fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, os.path.join(out, f"{name}_07_po_cf_and_ts_sum.png"), made)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, metavar="名稱=run1,run2,...")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="runs/plots_groups")
    a = ap.parse_args()
    from dataset import load_kimore
    samples, _ = load_kimore(a.pkl)
    os.makedirs(a.out, exist_ok=True)
    made = []
    for g in a.group:
        name, members = g.split("=", 1)
        runs = [P.load_run(m, pkl_samples=samples) for m in members.split(",")]
        g_learning_curves(name, runs, a.out, made)
        g_fold_bars(name, runs, a.out, made, samples)
        g_pred_vs_true(name, runs, a.out, made)
        g_bins(name, runs, a.out, made)
        g_per_exercise(name, runs, a.out, made, samples)
        g_subgroups(name, runs, a.out, made)
        if all(r["multitask"] for r in runs):
            g_po_cf(name, runs, a.out, made)
    P.write_index(a.out, made, L("分組圖：", "Group plots: ") + "；".join(a.group))
    print("已輸出：")
    for p in made:
        print("  " + p)


if __name__ == "__main__":
    main()
