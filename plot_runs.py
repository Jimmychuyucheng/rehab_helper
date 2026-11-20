#!/usr/bin/env python3
"""
plot_runs.py：把 runs/<name>/ 的訓練結果畫成圖表（PNG），方便看實際成果。

用法（在 CISLR-main 根目錄執行）：
    python plot_runs.py baseline                        # 一個實驗
    python plot_runs.py baseline v2_clip_sqrtinv --clip # 多個實驗 + 對照圖；--clip 把 TS 預測夾在 10–50
    python plot_runs.py baseline --pkl data/kimore/processed/kimore_full.pkl
        # 加 --pkl：畫「各動作 MAE vs 盲猜」需要用訓練集算每個動作的平均分
        # 用修正後的 pkl（kimore_full_pofix.pkl）時，真實分數也會換成修正後的值

輸出：
    runs/<name>/plots/*.png 與 runs/<name>/plots/README.md（在 VS Code 按預覽就能一次看全部圖）
    兩個以上實驗時另有 runs/compare_plots/
需要：numpy、matplotlib（pip install matplotlib）
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

from run_paths import run_dir

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 外觀
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_GRAY = "#b5b3aa"
SUBGROUP_ORDER = ["Expert", "NotExpert", "BackPain", "Parkinson", "Stroke"]
BINS = [(10, 25), (25, 35), (35, 45), (45, 50.01)]
BIN_NAMES = ["10–25", "25–35", "35–45", "45–50"]
RANGES = {"ts": (10, 50), "po": (3, 15), "cf": (7, 35)}

CJK_CANDIDATES = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC",
                  "Noto Sans CJK SC", "Microsoft JhengHei", "Microsoft YaHei"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_cjk = next((f for f in CJK_CANDIDATES if f in _available), None)
ZH = _cjk is not None


def L(zh, en):
    """有中文字型就用中文標籤，否則用英文（避免出現方塊字）。"""
    return zh if ZH else en


plt.rcParams.update({
    "font.sans-serif": ([_cjk] if _cjk else []) + ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
})


# ---------------------------------------------------------------- 讀檔
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_run(name, clip=False, pkl_samples=None):
    root = run_dir(name)
    folds = sorted(d for d in os.listdir(root) if d.startswith("fold") and
                   os.path.exists(os.path.join(root, d, "test_predictions.csv")))
    if not folds:
        sys.exit(f"找不到 {root}/fold*/test_predictions.csv")
    run = {"name": name, "root": root, "folds": []}
    for d in folds:
        k = int(d.replace("fold", ""))
        fd = {"k": k}
        lp = os.path.join(root, d, "log.csv")
        fd["log"] = read_csv(lp) if os.path.exists(lp) else []
        mp = os.path.join(root, d, "metrics.json")
        fd["metrics"] = json.load(open(mp)) if os.path.exists(mp) else {}
        rows = read_csv(os.path.join(root, d, "test_predictions.csv"))
        for r in rows:
            r["fold"] = k
            r["t"] = float(r["y_true"])
            r["p"] = float(r["y_pred"])
            if pkl_samples is not None:            # 改用 pkl 內（可能已修正）的真實分數
                s = pkl_samples[int(r["index"])]
                r["t"] = float(s["ts"])
                if "y_true_po" in r:
                    r["y_true_po"], r["y_true_cf"] = s["po"], s["cf"]
            if clip:
                r["p"] = float(np.clip(r["p"], *RANGES["ts"]))
        fd["rows"] = rows
        run["folds"].append(fd)
    run["rows"] = [r for fd in run["folds"] for r in fd["rows"]]
    run["multitask"] = "y_pred_po" in run["rows"][0]
    return run


def mae(t, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(t))))


def pearson(t, p):
    return float(np.corrcoef(t, p)[0, 1])


def slope(t, p):
    t, p = np.asarray(t), np.asarray(p)
    return float(np.polyfit(t, p, 1)[0])


def exercise_baseline(pkl_samples, fold, target="ts"):
    """與 dataset.mean_baseline 相同：用該折 train+val 的各動作平均當預測。"""
    from dataset import make_split
    sp = make_split(pkl_samples, fold)
    tr = [pkl_samples[i] for i in sp["train"] + sp["val"]]
    by = defaultdict(list)
    for s in tr:
        by[s["exercise"]].append(s[target])
    return {e: float(np.mean(v)) for e, v in by.items()}


def save(fig, path, made):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    made.append(path)


def note(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=INK2)


# ---------------------------------------------------------------- 單一實驗的圖
def plot_learning_curves(run, out, made):
    folds = [fd for fd in run["folds"] if fd["log"]]
    if not folds:
        return
    n = len(folds)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.2 * rows_n), squeeze=False)
    ymax = max(max(float(r["train_MAE"]), float(r["val_MAE"])) for fd in folds for r in fd["log"])
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        fd = folds[i]
        ep = [int(r["epoch"]) for r in fd["log"]]
        tr = [float(r["train_MAE"]) for r in fd["log"]]
        va = [float(r["val_MAE"]) for r in fd["log"]]
        ax.plot(ep, tr, color=SERIES[0], label=L("訓練 MAE", "Train MAE"))
        ax.plot(ep, va, color=SERIES[1], label=L("驗證 MAE", "Validation MAE"))
        best = int(np.argmin(va))
        ax.axvline(ep[best], color=MUTED, ls="--", lw=1)
        ax.scatter([ep[best]], [va[best]], s=36, color=SERIES[1], edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.text(0.98, 0.97, L(f"最佳 epoch {ep[best]}｜驗證 MAE {va[best]:.2f}",
                              f"best epoch {ep[best]} | val MAE {va[best]:.2f}"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_title(f"fold {fd['k']}")
        ax.set_xlabel("epoch")
        if i % cols == 0:
            ax.set_ylabel(L("MAE（分）", "MAE (points)"))
    axes.flat[0].legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.suptitle(L(f"{run['name']}：訓練與驗證 MAE 隨 epoch 的變化",
                   f"{run['name']}: train and validation MAE by epoch"), fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, os.path.join(out, "01_learning_curves.png"), made)


def plot_fold_bars(run, out, made, pkl_samples):
    ks, model, base = [], [], []
    for fd in run["folds"]:
        t = [r["t"] for r in fd["rows"]]
        p = [r["p"] for r in fd["rows"]]
        ks.append(f"fold {fd['k']}")
        model.append(mae(t, p))
        if pkl_samples is not None:
            em = exercise_baseline(pkl_samples, fd["k"])
            base.append(mae(t, [em[r["exercise"]] for r in fd["rows"]]))
        else:
            base.append(fd["metrics"].get("baseline_per_ex_mae", np.nan))
    ks.append(L("平均", "mean"))
    model.append(np.mean(model))
    base.append(np.nanmean(base))
    x = np.arange(len(ks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w / 2 - 0.01, base, w, color=BASELINE_GRAY, label=L("盲猜（依動作平均）", "Mean baseline (per exercise)"))
    b2 = ax.bar(x + w / 2 + 0.01, model, w, color=SERIES[0], label=L("模型", "Model"))
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, ks)
    ax.set_ylabel(L("測試集 MAE（分，越低越好）", "Test MAE (points, lower is better)"))
    wins = sum(m < b for m, b in zip(model[:-1], base[:-1]))
    ax.set_title(L(f"{run['name']}：各折模型 vs 盲猜（{wins}/{len(model) - 1} 折勝出）",
                   f"{run['name']}: model vs baseline per fold ({wins}/{len(model) - 1} folds won)"))
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, max(max(base), max(model)) * 1.2)
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(out, "02_fold_mae_vs_baseline.png"), made)


def scatter_true_pred(ax, rows, tkey, pkey, rng, title):
    lo, hi = rng
    for i, g in enumerate(SUBGROUP_ORDER):
        rs = [r for r in rows if r["subgroup"] == g]
        if rs:
            ax.scatter([float(r[tkey]) for r in rs], [float(r[pkey]) for r in rs], s=22, alpha=0.8,
                       color=SERIES[i], edgecolor=SURFACE, lw=0.6, label=g)
    t = np.array([float(r[tkey]) for r in rows])
    p = np.array([float(r[pkey]) for r in rows])
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
    a, b = np.polyfit(t, p, 1)
    ax.plot([lo, hi], [a * lo + b, a * hi + b], color=INK, lw=1.5)
    ax.set_xlim(lo - 1, hi + 1)
    ax.set_ylim(lo - 1, hi + 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    note(ax, f"MAE {mae(t, p):.2f}\nρ {pearson(t, p):.2f}\n" + L(f"斜率 {a:.2f}", f"slope {a:.2f}"))


def plot_pred_vs_true(run, out, made):
    rows = run["rows"]
    for r in rows:
        r["_t"], r["_p"] = r["t"], r["p"]
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter_true_pred(ax, rows, "_t", "_p", RANGES["ts"],
                      L(f"{run['name']}：臨床 TS vs 模型預測 TS", f"{run['name']}: clinical TS vs predicted TS"))
    ax.set_xlabel(L("臨床 TS（治療師評分）", "Clinical TS (therapist)"))
    ax.set_ylabel(L("CTR-GCN 預測 TS", "CTR-GCN predicted TS"))
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.30, L("虛線：完美預測（y = x）\n實線：實際趨勢", "dashed: perfect (y = x)\nsolid: fitted trend"),
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save(fig, os.path.join(out, "03_pred_vs_true_ts.png"), made)


def plot_bins(run, out, made):
    rows = run["rows"]
    n, m, bias = [], [], []
    for lo, hi in BINS:
        s = [r for r in rows if lo <= r["t"] < hi]
        n.append(len(s))
        m.append(mae([r["t"] for r in s], [r["p"] for r in s]) if s else np.nan)
        bias.append(float(np.mean([r["p"] - r["t"] for r in s])) if s else np.nan)
    x = np.arange(len(BINS))
    labels = [f"{b}\n(n={k})" for b, k in zip(BIN_NAMES, n)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    bars = a1.bar(x, m, 0.6, color=SERIES[0])
    for b in bars:
        a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}",
                ha="center", fontsize=8, color=INK2)
    a1.axhline(np.nanmean(m), color=MUTED, ls="--", lw=1)
    a1.text(0.98, 0.98, L(f"虛線：平衡 MAE {np.nanmean(m):.2f}", f"dashed: balanced MAE {np.nanmean(m):.2f}"),
            transform=a1.transAxes, va="top", ha="right", fontsize=8, color=MUTED)
    a1.set_ylim(0, np.nanmax(m) * 1.3)
    a1.set_xticks(x, labels)
    a1.set_xlabel(L("真實 TS 區間", "True TS range"))
    a1.set_ylabel(L("MAE（分）", "MAE (points)"))
    a1.set_title(L("各分數區間錯多少", "Error by score range"))
    a1.grid(axis="x", visible=False)
    colors = [SERIES[1] if v > 0 else SERIES[0] for v in bias]
    bars = a2.bar(x, bias, 0.6, color=colors)
    for b, v in zip(bars, bias):
        a2.text(b.get_x() + b.get_width() / 2, v + (0.3 if v >= 0 else -0.3), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    a2.axhline(0, color=AXIS, lw=1)
    a2.set_xticks(x, labels)
    a2.set_xlabel(L("真實 TS 區間", "True TS range"))
    a2.set_ylabel(L("平均偏差（預測 − 真實）", "Mean bias (pred − true)"))
    a2.set_title(L("往哪邊錯（正 = 給太高）", "Direction of error (+ = too high)"))
    lim = np.nanmax(np.abs(bias)) * 1.35
    a2.set_ylim(-lim, lim)
    a2.grid(axis="x", visible=False)
    fig.suptitle(L(f"{run['name']}：分數區間分析", f"{run['name']}: score-range analysis"),
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, os.path.join(out, "04_score_bins_mae_bias.png"), made)


def plot_per_exercise(run, out, made, pkl_samples):
    exs = sorted({r["exercise"] for r in run["rows"]})
    model = np.full((len(run["folds"]), len(exs)), np.nan)
    base = np.full_like(model, np.nan)
    for i, fd in enumerate(run["folds"]):
        em = exercise_baseline(pkl_samples, fd["k"]) if pkl_samples is not None else None
        for j, e in enumerate(exs):
            s = [r for r in fd["rows"] if r["exercise"] == e]
            if not s:
                continue
            model[i, j] = mae([r["t"] for r in s], [r["p"] for r in s])
            if em:
                base[i, j] = mae([r["t"] for r in s], [em[e]] * len(s))
    x = np.arange(len(exs))
    fig, ax = plt.subplots(figsize=(8, 4))
    has_base = not np.all(np.isnan(base))
    w = 0.38 if has_base else 0.6
    err = dict(ecolor=INK2, capsize=3, elinewidth=1)
    if has_base:
        bb = ax.bar(x - w / 2 - 0.01, np.nanmean(base, 0), w, yerr=np.nanstd(base, 0), color=BASELINE_GRAY,
                    label=L("盲猜（依動作平均）", "Mean baseline"), error_kw=err)
        for b in bb:
            ax.text(b.get_x() + b.get_width() / 2, 0.2, f"{b.get_height():.2f}", ha="center",
                    fontsize=8, color=INK)
    bars = ax.bar(x + (w / 2 + 0.01 if has_base else 0), np.nanmean(model, 0), w, yerr=np.nanstd(model, 0),
                  color=SERIES[0], label=L("模型", "Model"), error_kw=err)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, 0.2, f"{b.get_height():.2f}", ha="center",
                fontsize=8, color="#ffffff")
    ax.set_xticks(x, exs)
    ax.set_ylabel(L("測試集 MAE（分，五折平均 ± 標準差）", "Test MAE (5-fold mean ± sd)"))
    ax.set_title(L(f"{run['name']}：各動作 MAE" + (" vs 盲猜" if has_base else ""),
                   f"{run['name']}: MAE per exercise" + (" vs baseline" if has_base else "")))
    top = np.nanmax(np.nanmean(base, 0) + np.nanstd(base, 0)) if has_base else 0
    ax.set_ylim(0, max(top, np.nanmax(np.nanmean(model, 0) + np.nanstd(model, 0))) * 1.25)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    if not has_base:
        ax.text(0.99, -0.16, L("加 --pkl 可畫出盲猜對照", "add --pkl to draw the baseline"),
                transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save(fig, os.path.join(out, "05_per_exercise_mae.png"), made)


def plot_subgroups(run, out, made):
    rows = run["rows"]
    gs = [g for g in SUBGROUP_ORDER if any(r["subgroup"] == g for r in rows)]
    vals = [mae([r["t"] for r in rows if r["subgroup"] == g], [r["p"] for r in rows if r["subgroup"] == g]) for g in gs]
    ns = [sum(r["subgroup"] == g for r in rows) for g in gs]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(gs))
    ax.barh(y, [vals[i] for i in order], 0.6, color=SERIES[0])
    for yi, i in zip(y, order):
        ax.text(vals[i] + 0.1, yi, f"{vals[i]:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f"{gs[i]} (n={ns[i]})" for i in order])
    ax.set_xlabel(L("測試集 MAE（分）", "Test MAE (points)"))
    ax.set_title(L(f"{run['name']}：各組別 MAE", f"{run['name']}: MAE by subgroup"))
    ax.grid(axis="y", visible=False)
    save(fig, os.path.join(out, "06_subgroup_mae.png"), made)


def plot_po_cf(run, out, made):
    rows = run["rows"]
    for r in rows:
        for k in ("po", "cf"):
            r[f"_t{k}"] = float(r[f"y_true_{k}"])
            r[f"_p{k}"] = float(np.clip(float(r[f"y_pred_{k}"]), *RANGES[k]))
        r["_psum"] = float(np.clip(r["_ppo"] + r["_pcf"], *RANGES["ts"]))
        r["_t"] = r["t"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    scatter_true_pred(axes[0], rows, "_tpo", "_ppo", RANGES["po"], L("PO：臨床 vs 預測", "PO: clinical vs predicted"))
    scatter_true_pred(axes[1], rows, "_tcf", "_pcf", RANGES["cf"], L("CF：臨床 vs 預測", "CF: clinical vs predicted"))
    scatter_true_pred(axes[2], rows, "_t", "_psum", RANGES["ts"],
                      L("TS：臨床 vs（預測 PO + 預測 CF）", "TS: clinical vs (pred PO + pred CF)"))
    for ax, n in zip(axes, ["PO", "CF", "TS"]):
        ax.set_xlabel(L(f"臨床 {n}", f"Clinical {n}"))
        ax.set_ylabel(L(f"預測 {n}", f"Predicted {n}"))
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle(L(f"{run['name']}：PO / CF 多任務輸出", f"{run['name']}: PO / CF multi-task outputs"),
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, os.path.join(out, "07_po_cf_and_ts_sum.png"), made)


# ---------------------------------------------------------------- 多個實驗對照
def plot_compare(runs, made, out=os.path.join("runs", "compare_plots")):
    os.makedirs(out, exist_ok=True)
    names = [r["name"] for r in runs]
    # 各區間 MAE
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(BINS) + 1)
    w = 0.8 / len(runs)
    for i, run in enumerate(runs):
        vals = []
        for lo, hi in BINS:
            s = [r for r in run["rows"] if lo <= r["t"] < hi]
            vals.append(mae([r["t"] for r in s], [r["p"] for r in s]))
        vals.append(mae([r["t"] for r in run["rows"]], [r["p"] for r in run["rows"]]))
        ax.bar(x - 0.4 + w * (i + 0.5), vals, w - 0.02, color=SERIES[i % 8], label=run["name"])
    ax.set_xticks(x, BIN_NAMES + [L("整體", "overall")])
    ax.set_xlabel(L("真實 TS 區間", "True TS range"))
    ax.set_ylabel(L("MAE（分）", "MAE (points)"))
    ax.set_title(L("各實驗在各分數區間的 MAE", "MAE by score range across runs"))
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(out, "cmp_score_bins.png"), made)
    # 各折 MAE
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, run in enumerate(runs):
        ks = [fd["k"] for fd in run["folds"]]
        vals = [mae([r["t"] for r in fd["rows"]], [r["p"] for r in fd["rows"]]) for fd in run["folds"]]
        ax.plot(ks, vals, marker="o", ms=7, color=SERIES[i % 8], label=f"{run['name']}（{np.mean(vals):.2f}）")
    base = [fd["metrics"].get("baseline_per_ex_mae") for fd in runs[0]["folds"]]
    if all(b is not None for b in base):
        ax.plot([fd["k"] for fd in runs[0]["folds"]], base, marker="s", ms=6, color=BASELINE_GRAY,
                ls="--", label=L(f"盲猜（{np.mean(base):.2f}）", f"baseline ({np.mean(base):.2f})"))
    ax.set_xticks(sorted({fd["k"] for run in runs for fd in run["folds"]}))
    ax.set_xlabel("fold")
    ax.set_ylabel(L("測試集 MAE（分）", "Test MAE (points)"))
    ax.set_title(L("各實驗的各折 MAE（括號內為平均）", "Per-fold MAE across runs (mean in brackets)"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, len(runs) + 1))
    save(fig, os.path.join(out, "cmp_fold_mae.png"), made)
    write_index(out, [p for p in made if p.startswith(out)], L("實驗對照：", "Run comparison: ") + "、".join(names))


def plot_compare_groups(groups, made, out):
    """--group 模式：每組 (同設定不同 seed) 畫成一條線 = 各 run 的平均，淡色帶 = 最小～最大。"""
    os.makedirs(out, exist_ok=True)
    colors = {g: SERIES[i % 8] for i, g in enumerate(groups)}
    # 各區間 MAE：每組一根柱，誤差線 = 最小～最大
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(BINS) + 1)
    w = 0.8 / len(groups)
    for i, (g, runs) in enumerate(groups.items()):
        per_run = []
        for run in runs:
            v = []
            for lo, hi in BINS:
                sel = [r for r in run["rows"] if lo <= r["t"] < hi]
                v.append(mae([r["t"] for r in sel], [r["p"] for r in sel]))
            v.append(mae([r["t"] for r in run["rows"]], [r["p"] for r in run["rows"]]))
            per_run.append(v)
        a = np.array(per_run)
        m = a.mean(0)
        ax.bar(x - 0.4 + w * (i + 0.5), m, w - 0.02, color=colors[g],
               yerr=[m - a.min(0), a.max(0) - m], ecolor=INK2, capsize=3, error_kw={"lw": 1},
               label=f"{g}（{len(runs)} runs）")
    ax.set_xticks(x, BIN_NAMES + [L("整體", "overall")])
    ax.set_xlabel(L("真實 TS 區間", "True TS range"))
    ax.set_ylabel(L("MAE（分）", "MAE (points)"))
    ax.set_title(L("各組在各分數區間的 MAE（柱 = 平均，線 = 最小～最大）",
                   "MAE by score range (bar = mean, whisker = min–max)"))
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(out, "cmp_score_bins.png"), made)
    # 各折 MAE：盲猜 + 每組一條線
    fig, ax = plt.subplots(figsize=(8, 4))
    first = next(iter(groups.values()))[0]
    ks = [fd["k"] for fd in first["folds"]]
    base = [fd["metrics"].get("baseline_per_ex_mae") for fd in first["folds"]]
    if all(b is not None for b in base):
        ax.plot(ks, base, marker="s", ms=6, color=BASELINE_GRAY, ls="--",
                label=L(f"盲猜（{np.mean(base):.2f}）", f"guess ({np.mean(base):.2f})"))
    for g, runs in groups.items():
        a = np.array([[mae([r["t"] for r in fd["rows"]], [r["p"] for r in fd["rows"]]) for fd in run["folds"]]
                      for run in runs])
        m = a.mean(0)
        ax.fill_between(ks, a.min(0), a.max(0), color=colors[g], alpha=0.15, lw=0)
        ax.plot(ks, m, marker="o", ms=7, color=colors[g],
                label=f"{g}（{m.mean():.2f} ± {a.mean(1).std(ddof=1) if len(runs) > 1 else 0:.2f}）")
    ax.set_xticks(ks)
    ax.set_xlabel("fold")
    ax.set_ylabel(L("測試集 MAE（分）", "Test MAE (points)"))
    ax.set_title(L("各折 MAE（線 = 各 seed 平均，色帶 = 最小～最大）",
                   "Per-fold MAE (line = seed mean, band = min–max)"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(groups) + 1)
    save(fig, os.path.join(out, "cmp_fold_mae.png"), made)
    write_index(out, [p for p in made if p.startswith(out)],
                L("分組對照：", "Group comparison: ") + "；".join(
                    f"{g} = {', '.join(r['name'] for r in runs)}" for g, runs in groups.items()))


def write_index(out, paths, title):
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# {title}\n\n")
        for p in paths:
            fn = os.path.basename(p)
            f.write(f"## {fn}\n\n![{fn}]({fn})\n\n")


# ---------------------------------------------------------------- 主程式
def main():
    ap = argparse.ArgumentParser(description="把 runs/<name> 的結果畫成圖")
    ap.add_argument("names", nargs="*", help="runs/ 底下的實驗名稱")
    ap.add_argument("--group", action="append", default=[], metavar="名稱=run1,run2,...",
                    help="同設定不同 seed 合成一組，對照圖每組畫一條線 (平均) + 最小～最大色帶；可重複")
    ap.add_argument("--compare_only", action="store_true", help="只畫對照圖，不重畫各 run 自己的圖")
    ap.add_argument("--out", default=os.path.join("runs", "compare_plots"), help="對照圖輸出資料夾")
    ap.add_argument("--clip", action="store_true", help="把 TS 預測夾在 10–50（舊的 baseline 沒有 clip）")
    ap.add_argument("--pkl", default=None, help="KIMORE pkl：用來算各動作盲猜，並使用其中的真實分數")
    args = ap.parse_args()

    samples = None
    if args.pkl:
        sys.path.insert(0, os.getcwd())
        from dataset import load_kimore
        samples, _ = load_kimore(args.pkl)

    groups = {}
    for g in args.group:
        gname, members = g.split("=", 1)
        groups[gname] = [m for m in members.split(",") if m]
    names = list(dict.fromkeys(args.names + [m for ms in groups.values() for m in ms]))
    if not names:
        ap.error("請給 run 名稱或 --group")

    runs, made_all = [], []
    for name in names:
        run = load_run(name, clip=args.clip, pkl_samples=samples)
        if args.compare_only:
            runs.append(run)
            continue
        out = os.path.join(run["root"], "plots")
        os.makedirs(out, exist_ok=True)
        made = []
        plot_learning_curves(run, out, made)
        plot_fold_bars(run, out, made, samples)
        plot_pred_vs_true(run, out, made)
        plot_bins(run, out, made)
        plot_per_exercise(run, out, made, samples)
        plot_subgroups(run, out, made)
        if run["multitask"]:
            plot_po_cf(run, out, made)
        write_index(out, made, f"{name}" + (" (clip)" if args.clip else ""))
        runs.append(run)
        made_all += made
    if groups:
        by = {r["name"]: r for r in runs}
        plot_compare_groups({g: [by[m] for m in ms] for g, ms in groups.items()}, made_all, args.out)
    elif len(runs) > 1:
        plot_compare(runs, made_all, args.out)
    if not ZH:
        print("提醒：找不到中文字型，圖上改用英文標籤。")
    print("已輸出：")
    for p in made_all:
        print("  " + p)


if __name__ == "__main__":
    main()
