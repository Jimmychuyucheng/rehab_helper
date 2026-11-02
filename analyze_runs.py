"""
================================================================================
檔案名稱: analyze_runs.py
功能說明:
    讀 runs/<name>/fold*/test_predictions.csv，五折合併後計算整體與分數區間指標，
    可一次比較多個 run，輸出 markdown 對照表到 runs/compare.md。

    python analyze_runs.py baseline                       # 單一 run
    python analyze_runs.py baseline --clip                # 事後把預測夾在 10~50
    python analyze_runs.py baseline v2_clip_sqrtinv       # 對照表 (baseline 另附事後 clip 欄)
================================================================================
"""
import argparse
import glob
import os

import numpy as np

from train_rehab import (read_predictions, regression_metrics, range_metrics, spearman,
                         range_table, SCORE_BINS, bin_label)


def analyze(name, clip=False):
    files = sorted(glob.glob(os.path.join("runs", name, "fold*", "test_predictions.csv")))
    if not files:
        raise FileNotFoundError(f"runs/{name} 底下找不到 fold*/test_predictions.csv")
    folds = [read_predictions(f, clip) for f in files]
    yt = np.concatenate([f[0] for f in folds])
    yp = np.concatenate([f[1] for f in folds])
    subj = [s for f in folds for s in f[2]]
    fold_mae = [np.abs(f[1] - f[0]).mean() for f in folds]
    m = {"n_folds": len(files), "n": len(yt),
         "fold_mae_mean": float(np.mean(fold_mae)), "fold_mae_std": float(np.std(fold_mae)),
         **regression_metrics(yt, yp), "spearman": spearman(yt, yp),
         "pred_std": float(yp.std()), "pred_min": float(yp.min()), "pred_max": float(yp.max()),
         "n_out_of_range": int(((yp < 10) | (yp > 50)).sum()),
         **range_metrics(yt, yp, subj)}
    return m


def compare_table(cols):
    """cols: [(欄名, analyze 結果)] → markdown 行。"""
    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(m) for _, m in cols) + " |"
    lines = ["| 指標 | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|",
             row("折數 / 筆數", lambda m: f"{m['n_folds']} / {m['n']}"),
             row("MAE（五折 mean ± std）", lambda m: f"{m['fold_mae_mean']:.2f} ± {m['fold_mae_std']:.2f}"),
             row("MAE（合併）", lambda m: f"{m['mae']:.2f}"),
             row("RMSE（合併）", lambda m: f"{m['rmse']:.2f}"),
             row("Pearson ρ / Spearman ρ", lambda m: f"{m['pearson']:.3f} / {m['spearman']:.3f}"),
             row("預測 std", lambda m: f"{m['pred_std']:.2f}"),
             row("預測範圍（超出 10–50 筆數）",
                 lambda m: f"{m['pred_min']:.1f} ~ {m['pred_max']:.1f}（{m['n_out_of_range']}）"),
             row("平衡 MAE", lambda m: f"{m['balanced_mae']:.2f}"),
             row("斜率", lambda m: f"{m['slope']:.3f}"),
             row("以人為單位 MAE", lambda m: f"{m['subject_mae']:.2f}")]
    for lo, hi in SCORE_BINS:
        l = bin_label(lo, hi)
        lines.append(row(f"{l} MAE / bias（n）", lambda m, l=l: (
            f"{m['bins'][l]['mae']:.2f} / {m['bins'][l]['bias']:+.2f}（{m['bins'][l]['n']}）"
            if m["bins"][l]["n"] else "–")))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="runs/ 底下的 run 名稱")
    ap.add_argument("--clip", action="store_true", help="事後把所有 run 的預測夾在 10~50")
    ap.add_argument("--out", default="runs/compare.md")
    args = ap.parse_args()

    if len(args.names) == 1:
        name = args.names[0]
        m = analyze(name, args.clip)
        title = f"{name}{' + clip（事後）' if args.clip else ''}"
        lines = [f"# {title}", ""] + compare_table([(title, m)]) + ["", "## 分數區間", ""] + \
            range_table([(title, m)])
    else:
        cols = []
        for i, name in enumerate(args.names):
            cols.append((name + (" + clip" if args.clip else ""), analyze(name, args.clip)))
            # 第一個 run 視為對照組：未 clip 時另附一欄事後 clip，用來分離「clip 本身」的效果
            if i == 0 and not args.clip:
                cols.append((f"{name} + clip（事後）", analyze(name, clip=True)))
        lines = ["# Run 對照（五折合併的測試預測）", ""] + compare_table(cols)

    text = "\n".join(lines) + "\n"
    print(text)
    if len(args.names) > 1:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"已寫入 {args.out}")


if __name__ == "__main__":
    main()
