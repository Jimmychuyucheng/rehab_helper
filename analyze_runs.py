"""
================================================================================
檔案名稱: analyze_runs.py
功能說明:
    讀 runs/<name>/fold*/test_predictions.csv，五折合併後計算整體與分數區間指標，
    可一次比較多個 run，輸出 markdown 對照表到 runs/compare.md。

    python analyze_runs.py baseline                       # 單一 run
    python analyze_runs.py baseline --clip                # 事後把預測夾在 10~50
    python analyze_runs.py baseline v2_clip_sqrtinv       # 對照表 (baseline 另附事後 clip 欄)

    v4：加 --pkl 時依 csv 的 index 到 pkl 取真實分數 (例如修正 PO<3 後的標籤) 取代 csv 的 y_true，
    並額外輸出 TS/PO/CF 的盲猜 baseline (每折用訓練集各動作平均)。多任務 run (csv 有 y_pred_po /
    y_pred_cf) 會另外列出 ts_sum、PO/CF 表、「哪一類問題比較嚴重」的猜對率與 ts_head/ts_sum 一致性。

    python analyze_runs.py baseline --clip --pkl data/kimore/processed/kimore_full_pofix.pkl \
        --out runs/compare_v4_baseline.md
    python analyze_runs.py baseline v4_ts v4_mt --clip --pkl data/kimore/processed/kimore_full_pofix.pkl \
        --out runs/compare_v4.md
================================================================================
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

from train_rehab import (regression_metrics, range_metrics, spearman, pearson,
                         range_table, SCORE_BINS, bin_label, SCORE_MIN, TS_MIN, TS_MAX)
from dataset import SCORE_MAX

DIR_THRESHOLD = 0.20   # |PO/15 − CF/35| > 0.20 視為「PO、CF 差異大」


# ==================== 讀檔 ====================
def load_folds(name, clip=False, samples=None):
    """
    回傳 [{fold, index, subj, exercise, ts_true, ts_pred, (po_true, po_pred, cf_true, cf_pred, ts_sum)}]。
    samples 不為 None 時，真實分數一律依 index 從 pkl 取 (取代 csv 內的 y_true*)。
    clip=True 時事後把預測夾在各自範圍 (TS [10,50]、PO [3,15]、CF [7,35])。
    """
    files = sorted(glob.glob(os.path.join("runs", name, "fold*", "test_predictions.csv")))
    if not files:
        raise FileNotFoundError(f"runs/{name} 底下找不到 fold*/test_predictions.csv")
    out = []
    for path in files:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        fold = int(re.search(r"fold(\d+)", path).group(1))
        d = {"fold": fold, "index": np.array([int(r["index"]) for r in rows]),
             "subj": [r["subject_key"] for r in rows], "exercise": [r["exercise"] for r in rows],
             "ts_true": np.array([float(r["y_true"]) for r in rows]),
             "ts_pred": np.array([float(r["y_pred"]) for r in rows])}
        multi = "y_pred_po" in rows[0] and "y_pred_cf" in rows[0]
        if multi:
            for t in ("po", "cf"):
                d[f"{t}_true"] = np.array([float(r[f"y_true_{t}"]) for r in rows])
                d[f"{t}_pred"] = np.array([float(r[f"y_pred_{t}"]) for r in rows])
            d["ts_sum"] = (np.array([float(r["y_pred_ts_sum"]) for r in rows]) if "y_pred_ts_sum" in rows[0]
                           else d["po_pred"] + d["cf_pred"])
        if samples is not None:
            for i, r in zip(d["index"], rows):
                if samples[i]["subject_key"] != r["subject_key"] or samples[i]["exercise"] != r["exercise"]:
                    raise ValueError(f"{path}: index {i} 對應錯誤 ({r['subject_key']} {r['exercise']} vs "
                                     f"{samples[i]['subject_key']} {samples[i]['exercise']})")
            d["ts_true"] = np.array([samples[i]["ts"] for i in d["index"]], float)
            if multi:
                d["po_true"] = np.array([samples[i]["po"] for i in d["index"]], float)
                d["cf_true"] = np.array([samples[i]["cf"] for i in d["index"]], float)
        if clip:
            d["ts_pred"] = np.clip(d["ts_pred"], TS_MIN, TS_MAX)
            if multi:
                for t in ("po", "cf"):
                    d[f"{t}_pred"] = np.clip(d[f"{t}_pred"], SCORE_MIN[t], SCORE_MAX[t])
                d["ts_sum"] = np.clip(d["po_pred"] + d["cf_pred"], TS_MIN, TS_MAX)
        out.append(d)
    return out


def is_multi(folds):
    return all("po_pred" in d for d in folds)


# ==================== 盲猜 baseline ====================
def blind_predictions(samples, fold, target, indices):
    """第 fold 折：用 train+val 各動作的平均分猜 indices 的 target (與 dataset.mean_baseline 相同作法)。"""
    from dataset import make_split
    sp = make_split(samples, fold)
    tr = [samples[i] for i in sp["train"] + sp["val"]]
    g = np.mean([s[target] for s in tr])
    by_ex = defaultdict(list)
    for s in tr:
        by_ex[s["exercise"]].append(s[target])
    em = {e: np.mean(v) for e, v in by_ex.items()}
    return np.array([em.get(samples[i]["exercise"], g) for i in indices])


def blind_table(samples, targets=("ts", "po", "cf")):
    """mean_baseline 的每折與 mean ± std (依動作 / 全體平均)。"""
    from dataset import mean_baseline
    res = {t: mean_baseline(samples, n_folds=5, target=t) for t in targets}
    lines = ["| 分數 | " + " | ".join(f"fold {k}" for k in range(5)) +
             " | 依動作 mean ± std | 百分比 MAE | 一律猜平均 mean ± std |",
             "|---" * 9 + "|"]
    for t in targets:
        r = res[t]
        pf = r["per_fold"][:, 1]
        lines.append(f"| {t.upper()} (/{SCORE_MAX[t]:g}) | " + " | ".join(f"{v:.2f}" for v in pf) +
                     f" | {r['per_ex'][0]:.2f} ± {r['per_ex'][1]:.2f} | "
                     f"{r['per_ex'][0] / SCORE_MAX[t] * 100:.1f}% | "
                     f"{r['global'][0]:.2f} ± {r['global'][1]:.2f} |")
    return res, lines


# ==================== TS 指標 ====================
def analyze(folds, pred_key="ts_pred"):
    yt = np.concatenate([f["ts_true"] for f in folds])
    yp = np.concatenate([f[pred_key] for f in folds])
    subj = [s for f in folds for s in f["subj"]]
    fold_mae = [np.abs(f[pred_key] - f["ts_true"]).mean() for f in folds]
    return {"n_folds": len(folds), "n": len(yt), "fold_mae": fold_mae,
            "fold_mae_mean": float(np.mean(fold_mae)), "fold_mae_std": float(np.std(fold_mae)),
            **regression_metrics(yt, yp), "spearman": spearman(yt, yp),
            "pred_std": float(yp.std()), "pred_min": float(yp.min()), "pred_max": float(yp.max()),
            "n_out_of_range": int(((yp < 10) | (yp > 50)).sum()),
            **range_metrics(yt, yp, subj)}


def compare_table(cols, blind_ts=None):
    """cols: [(欄名, analyze 結果)] → markdown 行。blind_ts: TS 盲猜每折 MAE (有的話加「贏盲猜折數」)。"""
    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(m) for _, m in cols) + " |"
    lines = ["| 指標 | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|",
             row("折數 / 筆數", lambda m: f"{m['n_folds']} / {m['n']}"),
             row("MAE（五折 mean ± std）", lambda m: f"{m['fold_mae_mean']:.2f} ± {m['fold_mae_std']:.2f}"),
             row("MAE（各折）", lambda m: " / ".join(f"{v:.2f}" for v in m["fold_mae"]))]
    if blind_ts is not None:
        lines.append(row("贏 TS 盲猜的折數", lambda m: f"{sum(a < b for a, b in zip(m['fold_mae'], blind_ts))} / "
                                                     f"{len(m['fold_mae'])}"))
    lines += [row("MAE（合併）", lambda m: f"{m['mae']:.2f}"),
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


# ==================== PO / CF ====================
def po_cf_table(name, folds, samples):
    """每折與合併：MAE、Pearson、盲猜 MAE、百分比 MAE、贏盲猜折數。"""
    lines = [f"### {name}", "",
             "| 分數 | 各折 MAE | 各折盲猜 MAE | 贏盲猜折數 | MAE（五折 mean ± std） | 盲猜 MAE（mean ± std） | "
             "百分比 MAE（模型 / 盲猜） | Pearson（合併） | 各折 Pearson |",
             "|---" * 9 + "|"]
    res = {}
    for t in ("po", "cf"):
        maes, bl, rhos = [], [], []
        for f in folds:
            b = blind_predictions(samples, f["fold"], t, f["index"])
            maes.append(np.abs(f[f"{t}_pred"] - f[f"{t}_true"]).mean())
            bl.append(np.abs(b - f[f"{t}_true"]).mean())
            rhos.append(pearson(f[f"{t}_true"], f[f"{t}_pred"]))
        yt = np.concatenate([f[f"{t}_true"] for f in folds])
        yp = np.concatenate([f[f"{t}_pred"] for f in folds])
        rho = pearson(yt, yp)
        wins = sum(a < b for a, b in zip(maes, bl))
        res[t] = {"fold_mae": maes, "fold_blind": bl, "wins": wins, "pearson": rho, "fold_pearson": rhos,
                  "mae": float(np.mean(maes)), "blind": float(np.mean(bl))}
        mx = SCORE_MAX[t]
        lines.append(f"| {t.upper()} (/{mx:g}) | " + " / ".join(f"{v:.2f}" for v in maes) + " | " +
                     " / ".join(f"{v:.2f}" for v in bl) + f" | {wins} / {len(folds)} | "
                     f"{np.mean(maes):.2f} ± {np.std(maes):.2f} | {np.mean(bl):.2f} ± {np.std(bl):.2f} | "
                     f"{np.mean(maes) / mx * 100:.1f}% / {np.mean(bl) / mx * 100:.1f}% | {rho:.3f} | " +
                     " / ".join(f"{v:.3f}" for v in rhos) + " |")
    return res, lines


def direction_analysis(name, folds, samples, thr=DIR_THRESHOLD):
    """「哪一類問題比較嚴重」：真實 |PO% − CF%| > thr 的子集，sign(PO% − CF%) 猜對率 (模型 vs 盲猜 vs 多數方向)。"""
    def pct(f, kind):
        return f[f"po_{kind}"] / SCORE_MAX["po"] - f[f"cf_{kind}"] / SCORE_MAX["cf"]
    d_true = np.concatenate([pct(f, "true") for f in folds])
    d_pred = np.concatenate([pct(f, "pred") for f in folds])
    d_blind = np.concatenate([blind_predictions(samples, f["fold"], "po", f["index"]) / SCORE_MAX["po"] -
                              blind_predictions(samples, f["fold"], "cf", f["index"]) / SCORE_MAX["cf"]
                              for f in folds])
    ex = np.array([e for f in folds for e in f["exercise"]])
    m = np.abs(d_true) > thr
    n = int(m.sum())
    st, sp, sb = np.sign(d_true[m]), np.sign(d_pred[m]), np.sign(d_blind[m])
    acc, acc_b = float((sp == st).mean()), float((sb == st).mean())
    n_po_low = int((st < 0).sum())      # PO% < CF%：目標沒達成比較嚴重
    n_cf_low = int((st > 0).sum())      # CF% < PO%：姿勢代償比較嚴重
    majority = max(n_po_low, n_cf_low) / n if n else float("nan")
    # 混淆矩陣 (列 = 真實, 欄 = 模型)
    conf = {(a, b): int(((st == a) & (sp == b)).sum()) for a in (-1, 1) for b in (-1, 0, 1)}
    # 二項檢定：模型猜對率是否高於盲猜 / 多數方向 (單尾，常態近似僅供參考)
    def z(p, p0):
        return (p - p0) / np.sqrt(p0 * (1 - p0) / n) if n and 0 < p0 < 1 else float("nan")
    lines = [f"### {name}", "",
             f"- 子集：真實 |PO/15 − CF/35| > {thr:.2f} 的錄影 **{n} 筆**（共 {len(d_true)} 筆）；"
             f"其中 PO% 較低（目標沒達成較嚴重）{n_po_low} 筆、CF% 較低（姿勢代償較嚴重）{n_cf_low} 筆。",
             f"- **模型猜對率 {acc * 100:.1f}%**（{int((sp == st).sum())} / {n}）",
             f"- 盲猜（訓練集各動作平均 PO%、CF%）猜對率 {acc_b * 100:.1f}%（{int((sb == st).sum())} / {n}）",
             f"- 一律猜多數方向的猜對率 {majority * 100:.1f}%" +
             ("（**注意：多數方向本身就很高**，模型猜對率要跟它比）" if majority >= 0.6 else ""),
             f"- 模型 vs 盲猜 z ≈ {z(acc, acc_b):.2f}；模型 vs 多數方向 z ≈ {z(acc, majority):.2f}"
             "（單尾常態近似，僅供參考）",
             "", "混淆矩陣（列 = 真實較嚴重的一類，欄 = 模型判斷）：", "",
             "| 真實 \\ 模型 | PO 較低 | CF 較低 | 相同 |", "|---|---|---|---|",
             f"| PO 較低（{n_po_low}） | {conf[(-1, -1)]} | {conf[(-1, 1)]} | {conf[(-1, 0)]} |",
             f"| CF 較低（{n_cf_low}） | {conf[(1, -1)]} | {conf[(1, 1)]} | {conf[(1, 0)]} |",
             "", "各動作：", "", "| 動作 | 子集筆數 | 模型猜對率 | 盲猜猜對率 |", "|---|---|---|---|"]
    for e in sorted(set(ex[m])):
        k = ex[m] == e
        lines.append(f"| {e} | {int(k.sum())} | {(sp[k] == st[k]).mean() * 100:.1f}% | "
                     f"{(sb[k] == st[k]).mean() * 100:.1f}% |")
    return {"n": n, "acc": acc, "acc_blind": acc_b, "majority": majority}, lines


def consistency(folds):
    head = np.concatenate([f["ts_pred"] for f in folds])
    sm = np.concatenate([f["ts_sum"] for f in folds])
    yt = np.concatenate([f["ts_true"] for f in folds])
    return {"mean_abs_diff": float(np.abs(head - sm).mean()),
            "mae_head": float(np.abs(head - yt).mean()), "mae_sum": float(np.abs(sm - yt).mean())}


# ==================== 主程式 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="runs/ 底下的 run 名稱")
    ap.add_argument("--clip", action="store_true", help="事後把所有 run 的預測夾在合理範圍 (TS 10~50)")
    ap.add_argument("--pkl", default=None, help="依 index 從此 pkl 取真實分數 (例如修正後的標籤)，並算盲猜")
    ap.add_argument("--out", default=None, help="輸出 markdown；多個 run 時預設 runs/compare.md")
    args = ap.parse_args()

    samples = None
    if args.pkl:
        from dataset import load_kimore
        samples, meta = load_kimore(args.pkl)
    sfx = " + clip" if args.clip else ""

    lines = []
    if len(args.names) == 1:
        name = args.names[0]
        folds = load_folds(name, args.clip, samples)
        title = f"{name}{' + clip（事後）' if args.clip else ''}"
        if samples is None:
            m = analyze(folds)
            lines = [f"# {title}", ""] + compare_table([(title, m)]) + ["", "## 分數區間", ""] + \
                range_table([(title, m)])
        else:
            fixes = meta.get("label_fixes")
            bl, bl_lines = blind_table(samples)
            cols = [(f"{title}（csv 原標籤）", analyze(load_folds(name, args.clip, None))),
                    (f"{title}（{os.path.basename(args.pkl)}）", analyze(folds))]
            diff = cols[1][1]["fold_mae_mean"] - cols[0][1]["fold_mae_mean"]
            # csv 只存到小數 4 位，用容差判斷哪些標籤真的變了
            n_changed = int(sum((np.abs(f["ts_true"] - g["ts_true"]) > 1e-3).sum()
                                for f, g in zip(folds, load_folds(name, args.clip, None))))
            lines = [f"# {title}：用 {os.path.basename(args.pkl)} 的真實分數重算", ""]
            if fixes:
                lines += [f"標籤修正：{fixes['note']}", ""] + [
                    f"- {fx['subject_key']} {fx['exercise']}：PO {fx['po_orig']:.2f} → {fx['po_new']:g}，"
                    f"TS {fx['ts_orig']:.2f} → {fx['ts_new']:.2f}" for fx in fixes["fixes"]] + [""]
            lines += [f"測試預測中真實 TS 有變動的筆數：**{n_changed}**；五折 MAE 差異（修正 − 原標籤）："
                      f"**{diff:+.3f}**" + ("（< 0.05，index 對應正常）" if abs(diff) < 0.05 else
                                            "（**≥ 0.05，請檢查 index 對應**）"), "",
                      "## TS 指標", ""]
            lines += compare_table(cols, blind_ts=bl["ts"]["per_fold"][:, 1])
            lines += ["", "## 分數區間（修正後標籤）", ""] + range_table([(cols[1][0], cols[1][1])])
            lines += ["", f"## 盲猜 baseline（每折用 train+val 各動作平均；{os.path.basename(args.pkl)}）", ""]
            lines += bl_lines
    else:
        cols, mt_runs = [], []
        blind_ts = None
        if samples is not None:
            from dataset import mean_baseline
            blind_ts = mean_baseline(samples, n_folds=5, target="ts")["per_fold"][:, 1]
        for i, name in enumerate(args.names):
            folds = load_folds(name, args.clip, samples)
            if is_multi(folds):
                cols.append((f"{name}{sfx}（ts_head）", analyze(folds)))
                cols.append((f"{name}{sfx}（ts_sum）", analyze(folds, "ts_sum")))
                mt_runs.append((name, folds))
            else:
                cols.append((name + sfx, analyze(folds)))
            # 第一個 run 視為對照組：未 clip 時另附一欄事後 clip，用來分離「clip 本身」的效果
            if i == 0 and not args.clip:
                cols.append((f"{name} + clip（事後）", analyze(load_folds(name, True, samples))))
        lines = ["# Run 對照（五折合併的測試預測）", ""]
        if samples is not None:
            lines += [f"真實分數取自 `{args.pkl}`（依 index 對應）。", ""]
        lines += ["## TS", ""] + compare_table(cols, blind_ts)
        if blind_ts is not None:
            lines += ["", "TS 盲猜（依動作）各折 MAE：" + " / ".join(f"{v:.2f}" for v in blind_ts) +
                      f"（mean {np.mean(blind_ts):.2f}）"]
        if mt_runs:
            if samples is None:
                raise SystemExit("多任務分析需要 --pkl (算 PO/CF 盲猜)")
            lines += ["", "## PO / CF（盲猜 = 每折訓練集各動作平均）", ""]
            for name, folds in mt_runs:
                _, l = po_cf_table(name + sfx, folds, samples)
                lines += l + [""]
            lines += [f"## 哪一類問題比較嚴重（|PO/15 − CF/35| > {DIR_THRESHOLD:.2f} 的子集）", "",
                      "sign(PO% − CF%)：負 = PO 較低（目標沒達成較嚴重），正 = CF 較低（姿勢代償較嚴重）。", ""]
            for name, folds in mt_runs:
                _, l = direction_analysis(name + sfx, folds, samples)
                lines += l + [""]
            lines += ["## ts_head 與 ts_sum 一致性", "",
                      "| run | 平均 \\|ts_head − ts_sum\\| | ts_head MAE | ts_sum MAE | 較低者 |", "|---|---|---|---|---|"]
            for name, folds in mt_runs:
                c = consistency(folds)
                lines.append(f"| {name}{sfx} | {c['mean_abs_diff']:.2f} | {c['mae_head']:.2f} | {c['mae_sum']:.2f} | "
                             f"{'ts_head' if c['mae_head'] < c['mae_sum'] else 'ts_sum'} |")

    text = "\n".join(lines) + "\n"
    print(text)
    out = args.out or ("runs/compare.md" if len(args.names) > 1 else None)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(text)
        print(f"已寫入 {out}")


if __name__ == "__main__":
    main()
