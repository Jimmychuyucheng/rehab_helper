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
        --out runs/v4/compare_v4_baseline.md
    python analyze_runs.py baseline v4_ts v4_mt --clip --pkl data/kimore/processed/kimore_full_pofix.pkl \
        --out runs/v4/compare_v4.md

    v5：--group 把多個 run 視為同一設定的重複 (三次平均 ± 標準差、各 fold 範圍、avg vs best、組間比較)：
    python analyze_runs.py --group baseline=v5_base_s42,v5_base_s43,v5_base_s44 \
        --group 多任務=v5_mt_s42,v5_mt_s43,v5_mt_s44 --pkl data/kimore/processed/kimore_full_pofix.pkl \
        --status runs/v5/v5_status.txt --out runs/v5/compare_v5_seeds.md
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
from run_paths import run_dir

DIR_THRESHOLD = 0.20   # |PO/15 − CF/35| > 0.20 視為「PO、CF 差異大」


# ==================== 讀檔 ====================
RUNS_DIR = "runs"


def load_folds(name, clip=False, samples=None, sel="avg"):
    """
    回傳 [{fold, index, subj, exercise, ts_true, ts_pred, (po_true, po_pred, cf_true, cf_pred, ts_sum)}]。
    samples 不為 None 時，真實分數一律依 index 從 pkl 取 (取代 csv 內的 y_true*)。
    clip=True 時事後把預測夾在各自範圍 (TS [10,50]、PO [3,15]、CF [7,35])。
    sel="best" 時讀 v5 的 y_pred*_best 欄位 (best.pt)；預設 y_pred* (v5 為平均最後 K 個 epoch 的模型)。
    """
    files = sorted(glob.glob(os.path.join(run_dir(name, RUNS_DIR), "fold*", "test_predictions.csv")))
    if not files:
        raise FileNotFoundError(f"{RUNS_DIR}/{name} 底下找不到 fold*/test_predictions.csv")
    out = []
    for path in files:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        fold = int(re.search(r"fold(\d+)", path).group(1))
        sfx = ""
        if sel == "best":
            if "y_pred_best" not in rows[0]:
                raise ValueError(f"{path} 沒有 y_pred_best 欄位 (不是 --avg_k > 0 的 run)")
            sfx = "_best"
        rows = [{**r, **{k[:-len(sfx)]: v for k, v in r.items() if sfx and k.endswith(sfx)}} for r in rows]
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



# ==================== v5：同一設定的重複 (--group) ====================
SEL_NAMES = {"avg": "avg（平均最後 K 個 epoch）", "best": "best（驗證 MAE 最低 epoch）"}


def _msd(v, fmt="{:.2f}"):
    """平均 ± 樣本標準差 (ddof=1；只有一個值時 std 記為 –)。"""
    v = np.asarray(v, float)
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{fmt.format(v.mean())} ± " + ("–" if np.isnan(sd) else fmt.format(sd))


def _sd(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def direction_stats(folds, samples, thr=DIR_THRESHOLD):
    """「哪一類比較嚴重」子集的猜對率：模型 / 一律猜多數方向 / 依動作種類猜 (訓練集各動作平均)，
    並拆成真實 PO 較差、真實 CF 較差各自的猜對數。"""
    def pct(f, kind):
        return f[f"po_{kind}"] / SCORE_MAX["po"] - f[f"cf_{kind}"] / SCORE_MAX["cf"]
    d_true = np.concatenate([pct(f, "true") for f in folds])
    d_pred = np.concatenate([pct(f, "pred") for f in folds])
    d_blind = np.concatenate([blind_predictions(samples, f["fold"], "po", f["index"]) / SCORE_MAX["po"] -
                              blind_predictions(samples, f["fold"], "cf", f["index"]) / SCORE_MAX["cf"]
                              for f in folds])
    m = np.abs(d_true) > thr
    st, sp, sb = np.sign(d_true[m]), np.sign(d_pred[m]), np.sign(d_blind[m])
    n = int(m.sum())
    po_low, cf_low = st < 0, st > 0
    return {"n": n, "acc": float((sp == st).mean()), "acc_blind": float((sb == st).mean()),
            "majority": max(po_low.sum(), cf_low.sum()) / n if n else float("nan"),
            "n_po_low": int(po_low.sum()), "n_cf_low": int(cf_low.sum()),
            "ok_po_low": int((sp[po_low] == -1).sum()), "ok_cf_low": int((sp[cf_low] == 1).sum()),
            "blind_ok_po_low": int((sb[po_low] == -1).sum()), "blind_ok_cf_low": int((sb[cf_low] == 1).sum())}


def run_status(names, status_file):
    """各 run 完成的折數 (有 metrics.json)，以及 status 檔中的結束碼。"""
    codes = {}
    if status_file and os.path.exists(status_file):
        for line in open(status_file):
            mm = re.match(r"(\S+) exit=(\d+) minutes=(\d+)", line)
            if mm:
                codes[mm.group(1)] = (int(mm.group(2)), int(mm.group(3)))
    lines = ["| run | 完成折數 | 結束碼 | 耗時（分） | 狀態 |", "|---|---|---|---|---|"]
    ok_all = True
    for n in names:
        k = len(glob.glob(os.path.join(run_dir(n, RUNS_DIR), "fold*", "metrics.json")))
        rc, mins = codes.get(n, (None, None))
        ok = k == 5 and rc in (0, None)
        ok_all &= ok
        lines.append(f"| {n} | {k} / 5 | {'–' if rc is None else rc} | {'–' if mins is None else mins} | "
                     f"{'✓' if ok else '**✗ 失敗**'} |")
    return ok_all, lines


def group_report(groups, samples, clip=False, status_file=None):
    """groups: [(組名, [run 名稱])]。輸出 markdown 行。"""
    from dataset import mean_baseline
    blind_ts = mean_baseline(samples, n_folds=5, target="ts")["per_fold"][:, 1]
    all_names = [n for _, ns in groups for n in ns]
    ok_all, st_lines = run_status(all_names, status_file)
    lines = ["# v5：同一設定重複 3 次（seed 42/43/44）", "",
             f"真實分數取自 pkl（依 index 對應）。± 為三次之間的**樣本標準差**（ddof=1）。"
             f"TS 盲猜（依動作）各折 MAE：{' / '.join(f'{v:.2f}' for v in blind_ts)}（mean {np.mean(blind_ts):.2f}）。", "",
             "## 0. Run 狀態", ""] + st_lines + ["", "全部成功" if ok_all else "**有 run 失敗或未完成，見上表**", ""]

    data = {}   # (group, sel) -> {run: folds}
    for g, names in groups:
        for sel in ("avg", "best"):
            data[(g, sel)] = {}
            for n in names:
                try:
                    folds = load_folds(n, clip, samples, sel)
                except (FileNotFoundError, ValueError) as e:
                    lines.append(f"- ⚠ {g}/{sel}：{n} 無法讀取（{e}），不列入統計")
                    continue
                if sorted(f["fold"] for f in folds) != list(range(5)):
                    lines.append(f"- ⚠ {g}/{sel}：{n} 只有 fold {[f['fold'] for f in folds]}，不列入統計")
                    continue
                data[(g, sel)][n] = folds
    lines.append("")

    def ts_keys(folds_by_run):
        return ["ts_pred", "ts_sum"] if folds_by_run and all(is_multi(f) for f in folds_by_run.values()) \
            else ["ts_pred"]

    key_name = {"ts_pred": "ts_head", "ts_sum": "ts_sum"}
    summ = {}   # (group, sel, key) -> list of fold_mae_mean per run
    lines += ["## 1. 各組 TS：每個 run 與三次平均 ± 標準差", ""]
    for g, _ in groups:
        for sel in ("avg", "best"):
            fr = data[(g, sel)]
            for key in ts_keys(fr):
                multi_lbl = f"（{key_name[key]}）" if len(ts_keys(fr)) > 1 else ""
                lines += [f"### {g} · {SEL_NAMES[sel]}{multi_lbl}", "",
                          "| run | 五折平均 MAE | 平衡 MAE | 斜率 | 10–25 bias | 以人為單位 MAE | Pearson | 贏盲猜折數 |",
                          "|---|---|---|---|---|---|---|---|"]
                ms = []
                for n, folds in fr.items():
                    m = analyze(folds, key)
                    ms.append(m)
                    b = m["bins"]["10-25"]
                    wins = sum(a < c for a, c in zip(m["fold_mae"], blind_ts))
                    lines.append(f"| {n} | {m['fold_mae_mean']:.2f} | {m['balanced_mae']:.2f} | {m['slope']:.3f} | "
                                 f"{b['bias']:+.2f} | {m['subject_mae']:.2f} | {m['pearson']:.3f} | {wins} / 5 |")
                if ms:
                    lines.append(f"| **{len(ms)} 次 mean ± std** | **{_msd([m['fold_mae_mean'] for m in ms])}** | "
                                 f"{_msd([m['balanced_mae'] for m in ms])} | {_msd([m['slope'] for m in ms], '{:.3f}')} | "
                                 f"{_msd([m['bins']['10-25']['bias'] for m in ms])} | "
                                 f"{_msd([m['subject_mae'] for m in ms])} | {_msd([m['pearson'] for m in ms], '{:.3f}')} | – |")
                summ[(g, sel, key)] = ms
                lines.append("")

    # 2. 各 fold 在三次之間的範圍
    lines += ["## 2. 各 fold 在三次之間的 MAE 範圍（最小～最大）", ""]
    for g, _ in groups:
        for sel in ("avg", "best"):
            for key in ts_keys(data[(g, sel)]):
                ms = summ.get((g, sel, key), [])
                if not ms:
                    continue
                fm = np.array([m["fold_mae"] for m in ms])   # (runs, folds)
                multi_lbl = f"（{key_name[key]}）" if len(ts_keys(data[(g, sel)])) > 1 else ""
                lines += [f"### {g} · {SEL_NAMES[sel]}{multi_lbl}", "",
                          "| fold | " + " | ".join(data[(g, sel)].keys()) + " | 最小～最大 | 幅度 | std |",
                          "|---" * (len(ms) + 4) + "|"]
                for k in range(fm.shape[1]):
                    col = fm[:, k]
                    lines.append(f"| {k} | " + " | ".join(f"{v:.2f}" for v in col) +
                                 f" | {col.min():.2f}～{col.max():.2f} | {col.max() - col.min():.2f} | {_sd(col):.2f} |")
                w = int(np.argmax(fm.max(0) - fm.min(0)))
                lines += ["", f"起伏最大的是 fold {w}（幅度 {fm[:, w].max() - fm[:, w].min():.2f}）。", ""]

    # 3. avg vs best
    lines += ["## 3. avg vs best（挑模型方式）", "",
              "| 組 / 輸出 | avg MAE（mean ± std） | best MAE（mean ± std） | avg − best（各 run） | 較穩（std 小） | 較準（mean 小） |",
              "|---|---|---|---|---|---|"]
    thr_rows = []
    for g, _ in groups:
        for key in ts_keys(data[(g, "avg")]):
            a = [m["fold_mae_mean"] for m in summ.get((g, "avg", key), [])]
            b = [m["fold_mae_mean"] for m in summ.get((g, "best", key), [])]
            if not a or len(a) != len(b):
                continue
            lbl = g + (f"（{key_name[key]}）" if len(ts_keys(data[(g, 'avg')])) > 1 else "")
            lines.append(f"| {lbl} | {_msd(a)} | {_msd(b)} | " + " / ".join(f"{x - y:+.2f}" for x, y in zip(a, b)) +
                         f" | {'avg' if _sd(a) < _sd(b) else 'best'} | {'avg' if np.mean(a) < np.mean(b) else 'best'} |")
            thr_rows.append((lbl, _sd(a), _sd(b)))
    lines += ["", "### 建議的「有意義差距」門檻（2 × 三次標準差）", "",
              "| 組 / 輸出 | avg：std / 2×std | best：std / 2×std |", "|---|---|---|"]
    for lbl, sa, sb in thr_rows:
        lines.append(f"| {lbl} | {sa:.2f} / **{2 * sa:.2f}** | {sb:.2f} / {2 * sb:.2f} |")
    lines.append("")

    # 4. baseline vs 多任務
    if len(groups) >= 2:
        (g0, _), (g1, _) = groups[0], groups[1]
        lines += [f"## 4. {g0} vs {g1}", "", "### 4.1 TS", "",
                  "差距門檻 = 2 × 兩組標準差中較大者（保守）；同時列 2 × 合併標準差 sqrt((s1²+s2²)/2)。", "",
                  f"| 挑模型 | {g1} 輸出 | {g0} MAE | {g1} MAE | 差距（{g1} − {g0}） | 2 × max(std) | 2 × 合併 std | 超過門檻？ |",
                  "|---|---|---|---|---|---|---|---|"]
        for sel in ("avg", "best"):
            a = [m["fold_mae_mean"] for m in summ.get((g0, sel, "ts_pred"), [])]
            for key in ts_keys(data[(g1, sel)]):
                b = [m["fold_mae_mean"] for m in summ.get((g1, sel, key), [])]
                if not a or not b:
                    continue
                diff = np.mean(b) - np.mean(a)
                t1 = 2 * max(_sd(a), _sd(b))
                t2 = 2 * np.sqrt((_sd(a) ** 2 + _sd(b) ** 2) / 2)
                lines.append(f"| {sel} | {key_name[key]} | {_msd(a)} | {_msd(b)} | {diff:+.2f} | {t1:.2f} | {t2:.2f} | "
                             f"{'**是**' if abs(diff) > t1 else ('僅超過合併門檻' if abs(diff) > t2 else '否')} |")
        lines.append("")

    # 多任務組：PO/CF 與方向
    for g, _ in groups:
        for sel in ("avg", "best"):
            fr = data[(g, sel)]
            if not fr or not all(is_multi(f) for f in fr.values()):
                continue
            lines += [f"### 4.2 {g} · {SEL_NAMES[sel]}：PO / CF（盲猜 = 每折訓練集各動作平均）", "",
                      "| run | PO MAE | PO 盲猜 | PO Pearson | PO 贏盲猜折數 | CF MAE | CF 盲猜 | CF Pearson | CF 贏盲猜折數 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            pc = {}
            for n, folds in fr.items():
                r, _ = po_cf_table(n, folds, samples)
                pc[n] = r
                lines.append(f"| {n} | " + " | ".join(
                    f"{r[t]['mae']:.2f} | {r[t]['blind']:.2f} | {r[t]['pearson']:.3f} | "
                    f"{r[t]['wins']} / 5{' ✓' if r[t]['wins'] == 5 else ''}" for t in ("po", "cf")) + " |")
            lines.append(f"| **{len(pc)} 次 mean ± std** | " + " | ".join(
                f"{_msd([pc[n][t]['mae'] for n in pc])} | {_msd([pc[n][t]['blind'] for n in pc])} | "
                f"{_msd([pc[n][t]['pearson'] for n in pc], '{:.3f}')} | "
                f"{sum(pc[n][t]['wins'] == 5 for n in pc)} / {len(pc)} run 五折全贏" for t in ("po", "cf")) + " |")
            lines += ["", f"### 4.3 {g} · {SEL_NAMES[sel]}：哪一類比較嚴重（|PO/15 − CF/35| > {DIR_THRESHOLD:.2f}）", "",
                      "| run | 子集筆數 | 模型猜對率 | 一律猜多數方向 | 依動作種類猜 | 模型 > 依動作？ | "
                      "真實 PO 較差：模型猜對 / 依動作猜對 / 筆數 | 真實 CF 較差：模型猜對 / 依動作猜對 / 筆數 |",
                      "|---|---|---|---|---|---|---|---|"]
            ds = []
            for n, folds in fr.items():
                d = direction_stats(folds, samples)
                ds.append(d)
                lines.append(f"| {n} | {d['n']} | {d['acc'] * 100:.1f}% | {d['majority'] * 100:.1f}% | "
                             f"{d['acc_blind'] * 100:.1f}% | {'✓' if d['acc'] > d['acc_blind'] else '✗'} | "
                             f"{d['ok_po_low']} / {d['blind_ok_po_low']} / {d['n_po_low']} | "
                             f"{d['ok_cf_low']} / {d['blind_ok_cf_low']} / {d['n_cf_low']} |")
            lines += [f"| **{len(ds)} 次 mean ± std** | – | {_msd([d['acc'] * 100 for d in ds], '{:.1f}')} | "
                      f"{_msd([d['majority'] * 100 for d in ds], '{:.1f}')} | {_msd([d['acc_blind'] * 100 for d in ds], '{:.1f}')} | "
                      f"{sum(d['acc'] > d['acc_blind'] for d in ds)} / {len(ds)} | – | – |", ""]
    return lines


# ==================== 主程式 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="runs/ 底下的 run 名稱")
    ap.add_argument("--group", action="append", default=[], metavar="組名=run1,run2,...",
                    help="v5：把多個 run 視為同一設定的重複，可給多次 (第一組為對照組)；需 --pkl")
    ap.add_argument("--status", default=None, help="--group 時一併列出的 run 狀態檔 (例如 runs/v5_status.txt)")
    ap.add_argument("--runs_dir", default="runs", help="run 所在的資料夾 (預設 runs)")
    ap.add_argument("--clip", action="store_true", help="事後把所有 run 的預測夾在合理範圍 (TS 10~50)")
    ap.add_argument("--pkl", default=None, help="依 index 從此 pkl 取真實分數 (例如修正後的標籤)，並算盲猜")
    ap.add_argument("--out", default=None, help="輸出 markdown；多個 run 時預設 runs/compare.md")
    args = ap.parse_args()
    global RUNS_DIR
    RUNS_DIR = args.runs_dir
    if not args.names and not args.group:
        ap.error("請給 run 名稱或 --group")

    if args.group:
        if not args.pkl:
            ap.error("--group 需要 --pkl")
        from dataset import load_kimore
        samples, _ = load_kimore(args.pkl)
        groups = []
        for g in args.group:
            name, _, runs = g.partition("=")
            groups.append((name, [r for r in runs.split(",") if r]))
        text = "\n".join(group_report(groups, samples, args.clip, args.status)) + "\n"
        print(text)
        out = args.out or "runs/compare_v5_seeds.md"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(text)
        print(f"已寫入 {out}")
        return

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
