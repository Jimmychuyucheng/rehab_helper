"""
v5 (舊 pkl) vs v6 (v3 pkl) 在 20 筆資料修正樣本上的 TS 誤差對照。

以 subject_key + exercise 對應樣本 (兩個 pkl 的 index 不同)。
每個 run 的 5 個 fold 測試集合起來恰好涵蓋每個樣本一次；每組 3 個 seed，
表中列出 3 個 seed 的 |誤差| 平均 (與最小~最大)。預測不做事後 clip。

    python compare_fixed_samples.py --out runs/v6/compare_v5_v6_fixed.md
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from run_paths import run_dir

# (sample, 類別, 說明)；與 commit 1e21047 的清單一致
FIXED = [
    ("Expert/E_ID3/Es3", "標籤", "TS 48.33 → 43"),
    ("Expert/E_ID3/Es4", "標籤", "TS 50 → 34.67"),
    ("Expert/E_ID3/Es5", "標籤", "TS 48.33 → 44.67"),
    ("Expert/E_ID8/Es1", "新納入", "兩人幀"),
    ("Parkinson/P_ID5/Es1", "新納入", "路人; VERIFIED_TRIMS"),
    ("Stroke/S_ID6/Es5", "新納入", "兩人幀"),
    ("Parkinson/P_ID3/Es1", "新納入", "第一個檔壞掉"),
    ("Stroke/S_ID5/Es1", "換檔", "68 → 1222 幀"),
    ("NotExpert/NE_ID27/Es2", "換檔", "239 → 1258 幀 [不確定]"),
    ("NotExpert/NE_ID16/Es2", "換檔", "667 → 1348 幀"),
    ("NotExpert/NE_ID20/Es4", "換檔", "634 → 1829 幀"),
    ("NotExpert/NE_ID17/Es4", "換檔", "983 → 1041 幀 [不確定]"),
    ("NotExpert/NE_ID17/Es5", "換檔", "618 → 677 幀 [不確定]"),
    ("NotExpert/NE_ID23/Es4", "換檔", "799 → 841 幀 [不確定]"),
    ("NotExpert/NE_ID4/Es1", "換檔", "1094 → 927 幀 [不確定]"),
    ("NotExpert/NE_ID12/Es1", "中斷", "444 → 440 幀"),
    ("Parkinson/P_ID4/Es2", "中斷", "984 → 974 幀"),
    ("Stroke/S_ID2/Es2", "中斷", "803 → 796 幀"),
    ("Stroke/S_ID3/Es2", "中斷", "675 → 672 幀"),
    ("Stroke/S_ID2/Es5", "中斷", "589 → 485 幀 (VERIFIED_TRIMS)"),
]


def load_run(name, runs_dir, col):
    """回傳 DataFrame(index=sample, columns=[y_true, y_pred])；不完整回傳 None。"""
    files = sorted(glob.glob(os.path.join(run_dir(name, runs_dir), "fold*", "test_predictions.csv")))
    if len(files) != 5:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    df["sample"] = df["subject_key"] + "/" + df["exercise"]
    return df.set_index("sample")[["y_true", col]].rename(columns={col: "y_pred"})


def group(names, runs_dir, col):
    runs = {n: load_run(n, runs_dir, col) for n in names}
    ok = {n: r for n, r in runs.items() if r is not None}
    return ok, [n for n in names if runs[n] is None]


def fmt(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "—"
    return f"{np.mean(vals):.2f} ({min(vals):.2f}~{max(vals):.2f})"


def section(title, v5, v6, v3_true):
    lines = [f"## {title}", ""]
    for label, runs, miss in (("v5", *v5), ("v6", *v6)):
        lines.append(f"- {label} runs：{', '.join(runs) or '無'}"
                     + (f"；未完成而排除：{', '.join(miss)}" if miss else ""))
    lines += ["",
              "| # | 樣本 | 類別 | 修正內容 | v6 標籤 | v5 預測 | v5 \\|誤差\\| (對舊標籤) "
              "| v5 \\|誤差\\| (對 v3 標籤) | v6 預測 | v6 \\|誤差\\| | 變化 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    v5r, v6r = v5[0], v6[0]
    for i, (s, kind, note) in enumerate(FIXED, 1):
        t3 = v3_true.get(s, np.nan)
        p5 = [r.loc[s, "y_pred"] for r in v5r.values() if s in r.index]
        e5old = [abs(r.loc[s, "y_pred"] - r.loc[s, "y_true"]) for r in v5r.values() if s in r.index]
        e5new = [abs(p - t3) for p in p5]
        p6 = [r.loc[s, "y_pred"] for r in v6r.values() if s in r.index]
        e6 = [abs(p - t3) for p in p6]
        d = (f"{np.mean(e6) - np.mean(e5new):+.2f}" if e5new and e6 else "—")
        lines.append(f"| {i} | {s} | {kind} | {note} | {t3:.2f} | {fmt(p5)} | {fmt(e5old)} "
                     f"| {fmt(e5new)} | {fmt(p6)} | {fmt(e6)} | {d} |")

    # 參照：其餘未變動樣本與 20 筆整體的 MAE
    fixed = {s for s, _, _ in FIXED}
    def mae(runs, keep, true_from_v3):
        out = []
        for r in runs.values():
            idx = [s for s in r.index if keep(s)]
            if not idx:
                continue
            t = np.array([v3_true[s] for s in idx]) if true_from_v3 else r.loc[idx, "y_true"].values
            out.append(np.mean(np.abs(r.loc[idx, "y_pred"].values - t)))
        return out
    common = lambda s: s in fixed and any(s in r.index for r in v5r.values())
    lines += ["",
              "| 樣本集合 | v5 MAE | v6 MAE |", "|---|---|---|",
              f"| 修正樣本中 v5 也有的 16 筆 (皆對 v3 標籤) | {fmt(mae(v5r, common, True))} "
              f"| {fmt(mae(v6r, common, True))} |",
              f"| 全部 20 筆修正樣本 (v6) | — | {fmt(mae(v6r, lambda s: s in fixed, True))} |",
              f"| 其餘 360 筆未變動樣本 | {fmt(mae(v5r, lambda s: s not in fixed, False))} "
              f"| {fmt(mae(v6r, lambda s: s not in fixed, False))} |",
              ""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--sel", choices=["avg", "best"], default="avg")
    ap.add_argument("--out", default="runs/v6/compare_v5_v6_fixed.md")
    ap.add_argument("--pkl", default="data/kimore/processed/kimore_full_v3.pkl")
    a = ap.parse_args()
    suf = "" if a.sel == "avg" else "_best"
    col = "y_pred" + suf

    seeds = (42, 43, 44)
    out = ["# 20 筆資料修正樣本：v5 (舊 pkl) vs v6 (v3 pkl) 的 TS 誤差對照", "",
           f"模型選擇：{a.sel}；預測不做事後 clip。格式為 3 個 seed 的平均 (最小~最大)。",
           "「變化」= v6 |誤差| − v5 |誤差| (兩者都對 v3 標籤；負值代表 v6 較準)。",
           "新納入的 4 筆在 v5 沒有預測。E_ID3 三筆的 v5 誤差另列對舊 (錯誤) 標籤的值。", ""]
    v3 = pickle.load(open(a.pkl, "rb"))["samples"]
    v3_true = {f"{d['subject_key']}/{d['exercise']}": float(d["ts"]) for d in v3}
    for gname, tag in (("baseline", "base"), ("多任務 (TS head)", "mt")):
        v5 = group([f"v5_{tag}_s{s}" for s in seeds], a.runs_dir, col)
        v6 = group([f"v6_{tag}_s{s}" for s in seeds], a.runs_dir, col)
        out += section(gname, v5, v6, v3_true)
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
