"""
v7 T5 彙整（第 3.5 節、第 4.1 節）：K-common baseline × 3 seed。

  MAE_K   = 三個 seed「五折平均 MAE」的平均 ± 樣本標準差（ddof=1，同 v5/v6 慣例）；σ_seed = 該標準差
  T       = 2 × σ_seed
  MAE_blind = 依動作猜訓練集（train+val）平均的 test MAE，五折平均（fold = 樣本內存的 fold）
  Adv     = MAE_blind − MAE_K
  N       = 所有影片、所有 seed 配對 (s, s′) 的 |K_s(v) − K_s′(v)| 中位數（out-of-fold 預測，avg 模型）
另列 v6（25 點、380 筆）限縮到同一批 347 筆的對照，只作參考。

    python scripts/summarize_kcommon.py [--runs v7_kcommon_s42,v7_kcommon_s43,v7_kcommon_s44]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset import load_kimore, mean_baseline  # noqa: E402

PKL = "data/kimore/processed/kimore_k_common_347.pkl"


def load_run(name):
    folds = [json.load(open(f"runs/{name}/fold{k}/metrics.json")) for k in range(5)]
    preds = pd.concat([pd.read_csv(f"runs/{name}/fold{k}/test_predictions.csv") for k in range(5)])
    return np.array([m["mae"] for m in folds]), preds.set_index("index")


def noise_floor(preds):
    """preds: {run: DataFrame(index→y_pred)}，回傳 N 與所有配對差。"""
    d = []
    for a, b in itertools.combinations(preds, 2):
        pa, pb = preds[a]["y_pred"], preds[b]["y_pred"].reindex(preds[a].index)
        d.append(np.abs(pa.values - pb.values))
    d = np.concatenate(d)
    return float(np.median(d)), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v7_kcommon_s42,v7_kcommon_s43,v7_kcommon_s44")
    ap.add_argument("--v6", default="server_v6_base_s42,server_v6_base_s43,server_v6_base_s44")
    args = ap.parse_args()
    runs = args.runs.split(",")
    samples, _ = load_kimore(PKL)

    fold_mae, preds = {}, {}
    for r in runs:
        fold_mae[r], preds[r] = load_run(r)
        assert len(preds[r]) == len(samples) and set(preds[r].index) == set(range(len(samples))), r
    run_mae = np.array([fold_mae[r].mean() for r in runs])
    mae_k, sigma = run_mae.mean(), run_mae.std(ddof=1)
    bl = mean_baseline(samples, n_folds=5, target="ts", fold_key="fold")["per_fold"]
    blind = bl[:, 1].mean()
    N, d = noise_floor(preds)

    print(f"## K-common baseline（{len(samples)} 筆，seed {', '.join(r.rsplit('_s', 1)[1] for r in runs)}）\n")
    print("| run | " + " | ".join(f"fold {k}" for k in range(5)) + " | 五折平均 | 347 筆合併 MAE |")
    print("|---|" + "---|" * 7)
    for r in runs:
        yt = np.array([samples[i]["ts"] for i in preds[r].index])
        pooled = np.abs(preds[r]["y_pred"].values - yt).mean()
        print(f"| {r} | " + " | ".join(f"{v:.2f}" for v in fold_mae[r]) + f" | {fold_mae[r].mean():.2f} | {pooled:.2f} |")
    fm = np.stack([fold_mae[r] for r in runs])
    print("| 三次平均 | " + " | ".join(f"{v:.2f}" for v in fm.mean(0)) + f" | {mae_k:.2f} | |")
    print("| 盲猜（依動作） | " + " | ".join(f"{v:.2f}" for v in bl[:, 1]) + f" | {blind:.2f} | |")
    print(f"\n- MAE_K = {mae_k:.3f} ± {sigma:.3f}（σ_seed，ddof=1）")
    print(f"- 門檻 T = 2 × σ_seed = {2 * sigma:.3f}")
    print(f"- MAE_blind（347 筆，依動作）= {blind:.3f}（各折 {', '.join(f'{v:.2f}' for v in bl[:, 1])}）；"
          f"一律猜平均 = {bl[:, 0].mean():.3f}")
    print(f"- Adv = MAE_blind − MAE_K = {blind - mae_k:.3f}")
    print(f"- N（雜訊底線）= {N:.3f}（{len(d)} 個配對差；P75 {np.percentile(d, 75):.2f}、P90 {np.percentile(d, 90):.2f}、"
          f"最大 {d.max():.2f}）")

    # v6 對照（限縮到同一批 347 筆）
    v6 = args.v6.split(",")
    if all(os.path.exists(f"runs/{r}/fold0/metrics.json") for r in v6):
        id347 = [s["sample_id"] for s in samples]
        v3, _ = load_kimore("data/kimore/processed/kimore_full_v3.pkl")
        y3 = np.array([s["ts"] for s in v3])
        p6 = {r: load_run(r)[1] for r in v6}
        all_mae = [np.mean([json.load(open(f"runs/{r}/fold{k}/metrics.json"))["mae"] for k in range(5)]) for r in v6]
        sub = {r: p.loc[id347] for r, p in p6.items()}
        sub_mae = [np.abs(p["y_pred"].values - y3[id347]).mean() for p in sub.values()]
        N6, _ = noise_floor(sub)
        N6all, _ = noise_floor(p6)
        print(f"\n### 參考：v6（25 點，380 筆訓練）")
        print(f"- 380 筆五折平均 MAE = {np.mean(all_mae):.3f} ± {np.std(all_mae, ddof=1):.3f}，T_v6 = {2 * np.std(all_mae, ddof=1):.3f}，"
              f"N_v6 = {N6all:.3f}")
        print(f"- 限縮到同一批 347 筆的合併 MAE = {np.mean(sub_mae):.3f} ± {np.std(sub_mae, ddof=1):.3f}，N = {N6:.3f}"
              f"（注意：v6 的 fold 是 380 筆切法，347 筆的 fold 與之相同）")


if __name__ == "__main__":
    main()
