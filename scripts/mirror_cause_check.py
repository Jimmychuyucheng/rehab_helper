"""
v7 追加 15 A：整部鏡像影片的原因自動檢查。

對 T6（align_all.csv）判定為整部鏡像（腕部標反 > 50%）的每一列（影片 × 模型），在 T6 的共用對應下計算
「左右平均點在畫面上的 x 動作」與 Kinect X 動作的相關：
  - MediaPipe：6 對左右關節（肩、肘、腕、髖、膝、踝）各自的左右平均影像 x，再取 6 點平均 → 1 維序列。
  - Kinect：同 6 對關節左右平均的相機 X（未減髖部），取 6 點平均，乘上 −1 換成影像方向
    （align.KINECT_TO_IMAGE；影片已在 T2 翻轉，方向正確時影像 x 與 Kinect −X 同向）。
    所以相關 > 0 = 畫面上的水平動作方向與 Kinect 一致；影片檔本身左右翻轉時會變成負相關。
  - 只用 Kinect 保留段內、MediaPipe 偵測到並通過追加 6 B 骨架合理規則的幀（同追加 13 B）；少於 50 幀 → 算不出來。
  這個量與左右標記無關（左右對調時平均點不變）。
歸類：相關 > 0.3 → 「MediaPipe 標反」（不校正，照常納入）；≤ 0.3 或算不出來 → 「人工檢查」。

輸出 data/kimore/mediapipe/qc/mirror_cause.csv。

    python scripts/mirror_cause_check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import align as A  # noqa: E402

ALIGN_ALL = "data/kimore/mediapipe/qc/align_check/align_all.csv"
OUT = "data/kimore/mediapipe/qc/mirror_cause.csv"
CORR_MIN = 0.3


def x_motion_corr(key, model, a, b):
    d = A.load_video(key, model)
    n = len(d["image"])
    rows = A._rows(n, a, b)
    ks = A.kinect_signal(d)                      # 已是影像方向 (−X, −Y)、保留段外 NaN
    ms = A.mediapipe_signal(d)                   # 未偵測或不合理的幀 NaN
    i = A.overlap_pairs(ks, ms, rows)
    if len(i) < A.MIN_PAIRS:
        return np.nan, len(i)
    cols = A.SIGNAL_COL_X[:len(A.SIGNAL_PAIRS)]  # 6 個左右平均點的 x
    kx, mx = ks[rows[i]][:, cols].mean(1), ms[i][:, cols].mean(1)
    return float(np.corrcoef(kx, mx)[0, 1]), len(i)


def main():
    al = pd.read_csv(ALIGN_ALL)
    mir = al[al.mirrored].copy()
    out = []
    for _, r in mir.iterrows():
        c, n = x_motion_corr(r.key, r.model, int(r.a), float(r.b))
        out.append(dict(key=r.key, model=r.model, exercise=r.exercise, subgroup=r.subgroup, main_set=r.main_set,
                        reliable=r.reliable, rho=r.rho, flip_wrist=r.flip_wrist, flip_ankle=r.flip_ankle,
                        x_motion_corr=c, n_frames=n,
                        cause="MediaPipe 標反" if (np.isfinite(c) and c > CORR_MIN) else "人工檢查"))
    o = pd.DataFrame(out)
    man = pd.read_csv(A.MANIFEST)
    man["key"] = man.subject + "_" + man.exercise
    o = o.merge(man[["key", "sample_id"]], on="key", how="left").sort_values(["sample_id", "model"])
    o.to_csv(OUT, index=False)
    print(o.round(3).to_string(index=False))
    print("\n歸類（影片 × 模型）：", o.cause.value_counts().to_dict())
    print("→", OUT)


if __name__ == "__main__":
    main()
