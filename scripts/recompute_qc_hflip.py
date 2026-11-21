"""
v7 追加 11 C：在翻轉後重跑的 MediaPipe 輸出上，重算偵測失敗率與骨架合理率，重新確認主分析集。

  失敗率     = 未偵測幀 / 全部幀（第 3.8 節，> 20% 排除，用 heavy）
  骨架合理率 = 偵測到且通過追加 6 B 規則（align.plausible）的幀 / 全部幀（描述用；與 diag_t2/fail_features.csv 同定義）
  主分析集   = 347 筆中，場景 1～4 以外、且 heavy 失敗率 ≤ 20%（追加 6 A；場景標記不變）

輸出：
  data/kimore/mediapipe/qc/qc_hflip.csv     353 部：翻轉後的失敗率、合理率，並列鏡像畫面的舊值與新舊主分析集
  data/kimore/mediapipe/qc/excluded.csv     347 筆中翻轉後 heavy 失敗率 > 20% 者（舊檔先改名為 excluded_mirrored.csv）；
                                            原因沿用舊檔的目視判斷，新增者標「待目視（翻轉後新增）」
並印出與舊主分析集（259 部）的差異。

    python scripts/recompute_qc_hflip.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from align import plausible  # noqa: E402

QC = "data/kimore/mediapipe/qc"
RAW = "data/kimore/mediapipe/raw/{m}/{key}.npz"
FAIL_MAX = 0.2


def rates(path):
    z = np.load(path)
    det = z["detected"].astype(bool)
    pl = np.array([plausible(f) if d else False for f, d in zip(z["image"], det)])
    return float(1 - det.mean()), float(pl.mean())


def main():
    old = pd.read_csv(f"{QC}/diag_t2/fail_features.csv", index_col=0)
    sc = pd.read_csv(f"{QC}/scene_labels.csv", index_col=0)
    man = pd.read_csv("data/kimore/rgb_manifest.csv")
    man["key"] = man.subject + "_" + man.exercise
    man = man.set_index("key")
    rows = []
    for key in sc.index:
        r = dict(key=key, scene=int(sc.loc[key, "scene"]), backlit=bool(sc.loc[key, "backlit_scene_1_4"]),
                 in347=key in man.index)
        for m in ("heavy", "full"):
            r[f"fail_{m}"], r[f"plaus_{m}"] = rates(RAW.format(m=m, key=key))
            r[f"fail_{m}_mirrored"] = float(old.loc[key, f"fail_{m}"])
            r[f"plaus_{m}_mirrored"] = float(old.loc[key, f"plaus_{m}"])
        rows.append(r)
    q = pd.DataFrame(rows).set_index("key")
    q["main_set"] = q.in347 & ~q.backlit & (q.fail_heavy <= FAIL_MAX)
    q["main_set_mirrored"] = q.in347 & ~q.backlit & (q.fail_heavy_mirrored <= FAIL_MAX)
    for k in man.index:
        q.loc[k, "sample_id"] = int(man.loc[k, "sample_id"])
        q.loc[k, "subgroup"] = man.loc[k, "subject_key"].split("/")[0]
        q.loc[k, "ts"] = float(man.loc[k, "ts"])
    q.to_csv(f"{QC}/qc_hflip.csv")

    # excluded.csv（347 筆中 heavy 失敗率 > 20%）
    if not os.path.exists(f"{QC}/excluded_mirrored.csv"):
        os.rename(f"{QC}/excluded.csv", f"{QC}/excluded_mirrored.csv")
    eo = pd.read_csv(f"{QC}/excluded_mirrored.csv").set_index("key")
    ex = q[q.in347 & (q.fail_heavy > FAIL_MAX)].copy()
    out = []
    for k, r in ex.sort_values("fail_heavy", ascending=False).iterrows():
        if k in eo.index:
            reason, note = eo.loc[k, "reason"], eo.loc[k, "note"]
        elif r.backlit:
            reason, note = "逆光場景 1～4", "翻轉後新增"
        else:
            reason, note = "待目視（翻轉後新增）", ""
        out.append(dict(sample_id=int(r.sample_id), key=k, subgroup=r.subgroup, ts=r.ts,
                        fail_heavy=round(r.fail_heavy, 4), fail_full=round(r.fail_full, 4),
                        plaus_heavy=round(r.plaus_heavy, 4), scene=int(r.scene), reason=reason, note=note,
                        fail_heavy_mirrored=round(r.fail_heavy_mirrored, 4)))
    pd.DataFrame(out).to_csv(f"{QC}/excluded.csv", index=False)

    a = q[q.in347]
    print(f"347 筆：heavy 失敗率 > 20% 鏡像 {int((a.fail_heavy_mirrored > FAIL_MAX).sum())} 部 → 翻轉 {int((a.fail_heavy > FAIL_MAX).sum())} 部；"
          f"full {int((a.fail_full_mirrored > FAIL_MAX).sum())} → {int((a.fail_full > FAIL_MAX).sum())} 部")
    print(f"主分析集：鏡像 {int(a.main_set_mirrored.sum())} 部 → 翻轉 {int(a.main_set.sum())} 部")
    for lab, sel in [("新增", a.main_set & ~a.main_set_mirrored), ("移除", ~a.main_set & a.main_set_mirrored)]:
        s = a[sel].sort_values("fail_heavy")
        print(f"\n{lab} {len(s)} 部：")
        for k, r in s.iterrows():
            print(f"  {k:12s} {r.subgroup:10s} 場景 {int(r.scene):2d}  heavy 失敗率 {r.fail_heavy_mirrored:.1%} → {r.fail_heavy:.1%}"
                  f"（full {r.fail_full_mirrored:.1%} → {r.fail_full:.1%}）")
    m = a[a.main_set]
    print(f"\n翻轉後主分析集各組：{m.subgroup.value_counts().to_dict()}；真實 TS 10–25 分 {int((m.ts < 25).sum())} 部")
    print(f"翻轉後主分析集 heavy 骨架合理率 < 80%：{int((m.plaus_heavy < 0.8).sum())} 部（鏡像時 "
          f"{int((a[a.main_set_mirrored].plaus_heavy_mirrored < 0.8).sum())} 部）")
    d = a.fail_heavy - a.fail_heavy_mirrored
    print(f"heavy 失敗率變化（翻轉 − 鏡像，347 筆）：中位數 {d.median():+.3f}，"
          f"變化 > 10 個百分點 {int((d.abs() > 0.1).sum())} 部（變差 {int((d > 0.1).sum())}、變好 {int((d < -0.1).sum())}）")


if __name__ == "__main__":
    main()
