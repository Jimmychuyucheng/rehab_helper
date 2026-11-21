"""
v7 追加 15 B：無有效參考幀（追加 7 B）的影片，依 heavy（主分析模型）從主分析集排除，記入 excluded.csv。

  - 讀 T6 的 align_all.csv：ref_ok = False（共同參考幀不足 10 幀）。
  - qc_hflip.csv 新增欄位 no_ref_heavy、no_ref_full、main_set_final（= main_set 且 heavy 有參考幀）；
    full 的敏感度分析另用 main_set_final_full（= main_set 且 full 有參考幀）。
  - excluded.csv：heavy 無參考幀的影片若已因其他原因在清單中，備註加註；否則新增一列，原因「無有效參考幀」。
  - 印出主分析集部數、各組／各動作筆數、真實 TS 10–25 分的筆數（追加 1 B：少於 20 筆時重症警示只做描述）。

    python scripts/apply_ref_exclusion.py
"""
import pandas as pd

QC = "data/kimore/mediapipe/qc"
ALIGN_ALL = f"{QC}/align_check/align_all.csv"
NOTE = "heavy 無有效參考幀（追加 15 B）"


def main():
    al = pd.read_csv(ALIGN_ALL)
    q = pd.read_csv(f"{QC}/qc_hflip.csv", index_col=0)
    ref = {m: al[al.model == m].set_index("key").ref_ok for m in ("heavy", "full")}
    q["no_ref_heavy"] = ~ref["heavy"].reindex(q.index).fillna(True).astype(bool)
    q["no_ref_full"] = ~ref["full"].reindex(q.index).fillna(True).astype(bool)
    q.loc[~q.in347, ["no_ref_heavy", "no_ref_full"]] = False          # 347 筆以外沒有 Kinect 對照，不適用
    q["main_set_final"] = q.main_set & ~q.no_ref_heavy
    q["main_set_final_full"] = q.main_set & ~q.no_ref_full
    q.to_csv(f"{QC}/qc_hflip.csv")

    exc = pd.read_csv(f"{QC}/excluded.csv")
    nr = q[q.in347 & q.no_ref_heavy]
    for k, r in nr.iterrows():
        if k in set(exc.key):
            i = exc.index[exc.key == k][0]
            note = "" if pd.isna(exc.loc[i, "note"]) else str(exc.loc[i, "note"])
            if NOTE not in note:
                exc.loc[i, "note"] = (note + "；" if note else "") + NOTE
        else:
            exc.loc[len(exc)] = dict(sample_id=int(r.sample_id), key=k, subgroup=r.subgroup, ts=r.ts,
                                     fail_heavy=r.fail_heavy, fail_full=r.fail_full, plaus_heavy=r.plaus_heavy,
                                     scene=int(r.scene), reason="無有效參考幀", note=NOTE,
                                     fail_heavy_mirrored=r.fail_heavy_mirrored)
    exc.to_csv(f"{QC}/excluded.csv", index=False)

    ms = q[q.main_set_final]
    print(f"heavy 無有效參考幀：{len(nr)} 部（其中原主分析集 {int((nr.main_set).sum())} 部）；"
          f"full 無有效參考幀且在主分析集：{list(q[q.main_set & q.no_ref_full].index)}")
    print(f"主分析集：{len(ms)} 部（原 {int(q.main_set.sum())}）；full 敏感度分析：{int(q.main_set_final_full.sum())} 部")
    sev = ms.ts.between(10, 25)
    print(pd.crosstab(ms.subgroup, ms.index.str.rsplit("_", n=1).str[1], margins=True))
    print(f"真實 TS 10–25 分：{int(sev.sum())} 筆", ms[sev].subgroup.value_counts().to_dict(),
          "（< 20 筆 → 重症警示只做描述）" if sev.sum() < 20 else "（≥ 20 筆）")
    print(f"參考：347 筆中 TS 10–25 分 {int(q[q.in347].ts.between(10, 25).sum())} 筆；主分析集 TS 範圍 "
          f"{ms.ts.min():.1f}～{ms.ts.max():.1f}")


if __name__ == "__main__":
    main()
