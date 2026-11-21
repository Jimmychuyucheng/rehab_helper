"""
v7 追加 19 B：事後探索分析（看過 T8 結果後才決定；**不參與判定**，只用來理解原因與規劃下一步）。

1. Q1 系統性低估的來源（主分析集 × heavy）
   - M、K 預測分布（含夾在 10／50 分的比例）；r(M, K)、r(M, y)、r(K, y)（每個 seed 與 3 seed 平均）。
   - 正規化後（共同參考幀）各關節的平均位置差 M − K（逐幀對應、偵測幀），骨段長度（軀幹長單位）。
   - 診斷：把每部影片的「平均姿勢差」加到 Kinect 骨架上（K + mean(M − K)）再預測，看靜態姿勢差能解釋多少低估。
2. 只用骨架合理幀（追加 6 B 規則）重算角度摘要值與逐幀 MAE → Q2 分類、Q3 數值，與正式結果並列。
3. Q2 依特徵類型分組（正式結果重新整理）。

輸出 runs/v7_gap/explore/：pose_diff.csv、bone.csv、angles_plaus.csv、q1_dist.csv、explore.md（gap_report.py 會接進報告）。

    python scripts/gap_explore.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import align as A  # noqa: E402
import common_skeleton as cs  # noqa: E402
import paired_compare as PC  # noqa: E402
from common_skeleton import COMMON_JOINTS, J  # noqa: E402
from features_v7 import FEATURES  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_report as GR  # noqa: E402

OUT = "runs/v7_gap/explore"
GROUP = {   # 追加 19 B3：特徵類型（每個特徵只屬一類；左右標記類優先）
    "body": ["shoulder_width", "hip_width", "trunk_area", "ankle_dist"],
    "lr": ["angle_c", "pelvis_rotation", "pelvis_lateral_tilt", "knee_dy"],
    "frontal": ["trunk_incl_L", "trunk_incl_R", "trunk_lateral_tilt", "axilla_L", "axilla_R", "shoulder_flex_L",
                "shoulder_flex_R", "elbow_dist_x", "wrist_height_L", "wrist_height_R", "hip_dy"],
    "sagittal": ["shoulder_ext_L", "shoulder_ext_R", "knee_sag_L", "knee_sag_R", "knee_L", "knee_R", "hip_L", "hip_R",
                 "trunk_forward_lean", "trunk_rotation", "wrist_depth_L", "wrist_depth_R", "hip_dz", "shoulder_dz",
                 "elbow_L", "elbow_R", "arm_shoulder_L", "arm_shoulder_R"],
}
GROUP_NAME = {"body": "身體尺寸類", "frontal": "冠狀面角度／位置", "sagittal": "矢狀面與深度相關（含三維關節角）",
              "lr": "與左右標記有關"}
F2G = {f: g for g, fs in GROUP.items() for f in fs}
BONES = {"上臂": ("Shoulder", "Elbow"), "前臂": ("Elbow", "Wrist"), "大腿": ("Hip", "Knee"), "小腿": ("Knee", "Ankle")}


def video_arrays(key, ks, ms, r):
    """與 paired_compare.run_video 相同的前處理（共同參考幀正規化、保留段時段）。"""
    lo, hi = int(r.seg_video_first), int(r.seg_video_last) + 1
    kj = np.round(r.a + r.b * np.arange(ms["skeleton"].shape[1])).astype(int)[lo:hi] - ks["raw_start"]
    seg_K = ks["skeleton_centered"].transpose(1, 2, 0).astype(np.float64)
    Mseg = ms["skeleton"].transpose(1, 2, 0).astype(np.float64)[lo:hi]
    Kn, _ = cs.normalize(seg_K, ref_idx=np.array(list(map(int, str(r.ref_raw_rows).split()))) - ks["raw_start"])
    Mn, _ = cs.normalize(Mseg, ref_idx=np.array(list(map(int, str(r.ref_video_frames).split()))) - lo)
    det = ms["detected"][lo:hi].astype(bool)
    vis = ms["visibility"][lo:hi]
    img = np.load(A.MP_RAW.format(m="heavy", key=key))["image"][lo:hi]
    pl = np.array([bool(d) and A.plausible(f) for f, d in zip(img, det)])
    return dict(Kn=Kn, Mn=Mn, kj=kj, det=det, vis=vis, pl=pl)


def main():
    os.makedirs(OUT, exist_ok=True)
    pp, ang, chk, q, al, K = GR.load()
    S = GR.sets(q, al)
    keys = sorted(S["main"][0])
    Kd = {s["key"]: s for s in K}
    MP = {s["key"]: s for s in pickle.load(open(PC.MP_PKL.format(m="heavy"), "rb"))["samples"]}
    A_ = al.set_index(["key", "model"])
    C_ = chk.set_index(["key", "model"])
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pose, bone, angp, shift = [], [], [], []
    for n, key in enumerate(keys, 1):
        ks, ms, r = Kd[key], MP[key], A_.loc[(key, "heavy")]
        v = video_arrays(key, ks, ms, r)
        Ka, Ma, det = v["Kn"][v["kj"]], v["Mn"], v["det"]
        flip = bool(C_.loc[(key, "heavy"), "ref_lr_flipped"])
        dmean = np.nanmean((Ma - Ka)[det], axis=0)                               # (17,3)
        for j, jn in enumerate(COMMON_JOINTS):
            pose.append(dict(key=key, exercise=ks["exercise"], ref_lr_flipped=flip, joint=jn,
                             dx=dmean[j, 0], dy=dmean[j, 1], dz=dmean[j, 2]))
        b = dict(key=key, exercise=ks["exercise"], ref_lr_flipped=flip)
        for bn, (p, c) in BONES.items():
            for side in "LR":
                for nm, X in (("K", v["Kn"]), ("M", v["Mn"][det])):
                    b[f"{bn}{side}_{nm}"] = float(np.nanmedian(np.linalg.norm(X[:, J[p + side]] - X[:, J[c + side]], axis=-1)))
        bone.append(b)
        # 診斷：K + 平均姿勢差
        Kshift = v["Kn"] + dmean[None]
        f = int(ks["fold"])
        for s in PC.SEEDS:
            shift.append(dict(key=key, seed=s, K_shift=PC.predict(Kshift, s, f, device)))
        # 只用骨架合理幀的角度（追加 19 B2）
        Mm = v["Mn"].copy()
        Mm[~v["pl"]] = np.nan
        Mm[v["vis"] < cs.VIS_THRESHOLD] = np.nan
        for fname, (fn, joints, is_dir) in FEATURES[ks["exercise"]].items():
            ak, am = fn(v["Kn"]), fn(Mm)
            vm = float(np.isfinite(am).mean())
            rk, pk = PC.summary(ak)
            rm, pm = PC.summary(am)
            akr = ak[v["kj"]]
            both = np.isfinite(akr) & np.isfinite(am)
            d = PC._wrap(am - akr) if is_dir else am - akr
            angp.append(dict(key=key, model="heavy", exercise=ks["exercise"], subgroup=ks["subgroup"], feature=fname,
                             K_rom=rk, M_rom=rm, K_peak=pk, M_peak=pm, valid_M=vm, valid_ok=vm >= PC.VALID_MIN,
                             frame_mae=float(np.abs(d[both]).mean()) if (r.reliable and both.sum()) else np.nan,
                             frame_n=int(both.sum()), reliable=bool(r.reliable), fps=float(r.fps), plaus_frac=float(v["pl"].mean())))
        if n % 50 == 0:
            print(f"[{n}/{len(keys)}]", flush=True)
    pose, bone, angp, shift = map(pd.DataFrame, (pose, bone, angp, shift))
    for nm, df in (("pose_diff", pose), ("bone", bone), ("angles_plaus", angp), ("k_shift", shift)):
        df.to_csv(f"{OUT}/{nm}.csv", index=False)

    # ---------------------------------------------------------------- 彙整
    L = []
    w = L.append
    d = pp[(pp.model == "heavy") & pp.key.isin(keys)].copy()
    d = d.merge(shift, on=["key", "seed"])
    g = d.groupby("key").agg(K=("K_common", "mean"), M=("M", "mean"), Ks=("K_shift", "mean"), y=("ts", "first"),
                             ex=("exercise", "first"))
    w("## 9. 事後探索分析（追加 19 B；**不參與判定**，只用來理解原因與規劃下一步）")
    w("")
    w("以下分析都在看過 T8 結果之後才決定；判定仍以第 0 節為準（追加 19 A）。")
    w("")
    w("### 9.1 Q1 系統性低估的來源（主分析集 × heavy）")
    w("")
    w("**預測分布**（每部影片 × seed，共 {} 筆）：".format(len(d)))
    w("")
    w("| | 平均 | SD | P5 | 中位數 | P95 | 夾在 10 分 | 夾在 50 分 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for nm, col in (("真實分數 y", "ts"), ("K（Kinect）", "K_common"), ("M（MediaPipe）", "M"), ("K + 平均姿勢差（診斷）", "K_shift")):
        x = d[col]
        w(f"| {nm} | {x.mean():.2f} | {x.std():.2f} | {x.quantile(.05):.2f} | {x.median():.2f} | {x.quantile(.95):.2f} | "
          f"{(x <= 10.001).mean():.1%} | {(x >= 49.999).mean():.1%} |")
    w("")
    rr = []
    for s in list(PC.SEEDS) + ["3 seed 平均"]:
        x = g if s == "3 seed 平均" else d[d.seed == s].rename(columns={"K_common": "K", "ts": "y", "K_shift": "Ks"})
        rr.append((s, np.corrcoef(x.M, x.K)[0, 1], np.corrcoef(x.M, x.y)[0, 1], np.corrcoef(x.K, x.y)[0, 1],
                   np.corrcoef(x.Ks, x.K)[0, 1]))
    w("**相關係數**（Pearson）：")
    w("")
    w("| seed | r(M, K) | r(M, y) | r(K, y) | r(K+姿勢差, K) |")
    w("| --- | --- | --- | --- | --- |")
    for s, a, b, c, e in rr:
        w(f"| {s} | {a:.3f} | {b:.3f} | {c:.3f} | {e:.3f} |")
    w("")
    rM = pd.DataFrame([dict(ex=e, r_MK=np.corrcoef(x.M, x.K)[0, 1], r_My=np.corrcoef(x.M, x.y)[0, 1],
                            r_Ky=np.corrcoef(x.K, x.y)[0, 1], dTS=(x.M - x.K).mean(), n=len(x)) for e, x in g.groupby("ex")])
    w("依動作（3 seed 平均）：")
    w("")
    w("| 動作 | 部數 | r(M, K) | r(M, y) | r(K, y) | 平均 M − K |")
    w("| --- | --- | --- | --- | --- | --- |")
    for _, x in rM.iterrows():
        w(f"| {x.ex} | {x.n} | {x.r_MK:.3f} | {x.r_My:.3f} | {x.r_Ky:.3f} | {x.dTS:+.2f} |")
    w("")
    slope, icpt = np.polyfit(g.K, g.M, 1)
    mae_lin = np.abs(np.clip(icpt + slope * g.K, 10, 50) - g.M).mean()
    w(f"- M 對 K 的線性迴歸（3 seed 平均）：M ≈ {icpt:.2f} + {slope:.3f} × K；殘差 SD {np.std(g.M - (icpt + slope * g.K)):.2f} 分。"
      f"平均 M − K = {(g.M - g.K).mean():+.2f}，|M − K| 中位數 {(g.M - g.K).abs().median():.2f}。")
    w(f"- 診斷（K + 平均姿勢差）：平均 {g.Ks.mean():.2f}，相對 K 的平均差 {(g.Ks - g.K).mean():+.2f} 分"
      f"（M − K 為 {(g.M - g.K).mean():+.2f}）。也就是把每部影片的靜態姿勢差加到 Kinect 骨架上，"
      f"可重現約 {100 * (g.Ks - g.K).mean() / (g.M - g.K).mean():.0f}% 的整體低估。")
    w("")
    pm = pose.groupby("joint")[["dx", "dy", "dz"]].mean().reindex(COMMON_JOINTS)
    pmn = pose[~pose.ref_lr_flipped].groupby("joint")[["dx", "dy", "dz"]].mean().reindex(COMMON_JOINTS)
    w("**正規化後各關節的平均位置差 M − K**（軀幹長；x = 左右、y = 上下、z = 前後，+z 遠離鏡頭；"
      "先對每部影片的偵測幀平均，再對影片平均）：")
    w("")
    w("| 關節 | dx | dy | dz | |d| | dx（參考幀未標反） | dy（同） | dz（同） |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for jn in COMMON_JOINTS:
        a, b = pm.loc[jn], pmn.loc[jn]
        w(f"| {jn} | {a.dx:+.3f} | {a.dy:+.3f} | {a.dz:+.3f} | {np.sqrt((a ** 2).sum()):.3f} | {b.dx:+.3f} | {b.dy:+.3f} | {b.dz:+.3f} |")
    w("")
    top = (pmn ** 2).sum(1).pow(0.5).sort_values(ascending=False).head(5)
    w("- 參考幀未標反的影片中，平均位置差最大的 5 個關節：" + "、".join(f"{k} {v:.3f}" for k, v in top.items()) + "。")
    w("")
    w("**骨段長度**（軀幹長單位，兩邊各自以整段軀幹長中位數正規化；每部影片取中位數後平均）：")
    w("")
    w("| 骨段 | K | M | M / K |")
    w("| --- | --- | --- | --- |")
    for bn in BONES:
        for side in "LR":
            k, m = bone[f"{bn}{side}_K"].mean(), bone[f"{bn}{side}_M"].mean()
            w(f"| {bn}{side} | {k:.3f} | {m:.3f} | {m / k:.3f} |")
    w("")

    # ---- 9.2 只用骨架合理幀
    angp["ref_lr_flipped"] = False
    q2p = GR.q2(angp, set(keys), "heavy")
    q2o = pd.read_csv("runs/v7_gap/q2_main_final.csv")
    q2p.to_csv(f"{OUT}/q2_plaus.csv", index=False)
    rotp, ctrlp, finp = GR.q3(angp, set(keys), "heavy")
    _, (roto, ctrlo, fino) = None, GR.q3(ang, set(keys), "heavy")
    w("### 9.2 只用骨架合理幀重算 Q2、Q3（追加 19 B2）")
    w("")
    w("角度摘要值與逐幀 MAE 只用通過追加 6 B 骨架合理規則的 MediaPipe 幀（其餘規則不變：visibility < 0.5 遮罩、"
      "有效幀 ≥ 50%、P95／P5）。Kinect 端不變。正式 Q2 的 24 fps 處理（追加 3 B）這裡不套用，只並列全部影片的分類。")
    w("")
    vo = q2o.verdict_all.value_counts()
    vp = q2p.verdict.value_counts()
    w("| Q2 分類 | 正式（全部影片） | 只用合理幀 |")
    w("| --- | --- | --- |")
    for k in ["可沿用", "需重新校準", "手機版降級", "手機版停用", "樣本不足"]:
        w(f"| {k} | {vo.get(k, 0)} | {vp.get(k, 0)} |")
    w("")
    mg = q2o.merge(q2p, on=["exercise", "feature", "summary"], suffixes=("_o", "_p"))
    w(f"- r/s_E 中位數：正式 {mg.r_ratio_o.median():.2f} → 只用合理幀 {mg.r_ratio_p.median():.2f}；"
      f"分類改變 {int((mg.verdict_all != mg.verdict_p).sum())} 項（改善 "
      f"{int(sum(GR_rank(b) > GR_rank(a) for a, b in zip(mg.verdict_all, mg.verdict_p)))} 項、變差 "
      f"{int(sum(GR_rank(b) < GR_rank(a) for a, b in zip(mg.verdict_all, mg.verdict_p)))} 項）。")
    w("")
    w("| Q3 數值 | 特徵 | 納入 | Pearson r | 隨機誤差/s_E | 偏差 | 逐幀 MAE | 依第 4.3 節的分類（僅參考） |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for lab, x in (("正式", ctrlo), ("只用合理幀", ctrlp), ("正式", roto), ("只用合理幀", rotp)):
        w(f"| {lab} | {x['feature']} | {x['n']}/{x['n_set']} | {x['r']:.3f} | {x['rand_ratio']:.2f} | {x['bias']:+.2f}° | "
          f"{x['frame_mae']:.1f}° | {x['verdict']} |")
    w("")
    w(f"- 只用合理幀時，對照組 Es2 側彎為「{ctrlp['verdict']}」、Es3 旋轉為「{rotp['verdict']}」；"
      f"依同一套條件組合起來會是「{finp}」。**這不是 Q3 的判定**（追加 19 A：Q3 維持不判定）。")
    w("")

    # ---- 9.3 依特徵類型
    w("### 9.3 Q2 依特徵類型分組（追加 19 B3；正式結果重新整理，數值不變）")
    w("")
    q2o["group"] = q2o.feature.map(F2G)
    assert q2o.group.notna().all(), q2o[q2o.group.isna()].feature.unique()
    sw = pd.read_csv("runs/v7_gap/q2_swap.csv").set_index(["exercise", "feature", "summary"]).verdict
    w("| 類型 | 項數 | 可沿用 | 降級 | 停用 | r/s_E 中位數 | |b|/s_E 中位數 | 只用合理幀：可沿用／降級／停用 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    mg["group"] = mg.feature.map(F2G)
    for gk in ("body", "frontal", "sagittal", "lr"):
        x = q2o[q2o.group == gk]
        c = x.verdict.value_counts()
        cp = mg[mg.group == gk].verdict_p.value_counts()
        w(f"| {GROUP_NAME[gk]} | {len(x)} | {c.get('可沿用', 0)} | {c.get('手機版降級', 0)} | {c.get('手機版停用', 0)} | "
          f"{x.r_ratio.median():.2f} | {x.b_ratio.median():.2f} | {cp.get('可沿用', 0)}／{cp.get('手機版降級', 0)}／{cp.get('手機版停用', 0)} |")
    w("")
    w("各類型包含的特徵：" + "；".join(f"{GROUP_NAME[k]}：" + "、".join(f"`{f}`" for f in v) for k, v in GROUP.items()) + "。"
      "肘角、上臂–肩角等三維關節角含深度分量，歸入矢狀面與深度相關。")
    w("")
    x = q2o[q2o.group == "lr"]
    w("**與左右標記有關的特徵：左右換回前後**（換回只作說明，追加 11 E、18 C）：")
    w("")
    w("| 動作 | 特徵 | 摘要 | r/s_E | 判定（正式） | 判定（左右換回） |")
    w("| --- | --- | --- | --- | --- | --- |")
    for _, r in x.iterrows():
        w(f"| {r.exercise} | `{r.feature}` | {r.summary} | {r.r_ratio:.2f} | {r.verdict} | {sw.loc[(r.exercise, r.feature, r.summary)]} |")
    w("")
    bx = q2o[q2o.group == "body"]
    w(f"- 身體尺寸類的 s_E 中位數 {bx.s_E.median():.3f}（軀幹長或軀幹長²），r 中位數 {bx.r.median():.3f}："
      "Kinect 專家組幾乎沒有變異，任何雜訊都超過門檻。")
    w("")
    open(f"{OUT}/explore.md", "w").write("\n".join(L))
    print("→", f"{OUT}/explore.md")
    for s, a, b, c, e in rr:
        print(f"seed {s}: r(M,K) {a:.3f} r(M,y) {b:.3f} r(K,y) {c:.3f} r(Kshift,K) {e:.3f}")


RANK = {"手機版停用": 0, "手機版降級": 1, "需重新校準": 2, "可沿用": 3, "樣本不足": -1}


def GR_rank(v):
    return RANK.get(v, -1)


if __name__ == "__main__":
    main()
