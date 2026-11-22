"""
v7 T8：依第 4 節判斷標準與所有追加，彙整 T7 的逐部數值，產生 runs/v7_gap/gap_report.md 與圖。

分析集（qc_hflip.csv 與 align_all.csv）：
  main     主分析集（main_set_final，256 部）× heavy（追加 5 A、6 A、15 B）——**判定只用這組**
  full     full 敏感度分析（main_set_final_full，253 部）× full
  plaus80  追加 6 B 敏感度：主分析集再排除 heavy 骨架合理率 < 80%
  swap     追加 11 E／18 B 敏感度：左右標反換回（M_swap）
  backlit  逆光分析集：場景 1～4、heavy 失敗率 ≤ 20%、heavy 有參考幀（追加 6 A；只描述）
  ref347   347 部參考：heavy 有參考幀的全部影片（只描述）

Q1（第 4.1 節；追加 9 B：主分析集上、T5 的 out-of-fold 模型、T 固定 0.316）：
  K_s = K_common（追加 8 B：與 MediaPipe 同一組共同參考幀），M_s = M。
  MAE_K、MAE_M = 每個 seed 在分析集上的合併 MAE，再對 3 個 seed 平均；ΔMAE = MAE_M − MAE_K。
  MAE_blind = 依動作猜「非測試折」的平均（同 T5 / dataset.mean_baseline），在分析集上的 MAE；Adv = MAE_blind − MAE_K。
  N = 所有影片、所有 seed 配對 |K_s − K_s'| 的中位數；G = 所有影片、所有 seed |M_s − K_s| 的中位數。
  重症警示：真實 TS 10–25 分，偏差 mean(M − y) − mean(K − y) ≥ 2 → 「重症高估惡化」（< 20 筆只描述，追加 1 B）。
  追加 8 B 檢查：median |K_own − K_common| < N / 2。
Q2（第 4.2 節；追加 17 A、18 A 的 63 項特徵，摘要值 ROM、峰值各自判定）：
  d = M − K（方向角的峰值差 wrap 到 ±180°）；b = mean(d)，r = SD(d)，s_E = Kinect 專家組（同分析集）的 SD。
  只用 MediaPipe 有效幀 ≥ 50% 的影片（追加 7 D）；納入影片 < 該動作分析集的一半 → 「樣本不足」（追加 8 C）。
  追加 3 B：24 fps 組另列；25 fps 組單獨判定的結果若與全部不同，在報告中標出。
Q3（第 4.3 節）：Es3 trunk_rotation 的 ROM；Pearson r、隨機誤差 SD(M − K)、逐幀 MAE（只用對齊可靠的 25 fps 影片，
  以幀數加權；可靠影片 < 20 部 → 逐幀條件「樣本不足」，追加 10 C）。對照組 Es2 trunk_lateral_tilt 套用同一標準；
  對照組「撐不住」→ Q3 不判定。

    python scripts/gap_report.py
（若已先跑 scripts/gap_explore.py，會把 runs/v7_gap/explore/explore.md 接在最後作為第 9 節「事後探索分析」。）
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from features_v7 import FEATURE_SPECS  # noqa: E402

OUT = "runs/v7_gap"
FIG = f"{OUT}/fig"
QC = "data/kimore/mediapipe/qc"
T = 0.316                       # 追加 9 B：固定
SEV_LO, SEV_HI, SEV_MIN_N = 10, 25, 20
SEEDS = (42, 43, 44)
EX = ["Es1", "Es2", "Es3", "Es4", "Es5"]
SPEC = {(ex, f["name"]): f for ex, fs in FEATURE_SPECS.items() for f in fs}


# ---------------------------------------------------------------- 讀資料
def load():
    pp = pd.read_csv(f"{OUT}/paired_preds.csv")
    ang = pd.read_csv(f"{OUT}/angles.csv")
    chk = pd.read_csv(f"{OUT}/check.csv")
    q = pd.read_csv(f"{QC}/qc_hflip.csv", index_col=0)
    al = pd.read_csv(f"{QC}/align_check/align_all.csv")
    K = pickle.load(open("data/kimore/processed/kimore_k_common_347.pkl", "rb"))["samples"]
    return pp, ang, chk, q, al, K


def sets(q, al):
    h = al[al.model == "heavy"].set_index("key")
    in347 = q.in347.astype(bool)
    return {
        "main": (set(q.index[q.main_set_final.astype(bool)]), "heavy"),
        "full": (set(q.index[q.main_set_final_full.astype(bool)]), "full"),
        "plaus80": (set(q.index[q.main_set_final.astype(bool) & (q.plaus_heavy >= 0.8)]), "heavy"),
        "backlit": (set(q.index[in347 & q.backlit.astype(bool) & (q.fail_heavy <= 0.2) & ~q.no_ref_heavy.astype(bool)]), "heavy"),
        "ref347": (set(h.index[h.ref_ok.astype(bool)]), "heavy"),
    }


def blind_preds(K):
    """依動作猜非測試折的平均（同 dataset.mean_baseline 的 per_ex）。回傳 {key: 預測}。"""
    df = pd.DataFrame([dict(key=s["key"], ex=s["exercise"], fold=s["fold"], ts=s["ts"]) for s in K])
    out = {}
    for f in range(5):
        tr = df[df.fold != f]
        m = tr.groupby("ex").ts.mean()
        for _, r in df[df.fold == f].iterrows():
            out[r.key] = m.get(r.ex, tr.ts.mean())
    return out


# ---------------------------------------------------------------- Q1
def q1(pp, keys, model, blind, kcol="K_common", mcol="M"):
    d = pp[(pp.model == model) & pp.key.isin(keys)].copy()
    n_vid = d.key.nunique()
    mae_k = np.mean([np.abs(g[kcol] - g.ts).mean() for _, g in d.groupby("seed")])
    mae_m = np.mean([np.abs(g[mcol] - g.ts).mean() for _, g in d.groupby("seed")])
    yt = d.drop_duplicates("key").set_index("key").ts
    mae_b = float(np.mean([abs(blind[k] - yt[k]) for k in yt.index]))
    adv = mae_b - mae_k
    wk = d.pivot(index="key", columns="seed", values=kcol)
    wm = d.pivot(index="key", columns="seed", values=mcol)
    N = float(np.median(np.concatenate([np.abs(wk[a] - wk[b]).values for a, b in itertools.combinations(SEEDS, 2)])))
    G = float(np.median(np.abs(wm - wk).values.ravel()))
    dm = mae_m - mae_k
    if dm <= T and G <= 1.5 * N:
        v = "落差小"
    elif dm >= 0.5 * adv or G >= 3 * N:
        v = "落差大"
    else:
        v = "中間"
    sev = d[d.ts.between(SEV_LO, SEV_HI)]
    bias_k = float((sev[kcol] - sev.ts).mean()) if len(sev) else np.nan
    bias_m = float((sev[mcol] - sev.ts).mean()) if len(sev) else np.nan
    n_sev = sev.key.nunique()
    chk8b = float(np.median(np.abs(d.K_own - d.K_common)))
    return dict(n=n_vid, MAE_K=mae_k, MAE_M=mae_m, dMAE=dm, MAE_blind=mae_b, Adv=adv, N=N, G=G, verdict=v,
                cond_small=(dm <= T, G <= 1.5 * N), cond_large=(dm >= 0.5 * adv, G >= 3 * N),
                n_sev=n_sev, bias_K=bias_k, bias_M=bias_m, sev_flag=(bias_m - bias_k) >= 2 if n_sev else None,
                chk8b=chk8b, chk8b_ok=chk8b < N / 2,
                seed_mae_K=[np.abs(g[kcol] - g.ts).mean() for _, g in d.groupby("seed")],
                seed_mae_M=[np.abs(g[mcol] - g.ts).mean() for _, g in d.groupby("seed")])


# ---------------------------------------------------------------- Q2
def q2_verdict(b, r, sE):
    if not np.isfinite(sE) or sE <= 0:
        return "無法判定（s_E = 0）"
    if r > sE:
        return "手機版停用"
    if r > 0.5 * sE:
        return "手機版降級"
    if abs(b) > 0.5 * sE:
        return "需重新校準"
    return "可沿用"


def q2(ang, keys, model, mcol="M", fps=None):
    a = ang[(ang.model == model) & ang.key.isin(keys)]
    rows = []
    for (ex, f), g in a.groupby(["exercise", "feature"], sort=False):
        spec = SPEC[(ex, f)]
        n_set = g.key.nunique()
        gv = g[g.valid_ok.astype(bool)]
        if fps is not None:
            gv = gv[gv.fps == fps]
        for st in ("rom", "peak"):
            kc, mc = f"K_{st}", f"{mcol}_{st}"
            x = gv[np.isfinite(gv[kc]) & np.isfinite(gv[mc])]
            dd = x[mc] - x[kc]
            if st == "peak" and spec["is_dir"]:
                dd = (dd + 180) % 360 - 180
            b, r = float(dd.mean()) if len(dd) else np.nan, float(dd.std(ddof=1)) if len(dd) > 1 else np.nan
            ke = x[x.subgroup == "Expert"][kc]
            sE = float(ke.std(ddof=1)) if len(ke) > 1 else np.nan
            insufficient = len(x) < 0.5 * n_set
            v = "樣本不足" if insufficient else q2_verdict(b, r, sE)
            rows.append(dict(exercise=ex, feature=f, summary=st, source=spec["source"], unit=spec["unit"],
                             n_set=n_set, n=len(x), n_expert=len(ke), b=b, r=r, s_E=sE,
                             b_ratio=abs(b) / sE if sE else np.nan, r_ratio=r / sE if sE else np.nan, verdict=v,
                             verdict_if_enough=q2_verdict(b, r, sE)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Q3
def q3_feature(ang, keys, model, ex, feat, mcol="M"):
    a = ang[(ang.model == model) & ang.key.isin(keys) & (ang.exercise == ex) & (ang.feature == feat)]
    x = a[a.valid_ok.astype(bool) & np.isfinite(a.K_rom) & np.isfinite(a[f"{mcol}_rom"])]
    d = x[f"{mcol}_rom"] - x.K_rom
    r = float(np.corrcoef(x.K_rom, x[f"{mcol}_rom"])[0, 1]) if len(x) > 2 else np.nan
    rand = float(d.std(ddof=1))
    ke = x[x.subgroup == "Expert"].K_rom
    sE = float(ke.std(ddof=1))
    fm_col = "frame_mae" if mcol == "M" else "frame_mae_swap"
    fr = a[a.reliable.astype(bool) & (a.fps == 25) & a.valid_ok.astype(bool) & np.isfinite(a[fm_col])]
    n_rel = len(fr)
    fmae = float((fr[fm_col] * fr.frame_n).sum() / fr.frame_n.sum()) if n_rel else np.nan
    fmae_vid = float(fr[fm_col].mean()) if n_rel else np.nan
    frame_ok = None if n_rel < 20 else fmae <= 10
    hold = r >= 0.8 and rand <= 0.5 * sE and (frame_ok is not False)
    fail = r < 0.6 or rand > sE
    v = "撐不住" if fail else ("撐得住" if hold and frame_ok is not None else ("撐得住（逐幀條件樣本不足，只依摘要值）"
                                                                          if hold else "中間"))
    return dict(feature=feat, n=len(x), n_set=a.key.nunique(), r=r, rand=rand, s_E=sE, rand_ratio=rand / sE,
                bias=float(d.mean()), frame_mae=fmae, frame_mae_video_mean=fmae_vid, n_reliable25=n_rel, verdict=v,
                K_mean=float(x.K_rom.mean()), M_mean=float(x[f"{mcol}_rom"].mean()))


def q3(ang, keys, model, mcol="M"):
    ctrl = q3_feature(ang, keys, model, "Es2", "trunk_lateral_tilt", mcol)
    rot = q3_feature(ang, keys, model, "Es3", "trunk_rotation", mcol)
    if ctrl["verdict"] == "撐不住":
        final = "Q3 不判定（對照組 Es2 側彎撐不住；流程除錯見第 4.1 節）"
    else:
        final = rot["verdict"]
    return rot, ctrl, final


# ---------------------------------------------------------------- 描述
def bone_cv(keys, model, chk):
    """追加 5 C：上臂、前臂、大腿、小腿長度（偵測幀、保留段時段）的變異係數與偏離中位數 ±30% 的幀比例。"""
    S = {s["key"]: s for s in pickle.load(open(f"data/kimore/processed/kimore_mp_common_{model}_v1.pkl", "rb"))["samples"]}
    from common_skeleton import J
    bones = {"upper_arm": ("Shoulder", "Elbow"), "forearm": ("Elbow", "Wrist"), "thigh": ("Hip", "Knee"), "shank": ("Knee", "Ankle")}
    c = chk[(chk.model == model) & (chk.status == "ok")].set_index("key")
    rows = []
    for k in keys:
        if k not in c.index:
            continue
        s = S[k]
        lo, hi = int(c.loc[k, "seg_video_first"]), int(c.loc[k, "seg_video_last"]) + 1
        X = s["skeleton"].transpose(1, 2, 0)[lo:hi][s["detected"][lo:hi].astype(bool)]
        r = dict(key=k, subgroup=s["subgroup"])
        cvs, offs = [], []
        for bn, (a, b) in bones.items():
            for side in "LR":
                L = np.linalg.norm(X[:, J[a + side]] - X[:, J[b + side]], axis=-1)
                med = np.median(L)
                cvs.append(L.std() / L.mean())
                offs.append(np.mean(np.abs(L / med - 1) > 0.3))
        r.update(cv=float(np.mean(cvs)), off30=float(np.mean(offs)))
        rows.append(r)
    return pd.DataFrame(rows)


def pipeline_check(keys, chk, al):
    """
    第 4.3 節對照組失敗時的流程除錯（診斷用，不參與判定）：主分析集中對齊可靠、MediaPipe 參考幀未左右標反的影片，
    兩邊正規化後的逐幀一致性（四肢關節各軸誤差中位數、動作相關）與 Es2 軀幹側傾的逐幀相關。
    """
    import common_skeleton as cs
    from common_skeleton import J
    K = {s["key"]: s for s in pickle.load(open("data/kimore/processed/kimore_k_common_347.pkl", "rb"))["samples"]}
    MP = {s["key"]: s for s in pickle.load(open("data/kimore/processed/kimore_mp_common_heavy_v1.pkl", "rb"))["samples"]}
    A = al.set_index(["key", "model"])
    C = chk.set_index(["key", "model"])
    rows = []
    for key in sorted(keys):
        r = A.loc[(key, "heavy")]
        if not r.reliable or C.loc[(key, "heavy"), "ref_lr_flipped"]:
            continue
        ks, ms = K[key], MP[key]
        lo, hi = int(r.seg_video_first), int(r.seg_video_last) + 1
        kj = np.round(r.a + r.b * np.arange(ms["skeleton"].shape[1])).astype(int)[lo:hi] - ks["raw_start"]
        det = ms["detected"][lo:hi].astype(bool)
        Kn, _ = cs.normalize(ks["skeleton_centered"].transpose(1, 2, 0).astype(float),
                             ref_idx=np.array(list(map(int, r.ref_raw_rows.split()))) - ks["raw_start"])
        Mn, _ = cs.normalize(ms["skeleton"].transpose(1, 2, 0).astype(float)[lo:hi],
                             ref_idx=np.array(list(map(int, r.ref_video_frames.split()))) - lo)
        Ka, Ma = Kn[kj][det], Mn[det]
        d = dict(key=key, ex=ks["exercise"])
        for ax, nm in enumerate("xyz"):
            d[f"err_{nm}"] = float(np.nanmedian(np.abs(Ka - Ma)[:, 3:, ax]))
        v = lambda X: X[:, J["ShoulderMid"]] - X[:, J["HipMid"]]
        tk, tm = np.arctan2(v(Ka)[:, 0], v(Ka)[:, 1]), np.arctan2(v(Ma)[:, 0], v(Ma)[:, 1])
        d["tilt_corr"] = float(np.corrcoef(tk, tm)[0, 1])
        rows.append(d)
    return pd.DataFrame(rows)


def ctrl_outliers(ang, keys, q):
    x = ang[(ang.model == "heavy") & ang.key.isin(keys) & (ang.feature == "trunk_lateral_tilt") & ang.valid_ok.astype(bool)].copy()
    x["d"] = x.M_rom - x.K_rom
    out = x[x.d.abs() > 60]
    return x, out


def fmt(x, n=3):
    if x is None:
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "是" if x else "否"
    if isinstance(x, (float, np.floating)):
        return "—" if not np.isfinite(x) else f"{x:.{n}f}"
    return str(x)


# ---------------------------------------------------------------- 圖
def figures(pp, ang, al, keys, chk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    d = pp[(pp.model == "heavy") & pp.key.isin(keys)].groupby("key").agg(
        K=("K_common", "mean"), M=("M", "mean"), ts=("ts", "first"), ex=("exercise", "first"), grp=("subgroup", "first"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].scatter(d.K, d.M, s=10)
    ax[0].plot([10, 50], [10, 50], "k--", lw=1)
    ax[0].set_xlabel("K-common prediction (Kinect, 3-seed mean)")
    ax[0].set_ylabel("prediction from MediaPipe heavy")
    ax[0].set_title(f"TS: M vs K (main set, n={len(d)})")
    for a, (ex, f) in zip(ax[1:], [("Es3", "trunk_rotation"), ("Es2", "trunk_lateral_tilt")]):
        x = ang[(ang.model == "heavy") & ang.key.isin(keys) & (ang.feature == f) & (ang.exercise == ex) & ang.valid_ok.astype(bool)]
        a.scatter(x.K_rom, x.M_rom, s=10)
        lim = [0, max(x.K_rom.max(), x.M_rom.max()) * 1.05]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel("Kinect ROM [deg]")
        a.set_ylabel("MediaPipe heavy ROM [deg]")
        a.set_title(f"{ex} {f} ROM (n={len(x)})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/scatter_ts_es3_es2.png", dpi=110)
    plt.close(fig)

    d["dTS"] = d.M - d.K
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, col, order in ((ax[0], "grp", ["Expert", "NotExpert", "BackPain", "Parkinson", "Stroke"]), (ax[1], "ex", EX)):
        a.boxplot([d[d[col] == g].dTS for g in order], tick_labels=order)
        a.axhline(0, color="k", lw=0.8)
        a.set_ylabel("ΔTS = M − K (3-seed mean)")
        a.set_title(f"ΔTS by {'group' if col == 'grp' else 'exercise'} (main set)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/dts_box.png", dpi=110)
    plt.close(fig)

    h = al[al.model == "heavy"]
    fig, ax = plt.subplots(figsize=(7, 4))
    r = h.rho.replace(-np.inf, np.nan).dropna()
    ax.hist(r, bins=30)
    ax.axvline(0.7, color="r", ls="--")
    ax.set_xlabel("alignment ρ (shared mapping)")
    ax.set_title(f"T6 alignment ρ (347 videos, {int(np.isinf(h.rho).sum())} not computable)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/rho_hist.png", dpi=110)
    plt.close(fig)

    # Bland-Altman：每個特徵的 ROM（每個動作一張圖）
    for ex in EX:
        fs = [f["name"] for f in FEATURE_SPECS[ex]]
        n = len(fs)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
        for a, f in zip(axs.ravel(), fs):
            x = ang[(ang.model == "heavy") & ang.key.isin(keys) & (ang.exercise == ex) & (ang.feature == f) & ang.valid_ok.astype(bool)]
            m, dd = (x.M_rom + x.K_rom) / 2, x.M_rom - x.K_rom
            a.scatter(m, dd, s=6)
            if len(dd) > 1:
                for y, ls in ((dd.mean(), "-"), (dd.mean() + 1.96 * dd.std(), "--"), (dd.mean() - 1.96 * dd.std(), "--")):
                    a.axhline(y, color="r", ls=ls, lw=0.8)
            a.set_title(f"{f} (n={len(x)})", fontsize=8)
            a.tick_params(labelsize=7)
        for a in axs.ravel()[n:]:
            a.axis("off")
        fig.suptitle(f"Bland-Altman of ROM, {ex} (M − K vs mean; main set, heavy)")
        fig.tight_layout()
        fig.savefig(f"{FIG}/bland_altman_rom_{ex}.png", dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------- 報告
def main():
    pp, ang, chk, q, al, K = load()
    S = sets(q, al)
    blind = blind_preds(K)
    main_keys = S["main"][0]
    L = []
    w = L.append

    # ---- 計算
    Q1 = {nm: q1(pp, S[nm][0], S[nm][1], blind) for nm in ("main", "full", "plaus80", "backlit", "ref347")}
    Q1["swap"] = q1(pp, main_keys, "heavy", blind, mcol="M_swap")
    Q1["main_Kown"] = q1(pp, main_keys, "heavy", blind, kcol="K_own")
    Q2 = {nm: q2(ang, S[nm][0], S[nm][1]) for nm in ("main", "full", "plaus80", "backlit", "ref347")}
    Q2["swap"] = q2(ang, main_keys, "heavy", mcol="M_swap")
    Q2["main25"] = q2(ang, main_keys, "heavy", fps=25)
    Q2["main24"] = q2(ang, main_keys, "heavy", fps=24)
    Q3 = {nm: q3(ang, S[nm][0], S[nm][1]) for nm in ("main", "full", "plaus80", "backlit", "ref347")}
    Q3["swap"] = q3(ang, main_keys, "heavy", mcol="M_swap")
    os.makedirs(OUT, exist_ok=True)
    for nm, df in Q2.items():
        df.to_csv(f"{OUT}/q2_{nm}.csv", index=False)

    # angles_summary.csv（第 6 節）
    ang.to_csv(f"{OUT}/angles_summary.csv", index=False)

    m1, m2, (rot, ctrl, q3final) = Q1["main"], Q2["main"].copy(), Q3["main"]
    # 追加 3 B：25 fps 組單獨判定與全部不同的項目，決策只用 25 fps 組
    v25 = Q2["main25"].set_index(["exercise", "feature", "summary"]).verdict
    m2["verdict_all"] = m2.verdict
    m2["verdict"] = [v25.loc[(a, b, c)] for a, b, c in m2[["exercise", "feature", "summary"]].values]
    m2.to_csv(f"{OUT}/q2_main_final.csv", index=False)
    q2c = m2.verdict.value_counts()

    w("# v7 落差報告（gap_report）：Kinect ↔ MediaPipe")
    w("")
    w(f"產生：`scripts/gap_report.py`；資料：`runs/v7_gap/{{paired_preds,angles,check}}.csv`（T7，`paired_compare.py`）。")
    w("判定一律依第 4 節（凍結於 `4915ae9`）與追加 1～18；**判定只用主分析集 × heavy**（追加 5 A、6 A），其他分析集只並列。")
    w("")
    w("## 0. 三個判定")
    w("")
    w("| 問題 | 判定 | 依據 |")
    w("| --- | --- | --- |")
    w(f"| Q1 評分落差 | **{m1['verdict']}** | ΔMAE = {m1['dMAE']:.3f}（T = {T}，0.5 × Adv = {0.5 * m1['Adv']:.3f}）；"
      f"G = {m1['G']:.3f}（1.5 N = {1.5 * m1['N']:.3f}，3 N = {3 * m1['N']:.3f}） |")
    w(f"| Q2 特徵落差 | 逐特徵判定（{len(m2)} 項 = {len(m2.drop_duplicates(['exercise', 'feature']))} 個特徵 × ROM／峰值）："
      + "、".join(f"{k} {v}" for k, v in q2c.items()) + " | 第 4.2 節；見第 3 節 |")
    w(f"| Q3 旋轉落差 | **{q3final}** | Es3 ROM：r = {rot['r']:.3f}、隨機誤差 = {rot['rand']:.2f}° "
      f"（{rot['rand_ratio']:.2f} s_E）、逐幀 MAE = {rot['frame_mae']:.1f}°；對照組 Es2 側彎：{ctrl['verdict']} |")
    w("")
    sev_txt = ("可下結論" if m1["n_sev"] >= SEV_MIN_N else "只做描述")
    w(f"- 重症警示（真實 10–25 分，{m1['n_sev']} 部，{sev_txt}）：偏差 K {m1['bias_K']:+.2f}、M {m1['bias_M']:+.2f}，"
      f"M − K = {m1['bias_M'] - m1['bias_K']:+.2f} → {'**標記「重症高估惡化」**' if m1['sev_flag'] else '不標記'}。")
    w("")

    # ---- 1. 分析集
    w("## 1. 分析集與樣本")
    w("")
    w("| 分析集 | 模型 | 部數 | 用途 |")
    w("| --- | --- | --- | --- |")
    use = {"main": "**判定**", "full": "敏感度（追加 5 A）", "plaus80": "敏感度（追加 6 B）", "backlit": "描述（追加 6 A）",
           "ref347": "參考（heavy 有參考幀者）"}
    for nm in ("main", "full", "plaus80", "backlit", "ref347"):
        w(f"| {nm} | {S[nm][1]} | {len(S[nm][0])} | {use[nm]} |")
    w(f"| swap | heavy | {len(main_keys)} | 敏感度：左右標反換回（追加 11 E、18 B） |")
    w("")
    kdf = pd.DataFrame([dict(key=s["key"], grp=s["subgroup"], ex=s["exercise"], ts=s["ts"]) for s in K]).set_index("key")
    md = kdf.loc[sorted(main_keys)]
    w("主分析集各組 × 動作筆數、TS 分布（追加 6 A、16 A）：")
    w("")
    ct = pd.crosstab(md.grp, md.ex, margins=True)
    w("| 組別 | " + " | ".join(ct.columns) + " | TS 中位數 | TS 範圍 | 10–25 分 |")
    w("|" + " --- |" * (len(ct.columns) + 4))
    for g in ct.index:
        sub = md if g == "All" else md[md.grp == g]
        w(f"| {g} | " + " | ".join(str(v) for v in ct.loc[g]) +
          f" | {sub.ts.median():.1f} | {sub.ts.min():.1f}～{sub.ts.max():.1f} | {int(sub.ts.between(10, 25).sum())} |")
    w("")
    pk = kdf[kdf.grp == "Parkinson"]
    w(f"- 追加 5 B：巴金森組 347 筆 {len(pk)} 部（TS 中位數 {pk.ts.median():.1f}，10–25 分 {int(pk.ts.between(10, 25).sum())} 部）"
      f" → 主分析集 {int((md.grp == 'Parkinson').sum())} 部（TS 中位數 {md[md.grp == 'Parkinson'].ts.median():.1f}，"
      f"10–25 分 {int(md[md.grp == 'Parkinson'].ts.between(10, 25).sum())} 部）。")
    w(f"- 追加 1 A／B：v3 380 筆中 33 筆沒有影片（Parkinson 13/80、Stroke 6/49，其他組 ≤ 7.5%），含 3 筆重症樣本；347 筆中真實 10–25 分 "
      f"{int(kdf.ts.between(10, 25).sum())} 筆。")
    w("")

    # ---- 2. Q1
    w("## 2. Q1 評分落差（第 4.1 節）")
    w("")
    w("K_s = K-common 模型吃 Kinect 共同骨架（共同參考幀正規化，追加 8 B）；M_s = 同一模型吃 MediaPipe 共同骨架；"
      "out-of-fold（第 2 節第 2 點），測試設定同 T5。T 固定 0.316（追加 9 B）。")
    w("")
    w("| 分析集 | 部數 | MAE_K | MAE_M | ΔMAE | MAE_blind | Adv | N | G | G/N | 判定 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for nm in ("main", "full", "plaus80", "swap", "backlit", "ref347", "main_Kown"):
        r = Q1[nm]
        lab = {"main_Kown": "main（K = K_own，參考）"}.get(nm, nm)
        w(f"| {lab} | {r['n']} | {r['MAE_K']:.3f} | {r['MAE_M']:.3f} | {r['dMAE']:.3f} | {r['MAE_blind']:.3f} | "
          f"{r['Adv']:.3f} | {r['N']:.3f} | {r['G']:.3f} | {r['G'] / r['N']:.2f} | {r['verdict']} |")
    w("")
    w("**主分析集逐條對照：**")
    w("")
    w(f"- 落差小：ΔMAE ≤ T（{m1['dMAE']:.3f} ≤ {T}：{fmt(m1['cond_small'][0])}）且 G ≤ 1.5 N（{m1['G']:.3f} ≤ {1.5 * m1['N']:.3f}："
      f"{fmt(m1['cond_small'][1])}）")
    w(f"- 落差大：ΔMAE ≥ 0.5 × Adv（{m1['dMAE']:.3f} ≥ {0.5 * m1['Adv']:.3f}：{fmt(m1['cond_large'][0])}）或 G ≥ 3 N"
      f"（{m1['G']:.3f} ≥ {3 * m1['N']:.3f}：{fmt(m1['cond_large'][1])}）")
    w(f"- → **{m1['verdict']}**")
    w(f"- 各 seed MAE：K {', '.join(f'{v:.3f}' for v in m1['seed_mae_K'])}；M {', '.join(f'{v:.3f}' for v in m1['seed_mae_M'])}")
    w(f"- 追加 8 B 檢查：median |K_own − K_common| = {m1['chk8b']:.3f}，N/2 = {m1['N'] / 2:.3f} → "
      f"{'符合' if m1['chk8b_ok'] else '**不符合**（標出，由使用者決定如何解讀）'}")
    w(f"- 重症警示：見第 0 節。full：M − K = {Q1['full']['bias_M'] - Q1['full']['bias_K']:+.2f}；"
      f"swap：{Q1['swap']['bias_M'] - Q1['swap']['bias_K']:+.2f}。")
    w("")
    dm = pp[(pp.model == "heavy") & pp.key.isin(main_keys)].groupby("key").agg(
        K=("K_common", "mean"), M=("M", "mean"), ts=("ts", "first"), ex=("exercise", "first"), grp=("subgroup", "first"),
        ref_flip=("ref_lr_flipped", "first"))
    dm["dTS"] = dm.M - dm.K
    w("主分析集 ΔTS（M − K，3 seed 平均）依動作／組別（描述；第 6 節：場景與組別高度相關，不得解讀成組別差異）：")
    w("")
    w("| 分組 | 部數 | ΔTS 平均 | ΔTS 中位數 | |ΔTS| 中位數 |")
    w("| --- | --- | --- | --- | --- |")
    for col, order in (("ex", EX), ("grp", ["Expert", "NotExpert", "BackPain", "Parkinson", "Stroke"])):
        for g in order:
            x = dm[dm[col] == g].dTS
            w(f"| {g} | {len(x)} | {x.mean():+.2f} | {x.median():+.2f} | {x.abs().median():.2f} |")
    for flag in (False, True):
        x = dm[dm.ref_flip == flag].dTS
        w(f"| MediaPipe 參考幀左右標反 = {'是' if flag else '否'} | {len(x)} | {x.mean():+.2f} | {x.median():+.2f} | {x.abs().median():.2f} |")
    w("")
    n10 = int((dm.M <= 10.01).sum())
    w(f"- MediaPipe 預測被夾在 10 分（3 seed 平均 ≤ 10.01）的影片：{n10} 部。")
    w("")

    # ---- 3. Q2
    w("## 3. Q2 特徵落差（第 4.2 節）")
    w("")
    w("特徵清單：`docs/v7_feature_list.md`（追加 17 A、18 A，63 項）。摘要值 ROM = P95 − P5、峰值 = P95；"
      "b = mean(M − K)、r = SD(M − K)、s_E = Kinect 專家組 SD；MediaPipe 有效幀 < 50% 的影片不納入（追加 7 D），"
      "納入 < 該動作分析集一半 → 樣本不足（追加 8 C）。單位見清單（度／軀幹長／軀幹長²）。")
    w("")
    w("判定統計（主分析集，heavy）：")
    w("")
    w("| 判定 | " + " | ".join(EX) + " | 合計 |")
    w("| --- |" + " --- |" * (len(EX) + 1))
    for v in ["可沿用", "需重新校準", "手機版降級", "手機版停用", "樣本不足", "無法判定（s_E = 0）"]:
        cnt = [int(((m2.exercise == e) & (m2.verdict == v)).sum()) for e in EX]
        if sum(cnt):
            w(f"| {v} | " + " | ".join(map(str, cnt)) + f" | {sum(cnt)} |")
    w("")
    w("逐特徵（主分析集；heavy／full／左右換回並列，判定只看 heavy）：")
    w("")
    fq = Q2["full"].set_index(["exercise", "feature", "summary"])
    sq = Q2["swap"].set_index(["exercise", "feature", "summary"])
    q25 = Q2["main25"].set_index(["exercise", "feature", "summary"])
    w("「**判定**」= 最終判定（追加 3 B：與 25 fps 組單獨判定不同時採 25 fps 組）；「全部影片」= 含 24 fps 組的判定。")
    w("")
    w("| 動作 | 特徵 | 摘要 | 來源 | 納入/集 | b | r | s_E | |b|/s_E | r/s_E | **判定** | 全部影片 | full | 左右換回 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    diff25 = []
    for _, r in m2.iterrows():
        idx = (r.exercise, r.feature, r.summary)
        if r.verdict_all != r.verdict:
            diff25.append(idx)
        w(f"| {r.exercise} | `{r.feature}` | {r.summary} | {r.source} | {r.n}/{r.n_set} | {fmt(r.b, 2)} | {fmt(r.r, 2)} | "
          f"{fmt(r.s_E, 2)} | {fmt(r.b_ratio, 2)} | {fmt(r.r_ratio, 2)} | **{r.verdict}** | {r.verdict_all} | {fq.loc[idx, 'verdict']} | "
          f"{sq.loc[idx, 'verdict']} |")
    w("")
    w(f"- 追加 3 B（24 fps 組，{int((dm.index.isin(al[(al.fps == 24)].key)).sum())} 部）：25 fps 組單獨判定與全部不同的有 {len(diff25)} 項"
      + (("：" + "、".join(f"{a} {b} {c}（{m2.set_index(['exercise', 'feature', 'summary']).loc[(a, b, c), 'verdict_all']} → "
                         f"{m2.set_index(['exercise', 'feature', 'summary']).loc[(a, b, c), 'verdict']}）" for a, b, c in diff25)
         + "。依追加 3 B，這些項目的決策只用 25 fps 組（已反映在「判定」欄與統計）。") if diff25 else "。"))
    insuff = m2[m2.verdict == "樣本不足"]
    if len(insuff):
        w(f"- 樣本不足（追加 8 C：MediaPipe 有效幀 ≥ 50% 的影片少於該動作主分析集的一半）：" +
          "、".join(f"{r.exercise} `{r.feature}` {r.summary}（{r.n}/{r.n_set}）" for _, r in insuff.iterrows()) + "。")
    nv = m2.drop_duplicates(["exercise", "feature"])[["exercise", "feature", "n", "n_set"]]
    nvf = Q2["full"].drop_duplicates(["exercise", "feature"]).set_index(["exercise", "feature"]).n
    w("- 追加 8 C：每個特徵 heavy／full 各納入幾部，見上表「納入/集」與 `q2_full.csv`；"
      f"heavy 納入比例中位數 {np.median(nv.n / nv.n_set):.2f}，full {np.median([nvf.loc[(a, b)] / c for a, b, c in nv[['exercise', 'feature', 'n_set']].values]):.2f}。")
    w("- 追加 18 C：與左右有關的特徵（angle_c、骨盆／軀幹方向角、左右成對的角度），MediaPipe 大量左右標反的影片 ROM 接近 180°；"
      "「左右換回」欄是若左右判斷正確時的判定，只作說明。")
    w("- 追加 7 C：依賴骨盆位移的官方特徵（Es4 SpineBase 位移等）兩邊不可比，不在清單中（見特徵清單「不納入」）。")
    w("")

    # ---- 4. Q3
    w("## 4. Q3 旋轉落差（第 4.3 節）")
    w("")
    w("| 分析集 | 特徵 | 納入/集 | Pearson r | 隨機誤差 SD(M−K) | s_E | 隨機誤差/s_E | 偏差 | 逐幀 MAE（幀加權） | 可靠 25 fps 部數 | 判定 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for nm in ("main", "full", "plaus80", "swap", "backlit", "ref347"):
        rr, cc, fin = Q3[nm]
        for x in (rr, cc):
            w(f"| {nm} | {x['feature']} | {x['n']}/{x['n_set']} | {x['r']:.3f} | {x['rand']:.2f}° | {x['s_E']:.2f}° | "
              f"{x['rand_ratio']:.2f} | {x['bias']:+.2f}° | {fmt(x['frame_mae'], 1)}° | {x['n_reliable25']} | {x['verdict']} |")
    w("")
    w("**主分析集逐條對照：**")
    w("")
    w(f"- 對照組 Es2 側彎（`trunk_lateral_tilt` ROM）：r = {ctrl['r']:.3f}、隨機誤差 = {ctrl['rand_ratio']:.2f} s_E、"
      f"逐幀 MAE = {ctrl['frame_mae']:.1f}° → **{ctrl['verdict']}**")
    w(f"- Es3 旋轉（`trunk_rotation` ROM）：撐得住需 r ≥ 0.8（{rot['r']:.3f}）、隨機誤差 ≤ 0.5 s_E（{rot['rand_ratio']:.2f} s_E）、"
      f"逐幀 MAE ≤ 10°（{rot['frame_mae']:.1f}°，{rot['n_reliable25']} 部可靠 25 fps 影片 ≥ 20：逐幀條件照常使用）；"
      f"撐不住：r < 0.6 或隨機誤差 > 1 s_E → **{rot['verdict']}**")
    w(f"- → Q3：**{q3final}**")
    w(f"- 平均 ROM：Es3 Kinect {rot['K_mean']:.1f}°、MediaPipe {rot['M_mean']:.1f}°；Es2 Kinect {ctrl['K_mean']:.1f}°、MediaPipe {ctrl['M_mean']:.1f}°。")
    w("")
    if ctrl["verdict"] == "撐不住":
        pc = pipeline_check(main_keys, chk, al)
        xs, out = ctrl_outliers(ang, main_keys, q)
        rest = xs[~xs.key.isin(out.key)]
        w("### 4.1 對照組失敗後的流程除錯（第 4.3 節「必須先除錯」；**診斷用，不參與判定**）")
        w("")
        w(f"1. **流程本身**：主分析集中對齊可靠、MediaPipe 參考幀未左右標反的 {len(pc)} 部，兩邊正規化後四肢關節的逐幀誤差中位數"
          f"（軀幹長）x {pc.err_x.median():.3f}、y {pc.err_y.median():.3f}、z {pc.err_z.median():.3f}；"
          f"Es2 軀幹側傾的逐幀相關中位數 {pc[pc.ex == 'Es2'].tilt_corr.median():.3f}（{int((pc.ex == 'Es2').sum())} 部）。"
          "對齊、正規化、座標軸在這些影片上一致，沒有發現流程錯誤。K_own 也完全重現 T5 的預測（T7 試跑）。")
        w(f"2. **對照組失敗的來源**：Es2 側彎 ROM 的 M − K 有 {len(out)} 部超過 60°（"
          + "、".join(f"{k} {d:+.0f}°" for k, d in zip(out.key, out.d)) +
          "）。這些影片中 MediaPipe 有 7%～61% 的幀軀幹上下顛倒（側傾角 > 90°），**這些幀全部不通過追加 6 B 的骨架合理規則**；"
          "依追加 7 D 只遮罩 visibility < 0.5 的關節，這些幀照常計入 P95／P5。")
        w(f"3. 其餘 {len(rest)} 部：M − K 的 SD = {rest.d.std(ddof=1):.2f}°（{rest.d.std(ddof=1) / ctrl['s_E']:.2f} s_E）、"
          f"r = {np.corrcoef(rest.K_rom, rest.M_rom)[0, 1]:.3f}、偏差 {rest.d.mean():+.2f}°。")
        w("4. 結論：對照組失敗主要是 MediaPipe 的嚴重骨架錯誤（上下顛倒），不是對齊／正規化／座標軸的錯誤。"
          "但依第 4.3 節的規則，對照組撐不住時 Q3 **不判定**；是否改用其他處理（例如角度只用通過骨架合理規則的幀）需由使用者決定並寫入追加。")
        w("")

    # ---- 5. 對齊、左右、品質（描述）
    h = al[al.model == "heavy"].set_index("key")
    hm = h.loc[sorted(main_keys)]
    w("## 5. 對齊、左右標記與骨架品質（描述）")
    w("")
    w("| 動作 | 主分析集 | 對齊可靠 | 不可靠 | 整部鏡像（heavy） | 腕標反比例中位數 | 踝標反比例中位數 | MediaPipe 參考幀左右標反 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    cm = chk[(chk.model == "heavy") & chk.key.isin(main_keys)].set_index("key")
    for e in EX:
        x = hm[hm.exercise == e]
        w(f"| {e} | {len(x)} | {int(x.reliable.sum())} | {int((~x.reliable.astype(bool)).sum())} | {int(x.mirrored.sum())} | "
          f"{x.flip_wrist.median():.3f} | {x.flip_ankle.median():.3f} | {int(cm.loc[cm.index.intersection(x.index)].ref_lr_flipped.sum())} |")
    w("")
    w(f"- 追加 10 C／15 C：主分析集對齊不可靠 {int((~hm.reliable.astype(bool)).sum())} 部，納入 Q1 與摘要值，排除於逐幀指標。")
    cov = hm[hm.kinect_coverage < 0.5]
    w(f"- 追加 14 B：Kinect 覆蓋率 < 50%：{len(cov)} 部（" + "、".join(f"{k} {v:.3f}" for k, v in cov.kinect_coverage.items())
      + "）。這些樣本的 v3 骨架只涵蓋部分動作，是 v3 資料本身的限制；不排除。")
    w(f"- 追加 13 A：Expert 的 Es2 大量左右標反；主分析集 Es2 整部鏡像 {int(hm[hm.exercise == 'Es2'].mirrored.sum())} 部。")
    w(f"- 追加 15 A／16：整部鏡像的原因自動檢查＋人工檢查，沒有影片翻轉；全部歸為「MediaPipe 標反」或「原因無法判斷」，不校正。")
    nflip = int(cm.ref_lr_flipped.sum())
    w(f"- **MediaPipe 參考幀左右標反**：主分析集 {nflip} 部 heavy 的共同參考幀中，MediaPipe 髖部過半左右標反。主分析不校正（追加 4 B、11 E），"
      "這些影片的 MediaPipe 正規化左右軸與 Kinect 相反（整段繞垂直軸轉 180°），左右與前後方向的量都受影響。"
      "左右換回的敏感度分析（swap）先換回再正規化，量的是若左右判斷正確時的落差。")
    bc = bone_cv(sorted(main_keys), "heavy", chk)
    bcf = bone_cv(sorted(main_keys), "full", chk)
    w(f"- 追加 5 C 骨架長度穩定度（上臂、前臂、大腿、小腿；偵測幀、保留段時段）：heavy 變異係數中位數 {bc.cv.median():.3f}、"
      f"偏離中位數 ±30% 的幀比例中位數 {bc.off30.median():.3f}；full {bcf.cv.median():.3f}、{bcf.off30.median():.3f}。")
    w(f"- 追加 6 B 骨架合理率 < 80% 的主分析集影片：{len(main_keys) - len(S['plaus80'][0])} 部；排除後的判定見 plaus80 列。")
    w("")

    # ---- 6. 敏感度分析
    w("## 6. 敏感度分析與並列結果")
    w("")
    w("| 分析集 | Q1 | Q2：可沿用／重新校準／降級／停用／樣本不足 | Q3 |")
    w("| --- | --- | --- | --- |")
    for nm in ("main", "full", "plaus80", "swap", "backlit", "ref347"):
        c = (m2 if nm == "main" else Q2[nm]).verdict.value_counts()
        w(f"| {nm} | {Q1[nm]['verdict']} | {c.get('可沿用', 0)}／{c.get('需重新校準', 0)}／{c.get('手機版降級', 0)}／"
          f"{c.get('手機版停用', 0)}／{c.get('樣本不足', 0)} | {Q3[nm][2]} |")
    w("")
    chg = [nm for nm in ("plaus80",) if (Q1[nm]["verdict"] != m1["verdict"] or Q3[nm][2] != q3final)]
    w(f"- 追加 6 B：排除骨架合理率 < 80% 後，Q1／Q3 判定{'**改變**（' + '、'.join(chg) + '），由使用者決定如何解讀' if chg else '不變'}。")
    w("- full、swap、backlit、ref347 只並列，不得據以改變判定（追加 5 A、6 A、11 E）。逆光分析集與主分析集的差異不得解讀成組別差異（追加 6 A）。")
    w("")

    # ---- 7. 解讀限制
    w("## 7. 解讀限制（第 5 節與追加）")
    w("")
    for t in [
        "Kinect 不是真值：量的是兩種裝置的差異，不是 MediaPipe 的誤差；旋轉尤其如此。",
        "落差大不代表 MediaPipe 版一定較差：從頭訓練後的 MediaPipe 模型以自己的交叉驗證 MAE 為準。",
        "只有 347/380 筆有影片；缺影片集中在 Parkinson 與 Stroke（追加 1 B）。K-common 的 MAE 不能和 v6（380 筆）直接比較。",
        "RGB 為 960×540，人在畫面中偏小，且臉部打了馬賽克：MediaPipe 的偵測與左右判斷可能比手機 demo 悲觀（追加 1 B、5 A、6 C、11 E）。",
        "KIMORE 是正面拍攝，對旋轉最不利；Es3 為站姿、KERAAL 為坐姿，Q3 只是代理指標（第 4.3 節）。",
        "摘要值用 P95／P5，不是官方的每次重複峰值平均；不套用官方 1 Hz 濾波（追加 17 A、18 A）。",
        "MediaPipe 對鏡像不對稱：同一部影片翻轉前後偵測結果明顯不同（例如 S_ID9_Es5：11.9% → 66.5%）（追加 12 A）。",
        "臉部馬賽克是否造成左右誤判：visibility 在 KIMORE 上不可靠，無法判斷，改由 demo 階段驗證（追加 13 C）。",
        "左右標記主分析一律不校正；MediaPipe 參考幀左右標反時，正規化方向整段相反（第 5 節）。",
        "第 2 節第 1 點的「avg.pt + MediaPipe 補點版」參考結果本次未跑（不參與決策）。",
    ]:
        w(f"- {t}")
    w("")
    w("## 8. 圖")
    w("")
    for fn, t in [("scatter_ts_es3_es2.png", "M vs K：TS、Es3 ROM、Es2 ROM"), ("dts_box.png", "ΔTS 依組別／動作"),
                  ("rho_hist.png", "對齊 ρ 分布"), *[(f"bland_altman_rom_{e}.png", f"Bland-Altman（ROM，{e}）") for e in EX]]:
        w(f"- `fig/{fn}`：{t}")
    w("")
    ex_md = f"{OUT}/explore/explore.md"          # 追加 19 B：事後探索分析（scripts/gap_explore.py 產生；不參與判定）
    if os.path.exists(ex_md):
        L.append(open(ex_md).read())
    open(f"{OUT}/gap_report.md", "w").write("\n".join(L))
    figures(pp, ang, al, main_keys, chk)
    print("→", f"{OUT}/gap_report.md")
    print(f"Q1 {m1['verdict']} ΔMAE {m1['dMAE']:.3f} G {m1['G']:.3f} N {m1['N']:.3f} Adv {m1['Adv']:.3f}")
    print("Q2", q2c.to_dict())
    print("Q3", q3final, "| rot", rot["verdict"], "| ctrl", ctrl["verdict"])


if __name__ == "__main__":
    main()
