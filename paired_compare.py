"""
v7 T7：配對評分（Q1）與角度比較（Q2、Q3）的逐部資料產生（第 3.6、3.7 節；追加 7、8 B、9 B、10 C、11 E、14）。
本程式只產生「每部影片」的配對數值；ΔMAE、G、b、r、s_E 等判定用的彙整另寫（T8 前），不在這裡算。

共同前處理（每部影片 v、每個骨架模型 m ∈ {heavy, full}；主分析 = heavy）：
  - 對應與參考幀：T6 的 align_all.csv（共用對應 a、b；此模型自己的共同參考幀 ref_video_frames / ref_raw_rows）。
  - 追加 14 A：MediaPipe 只取對應後落在 Kinect 保留段 [start, end) 內的影片幀（seg_video_first～seg_video_last）。
  - 追加 8 B：Kinect（skeleton_centered）與 MediaPipe（skeleton，已逐幀減 HipMid、未正規化）都用同一組
    共同參考幀（追加 7 B）重新正規化（common_skeleton.normalize，原點/方向 = 參考幀中位數，尺度 = 整段軀幹長中位數）。
    Kinect 參考幀 index = ref_raw_rows − raw_start；MediaPipe = ref_video_frames − seg_video_first。
  - 參考幀不足 10 幀（ref_ok = False）→ 該模型不做（追加 7 B、15 B：已在主分析集外）。

Q1（第 3.6 節）：每部影片 v、每個 seed s，用 v 所在測試折的 K-common 模型（runs/v7_kcommon_s{s}/fold{f}/avg.pt，
  即 test_predictions.csv 的 y_pred）預測，測試設定與 T5 相同（5 個窗口、窗口長 = 50%、重採樣 100 幀、夾在 10～50）：
  - K_own    ：Kinect，自己的前 10 幀正規化（= T5 的輸入）；應與 T5 的 y_pred 相同（流程檢查 own_vs_t5）。
  - K_common ：Kinect，共同參考幀正規化（追加 8 B）。
  - M        ：MediaPipe，共同參考幀正規化；未偵測的幀線性內插（追加 7 D，fill_undetected）。
  - M_swap   ：追加 11 E 敏感度分析——以 Kinect 為參考，把 MediaPipe 左右標反的幀左右換回後再預測。
               逐幀判斷：6 對左右關節（肩、肘、腕、髖、膝、踝）合計的 x、y 距離，交叉配對比同側配對小 → 整幀左右對調
               （未正規化、逐幀減 HipMid 的相機座標，同 align.lr_flip）。只用於「若左右判斷正確時的落差」，不參與判定。
  追加 8 B 的檢查量 |K_own − K_common| 逐部列出；中位數與 N/2 的比較在彙整時做。
Q2／Q3（第 3.7 節；特徵定義見 FEATURES，**暫定清單，待使用者依特徵函式庫確認**）：
  - 角度在共同參考幀正規化後的座標上計算。MediaPipe 端 visibility < 0.5 的關節視為缺漏（追加 7 D），
    該幀用到此關節的角度為 NaN；有效幀比例 < 50% → 該影片不納入該特徵（valid_ok = False）。Kinect 不遮罩。
  - 摘要值（決策用）：ROM = P95 − P5、峰值 = P95（暫定，待確認）。兩邊各自在保留段時段上計算（追加 14 A）。
  - 逐幀：MediaPipe 第 i 幀 ↔ Kinect 第 r(i) 行，兩邊都有值的幀上的 MAE（度）；只在對齊可靠（追加 10 C、13 B）的影片計算，
    不可靠者 frame_mae = NaN（仍輸出摘要值）。
  - 角度差的週期性：方向角（旋轉、傾角）以 (M − K + 180) mod 360 − 180 計算差值。
  - 追加 11 E：M_swap_rom／M_swap_peak／frame_mae_swap = 左右標反幀換回後的同一組量（敏感度分析，不參與判定）。

輸出（--out，預設 runs/v7_gap/）：
  paired_preds.csv  每部影片 × 模型 × seed：K_own、K_t5、K_common、M、M_swap、左右對調幀比例、覆蓋率、截掉行數…
  angles.csv        每部影片 × 模型 × 特徵：K／M 的 ROM、峰值、有效幀比例、逐幀 MAE、對齊方式、ρ、是否可靠
  check.csv         每部影片 × 模型：流程檢查（長度、參考幀、對應範圍）

    python paired_compare.py --keys E_ID3_Es1,E_ID14_Es2 --out runs/v7_gap_trial [--plot]
    python paired_compare.py --keys main --out runs/v7_gap          # 主分析集（main_set_final）
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import common_skeleton as cs
from common_skeleton import J
from dataset import crop, resample

K_PKL = "data/kimore/processed/kimore_k_common_347.pkl"
MP_PKL = "data/kimore/processed/kimore_mp_common_{m}_v1.pkl"
ALIGN_ALL = "data/kimore/mediapipe/qc/align_check/align_all.csv"
QC_HFLIP = "data/kimore/mediapipe/qc/qc_hflip.csv"
RUNS = "runs/v7_kcommon_s{s}/fold{f}/avg.pt"
T5_PRED = "runs/v7_kcommon_s{s}/fold{f}/test_predictions.csv"
SEEDS = (42, 43, 44)
MODELS = ("heavy", "full")
OUT_FRAMES, EVAL_WINDOWS, EVAL_CROP = 100, 5, 0.5          # 與 T5（train_rehab.run_fold）相同
TS_MIN, TS_MAX = 10.0, 50.0
VALID_MIN = 0.5
LR_PAIRS = ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"]


# ---------------------------------------------------------------- 角度
def _ang3(X, a, b, c):
    """三點角度 ∠abc（度），X: (T,17,3)。"""
    u, v = X[:, J[a]] - X[:, J[b]], X[:, J[c]] - X[:, J[b]]
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _seg_vs_down(X, a, b):
    """a→b 段與「向下」軸（−y）的夾角（度）：上臂 Shoulder→Elbow 用來量肩屈曲／外展。"""
    v = X[:, J[b]] - X[:, J[a]]
    return np.degrees(np.arccos(np.clip(-v[:, 1] / np.linalg.norm(v, axis=-1), -1, 1)))


def _tilt_frontal(X, a, b):
    """a→b 段在冠狀面（x–y）相對垂直軸的側傾角（度，帶正負）。"""
    v = X[:, J[b]] - X[:, J[a]]
    return np.degrees(np.arctan2(v[:, 0], v[:, 1]))


def _tilt_sagittal(X, a, b):
    """a→b 段在矢狀面（z–y）相對垂直軸的前傾角（度，帶正負）。"""
    v = X[:, J[b]] - X[:, J[a]]
    return np.degrees(np.arctan2(v[:, 2], v[:, 1]))


def _yaw(X, a, b):
    """a→b 連線在水平面（x–z）上的方向角（度）。"""
    v = X[:, J[b]] - X[:, J[a]]
    return np.degrees(np.arctan2(v[:, 2], v[:, 0]))


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def trunk_rotation(X):
    return _wrap(_yaw(X, "ShoulderL", "ShoulderR") - _yaw(X, "HipL", "HipR"))


# 暫定特徵清單（待使用者依特徵函式庫確認）：名稱 → (動作, 函式, 用到的關節, 是否為方向角)
FEATURES = {
    "Es1": {
        "shoulder_flex_L": (lambda X: _seg_vs_down(X, "ShoulderL", "ElbowL"), ["ShoulderL", "ElbowL"], False),
        "shoulder_flex_R": (lambda X: _seg_vs_down(X, "ShoulderR", "ElbowR"), ["ShoulderR", "ElbowR"], False),
        "elbow_L": (lambda X: _ang3(X, "ShoulderL", "ElbowL", "WristL"), ["ShoulderL", "ElbowL", "WristL"], False),
        "elbow_R": (lambda X: _ang3(X, "ShoulderR", "ElbowR", "WristR"), ["ShoulderR", "ElbowR", "WristR"], False),
    },
    "Es2": {
        "trunk_lateral_tilt": (lambda X: _tilt_frontal(X, "HipMid", "ShoulderMid"), ["HipL", "HipR", "ShoulderL", "ShoulderR"], True),
        "shoulder_flex_L": (lambda X: _seg_vs_down(X, "ShoulderL", "ElbowL"), ["ShoulderL", "ElbowL"], False),
        "shoulder_flex_R": (lambda X: _seg_vs_down(X, "ShoulderR", "ElbowR"), ["ShoulderR", "ElbowR"], False),
    },
    "Es3": {
        "trunk_rotation": (trunk_rotation, ["HipL", "HipR", "ShoulderL", "ShoulderR"], True),
    },
    "Es4": {
        "pelvis_rotation": (lambda X: _yaw(X, "HipL", "HipR"), ["HipL", "HipR"], True),
        "pelvis_lateral_tilt": (lambda X: _wrap(np.degrees(np.arctan2(
            (X[:, J["HipR"]] - X[:, J["HipL"]])[:, 1], (X[:, J["HipR"]] - X[:, J["HipL"]])[:, 0]))), ["HipL", "HipR"], True),
    },
    "Es5": {
        "knee_L": (lambda X: _ang3(X, "HipL", "KneeL", "AnkleL"), ["HipL", "KneeL", "AnkleL"], False),
        "knee_R": (lambda X: _ang3(X, "HipR", "KneeR", "AnkleR"), ["HipR", "KneeR", "AnkleR"], False),
        "hip_L": (lambda X: _ang3(X, "ShoulderL", "HipL", "KneeL"), ["ShoulderL", "HipL", "KneeL"], False),
        "hip_R": (lambda X: _ang3(X, "ShoulderR", "HipR", "KneeR"), ["ShoulderR", "HipR", "KneeR"], False),
        "trunk_forward_lean": (lambda X: _tilt_sagittal(X, "HipMid", "ShoulderMid"), ["HipL", "HipR", "ShoulderL", "ShoulderR"], True),
    },
}


def summary(a):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    p5, p95 = np.percentile(a, [5, 95])
    return float(p95 - p5), float(p95)


# ---------------------------------------------------------------- 模型
_models = {}


def load_model(seed, fold, device):
    key = (seed, fold)
    if key not in _models:
        from model.rehab_ctrgcn import Model
        m = Model(num_class=1, num_point=cs.NUM_COMMON, in_channels=3, init_bias=0.5, gc_type="alter",
                  graph="common17").to(device)
        m.load_state_dict(torch.load(RUNS.format(s=seed, f=fold), map_location=device))
        m.eval()
        _models[key] = m
    return _models[key]


@torch.no_grad()
def predict(x_tvc, seed, fold, device):
    """x_tvc: (T,17,3) 已正規化、無 NaN。回傳 TS（10～50），測試設定同 T5。"""
    x = np.ascontiguousarray(x_tvc.transpose(2, 0, 1), dtype=np.float32)       # (3,T,V)
    L = x.shape[1]
    win = int(np.clip(round(L * EVAL_CROP), min(L, 15), L))
    starts = np.linspace(0, L - win, EVAL_WINDOWS).round().astype(int)
    w = np.stack([resample(crop(x, st, win), OUT_FRAMES) for st in starts])
    out = load_model(seed, fold, device)(torch.from_numpy(w).to(device)).mean(0)
    return float(out.clamp(TS_MIN / TS_MAX, 1.0).item() * TS_MAX)


# ---------------------------------------------------------------- 左右對調（追加 11 E）
def lr_swap_frames(Ka, Mc):
    """逐幀：6 對左右關節合計的 x、y 距離，交叉 < 同側 → True。Ka、Mc：(T,17,3) 逐幀減 HipMid、未正規化。"""
    same = cross = 0.0
    ok = np.ones(len(Mc), bool)
    for p in LR_PAIRS:
        kl, kr = Ka[:, J[p + "L"], :2], Ka[:, J[p + "R"], :2]
        ml, mr = Mc[:, J[p + "L"], :2], Mc[:, J[p + "R"], :2]
        ok &= np.isfinite(kl[:, 0]) & np.isfinite(kr[:, 0]) & np.isfinite(ml[:, 0]) & np.isfinite(mr[:, 0])
        d = lambda u, v: np.linalg.norm(u - v, axis=1)
        same = same + d(ml, kl) + d(mr, kr)
        cross = cross + d(ml, kr) + d(mr, kl)
    return ok & (cross < same), ok


def swap_lr(X, frames):
    X = X.copy()
    for a, b in cs.COMMON_FLIP_PAIRS:
        X[frames, a], X[frames, b] = X[frames, b].copy(), X[frames, a].copy()
    return X


# ---------------------------------------------------------------- 單部影片
def run_video(key, ks, mps, al, t5, device):
    pp, ang, chk = [], [], []
    seg_K = ks["skeleton_centered"].transpose(1, 2, 0).astype(np.float64)        # (L,17,3)，第 j 幀 = raw_start + j
    K_own = ks["skeleton"].transpose(1, 2, 0)
    raw0 = ks["raw_start"]
    for m in MODELS:
        r = al.loc[(key, m)]
        ms = mps[m]
        c = dict(key=key, model=m, exercise=ks["exercise"], ref_ok=bool(r.ref_ok), reliable=bool(r.reliable))
        if not r.ref_ok:
            chk.append(dict(c, status="無有效參考幀，跳過"))
            continue
        lo, hi = int(r.seg_video_first), int(r.seg_video_last) + 1
        rows = np.round(r.a + r.b * np.arange(len(ms["skeleton"][0]))).astype(int)[lo:hi]      # 段內每幀的 Kinect 行
        Mfull = ms["skeleton"].transpose(1, 2, 0).astype(np.float64)
        Mseg, det, vis = Mfull[lo:hi], ms["detected"][lo:hi].astype(bool), ms["visibility"][lo:hi]
        ref_v = np.array(list(map(int, str(r.ref_video_frames).split())))
        ref_k = np.array(list(map(int, str(r.ref_raw_rows).split())))
        Kn, pk = cs.normalize(seg_K, ref_idx=ref_k - raw0)
        Mn, pm = cs.normalize(Mseg, ref_idx=ref_v - lo)
        Mfill = cs.fill_undetected(Mn, det)
        # Kinect 與 MediaPipe 段內逐幀對應（追加 14 A）：MediaPipe 第 i 幀 ↔ Kinect 第 rows[i] − raw0 幀
        kj = rows - raw0
        assert kj.min() >= 0 and kj.max() < len(seg_K), (key, m, kj.min(), kj.max(), len(seg_K))
        Ka_c = seg_K[kj]                                                            # 未正規化、逐幀減 HipMid
        swap, swap_ok = lr_swap_frames(Ka_c, Mseg)
        Mswap = cs.fill_undetected(swap_lr(Mn, swap), det)
        c.update(status="ok", n_seg_video=hi - lo, n_kinect=len(seg_K), seg_video_first=lo, seg_video_last=hi - 1,
                 map_rows_first=int(rows[0]), map_rows_last=int(rows[-1]), raw_start=raw0, raw_end=int(ks["raw_end"]),
                 n_seg_truncated=int(r.n_seg_truncated), kinect_coverage=float(r.kinect_coverage),
                 ref_k_idx=" ".join(map(str, ref_k - raw0)), ref_m_idx=" ".join(map(str, ref_v - lo)),
                 scale_K=pk["scale"], scale_M=pm["scale"], mp_undetected_in_seg=float(1 - det.mean()),
                 ref_hipmid_K=float(np.abs(np.median(Kn[pk["ref_idx"], J["HipMid"]], 0)).max()),
                 ref_hipmid_M=float(np.abs(np.median(Mn[pm["ref_idx"], J["HipMid"]], 0)).max()),
                 swap_frac=float(swap[swap_ok].mean()) if swap_ok.any() else np.nan, swap_n=int(swap_ok.sum()))
        chk.append(c)

        # ---- Q1
        f = int(ks["fold"])
        for s in SEEDS:
            pp.append(dict(key=key, model=m, seed=s, fold=f, sample_id=ks["sample_id"], exercise=ks["exercise"],
                           subgroup=ks["subgroup"], ts=ks["ts"], reliable=bool(r.reliable), mirrored=bool(r.mirrored),
                           kinect_coverage=float(r.kinect_coverage), n_seg_truncated=int(r.n_seg_truncated),
                           K_t5=float(t5[(s, f)].loc[ks["_idx"], "y_pred"]),
                           K_own=predict(K_own, s, f, device), K_common=predict(Kn, s, f, device),
                           M=predict(Mfill, s, f, device), M_swap=predict(Mswap, s, f, device),
                           swap_frac=c["swap_frac"]))

        # ---- Q2／Q3
        Mmask = Mn.copy()
        Mmask[~det] = np.nan
        Mmask[vis < cs.VIS_THRESHOLD] = np.nan                                    # 追加 7 D
        Mmask_sw = swap_lr(Mmask, swap)                                           # 追加 11 E 敏感度分析
        for fname, (fn, joints, is_dir) in FEATURES[ks["exercise"]].items():
            ak, am, asw = fn(Kn), fn(Mmask), fn(Mmask_sw)
            vm = float(np.isfinite(am).mean())
            rk, pk95 = summary(ak)
            rm, pm95 = summary(am)
            akr = ak[kj]                                                            # 對到 MediaPipe 幀的 Kinect 角度
            both = np.isfinite(akr) & np.isfinite(am)
            d = _wrap(am - akr) if is_dir else am - akr
            dsw = _wrap(asw - akr) if is_dir else asw - akr
            bsw = np.isfinite(akr) & np.isfinite(asw)
            rsw, psw = summary(asw)
            ang.append(dict(key=key, model=m, exercise=ks["exercise"], subgroup=ks["subgroup"], feature=fname,
                            K_rom=rk, M_rom=rm, K_peak=pk95, M_peak=pm95, valid_M=vm, valid_ok=vm >= VALID_MIN,
                            frame_mae=float(np.abs(d[both]).mean()) if (r.reliable and both.sum()) else np.nan,
                            frame_n=int(both.sum()), M_swap_rom=rsw, M_swap_peak=psw,
                            frame_mae_swap=float(np.abs(dsw[bsw]).mean()) if (r.reliable and bsw.sum()) else np.nan,
                            reliable=bool(r.reliable), method=r.method, rho=float(r.rho),
                            fps=float(r.fps), mirrored=bool(r.mirrored)))
    return pp, ang, chk


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help="逗號分隔；main = 主分析集（qc_hflip.main_set_final）")
    ap.add_argument("--out", default="runs/v7_gap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    K = pickle.load(open(K_PKL, "rb"))["samples"]
    kidx = {s["key"]: i for i, s in enumerate(K)}
    MP = {m: {s["key"]: s for s in pickle.load(open(MP_PKL.format(m=m), "rb"))["samples"]} for m in MODELS}
    al = pd.read_csv(ALIGN_ALL).set_index(["key", "model"])
    if args.keys == "main":
        q = pd.read_csv(QC_HFLIP, index_col=0)
        keys = [k for k in q.index[q.main_set_final.astype(bool)]]
    else:
        keys = args.keys.split(",")
    t5 = {(s, f): pd.read_csv(T5_PRED.format(s=s, f=f)).set_index("index") for s in SEEDS for f in range(5)}
    os.makedirs(args.out, exist_ok=True)
    PP, ANG, CHK = [], [], []
    for n, key in enumerate(keys, 1):
        ks = dict(K[kidx[key]], _idx=kidx[key])
        pp, ang, chk = run_video(key, ks, {m: MP[m][key] for m in MODELS}, al, t5, args.device)
        PP += pp
        ANG += ang
        CHK += chk
        print(f"[{n}/{len(keys)}] {key}: " + "；".join(f"{c['model']} {c['status']}" for c in chk), flush=True)
    for name, rows in (("paired_preds", PP), ("angles", ANG), ("check", CHK)):
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
    print("→", args.out)


if __name__ == "__main__":
    main()
