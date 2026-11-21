"""
v7 T6：Kinect ↔ MediaPipe 時間對齊、整部鏡像、逐幀左右標反比例、共同參考幀
（第 3.4 節；追加 3、4 B、7 B、7 E、8 B、10）。

對應關係（追加 10 B）：影片第 i 幀 ↔ Kinect 原始 CSV 第 r(i) 行，是錄影本身的屬性：
  heavy 與 full 各求一次，取峰值 ρ 較高者，兩個模型共用同一組對應。
  - 25 fps（追加 3 B）：r(i) = i + k，k ∈ [0, 行數 − 影片幀數]，整數。
    漂移檢查（追加 10 D）：重疊段前半、後半各找一次 k；兩半段 ρ 都 ≥ 0.7 且 k 相差 > 2 幀 → 改用線性對應；
    否則不判定漂移，沿用整部的 k。
  - 24 fps（追加 7 E）：r(i) = round(a + b·i)，每部各自搜尋 a、b。
  - 峰值 ρ < 0.7 → 「對齊不可靠」（追加 10 C：不做 DTW；只從逐幀指標排除）。
對齊訊號（追加 10 A）：畫面上的絕對位置。MediaPipe 影像座標 (x, y)，對 Kinect **未減髖部**的相機座標 (−X, −Y)
  （軸向依資料確認：Kinect 左肩 X < 右肩 X、MediaPipe 影像左肩 x > 右肩 x；Kinect Y 向上、影像 y 向下）。
  關節 = 肩、肘、腕、髖、膝、踝 12 點；每個關節的 x、y 各自 z-score（MediaPipe 用偵測到的幀，Kinect 用保留段），
  串接成 24 維訊號，ρ = 兩邊都有值的幀上 24 維合併的 Pearson 相關。影像座標只用來求對應，不進入任何量測。

鏡像與左右標反：兩邊都用共同骨架、逐幀減 HipMid（追加 7 C）、未正規化的相機軸向（公尺）。
  （正規化的左右軸取自 HipL→HipR；MediaPipe 左右標反時，正規化會把 x 一起翻過來，左右就看不出來。）
  逐幀左右標反（追加 4 B）：每幀比較「同側配對」與「交叉配對」的 x、y 距離和，交叉較小 → 標反。
    左右腕、左右踝分開算；分母 = MediaPipe 偵測到且 Kinect 有值的幀。另列「明確幀」版本（Kinect 左右點距離 ≥ 0.10 m）。
  整部鏡像（追加 10 E）：腕部標反 > 50%。判定為鏡像的影片須看 RGB 判斷原因（本程式不翻轉）。
共同參考幀（追加 7 B）：Kinect 保留段開頭起算、依對應關係排列，取前 10 個「兩邊都有效」的幀：
  Kinect 有值、MediaPipe 偵測到且通過追加 6 B 的骨架合理規則（plausible()）。不足 10 幀 → 「無有效參考幀」。

    python align.py --keys E_ID14_Es2,B_ID1_Es1,NE_ID6_Es3,P_ID15_Es4,S_ID9_Es5 --plot
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

import common_skeleton as cs
from common_skeleton import J

K_PKL = "data/kimore/processed/kimore_k_common_347.pkl"
MP_PKL = "data/kimore/processed/kimore_mp_common_{m}_v1.pkl"
MP_RAW = "data/kimore/mediapipe/raw/{m}/{key}.npz"
FRAME_CHECK = "data/kimore/frame_check.csv"
MANIFEST = "data/kimore/rgb_manifest.csv"
CROP = "data/kimore/v3_crop_index.csv"
OUT_DIR = "data/kimore/mediapipe/qc/align_check"
MODELS = ("heavy", "full")

RHO_MIN = 0.7
DRIFT_MAX = 2
MIN_PAIRS = 50
CLEAR_SEP_M = 0.10
MIRROR_FRAC = 0.5
B_GRID = np.round(np.arange(0.95, 1.10 + 1e-9, 0.0005), 4)
IMG_W, IMG_H = 960, 540
# 對齊訊號的 12 個關節：共同骨架名稱 → MediaPipe 33 點 index
SIGNAL_JOINTS = {"ShoulderL": 11, "ShoulderR": 12, "ElbowL": 13, "ElbowR": 14, "WristL": 15, "WristR": 16,
                 "HipL": 23, "HipR": 24, "KneeL": 25, "KneeR": 26, "AnkleL": 27, "AnkleR": 28}
KINECT_TO_IMAGE = np.array([-1.0, -1.0])     # Kinect (X, Y) → 與影像 (x, y) 同向


# ---------------------------------------------------------------- 資料
_cache = {}


def load_video(key, model):
    """回傳 dict：Kc (n_raw,17,3) 原始行號、保留段外 NaN、已減 HipMid；Ku 同 Kc 但未減 HipMid；
    Mc (T,17,3) 已減 HipMid；image (T,33,3)；detected；資訊。"""
    if "k" not in _cache:
        _cache["k"] = {s["key"]: s for s in pickle.load(open(K_PKL, "rb"))["samples"]}
        _cache["fc"] = pd.read_csv(FRAME_CHECK).set_index("sample_id")
        _cache["man"] = pd.read_csv(MANIFEST).set_index("sample_id")
        _cache["crop"] = pd.read_csv(CROP).set_index("sample_id")
    if model not in _cache:
        _cache[model] = {s["key"]: s for s in pickle.load(open(MP_PKL.format(m=model), "rb"))["samples"]}
    ks, ms = _cache["k"][key], _cache[model][key]
    sid = ks["sample_id"]
    if ("Ku", key) not in _cache:
        mr, cr = _cache["man"].loc[sid], _cache["crop"].loc[sid]
        K25 = cs.load_kinect_raw(os.path.join("data/kimore/raw", mr.group, mr.subgroup, mr.subject, mr.exercise,
                                              "Raw", cr.source_file), int(cr.start), int(cr.end))
        _cache[("Ku", key)] = cs.kinect_to_common(K25, center=False)
    Ku = _cache[("Ku", key)]
    Kc = np.full_like(Ku, np.nan)
    Kc[ks["raw_start"]:ks["raw_end"]] = ks["skeleton_centered"].transpose(1, 2, 0)
    assert np.allclose(cs.center_on_hipmid(Ku)[ks["raw_start"]:ks["raw_end"]], Kc[ks["raw_start"]:ks["raw_end"]],
                       atol=1e-5), key
    f = _cache["fc"].loc[sid]
    z = np.load(MP_RAW.format(m=model, key=key))
    Mc = ms["skeleton"].transpose(1, 2, 0)
    assert len(Mc) == len(z["image"]) == int(f.video_frames), key
    return dict(key=key, model=model, Kc=Kc, Ku=Ku, Mc=Mc, image=z["image"], detected=ms["detected"].astype(bool),
                start=ks["raw_start"], end=ks["raw_end"], n_raw=ks["n_raw"], fps=float(f.fps),
                diff=int(f.n_raw - f.video_frames), subgroup=ks["subgroup"], exercise=ks["exercise"],
                main_set=bool(ks["main_set"]), backlit=bool(ks["backlit"]))


def plausible(img33):
    """追加 6 B 骨架合理規則（與 qc/diag_t2/diag.py 的 plaus() 相同）：肩中點在髖中點上方、
    軀幹高 > 40 px、肩寬 > 15 px、肩髖都在畫面內（容許 2%）。img33：單幀 (33,3) 正規化影像座標。"""
    if not np.all(np.isfinite(img33[[11, 12, 23, 24], :2])):
        return False
    s, h = (img33[11] + img33[12]) / 2, (img33[23] + img33[24]) / 2
    inside = np.all((img33[[11, 12, 23, 24], :2] > -0.02) & (img33[[11, 12, 23, 24], :2] < 1.02))
    return bool(inside and (h[1] - s[1]) * IMG_H > 40 and abs(img33[11, 0] - img33[12, 0]) * IMG_W > 15)


def _z(x):
    """逐欄 z-score（忽略 NaN）。"""
    return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)


def kinect_signal(d):
    """(n_raw, 24)：Kinect 未減髖部的 (−X, −Y)，逐欄 z-score；保留段外 NaN。"""
    Ku = d["Ku"]
    cols = [Ku[:, J[n], c] * KINECT_TO_IMAGE[c] for n in SIGNAL_JOINTS for c in (0, 1)]
    return _z(np.stack(cols, axis=1).astype(np.float64))


def mediapipe_signal(d):
    """(T, 24)：MediaPipe 影像座標 (x, y)，逐欄 z-score；未偵測幀 NaN。"""
    img = d["image"].astype(np.float64).copy()
    img[~d["detected"]] = np.nan
    cols = [img[:, i, c] for i in SIGNAL_JOINTS.values() for c in (0, 1)]
    return _z(np.stack(cols, axis=1))


# ---------------------------------------------------------------- 對應搜尋
def _rows(n, a, b):
    return np.round(a + b * np.arange(n)).astype(int)


def corr_at(ks, ms, rows, sel=None):
    """多維 Pearson：ks[rows[i]] 與 ms[i] 在兩邊都有值的 i 上，所有維度合併計算（可用 sel 限縮）。回傳 (ρ, 幀數)。"""
    v = (rows >= 0) & (rows < len(ks))
    if sel is not None:
        v &= sel
    i = np.flatnonzero(v)
    a, b = ks[rows[i]], ms[i]
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if ok.sum() < MIN_PAIRS:
        return -np.inf, int(ok.sum())
    return float(np.corrcoef(a[ok].ravel(), b[ok].ravel())[0, 1]), int(ok.sum())


def search_offset(ks, ms, k_max, sel=None):
    best = (-np.inf, 0, 0)
    for k in range(0, k_max + 1):
        c, n = corr_at(ks, ms, _rows(len(ms), k, 1.0), sel)
        if c > best[0]:
            best = (c, k, n)
    return best                 # (ρ, k, 幀數)


def search_linear(ks, ms, a_range):
    best = (-np.inf, 0, 1.0, 0)
    for b in B_GRID:
        for a in a_range:
            c, n = corr_at(ks, ms, _rows(len(ms), a, b))
            if c > best[0]:
                best = (c, a, float(b), n)
    return best                 # (ρ, a, b, 幀數)


def align(d, ks):
    """單一模型的對應。回傳 dict（含 rows：每個影片幀對應的 Kinect 原始行號）。"""
    ms = mediapipe_signal(d)
    n = len(ms)
    r = {}
    if d["fps"] == 25:
        rho, k, npair = search_offset(ks, ms, d["diff"])
        rows = _rows(n, k, 1.0)
        both = np.flatnonzero(np.isfinite(ms).all(1) & (rows < len(ks))
                              & np.isfinite(ks[np.clip(rows, 0, len(ks) - 1)]).all(1))
        mid = both[len(both) // 2]
        t = np.arange(n)
        r1 = search_offset(ks, ms, d["diff"], sel=t < mid)
        r2 = search_offset(ks, ms, d["diff"], sel=t >= mid)
        r.update(k_half1=r1[1], rho_half1=r1[0], k_half2=r2[1], rho_half2=r2[0],
                 drift=bool(abs(r1[1] - r2[1]) > DRIFT_MAX and min(r1[0], r2[0]) >= RHO_MIN))
        if r["drift"]:
            rho, a, b, npair = search_linear(ks, ms, range(-DRIFT_MAX * 5, d["diff"] + 1))
            r.update(method="linear(drift)", k=np.nan, a=a, b=b)
        else:
            r.update(method="offset", k=k, a=k, b=1.0)
    else:
        rho, a, b, npair = search_linear(ks, ms, range(0, d["diff"] + 1))
        r.update(method="linear(24fps)", k=np.nan, a=a, b=b)
    r.update(rho=rho, n_pairs=npair, rows=_rows(n, r["a"], r["b"]))
    return r


# ---------------------------------------------------------------- 對齊後的檢查
def aligned_kinect(d, rows):
    """與 Mc 同長的 Kinect（第 i 幀 = 原始第 rows[i] 行；超出範圍或保留段外為 NaN）。"""
    out = np.full_like(d["Mc"], np.nan)
    v = (rows >= 0) & (rows < len(d["Kc"]))
    out[v] = d["Kc"][rows[v]]
    return out


def lr_flip(Ka, Mc, joint):
    """逐幀左右標反比例（x、y 距離）。回傳 (比例, 分母, 明確幀比例, 明確幀分母, 每幀標反 bool)。"""
    kl, kr = Ka[:, J[joint + "L"], :2], Ka[:, J[joint + "R"], :2]
    ml, mr = Mc[:, J[joint + "L"], :2], Mc[:, J[joint + "R"], :2]
    ok = np.isfinite(kl[:, 0]) & np.isfinite(kr[:, 0]) & np.isfinite(ml[:, 0]) & np.isfinite(mr[:, 0])
    dist = lambda p, q: np.linalg.norm(p - q, axis=1)
    flip = ok & (dist(ml, kr) + dist(mr, kl) < dist(ml, kl) + dist(mr, kr))
    clear = ok & (dist(kl, kr) >= CLEAR_SEP_M)
    return (flip[ok].mean() if ok.any() else np.nan, int(ok.sum()),
            flip[clear].mean() if clear.any() else np.nan, int(clear.sum()), flip)


def common_ref_frames(d, rows, n_ref=cs.REF_FRAMES):
    """追加 7 B：回傳 (影片幀 index, Kinect 原始行號)，各 ≤ n_ref 個，依 Kinect 行號由小到大。"""
    need = [J["HipMid"], J["ShoulderMid"], J["HipL"], J["HipR"]]
    vid, raw = [], []
    for i in np.argsort(rows, kind="stable"):
        r = rows[i]
        if r < d["start"] or r >= d["end"]:
            continue
        if not d["detected"][i] or not np.isfinite(d["Mc"][i, need, 0]).all():
            continue
        if not np.isfinite(d["Kc"][r, need, 0]).all() or not plausible(d["image"][i]):
            continue
        vid.append(int(i))
        raw.append(int(r))
        if len(vid) == n_ref:
            break
    return np.array(vid, dtype=int), np.array(raw, dtype=int)


def checks(d, rows):
    """共用對應下，單一模型的鏡像、左右標反、參考幀。"""
    Ka = aligned_kinect(d, rows)
    r, flips = {}, {}
    for jn in ("Wrist", "Ankle"):
        f, n, fc_, nc, fl = lr_flip(Ka, d["Mc"], jn)
        r.update({f"flip_{jn.lower()}": f, f"flip_{jn.lower()}_n": n,
                  f"flip_{jn.lower()}_clear": fc_, f"flip_{jn.lower()}_clear_n": nc})
        flips[jn] = fl
    r["mirrored"] = bool(r["flip_wrist"] > MIRROR_FRAC)
    vid, raw = common_ref_frames(d, rows)
    r.update(n_ref=len(vid), ref_ok=len(vid) == cs.REF_FRAMES,
             ref_video_frames=" ".join(map(str, vid)), ref_raw_rows=" ".join(map(str, raw)),
             ref_first_raw_offset=int(raw[0] - d["start"]) if len(raw) else np.nan,
             mp_fail=float(1 - d["detected"].mean()))
    return r, dict(Ka=Ka, flips=flips, ref_vid=vid)


def run_one(key):
    """回傳 (每個模型一列的結果 list, 疊圖用資料 dict)。"""
    ds = {m: load_video(key, m) for m in MODELS}
    d0 = ds[MODELS[0]]
    ks = kinect_signal(d0)
    al = {m: align(ds[m], ks) for m in MODELS}
    src = max(MODELS, key=lambda m: al[m]["rho"])           # 追加 10 B：取 ρ 較高者
    shared = al[src]
    rows = shared["rows"]
    base = dict(key=key, exercise=d0["exercise"], subgroup=d0["subgroup"], main_set=d0["main_set"],
                backlit=d0["backlit"], fps=d0["fps"], diff=d0["diff"], n_video=len(rows), n_raw=d0["n_raw"],
                seg_start=d0["start"], seg_end=d0["end"],
                map_from=src, method=shared["method"], k=shared["k"], a=shared["a"], b=shared["b"],
                rho=shared["rho"], reliable=bool(shared["rho"] >= RHO_MIN),
                **{f"{f}_{m}": al[m][f] for m in MODELS for f in ("rho", "method", "k", "a", "b")},
                **{f"{f}_{src}": shared.get(f, np.nan) for f in ("k_half1", "rho_half1", "k_half2", "rho_half2",
                                                                 "drift")})
    res, extra = [], {}
    for m in MODELS:
        c, x = checks(ds[m], rows)
        c["rho_own_signal"] = corr_at(ks, mediapipe_signal(ds[m]), rows)[0]   # 此模型在共用對應下的 ρ
        res.append(dict(base, model=m, **c))
        extra[m] = dict(x, d=ds[m], ks=ks, rows=rows)
    return res, extra


# ---------------------------------------------------------------- 疊圖
def plot(r, x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, Ka, Mc, rows = x["d"], x["Ka"], x["d"]["Mc"], x["rows"]
    t = np.arange(len(Mc)) / 30.0
    ksig = np.full((len(Mc), x["ks"].shape[1]), np.nan)
    v = (rows >= 0) & (rows < len(x["ks"]))
    ksig[v] = x["ks"][rows[v]]
    msig = mediapipe_signal(d)
    fig, ax = plt.subplots(6, 1, figsize=(14, 15), sharex=True)
    for a, c, lab in ((ax[0], 1, "image y (12 joints, mean of z)\n+ = down"),
                      (ax[1], 0, "image x (12 joints, mean of z)\n+ = image right")):
        a.plot(t, np.nanmean(ksig[:, c::2], axis=1), "k-", lw=1, label="Kinect (−X, −Y)")
        a.plot(t, np.nanmean(msig[:, c::2], axis=1), "r-", lw=1, alpha=0.8, label=f"MediaPipe {r['model']} image")
        a.set_ylabel(lab)
    panels = [("WristL", 1, "WristL  y (up)"), ("WristR", 1, "WristR  y (up)"),
              ("WristL", 0, "WristL  x (subject-left < 0)"), ("WristR", 0, "WristR  x")]
    fl = x["flips"]["Wrist"]
    for a, (jn, c, lab) in zip(ax[2:], panels):
        a.plot(t, Ka[:, J[jn], c], "k-", lw=1, label="Kinect")
        a.plot(t, Mc[:, J[jn], c], "r-", lw=1, alpha=0.8, label="MediaPipe")
        if fl.any():
            a.plot(t[fl], Mc[fl, J[jn], c], "b.", ms=3, label="MP wrist L/R flipped frame")
        a.set_ylabel(lab + "\nrel. HipMid [m]")
    undet = ~d["detected"]
    for a in ax:
        if len(x["ref_vid"]):
            a.axvspan(t[x["ref_vid"][0]], t[x["ref_vid"][-1]], color="g", alpha=0.15, lw=0)
        if undet.any():
            yl = a.get_ylim()
            a.plot(t[undet], np.full(undet.sum(), yl[0]), "|", color="orange", ms=8)
        a.grid(alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[2].legend(loc="upper right", fontsize=8)
    if r["method"] == "offset":
        src = r["map_from"]
        m = (f"offset k={r['k']} (halves k={r[f'k_half1_{src}']}/{r[f'k_half2_{src}']}, "
             f"ρ={r[f'rho_half1_{src}']:.2f}/{r[f'rho_half2_{src}']:.2f})")
    else:
        m = f"{r['method']} row = {r['a']} + {r['b']:.4f}·i"
    if not r["reliable"]:
        m += "  [ρ<0.7: UNRELIABLE → excluded from per-frame metrics]"
    fig.suptitle(f"{r['key']} [{r['subgroup']}, {r['fps']:.0f} fps]  plotted: MediaPipe {r['model']} "
                 f"(fail {r['mp_fail']:.1%})\nshared mapping from {r['map_from']}: {m}  ρ={r['rho']:.3f} "
                 f"(heavy {r['rho_heavy']:.3f} / full {r['rho_full']:.3f})\n"
                 f"wrist L/R flip {r['flip_wrist']:.1%} → {'MIRRORED (>50%)' if r['mirrored'] else 'not mirrored'};  "
                 f"ankle flip {r['flip_ankle']:.1%};  ref frames {r['n_ref']}/10 (green);  orange ticks = MP undetected",
                 fontsize=10)
    ax[-1].set_xlabel("time from video frame 0 [s] (Kinect plotted at mapped CSV row)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help="逗號分隔，例如 E_ID14_Es2,B_ID1_Es1；all = 347 部")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "align.csv"))
    args = ap.parse_args()
    if args.keys == "all":
        keys = [s["key"] for s in pickle.load(open(K_PKL, "rb"))["samples"]]
    else:
        keys = args.keys.split(",")
    os.makedirs(OUT_DIR, exist_ok=True)
    res = []
    for key in keys:
        rr, extra = run_one(key)
        res += rr
        for r in rr:
            if args.plot:
                plot(r, extra[r["model"]], os.path.join(OUT_DIR, f"{key}_{r['model']}.png"))
        r = rr[0]
        print(f"{key}: ρ heavy {r['rho_heavy']:.3f} ({r['method_heavy']} a={r['a_heavy']} b={r['b_heavy']:.4f}) | "
              f"full {r['rho_full']:.3f} ({r['method_full']} a={r['a_full']} b={r['b_full']:.4f}) → 用 {r['map_from']}"
              f"{'' if r['reliable'] else '（不可靠）'}；"
              + "；".join(f"{x['model']} 腕標反 {x['flip_wrist']:.1%} 踝 {x['flip_ankle']:.1%}"
                         f"{' 鏡像' if x['mirrored'] else ''} 參考幀 {x['n_ref']}" for x in rr))
    pd.DataFrame(res).to_csv(args.out, index=False)
    print("→", args.out)


if __name__ == "__main__":
    main()
