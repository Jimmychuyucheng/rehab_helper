"""
================================================================================
檔案名稱: preprocess_kimore.py  (v2)
功能說明:
    掃描 KIMORE 原始資料夾，讀取 Kinect v2 25 關節 3D 座標與物理治療師評分，
    完成「整段序列共用參數」的視角幾何校準後，保留原始長度打包成單一 .pkl。
    隨機裁窗、資料增強、重採樣到固定 T 等「每個 epoch 都不一樣」的處理，
    全部移到 dataset.py 的 __getitem__ 中進行。

--------------------------------------------------------------------------------
【v2 相對 v1 的修正】
1. [Bug] 標籤欄位比對：v1 的 `"clinicalts" in col` 會讓 Es2~Es5 全部讀到
   第一個欄位 (Ex#1) 的分數。v2 改用正規表示式精準比對 `clinical TS/PO/CF Ex#n`。
2. [Bug] 旋轉方向：v1 的 R_y 用 -θ，會把鏡頭偏角「加倍」而不是消除
   (校準後髖部連線的 Z 分量不為 0)。v2 改用 +θ。
3. [Bug] 幾何校準：v1 每一幀各自平移、旋轉、縮放，會把 Es4 骨盆畫圈 (PO 就是
   SpineBase 軌跡) 與 Es3 軀幹旋轉的動作本身抹掉。v2 只用「開頭 REF_FRAMES 幀」
   算一次平移與旋轉、用整段中位數算一次尺度，再套用到整段序列。
4. 不再內插到 T=100，保留原始長度 (3, T_orig, 25)，由 dataset.py 決定如何取樣。
5. 額外讀取 PO (Primary Outcome) 與 CF (Control Factors) 分數，供多任務學習。
6. Kinect 追蹤中斷時會輸出全 0 的幀，v2 會把它視為缺值再內插補齊。
7. 標籤讀不到的樣本直接排除 (v1 會留下 clinical_score=None)。
8. subject_key = "{subgroup}/{subject}"，保證跨組別唯一，供受試者切分使用。
9. 結尾印出統計與「同一受試者五個動作分數完全相同」的警告，方便抓標籤錯誤。

【v4 標籤修正】fix_scores()
    PO 依 EAAQ 計分規則為三題各 1–5 分，最小 3。原始資料有 3 筆 PO < 3，
    經 KIMORE 作者 (L. Romeo, 2026-11-13) 確認：補成 3 (不刪除)，TS 重算為 PO + CF。
    每筆樣本帶 sample["label_fix"] = {"po_orig", "ts_orig"} 或 None；meta["label_fixes"] 彙整。
    其他不一致 (例如 Expert/E_ID4 Es5 的 TS 與 PO+CF 差 0.67) 一律不動。
    v2 時輸出 kimore_full_pofix.pkl (--no_fix: kimore_full.pkl)。

【v3 資料正確性修正】(逐檔檢查 400 個 JointPosition 檔、所有 ClinicalAssessment 檔)
1. [Bug] 標籤檔：v2 取排序後第一個 ClinicalAssessment*.xlsx，E_ID3 的 Es3/Es4/Es5 資料夾混有
   ClinicalAssessment_E_ID1.xlsx，因此讀到 E_ID1 的分數。v3 只用檔名屬於本人的檔，多個時內容須相同。
2. [Bug] 多個骨架檔：v2 取排序後第一個 (S_ID5/Es1 讀到 68 幀的中斷錄影；P_ID3/Es1 第一個檔壞掉整筆被跳過)。
   v3：去除內容相同的 "(1)" 複本 → 排除不合格的檔 → 取錄影時間最晚的合格檔；長度相近的候選標記為無法確定。
3. [Bug] 多人幀：一行 200 欄 = 同一幀追蹤到兩個人。v2 直接讀檔失敗，整筆跳過 (E_ID8/Es1、P_ID5/Es1、S_ID6/Es5)。
   v3 依位置選受試者那一具。
4. [Bug] 中段空行 (追蹤中斷)：v2 經 pandas 直接略過，時間軸被壓縮。v3 只保留最長連續段 (丟掉的都是開頭 3~10 幀的零星偵測)。
5. 每筆樣本記錄 source_file、skeleton_info、ambiguous_source、notes；meta 記錄跳過原因與備註。

--------------------------------------------------------------------------------
【座標系說明】(Kinect v2 camera space，單位公尺)
    X: 受試者左右方向   Y: 垂直向上   Z: 鏡頭到受試者的深度
    校準後：原點 = 開頭幾幀的 SpineBase 平均位置；左右髖連線對齊 X 軸；
            長度單位 = 「SpineBase → SpineShoulder」軀幹長 (約 0.5 m)。

【輸出】data/kimore/processed/kimore_full_v3_1.pkl (--no_fix: kimore_full_v3_1_nofix.pkl)；v3 檔不覆蓋
    {
      "meta":    {...版本、關節名稱、左右對稱表、校準參數...},
      "samples": [ {
          "group", "subgroup", "subject", "subject_key",
          "exercise" (如 "Es1"), "exercise_id" (0~4),
          "skeleton": float32 (3, T_orig, 25),
          "num_frames": T_orig,
          "ts", "po", "cf": float  (clinical Total / Primary Outcome / Control Factors)
          "label_fix": None 或 {"po_orig", "ts_orig"}
      }, ... ]
    }
================================================================================
"""

import os
import re
import glob
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# ==================== 路徑與參數設定 ====================
RAW_DATA_DIR = "./data/kimore/raw"
PROCESSED_DIR = "./data/kimore/processed"
OUTPUT_NAME = "kimore_full_v3_1.pkl"       # v3.1 = v3 + VERIFIED_START_TRIMS；fix_scores()
OUTPUT_NAME_NOFIX = "kimore_full_v3_1_nofix.pkl"   # --no_fix：v3.1 規則、原始標籤
# kimore_full_v3.pkl (v3，無開頭裁切) 不覆蓋，v6 run 的 index 對應以它為準
# 舊檔 kimore_full_pofix.pkl / kimore_full.pkl (v2 規則) 不覆蓋，舊 run 的 index 對應仍以它們為準
PO_MIN = 3.0                               # EAAQ：PO 三題各 1–5 分
LABEL_FIX_NOTE = "PO<3 set to 3 per KIMORE author (L. Romeo, 2026-11-13); TS recomputed as PO+CF"

MIN_FRAMES = 15          # 少於此幀數視為無效錄影
REF_FRAMES = 10          # 用開頭幾幀計算平移與旋轉 (受試者此時為站立預備姿勢)
MAX_ZERO_RATIO = 0.3     # 追蹤失敗 (全 0) 幀超過此比例就排除

# ==================== Kinect v2 關節定義 ====================
JOINT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head",                       # 0-3
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",          # 4-7
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",      # 8-11
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",                # 12-15
    "HipRight", "KneeRight", "AnkleRight", "FootRight",            # 16-19
    "SpineShoulder",                                               # 20
    "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",      # 21-24
]
FLIP_PAIRS = [(4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17),
              (14, 18), (15, 19), (21, 23), (22, 24)]

SPINE_BASE, SPINE_MID, SPINE_SHOULDER = 0, 1, 20
HIP_LEFT, HIP_RIGHT = 12, 16

# CSV 每個關節 4 欄 [X, Y, Z, TrackingState]，取前 3 欄
XYZ_INDICES = [j * 4 + k for j in range(25) for k in range(3)]


# ==================== 1. 讀取骨架 ====================
# JointPosition CSV 格式 (逐行檢查 400 個檔得到)：
#   - 每行一幀；每具骨架 25 關節 × [X, Y, Z, TrackingState] = 100 欄，行尾多一個逗號。
#   - 空行 = 該幀沒有追蹤到人 (錄影開頭常見)。pandas 會直接略過空行，若空行在序列中間會把時間軸壓縮。
#   - 一行 200 / 300 欄 = 同一幀追蹤到 2 / 3 個人 (TimeStamp 同一行也有相同的 2 個時間戳)，
#     例如治療師走進畫面。受試者 = SpineBase 水平位置與單人幀一致的那一具；骨架順序不固定。
#   - 單人幀也不一定是受試者：P_ID5/Es1 有約 11 秒受試者沒被追蹤到、只追蹤到走向鏡頭的路人；
#     S_ID2/Es5 結尾也有一段換成別人。換人的特徵是 SpineBase 瞬間跳動，而不是離中心多遠
#     (受試者開頭走到定位、坐姿追蹤抖動都會偏離中心但連續)。
#     因此依幀間連續性把骨架串成軌跡，幀數最多的軌跡 = 受試者，其他軌跡的幀視為缺幀。
TRACK_JUMP = 0.50        # 相鄰 (有骨架的) 幀之間 SpineBase 水平 (x, z) 移動超過此值 (公尺) 視為不同人
MAX_DROPPED = 30         # 中段斷掉時只保留最長連續段；丟掉的幀數超過此值就排除該錄影
# 人工逐幀確認過「最長連續段已包含完整動作、丟掉的部分沒有動作」的錄影，允許超過 MAX_DROPPED。
# 新出現的同類情況仍會被排除，需人工確認後再加進來。
VERIFIED_TRIMS = {
    "Parkinson/P_ID5/Es1": "行 1018–1353 只追蹤到走向鏡頭的路人；保留段 (行 18–1017) 有完整 5 次舉手，"
                           "受試者回到畫面後 (行 1354–1543) 雙手一直舉著、沒有新的重複",
    "Stroke/S_ID2/Es5": "行 549–566 為路人；之後 86 幀受試者站立 (無下蹲) 並往左走出畫面，動作已在保留段 (行 64–548) 結束",
}


def _rows_to_bodies(line):
    """一行 → list[(25, 3) ndarray]；空行回傳 []。欄位數不是 100 的倍數 (去掉行尾空欄) 時丟出 ValueError。"""
    cells = line.strip().split(",")
    while cells and cells[-1] == "":
        cells.pop()
    if not cells:
        return []
    if len(cells) % 100:
        raise ValueError(f"欄位數 {len(cells)} 不是 100 的倍數")
    out = []
    for k in range(len(cells) // 100):
        v = np.asarray(cells[k * 100:(k + 1) * 100], dtype=np.float64).reshape(25, 4)
        out.append(v[:, :3])
    return out


# 開頭受試者還在走向定位的錄影：從定位後的靜止預備姿勢開始。
# 原因：body_centric_alignment 用開頭 REF_FRAMES 幀定原點，走位中的開頭會讓整段動作偏離原點
# (這兩筆偏移 1.7 / 1.55 軀幹長，全部樣本中位數 0.04、第 99 百分位 0.53)。
# key = 動作；值 = (選用檔名, 起始幀, 理由)。起始幀以 load_joint_positions 輸出 (已去掉開頭空行) 的索引計，
# 理由中的「行」為 CSV 行號 (1 起算)。選用檔名不符時直接報錯，避免選檔規則改變後裁錯檔。
VERIFIED_START_TRIMS = {
    "Parkinson/P_ID3/Es1": ("JointPosition060616_120220.csv", 390,
                            "行 20–399 受試者從離鏡頭 1.7 m 處往後、往左走到 2.55 m 定位 (水平移動約 0.9 m)；"
                            "行 400–455 定位後雙手下垂靜止，行 457 起開始舉手 (共 7 次)。從行 410 (幀 390) 開始，"
                            "保留約 1.5 秒預備姿勢 (全部 Es1 第一次舉手的中位數為第 45 幀)"),
    "Stroke/S_ID1/Es1": ("JointPosition060616_122123.csv", 530,
                         "行 31–560 受試者分多步從 2.54 m 退到 3.3 m 定位；行 561 起位置穩定 (距最終位置 < 0.05 m)，"
                         "雙手下垂直到約行 630 才開始動作。從行 561 (幀 530) 開始"),
}


def load_joint_positions(csv_path: str, max_dropped: int = MAX_DROPPED, return_index: bool = False):
    """
    讀取 JointPosition.csv，回傳 ((T, 25, 3) float32, info)；無法使用時丟出 ValueError。
    return_index=True 時多回傳 (start, end)：保留段在原始 CSV 的幀號 (= 行號，0 起算，含頭不含尾，
    end - start == T)；預設 False，回傳值與 v3 完全相同。
    1. 逐行讀，空行記為缺幀 (保留時間位置)，多人幀先全部保留。
    2. 軌跡追蹤：每具骨架接到「最後位置水平距離最近且 < TRACK_JUMP」的軌跡 (同一幀不重複)，否則開新軌跡；
       軌跡不會過期 (受試者中斷後回到原位會接回)。幀數最多的軌跡 = 受試者，其他幀視為缺幀。
    3. 去掉頭尾缺幀；中段若有缺幀，只保留最長的連續段 (不跨大段空白內插、也不壓縮時間)，
       丟掉的幀數 > MAX_DROPPED 時排除。
    4. 全 0 的幀 (追蹤失敗) 沿用舊作法：比例 > MAX_ZERO_RATIO 排除，否則線性內插。
    """
    with open(csv_path, newline="") as f:
        lines = f.read().replace("\r\n", "\n").replace("\r", "\n").split("\n")
    frames = []
    for i, line in enumerate(lines):
        try:
            frames.append(_rows_to_bodies(line))
        except ValueError as e:
            raise ValueError(f"第 {i + 1} 行：{e}")
    if not any(frames):
        raise ValueError("沒有任何骨架幀")
    # 軌跡追蹤 → 幀數最多的軌跡為受試者
    n_multi = sum(len(b) > 1 for b in frames)
    tracks = []                       # 每條軌跡: {"last": (x, z), "frames": {幀: 骨架}}
    for i, b in enumerate(frames):
        pos = [x[SPINE_BASE][[0, 2]] for x in b]
        pairs = sorted((float(np.linalg.norm(p - t["last"])), j, k)
                       for j, p in enumerate(pos) for k, t in enumerate(tracks))
        used_b, used_t = set(), set()
        for d, j, k in pairs:
            if d < TRACK_JUMP and j not in used_b and k not in used_t:
                tracks[k]["frames"][i] = b[j]
                tracks[k]["last"] = pos[j]
                used_b.add(j)
                used_t.add(k)
        for j in range(len(b)):
            if j not in used_b:
                tracks.append({"last": pos[j], "frames": {i: b[j]}})
    main = max(tracks, key=lambda t: len(t["frames"]))
    n_other_tracks = len(tracks) - 1
    picked = [main["frames"].get(i) for i in range(len(frames))]
    n_off = sum(1 for i, b in enumerate(frames) if b and picked[i] is None)   # 有骨架但不是受試者的幀
    have = [i for i, b in enumerate(picked) if b is not None]
    if not have:
        raise ValueError("沒有任何幀是受試者")
    n_off_inside = sum(1 for i in range(have[0], have[-1] + 1)
                       if picked[i] is None and frames[i])      # 頭尾以外的非受試者幀
    picked = picked[have[0]:have[-1] + 1]

    # 中段缺幀 → 切段，保留最長連續段
    segs, cur = [], []
    for i, b in enumerate(picked):
        if b is not None:
            cur.append(i)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    keep = max(segs, key=len)
    n_gap = len(picked) - sum(len(sg) for sg in segs)
    dropped = sum(len(sg) for sg in segs) - len(keep)
    if dropped > max_dropped:
        raise ValueError(f"中段追蹤中斷 {n_gap} 幀 (其中非受試者 {n_off_inside} 幀)，"
                         f"最長連續段 {len(keep)} 幀之外還有 {dropped} 幀 (> {max_dropped})")
    picked = [picked[i] for i in keep]
    vals = np.stack(picked).reshape(len(picked), -1)

    # Kinect 追蹤失敗時整幀為 0 → 視為缺值
    zero_rows = np.all(np.abs(vals) < 1e-8, axis=1)
    if zero_rows.mean() > MAX_ZERO_RATIO:
        raise ValueError(f"追蹤失敗幀比例 {zero_rows.mean():.0%} 過高")
    vals[zero_rows] = np.nan
    vals = (pd.DataFrame(vals)
            .interpolate(method="linear", limit_direction="both", axis=0)
            .bfill().ffill().values)
    if np.isnan(vals).any():
        raise ValueError("補值後仍含 NaN")
    T = vals.shape[0]
    if T < MIN_FRAMES:
        raise ValueError(f"幀數 {T} < {MIN_FRAMES}")
    info = {"frames": T, "gap_frames": n_gap, "dropped_frames": dropped, "multi_body_frames": n_multi,
            "off_subject_frames": n_off, "other_tracks": n_other_tracks, "zero_frames": int(zero_rows.sum())}
    if return_index:
        start = have[0] + keep[0]          # keep 為 picked (從 have[0] 起算) 中的連續索引
        return vals.reshape(T, 25, 3).astype(np.float32), info, (start, start + T)
    return vals.reshape(T, 25, 3).astype(np.float32), info


_STAMP = re.compile(r"(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def _stamp_key(path):
    """檔名中的 DDMMYY_HHMMSS → 可排序的 (YY, MM, DD, hh, mm, ss)；沒有時間戳回傳 None。"""
    m = _STAMP.search(os.path.basename(path))
    if not m:
        return None
    d, mo, y, h, mi, se = (int(g) for g in m.groups())
    return (y, mo, d, h, mi, se)


def select_joint_position(csv_paths, max_dropped: int = MAX_DROPPED):
    """
    一個動作資料夾內有多個 JointPosition 檔時的選檔規則：
      1. 內容逐位元相同的檔 (例如 "xxx(1).csv" 複本) 只留一份；
      2. 每個檔各自讀取，讀不了 / 不合格的排除；
      3. 合格的檔中取**錄影時間最晚**的一個 (前面的視為中斷或重錄)。
    回傳 (skeleton, info, choice)；choice 記錄所有候選與理由。全部不合格時丟出 ValueError。
    """
    import filecmp
    uniq = []
    for p in sorted(csv_paths):
        if not any(filecmp.cmp(p, q, shallow=False) for q in uniq):
            uniq.append(p)
    dup = sorted(set(csv_paths) - set(uniq))
    cands = []
    for p in uniq:
        try:
            skel, info = load_joint_positions(p, max_dropped)
            cands.append({"file": os.path.basename(p), "ok": True, "frames": info["frames"],
                          "_skel": skel, "_info": info, "_key": _stamp_key(p)})
        except Exception as e:
            cands.append({"file": os.path.basename(p), "ok": False, "error": str(e), "_key": _stamp_key(p)})
    good = [c for c in cands if c["ok"]]
    if not good:
        raise ValueError("; ".join(f"{c['file']}: {c['error']}" for c in cands))
    if len(good) > 1 and any(c["_key"] is None for c in good):
        raise ValueError(f"多個合格骨架檔但無法從檔名判斷錄影時間：{[c['file'] for c in good]}")
    best = max(good, key=lambda c: c["_key"] or ())
    # 有兩個以上合格檔、且較早的檔長度不到最晚檔的一半以下 → 很可能只是中斷；否則標記為「無法確定」
    ambiguous = [c["file"] for c in good if c is not best and c["frames"] >= 0.5 * best["frames"]]
    choice = {"chosen": best["file"], "duplicates_removed": [os.path.basename(p) for p in dup],
              "candidates": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands],
              "ambiguous_with": ambiguous}
    return best["_skel"], best["_info"], choice


# ==================== 2. 整段序列共用參數的幾何校準 ====================
def body_centric_alignment(skeleton: np.ndarray):
    """
    輸入 (T, 25, 3)，輸出 (T, 25, 3) 與校準參數。
    平移/旋轉只由開頭 REF_FRAMES 幀決定，尺度由整段中位數決定，
    因此骨盆位移、軀幹旋轉等「動作本身」會被完整保留。
    """
    ref = skeleton[:REF_FRAMES]

    # (1) 平移：開頭幾幀 SpineBase 的平均位置
    origin = ref[:, SPINE_BASE].mean(axis=0)
    out = skeleton - origin

    # (2) 旋轉：開頭幾幀左右髖連線在 X-Z 平面的平均方向，繞 Y 軸轉到與 X 軸平行
    hip_vec = (ref[:, HIP_RIGHT] - ref[:, HIP_LEFT]).mean(axis=0)
    theta = 0.0
    if np.linalg.norm(hip_vec[[0, 2]]) > 1e-4:
        theta = float(np.arctan2(hip_vec[2], hip_vec[0]))
        # R_y(φ) 會把 X-Z 平面上角度 θ 的向量轉到 θ-φ，所以 φ=θ 才會轉回 X 軸。
        # (v1 用 -θ，結果是把偏角加倍而不是消除)
        c, s = np.cos(theta), np.sin(theta)
        R_y = np.array([[c, 0.0, s],
                        [0.0, 1.0, 0.0],
                        [-s, 0.0, c]], dtype=np.float32)
        out = out @ R_y.T

    # (3) 尺度：整段 SpineBase→SpineShoulder 長度的中位數 (比 SpineMid 段長、較不受雜訊影響)
    torso = np.linalg.norm(skeleton[:, SPINE_SHOULDER] - skeleton[:, SPINE_BASE], axis=1)
    scale = float(np.median(torso))
    if scale < 1e-3:
        scale = 0.5  # 成人 SpineBase→SpineShoulder 約 0.45~0.55 m
    out = out / scale

    params = {"origin": origin.astype(np.float32), "yaw_rad": theta, "scale_m": scale}
    return out.astype(np.float32), params


# ==================== 3. 讀取臨床評分 ====================
_SCORE_COL = re.compile(r"(ts|po|cf)ex#?(\d)")


def _to_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        val = val.strip().replace(",", ".")
        if val == "":
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def label_files_for(label_dir: str, subject: str):
    """
    資料夾內屬於 subject 的 ClinicalAssessment 檔 (檔名 ClinicalAssessment_<subject>.xlsx 或 ..._<subject>(n).xlsx)。
    其他受試者的檔 (例如 E_ID3/Es3 裡混進的 ClinicalAssessment_E_ID1.xlsx) 一律不用。
    回傳 (屬於本人的檔, 其他人的檔)。
    """
    own, other = [], []
    for p in sorted(glob.glob(os.path.join(label_dir, "ClinicalAssessment*.xlsx"))):
        m = re.fullmatch(r"ClinicalAssessment_(.+?)(\(\d+\))?\.xlsx", os.path.basename(p))
        (own if m and m.group(1) == subject else other).append(p)
    return own, other


def parse_clinical_scores(label_dir: str, ex_num: int, subject: str, notes=None):
    """
    從屬於 subject 的 ClinicalAssessment_*.xlsx 讀取第 ex_num 個動作的 TS / PO / CF。
    欄位名稱如 "clinical TS Ex#1"，清理後為 "clinicaltsex#1"。
    本人有多個檔 (例如 "(1)" 複本) 時，內容必須相同，否則丟出 ValueError。
    表內 "Subject ID" 欄與受試者不符時只記錄在 notes (已確認 NE_ID2、E_ID16/Es2 只是該欄填錯，分數是本人的)。
    回傳 dict(ts=..., po=..., cf=...)；讀不到 TS 時回傳 None。
    """
    notes = notes if notes is not None else []
    own, other = label_files_for(label_dir, subject)
    if other:
        notes.append(f"忽略其他受試者的標籤檔 {[os.path.basename(p) for p in other]}")
    if not own:
        return None
    dfs = [pd.read_excel(p) for p in own]
    if any(not d.equals(dfs[0]) for d in dfs[1:]):
        raise ValueError(f"本人有多個內容不同的標籤檔 {[os.path.basename(p) for p in own]}")
    if len(own) > 1:
        notes.append(f"本人有 {len(own)} 個內容相同的標籤檔")
    df = dfs[0]
    if df.empty:
        return None
    if "Subject ID" in df.columns and str(df["Subject ID"].iloc[0]).strip() != subject:
        notes.append(f"標籤檔內 Subject ID 寫成 {str(df['Subject ID'].iloc[0]).strip()} (檔名為本人，使用其分數)")

    scores = {}
    for col in df.columns:
        clean = str(col).lower().replace(" ", "").replace("_", "")
        m = _SCORE_COL.search(clean)
        if m and int(m.group(2)) == ex_num:
            scores[m.group(1)] = _to_float(df[col].iloc[0])

    if scores.get("ts") is None:
        return None
    return {"ts": scores["ts"], "po": scores.get("po"), "cf": scores.get("cf")}


def fix_scores(sample):
    """
    PO < 3 不符合 EAAQ 計分規則 (最小 3)：設為 3 並重算 TS = PO + CF。
    原值記在 sample["label_fix"]；沒修正的為 None。其他情況一律不動。原地修改並回傳 sample。
    """
    sample["label_fix"] = None
    po, cf = sample.get("po"), sample.get("cf")
    if po is not None and po < PO_MIN:
        if cf is None:
            raise ValueError(f"{sample['subject_key']}/{sample['exercise']}: PO<3 但缺 CF，無法重算 TS")
        sample["label_fix"] = {"po_orig": po, "ts_orig": sample["ts"]}
        sample["po"] = PO_MIN
        sample["ts"] = PO_MIN + cf
    return sample


# ==================== 4. 主流程 ====================
def run_preprocessing(raw_dir: str = RAW_DATA_DIR, out_dir: str = PROCESSED_DIR, apply_fix: bool = True):
    print(f"🚀 開始掃描 KIMORE 資料夾: {raw_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # 路徑規格: raw/{Group}/{SubGroup}/{Subject}/{Es1~Es5}
    exercise_dirs = sorted(glob.glob(os.path.join(raw_dir, "*", "*", "*", "[Ee][Ss][1-5]")))
    print(f"🔍 總共找到 {len(exercise_dirs)} 個動作資料夾。")

    samples, skipped, issues = [], defaultdict(list), []

    for ex_dir in exercise_dirs:
        parts = os.path.normpath(ex_dir).split(os.sep)
        exercise, subject, subgroup, group = parts[-1], parts[-2], parts[-3], parts[-4]
        exercise = "Es" + exercise[-1]
        ex_num = int(exercise[-1])
        tag = f"{subgroup}/{subject}/{exercise}"

        notes = []
        try:
            labels = parse_clinical_scores(os.path.join(ex_dir, "Label"), ex_num, subject, notes)
        except Exception as e:
            labels = None
            notes.append(f"讀取評分失敗: {e}")
        csv_candidates = sorted(glob.glob(os.path.join(ex_dir, "Raw", "*[Jj]oint[Pp]osition*.csv")))
        if not csv_candidates:
            raw_files = sorted(os.listdir(os.path.join(ex_dir, "Raw"))) if os.path.isdir(os.path.join(ex_dir, "Raw")) else []
            skipped["缺 JointPosition.csv"].append(
                f"{tag} (Raw 內: {', '.join(raw_files) or '空'}；{'有' if labels else '無'}評分)")
            continue
        try:
            verified = VERIFIED_TRIMS.get(tag)
            skel, skel_info, choice = select_joint_position(csv_candidates, 10 ** 9 if verified else MAX_DROPPED)
            if verified:
                notes.append(f"人工確認後保留最長連續段：{verified}")
            start_trim = VERIFIED_START_TRIMS.get(tag)
            if start_trim:
                fname, start, why = start_trim
                if choice["chosen"] != fname:
                    raise ValueError(f"VERIFIED_START_TRIMS 預期 {fname}，實際選到 {choice['chosen']}")
                if start >= skel.shape[0] - MIN_FRAMES:
                    raise ValueError(f"VERIFIED_START_TRIMS 起始幀 {start} 超過長度 {skel.shape[0]}")
                skel = skel[start:]
                skel_info = {**skel_info, "start_trim": start, "frames": int(skel.shape[0])}
                notes.append(f"人工確認後去掉開頭走位 {start} 幀：{why}")
        except Exception as e:
            skipped["骨架異常"].append(f"{tag} ({e})")
            continue
        if len(csv_candidates) > 1:
            cs = ", ".join(f"{c['file']}({c['frames']} 幀)" if c["ok"] else f"{c['file']}(不合格: {c['error'][:40]})"
                           for c in choice["candidates"])
            notes.append(f"多個骨架檔 → 選 {choice['chosen']}；候選 {cs}" +
                         (f"；已去除相同複本 {choice['duplicates_removed']}" if choice["duplicates_removed"] else "") +
                         (f"；**無法確定**：{choice['ambiguous_with']} 長度相近" if choice["ambiguous_with"] else ""))
        if skel_info["dropped_frames"] or skel_info["gap_frames"]:
            notes.append(f"中段追蹤中斷 {skel_info['gap_frames']} 幀，保留最長連續段，丟掉 {skel_info['dropped_frames']} 幀")
        if skel_info["multi_body_frames"]:
            notes.append(f"{skel_info['multi_body_frames']} 幀同時追蹤到多人，依水平位置選受試者")
        if skel_info["off_subject_frames"]:
            notes.append(f"{skel_info['off_subject_frames']} 幀追蹤到的不是受試者 (SpineBase 瞬間跳動 > {TRACK_JUMP} m 的其他軌跡)，視為缺幀")
        if labels is None:
            skipped["缺評分"].append(tag + (f" ({'; '.join(notes)})" if notes else ""))
            continue

        aligned, params = body_centric_alignment(skel)
        if not np.isfinite(aligned).all():
            skipped["校準後非有限值"].append(tag)
            continue

        sample = {
            "group": group,
            "subgroup": subgroup,
            "subject": subject,
            "subject_key": f"{subgroup}/{subject}",
            "exercise": exercise,
            "exercise_id": ex_num - 1,
            "skeleton": np.ascontiguousarray(aligned.transpose(2, 0, 1)),  # (3, T, 25)
            "num_frames": int(aligned.shape[0]),
            "ts": labels["ts"],
            "po": labels["po"],
            "cf": labels["cf"],
            "align_params": params,
            "source_file": choice["chosen"],
            "skeleton_info": skel_info,
            "ambiguous_source": choice["ambiguous_with"],
            "notes": notes,
        }
        if apply_fix:
            sample = fix_scores(sample)
        samples.append(sample)
        if notes:
            issues.append((tag, notes))

    report(samples, skipped)
    print(f"\n📝 有備註的樣本 {len(issues)} 筆：")
    for tag, ns in issues:
        print(f"   {tag}")
        for n in ns:
            print(f"      - {n}")

    payload = {
        "meta": {
            "version": 3.1,
            "layout": "(C=3, T_orig, V=25)",
            "joint_names": JOINT_NAMES,
            "flip_pairs": FLIP_PAIRS,
            "unit": "SpineBase→SpineShoulder torso length",
            "ref_frames": REF_FRAMES,
            "selection_rule": "JointPosition: dedupe identical copies, drop unusable, take latest timestamp; "
                              "labels: only ClinicalAssessment_<subject>*.xlsx; "
                              "v3.1: VERIFIED_START_TRIMS drops pre-exercise walk-in at recording start",
            "skipped": {k: list(v) for k, v in skipped.items()},
            "notes": [{"tag": t, "notes": n} for t, n in issues],
        },
        "samples": samples,
    }
    if apply_fix:
        fixes = [{"subject_key": s["subject_key"], "exercise": s["exercise"],
                  "po_orig": s["label_fix"]["po_orig"], "po_new": s["po"],
                  "ts_orig": s["label_fix"]["ts_orig"], "ts_new": s["ts"]}
                 for s in samples if s["label_fix"] is not None]
        payload["meta"]["label_fixes"] = {"note": LABEL_FIX_NOTE, "fixes": fixes}
        print(f"\n🩹 標籤修正 {len(fixes)} 筆 ({LABEL_FIX_NOTE}):")
        for fx in fixes:
            print(f"   {fx['subject_key']} {fx['exercise']}: PO {fx['po_orig']:.2f} → {fx['po_new']:g}, "
                  f"TS {fx['ts_orig']:.2f} → {fx['ts_new']:.2f}")
    out_path = os.path.join(out_dir, OUTPUT_NAME if apply_fix else OUTPUT_NAME_NOFIX)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"💾 已輸出: {out_path}")
    return payload


def report(samples, skipped):
    print("\n" + "=" * 60)
    print(f"✅ 成功 {len(samples)} 筆，⚠️ 跳過 {sum(len(v) for v in skipped.values())} 筆")
    for reason, items in skipped.items():
        print(f"   - {reason}: {len(items)} 筆")
        for it in items[:5]:
            print(f"       {it}")
        if len(items) > 5:
            print(f"       ...")
    if not samples:
        return

    subjects = {s["subject_key"] for s in samples}
    print(f"\n👤 受試者數: {len(subjects)}")
    by_sub = defaultdict(int)
    for s in samples:
        by_sub[s["subgroup"]] += 1
    print("   各子組樣本數: " + ", ".join(f"{k}={v}" for k, v in sorted(by_sub.items())))

    print("\n📊 各動作統計 (幀數 / TS):")
    for ex in sorted({s["exercise"] for s in samples}):
        sub = [s for s in samples if s["exercise"] == ex]
        T = np.array([s["num_frames"] for s in sub])
        ts = np.array([s["ts"] for s in sub])
        print(f"   {ex}: n={len(sub):3d} | 幀數 中位數 {int(np.median(T)):5d} "
              f"(min {T.min()}, max {T.max()}) ≈ {np.median(T) / 30:.1f}s"
              f" | TS {ts.min():.1f}~{ts.max():.1f}, 平均 {ts.mean():.1f} ± {ts.std():.1f}")

    # 重複錄影檢查
    keys = defaultdict(int)
    for s in samples:
        keys[(s["subject_key"], s["exercise"])] += 1
    dup = [k for k, v in keys.items() if v > 1]
    if dup:
        print(f"\n⚠️ 有 {len(dup)} 組 (受試者, 動作) 重複出現，例如 {dup[:3]}")

    # 標籤錯誤檢查：同一受試者五個動作的 TS 完全相同 → 很可能讀錯欄位
    ts_by_subj = defaultdict(list)
    for s in samples:
        ts_by_subj[s["subject_key"]].append(s["ts"])
    same = [k for k, v in ts_by_subj.items() if len(v) >= 3 and len(set(v)) == 1]
    if same:
        print(f"\n🚨 {len(same)} 位受試者的各動作 TS 完全相同 (可能讀錯欄位): {same[:5]}")
    else:
        print("\n✅ 標籤檢查: 沒有受試者的各動作 TS 完全相同")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_fix", action="store_true",
                    help="不套用 PO<3 修正，輸出原始標籤的 kimore_full_v3_1_nofix.pkl")
    run_preprocessing(apply_fix=not ap.parse_args().no_fix)