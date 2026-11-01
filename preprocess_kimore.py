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

--------------------------------------------------------------------------------
【座標系說明】(Kinect v2 camera space，單位公尺)
    X: 受試者左右方向   Y: 垂直向上   Z: 鏡頭到受試者的深度
    校準後：原點 = 開頭幾幀的 SpineBase 平均位置；左右髖連線對齊 X 軸；
            長度單位 = 「SpineBase → SpineShoulder」軀幹長 (約 0.5 m)。

【輸出】data/kimore/processed/kimore_full.pkl
    {
      "meta":    {...版本、關節名稱、左右對稱表、校準參數...},
      "samples": [ {
          "group", "subgroup", "subject", "subject_key",
          "exercise" (如 "Es1"), "exercise_id" (0~4),
          "skeleton": float32 (3, T_orig, 25),
          "num_frames": T_orig,
          "ts", "po", "cf": float  (clinical Total / Primary Outcome / Control Factors)
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
OUTPUT_NAME = "kimore_full.pkl"

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
def load_joint_positions(csv_path: str) -> np.ndarray:
    """讀取 JointPosition.csv，回傳 (T, 25, 3) float32；失敗時丟出 ValueError。"""
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 100:
        raise ValueError(f"欄位數僅 {df.shape[1]} < 100")

    xyz = df.iloc[:, XYZ_INDICES].apply(pd.to_numeric, errors="coerce")
    vals = xyz.values.astype(np.float64)

    # Kinect 追蹤失敗時整幀為 0 → 視為缺值
    zero_rows = np.all(np.abs(vals) < 1e-8, axis=1)
    if zero_rows.mean() > MAX_ZERO_RATIO:
        raise ValueError(f"追蹤失敗幀比例 {zero_rows.mean():.0%} 過高")
    vals[zero_rows] = np.nan

    # 沿時間軸線性內插補缺值，頭尾用最近值補
    vals = (pd.DataFrame(vals)
            .interpolate(method="linear", limit_direction="both", axis=0)
            .bfill().ffill().values)

    if np.isnan(vals).any():
        raise ValueError("補值後仍含 NaN")
    T = vals.shape[0]
    if T < MIN_FRAMES:
        raise ValueError(f"幀數 {T} < {MIN_FRAMES}")
    return vals.reshape(T, 25, 3).astype(np.float32)


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


def parse_clinical_scores(label_dir: str, ex_num: int):
    """
    從 ClinicalAssessment_*.xlsx 讀取第 ex_num 個動作的 TS / PO / CF。
    欄位名稱如 "clinical TS Ex#1"，清理後為 "clinicaltsex#1"。
    回傳 dict(ts=..., po=..., cf=...)；讀不到 TS 時回傳 None。
    """
    xlsx_files = sorted(glob.glob(os.path.join(label_dir, "ClinicalAssessment*.xlsx")))
    if not xlsx_files:
        return None
    if len(xlsx_files) > 1:
        print(f"   ⚠️ {label_dir} 有多個 ClinicalAssessment 檔，使用 {os.path.basename(xlsx_files[0])}")

    df = pd.read_excel(xlsx_files[0])
    if df.empty:
        return None

    scores = {}
    for col in df.columns:
        clean = str(col).lower().replace(" ", "").replace("_", "")
        m = _SCORE_COL.search(clean)
        if m and int(m.group(2)) == ex_num:
            scores[m.group(1)] = _to_float(df[col].iloc[0])

    if scores.get("ts") is None:
        return None
    return {"ts": scores["ts"], "po": scores.get("po"), "cf": scores.get("cf")}


# ==================== 4. 主流程 ====================
def run_preprocessing(raw_dir: str = RAW_DATA_DIR, out_dir: str = PROCESSED_DIR):
    print(f"🚀 開始掃描 KIMORE 資料夾: {raw_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # 路徑規格: raw/{Group}/{SubGroup}/{Subject}/{Es1~Es5}
    exercise_dirs = sorted(glob.glob(os.path.join(raw_dir, "*", "*", "*", "[Ee][Ss][1-5]")))
    print(f"🔍 總共找到 {len(exercise_dirs)} 個動作資料夾。")

    samples, skipped = [], defaultdict(list)

    for ex_dir in exercise_dirs:
        parts = os.path.normpath(ex_dir).split(os.sep)
        exercise, subject, subgroup, group = parts[-1], parts[-2], parts[-3], parts[-4]
        exercise = "Es" + exercise[-1]
        ex_num = int(exercise[-1])
        tag = f"{subgroup}/{subject}/{exercise}"

        csv_candidates = sorted(glob.glob(os.path.join(ex_dir, "Raw", "*[Jj]oint[Pp]osition*.csv")))
        if not csv_candidates:
            skipped["缺 JointPosition.csv"].append(tag)
            continue
        if len(csv_candidates) > 1:
            print(f"   ⚠️ {tag} 有 {len(csv_candidates)} 個 JointPosition 檔，使用第一個")

        try:
            skel = load_joint_positions(csv_candidates[0])
        except Exception as e:
            skipped[f"骨架異常"].append(f"{tag} ({e})")
            continue

        try:
            labels = parse_clinical_scores(os.path.join(ex_dir, "Label"), ex_num)
        except Exception as e:
            labels = None
            print(f"   ❌ {tag} 讀取評分失敗: {e}")
        if labels is None:
            skipped["缺評分"].append(tag)
            continue

        aligned, params = body_centric_alignment(skel)
        if not np.isfinite(aligned).all():
            skipped["校準後非有限值"].append(tag)
            continue

        samples.append({
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
        })

    report(samples, skipped)

    payload = {
        "meta": {
            "version": 2,
            "layout": "(C=3, T_orig, V=25)",
            "joint_names": JOINT_NAMES,
            "flip_pairs": FLIP_PAIRS,
            "unit": "SpineBase→SpineShoulder torso length",
            "ref_frames": REF_FRAMES,
        },
        "samples": samples,
    }
    out_path = os.path.join(out_dir, OUTPUT_NAME)
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
    run_preprocessing()