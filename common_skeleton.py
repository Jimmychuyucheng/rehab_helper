"""
v7 第 3.2 節：Kinect v2（25 點）與 MediaPipe Pose（33 點）轉成同一套 17 點「共同骨架」。

座標系（以 T1 真實資料決定，見 tests/test_common_skeleton.py）：
  共同骨架沿用 Kinect 相機座標的軸向：y 向上；左肩 x < 右肩 x；z 越小越靠近鏡頭。
  MediaPipe world 轉換為 (x, y, z) → (-x, -y, z)：y 原本向下；x 原本左肩 > 右肩；z 原本就是越小越近。
  這是繞 z 軸轉 180°，座標系左右手性不變。
  追加 7 C：MediaPipe world 每幀以髖部中點為原點、沒有骨盆整體位移，所以兩邊都逐幀減去 HipMid
  （center=True，預設），輸入資訊一致。依賴骨盆位移的特徵兩邊不可比。

Kinect 幀的座標一律是原始 CSV 行號（追加 2 C）：load_kinect_raw() 回傳 (n_raw, 25, 3)，
v3 保留段 [start, end) 以外為 NaN；影片第 i 幀 ↔ 原始 CSV 第 i + k 行。

    from common_skeleton import COMMON_JOINTS, kinect_to_common, mediapipe_to_common, fill_short_gaps, normalize
"""
import importlib.util
import os

import numpy as np

COMMON_JOINTS = ["HipMid", "ShoulderMid", "Head",
                 "ShoulderL", "ShoulderR", "ElbowL", "ElbowR", "WristL", "WristR",
                 "HipL", "HipR", "KneeL", "KneeR", "AnkleL", "AnkleR", "FootL", "FootR"]
J = {n: i for i, n in enumerate(COMMON_JOINTS)}
NUM_COMMON = len(COMMON_JOINTS)

# 每個共同骨架點的來源：int = 單一點；tuple = 兩點的中點（兩邊用同一種算法）
KINECT_SRC = [(12, 16), (4, 8), 3,
              4, 8, 5, 9, 6, 10,
              12, 16, 13, 17, 14, 18, 15, 19]
MP_SRC = [(23, 24), (11, 12), (7, 8),          # Head = 兩耳中點
          11, 12, 13, 14, 15, 16,
          23, 24, 25, 26, 27, 28, 31, 32]      # Foot = foot index
MP_TO_KINECT_AXES = np.array([-1.0, -1.0, 1.0], dtype=np.float32)

VIS_THRESHOLD = 0.5
MAX_SHORT_GAP = 5
REF_FRAMES = 10

# 骨架圖（給 CTR-GCN）：(子, 父)，中心點 HipMid
COMMON_INWARD = [(J["ShoulderMid"], J["HipMid"]), (J["Head"], J["ShoulderMid"]),
                 (J["ShoulderL"], J["ShoulderMid"]), (J["ShoulderR"], J["ShoulderMid"]),
                 (J["ElbowL"], J["ShoulderL"]), (J["ElbowR"], J["ShoulderR"]),
                 (J["WristL"], J["ElbowL"]), (J["WristR"], J["ElbowR"]),
                 (J["HipL"], J["HipMid"]), (J["HipR"], J["HipMid"]),
                 (J["KneeL"], J["HipL"]), (J["KneeR"], J["HipR"]),
                 (J["AnkleL"], J["KneeL"]), (J["AnkleR"], J["KneeR"]),
                 (J["FootL"], J["AnkleL"]), (J["FootR"], J["AnkleR"])]
COMMON_FLIP_PAIRS = [(J[a + "L"], J[a + "R"]) for a in ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle", "Foot"]]


def _gather(x, src):
    out = []
    for s in src:
        out.append((x[:, s[0]] + x[:, s[1]]) / 2 if isinstance(s, tuple) else x[:, s])
    return np.stack(out, axis=1)


def center_on_hipmid(x17):
    """追加 7 C：逐幀減去 HipMid（HipMid 本身變成 0）。"""
    x17 = np.asarray(x17, dtype=np.float32)
    return (x17 - x17[:, J["HipMid"]:J["HipMid"] + 1]).astype(np.float32)


def kinect_to_common(x25, center=True):
    """(T, 25, 3) Kinect 相機座標 → (T, 17, 3)；center=True 時逐幀減去 HipMid。NaN 原樣傳遞。"""
    x25 = np.asarray(x25, dtype=np.float32)
    assert x25.ndim == 3 and x25.shape[1:] == (25, 3), x25.shape
    x = _gather(x25, KINECT_SRC).astype(np.float32)
    return center_on_hipmid(x) if center else x


def mediapipe_to_common(x33, visibility, vis_threshold=VIS_THRESHOLD, center=True):
    """
    (T, 33, 3) pose_world_landmarks、(T, 33) visibility → ((T, 17, 3), (T, 17) visibility)。
    軸向轉成 Kinect 的方向；中點的 visibility 取兩端點的最小值；center=True 時逐幀減去 HipMid
    （在遮罩前做，用 MediaPipe 輸出的髖座標）。
    vis_threshold：visibility 低於此值的點設為 NaN（角度特徵用，追加 7 D）；None = 不遮罩
    （模型輸入用：偵測到的幀一律使用輸出座標）。偵測失敗的幀 world 本來就是 NaN、visibility 為 0。
    """
    x33 = np.asarray(x33, dtype=np.float32) * MP_TO_KINECT_AXES
    visibility = np.asarray(visibility, dtype=np.float32)
    assert x33.ndim == 3 and x33.shape[1:] == (33, 3), x33.shape
    assert visibility.shape == x33.shape[:2], visibility.shape
    x = _gather(x33, MP_SRC)
    if center:
        x = center_on_hipmid(x)
    vis = np.stack([np.minimum(visibility[:, s[0]], visibility[:, s[1]]) if isinstance(s, tuple)
                    else visibility[:, s] for s in MP_SRC], axis=1)
    if vis_threshold is not None:
        x[vis < vis_threshold] = np.nan
    return x.astype(np.float32), vis.astype(np.float32)


def fill_undetected(x, detected):
    """
    追加 7 D（模型輸入）：沒偵測到的幀線性內插補齊，頭尾用最近的有效幀延伸。
    x: (T, V, 3)，detected: (T,) bool。偵測到的幀原樣保留。
    """
    x = np.array(x, dtype=np.float32, copy=True)
    detected = np.asarray(detected, dtype=bool)
    if not detected.any():
        raise ValueError("沒有任何偵測到的幀")
    t, good = np.arange(len(x)), np.flatnonzero(detected)
    for v in range(x.shape[1]):
        for c in range(3):
            x[~detected, v, c] = np.interp(t[~detected], good, x[good, v, c])   # np.interp 在頭尾取端點值
    return x


def fill_short_gaps(x, max_gap=MAX_SHORT_GAP):
    """
    逐關節沿時間軸：兩端都有值、長度 ≤ max_gap 的缺漏線性內插；更長的缺漏與頭尾缺漏保留 NaN。
    （一個關節的 x、y、z 同時缺漏，以 x 判斷。）回傳 (填補後陣列, info)。
    """
    x = np.array(x, dtype=np.float32, copy=True)
    T, V, _ = x.shape
    t = np.arange(T)
    info = {"filled_points": 0, "long_gap_points": 0, "edge_gap_points": 0, "longest_gap": 0,
            "long_gaps": []}       # long_gaps: (joint, start, length)
    for v in range(V):
        miss = ~np.isfinite(x[:, v, 0])
        if not miss.any():
            continue
        if miss.all():
            info["edge_gap_points"] += T
            continue
        d = np.diff(np.r_[0, miss.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            n = e - s
            if s == 0 or e == T:
                info["edge_gap_points"] += n
                continue
            info["longest_gap"] = max(info["longest_gap"], int(n))
            if n <= max_gap:
                for c in range(3):
                    x[s:e, v, c] = np.interp(t[s:e], [s - 1, e], [x[s - 1, v, c], x[e, v, c]])
                info["filled_points"] += int(n)
            else:
                info["long_gap_points"] += int(n)
                info["long_gaps"].append((COMMON_JOINTS[v], int(s), int(n)))
    return x, info


def normalize(x17, ref_idx=None, ref_frames=REF_FRAMES):
    """
    第 3.2 節正規化（依追加 7 B 修改），Kinect 與 MediaPipe 完全相同：
      原點 = 參考幀 HipMid 的中位數；
      方向 = 參考幀 HipL→HipR 向量的中位數定左右軸 (x)、HipMid→ShoulderMid 向量的中位數定垂直軸 (y)，
             Gram-Schmidt 正交化，z = x × y；
      尺度 = 整段 HipMid→ShoulderMid 長度的中位數。
    ref_idx：參考幀（陣列 index）。追加 7 B 的配對用參考幀 = Kinect 保留段開頭起算、兩邊都有效的前 10 幀，
             由 T6 依對應關係算好後傳入；給定時必須全部有效、且恰為 ref_frames 幀，否則丟出 ValueError
             （「無有效參考幀」由呼叫端記入 excluded.csv）。
             None：取 HipMid、ShoulderMid、HipL、HipR 都有值的前 ref_frames 幀（單邊使用，例如 T5 的 Kinect 訓練資料）。
    回傳 (T, 17, 3) 與參數 dict（origin、R (3×3, 列為新軸)、scale、ref_idx）。
    """
    x17 = np.asarray(x17, dtype=np.float64)
    need = [J["HipMid"], J["ShoulderMid"], J["HipL"], J["HipR"]]
    ok = np.all(np.isfinite(x17[:, need, 0]), axis=1)
    if ref_idx is None:
        ref_idx = np.flatnonzero(ok)[:ref_frames]
        if len(ref_idx) == 0:
            raise ValueError("沒有任何幀同時有 HipMid、ShoulderMid、HipL、HipR")
    else:
        ref_idx = np.asarray(ref_idx, dtype=int)
        if len(ref_idx) != ref_frames or not ok[ref_idx].all():
            raise ValueError(f"參考幀須為 {ref_frames} 個有效幀，收到 {len(ref_idx)} 個、有效 {int(ok[ref_idx].sum())} 個")
    ref = x17[ref_idx]
    origin = np.median(ref[:, J["HipMid"]], axis=0)
    lateral = np.median(ref[:, J["HipR"]] - ref[:, J["HipL"]], axis=0)
    vertical = np.median(ref[:, J["ShoulderMid"]] - ref[:, J["HipMid"]], axis=0)
    ey = vertical / np.linalg.norm(vertical)
    ex = lateral - lateral.dot(ey) * ey
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    R = np.stack([ex, ey, ez])
    torso = np.linalg.norm(x17[:, J["ShoulderMid"]] - x17[:, J["HipMid"]], axis=-1)
    scale = float(np.nanmedian(torso))
    out = (x17 - origin) @ R.T / scale
    return out.astype(np.float32), {"origin": origin.astype(np.float32), "R": R.astype(np.float32),
                                    "scale": scale, "ref_idx": [int(i) for i in ref_idx]}


def common_adjacency():
    """(3, 17, 17)：自身、向心、離心；建法與正規化同 model/graph/kimore_25_graph.py（列正規化 D^-1 A）。"""
    A = np.zeros((3, NUM_COMMON, NUM_COMMON))
    for i in range(NUM_COMMON):
        A[0, i, i] = 1
    for i, j in COMMON_INWARD:
        A[1, i, j] = 1
        A[2, j, i] = 1
    for k in range(3):
        deg = A[k].sum(axis=1, keepdims=True)
        deg[deg == 0] = 1
        A[k] = A[k] / deg
    return A


_crop = None


def _crop_module():
    global _crop
    if _crop is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "build_v3_crop_index.py")
        spec = importlib.util.spec_from_file_location("build_v3_crop_index", p)
        _crop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_crop)
    return _crop


def load_kinect_raw(csv_path, start, end):
    """
    原始 JointPosition CSV → (n_raw, 25, 3) 相機座標，index = 原始 CSV 行號（0 起算）。
    只填 v3 保留段 [start, end)（data/kimore/v3_crop_index.csv），取法與 v3 相同
    （scripts/build_v3_crop_index.slice_skeleton：多人幀依連續性取受試者、全 0 幀內插），其餘行為 NaN。
    """
    m = _crop_module()
    lines = m.read_lines(csv_path)
    out = np.full((len(lines), 25, 3), np.nan, dtype=np.float32)
    out[start:end] = m.slice_skeleton(lines, start, end)
    return out
