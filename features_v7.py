"""
v7 追加 17 A：Q2／Q3 的特徵清單（以 KIMORE 官方 MATLAB 腳本 feat_extract_Ex1～Ex5.m 為準）。

座標：共同骨架 17 點，已用共同參考幀正規化（追加 8 B）：x = 參考幀 HipL→HipR（約等於 Kinect +X）、
y = 向上（約 Kinect +Y）、z = x × y（約 Kinect +Z，遠離鏡頭）。官方公式的第 1、2、3 分量直接對應 x、y、z。
長度單位 = 軀幹長（HipMid→ShoulderMid 整段中位數），面積單位 = 軀幹長²，角度單位 = 度。
官方用到 SpineShoulder／SpineBase 的特徵，依第 3.2 節改用 ShoulderMid／HipMid（「以共同骨架近似」）。
摘要值（追加 17 A）：ROM = P95 − P5、峰值 = P95；不使用官方的找峰值（findpeaks）、低通濾波（filtering.m）與頭尾裁切。

    python features_v7.py            # 產生 runs/v7_gap/feature_list.md
"""
import os

import numpy as np

from common_skeleton import J


def P(X, n):
    return X[:, J[n]]


def dist(X, a, b):
    return np.linalg.norm(P(X, a) - P(X, b), axis=-1)


def ang3(X, a, b, c):
    """∠abc（度），三維；等同官方以餘弦定理（三段長）計算的角度。"""
    u, v = P(X, a) - P(X, b), P(X, c) - P(X, b)
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def atan2d(y, x):
    return np.degrees(np.arctan2(y, x))


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def heron_trunk_area(X):
    """官方 Ex1／Ex5 的軀幹面積（照抄三角形組合，含官方原式的邊長選擇）。"""
    hip, sh = dist(X, "HipL", "HipR"), dist(X, "ShoulderL", "ShoulderR")
    shR_hipR, shL_hipL, shR_hipL = dist(X, "ShoulderR", "HipR"), dist(X, "ShoulderL", "HipL"), dist(X, "ShoulderR", "HipL")
    sR = (hip + shR_hipL + shR_hipR) / 2
    aR = np.sqrt(np.clip(sR * (sR - hip) * (sR - shR_hipL) * (sR - shR_hipR), 0, None))
    sL = (sh + shR_hipL + shL_hipL) / 2
    aL = np.sqrt(np.clip(sL * (sL - sh) * (sL - shR_hipL) * (sL - shL_hipL), 0, None))
    return aR + aL


# ---------------------------------------------------------------- 特徵
# 每項：(名稱, 函式, 用到的關節, 方向角?, 單位, 來源, 公式說明, 備註)
# 來源：官方 PO = 官方 primary outcome；官方 CF = 官方 control factor；
#       3.7 = 第 3.7 節原定義（Es3 Q3、Es2 對照組）；補充 = 本專題補充（追加 17 A 第 4 點）
def _f(name, fn, joints, is_dir, unit, src, formula, note=""):
    return dict(name=name, fn=fn, joints=joints, is_dir=is_dir, unit=unit, source=src, formula=formula, note=note)


APPROX = "以共同骨架近似（SpineShoulder→ShoulderMid）"
APPROX_B = "以共同骨架近似（SpineBase→HipMid）"


def _elbow(s):
    return _f(f"elbow_{s}", lambda X: ang3(X, f"Shoulder{s}", f"Elbow{s}", f"Wrist{s}"),
              [f"Shoulder{s}", f"Elbow{s}", f"Wrist{s}"], False, "度", "官方 CF", f"∠(Shoulder{s}, Elbow{s}, Wrist{s})，三維")


def _knee3d(s, src="官方 CF", note=""):
    return _f(f"knee_{s}", lambda X: ang3(X, f"Hip{s}", f"Knee{s}", f"Ankle{s}"),
              [f"Hip{s}", f"Knee{s}", f"Ankle{s}"], False, "度", src, f"∠(Hip{s}, Knee{s}, Ankle{s})，三維", note)


def _angle_c():
    return [
        _f("angle_c_L", lambda X: atan2d(P(X, "HipMid")[:, 0] - P(X, "HipL")[:, 0], P(X, "HipMid")[:, 1] - P(X, "HipL")[:, 1]),
           ["HipL", "HipR"], True, "度", "官方 CF", "atan2(HipMid.x − HipL.x, HipMid.y − HipL.y)", APPROX_B),
        _f("angle_c_R", lambda X: atan2d(P(X, "HipR")[:, 0] - P(X, "HipMid")[:, 0], P(X, "HipR")[:, 1] - P(X, "HipMid")[:, 1]),
           ["HipL", "HipR"], True, "度", "官方 CF", "atan2(HipR.x − HipMid.x, HipR.y − HipMid.y)", APPROX_B),
    ]


def _axilla(s):
    o = "R" if s == "L" else "L"
    return _f(f"axilla_{s}", lambda X: ang3(X, f"Elbow{s}", f"Shoulder{s}", f"Hip{o}"),
              [f"Elbow{s}", f"Shoulder{s}", f"Hip{o}"], False, "度", "官方 CF",
              f"∠(Elbow{s}, Shoulder{s}, Hip{o})，三維（官方以對側髖為「腋下角」的一邊）")


def _shoulder_flex(s):
    return _f(f"shoulder_flex_{s}", lambda X: np.degrees(np.arccos(np.clip(
        -(P(X, f"Elbow{s}") - P(X, f"Shoulder{s}"))[:, 1] / dist(X, f"Elbow{s}", f"Shoulder{s}"), -1, 1))),
        [f"Shoulder{s}", f"Elbow{s}"], False, "度", "補充", f"上臂 Shoulder{s}→Elbow{s} 與向下軸 (−y) 的夾角，三維")


def _dist(name, a, b, src, note=""):
    return _f(name, lambda X: dist(X, a, b), [a, b], False, "軀幹長", src, f"|{a} − {b}|，三維", note)


def _shoulder_ext(s):
    def fn(X):
        a = atan2d(P(X, "ShoulderMid")[:, 2] - P(X, f"Elbow{s}")[:, 2], P(X, "ShoulderMid")[:, 1] - P(X, f"Elbow{s}")[:, 1])
        return np.where(a < -90, a + 360, a)                  # 連續到 [−90°, 270°)，取代官方的「跳動 > 100° 反號」
    return _f(f"shoulder_ext_{s}", fn, ["ShoulderL", "ShoulderR", f"Elbow{s}"], True, "度", "官方 PO",
              f"atan2(ShoulderMid.z − Elbow{s}.z, ShoulderMid.y − Elbow{s}.y)（矢狀面），映射到 [−90°, 270°)",
              APPROX + "；官方以「相鄰幀跳動 > 100° 時反號」處理 ±180° 跳動，這裡改用連續映射")


def _trunk_side(s):
    sign = -1.0 if s == "L" else 1.0
    return _f(f"trunk_incl_{s}", lambda X: sign * atan2d(P(X, f"Shoulder{s}")[:, 0] - P(X, f"Hip{s}")[:, 0],
                                                         P(X, f"Shoulder{s}")[:, 1] - P(X, f"Hip{s}")[:, 1]),
              [f"Shoulder{s}", f"Hip{s}"], True, "度", "官方 PO",
              f"{'−' if s == 'L' else ''}atan2(Shoulder{s}.x − Hip{s}.x, Shoulder{s}.y − Hip{s}.y)（冠狀面）"
              + ("；官方左側取負號" if s == "L" else ""))


def _knee_sag(s):
    def fn(X):
        h, k, a = P(X, f"Hip{s}"), P(X, f"Knee{s}"), P(X, f"Ankle{s}")
        return (atan2d(h[:, 1] - k[:, 1], h[:, 2] - k[:, 2]) + atan2d(k[:, 1] - a[:, 1], a[:, 2] - k[:, 2])) % 360.0
    return _f(f"knee_sag_{s}", fn, [f"Hip{s}", f"Knee{s}", f"Ankle{s}"], True, "度", "官方 PO",
              f"atan2(Hip{s}.y − Knee{s}.y, Hip{s}.z − Knee{s}.z) + atan2(Knee{s}.y − Ankle{s}.y, Ankle{s}.z − Knee{s}.z)"
              "（矢狀面），取 mod 360",
              "官方以「跳動 > 100° 反號」處理跳動，這裡改用 mod 360（站直 ≈ 180°）")


TRUNK = ["HipL", "HipR", "ShoulderL", "ShoulderR"]

FEATURE_SPECS = {
    "Es1": [_shoulder_ext("L"), _shoulder_ext("R"), _elbow("L"), _elbow("R"), _knee3d("L"), _knee3d("R"),
            *_angle_c(),
            _f("trunk_area", heron_trunk_area, TRUNK, False, "軀幹長²", "官方 CF",
               "Heron 公式：三角形 (HipL–HipR, ShoulderR–HipL, ShoulderR–HipR) + (ShoulderL–ShoulderR, ShoulderR–HipL, ShoulderL–HipL)",
               "照抄官方的邊長組合"),
            _dist("ankle_dist", "AnkleL", "AnkleR", "官方 CF")],
    "Es2": [_trunk_side("L"), _trunk_side("R"),
            _f("trunk_lateral_tilt", lambda X: atan2d(P(X, "ShoulderMid")[:, 0] - P(X, "HipMid")[:, 0],
                                                      P(X, "ShoulderMid")[:, 1] - P(X, "HipMid")[:, 1]),
               TRUNK, True, "度", "3.7", "atan2(ShoulderMid.x − HipMid.x, ShoulderMid.y − HipMid.y)（冠狀面）",
               "第 3.7 節：Es2 側彎，Q3 對照組"),
            _elbow("L"), _elbow("R"), _knee3d("L"), _knee3d("R"), *_angle_c(), _axilla("L"), _axilla("R"),
            _f("wrist_depth_L", lambda X: P(X, "WristL")[:, 2] - P(X, "ShoulderMid")[:, 2], ["WristL", "ShoulderL", "ShoulderR"],
               False, "軀幹長", "官方 CF", "WristL.z − ShoulderMid.z", APPROX),
            _f("wrist_depth_R", lambda X: P(X, "WristR")[:, 2] - P(X, "ShoulderMid")[:, 2], ["WristR", "ShoulderL", "ShoulderR"],
               False, "軀幹長", "官方 CF", "WristR.z − ShoulderMid.z", APPROX),
            _dist("shoulder_width", "ShoulderL", "ShoulderR", "官方 CF"), _dist("hip_width", "HipL", "HipR", "官方 CF"),
            _shoulder_flex("L"), _shoulder_flex("R")],
    "Es3": [_f("trunk_rotation", lambda X: wrap180(
                atan2d((P(X, "ShoulderR") - P(X, "ShoulderL"))[:, 2], (P(X, "ShoulderR") - P(X, "ShoulderL"))[:, 0])
                - atan2d((P(X, "HipR") - P(X, "HipL"))[:, 2], (P(X, "HipR") - P(X, "HipL"))[:, 0])),
               TRUNK, True, "度", "3.7", "肩連線與骨盆連線在水平面 (x–z) 的方向角差，wrap 到 ±180°", "第 3.7 節：Q3"),
            _f("elbow_dist_x", lambda X: np.abs(P(X, "ElbowL")[:, 0] - P(X, "ElbowR")[:, 0]), ["ElbowL", "ElbowR"],
               False, "軀幹長", "官方 PO", "|ElbowL.x − ElbowR.x|",
               "官方程式算的是兩肘（變數 linkelbow），圖例寫 wrist；依程式用肘"),
            _elbow("L"), _elbow("R"), _knee3d("L"), _knee3d("R"), *_angle_c(),
            _f("arm_shoulder_L", lambda X: ang3(X, "ElbowL", "ShoulderL", "ShoulderMid"), ["ElbowL", "ShoulderL", "ShoulderR"],
               False, "度", "官方 CF", "∠(ElbowL, ShoulderL, ShoulderMid)，三維", APPROX),
            _f("arm_shoulder_R", lambda X: ang3(X, "ElbowR", "ShoulderR", "ShoulderMid"), ["ElbowR", "ShoulderR", "ShoulderL"],
               False, "度", "官方 CF", "∠(ElbowR, ShoulderR, ShoulderMid)，三維", APPROX),
            _axilla("L"), _axilla("R"),
            _f("wrist_height_L", lambda X: 2 * P(X, "WristL")[:, 1] - P(X, "ElbowL")[:, 1] - P(X, "ShoulderL")[:, 1],
               ["WristL", "ElbowL", "ShoulderL"], False, "軀幹長", "官方 CF", "2·WristL.y − ElbowL.y − ShoulderL.y"),
            _f("wrist_height_R", lambda X: 2 * P(X, "WristR")[:, 1] - P(X, "ElbowR")[:, 1] - P(X, "ShoulderR")[:, 1],
               ["WristR", "ElbowR", "ShoulderR"], False, "軀幹長", "官方 CF", "2·WristR.y − ElbowR.y − ShoulderR.y"),
            _dist("hip_width", "HipL", "HipR", "官方 CF")],
    "Es4": [_elbow("L"), _elbow("R"), _knee3d("L"), _knee3d("R"), _axilla("L"), _axilla("R"),
            _dist("shoulder_width", "ShoulderL", "ShoulderR", "官方 CF"),
            _f("hip_dz", lambda X: np.abs(P(X, "HipL")[:, 2] - P(X, "HipR")[:, 2]), ["HipL", "HipR"], False, "軀幹長",
               "官方 CF", "|HipL.z − HipR.z|"),
            _f("hip_dy", lambda X: np.abs(P(X, "HipL")[:, 1] - P(X, "HipR")[:, 1]), ["HipL", "HipR"], False, "軀幹長",
               "官方 CF", "|HipL.y − HipR.y|"),
            _f("shoulder_dz", lambda X: np.abs(P(X, "ShoulderL")[:, 2] - P(X, "ShoulderR")[:, 2]), ["ShoulderL", "ShoulderR"],
               False, "軀幹長", "官方 CF", "|ShoulderL.z − ShoulderR.z|"),
            _f("pelvis_rotation", lambda X: atan2d((P(X, "HipR") - P(X, "HipL"))[:, 2], (P(X, "HipR") - P(X, "HipL"))[:, 0]),
               ["HipL", "HipR"], True, "度", "補充", "HipL→HipR 在水平面 (x–z) 的方向角"),
            _f("pelvis_lateral_tilt", lambda X: atan2d((P(X, "HipR") - P(X, "HipL"))[:, 1], (P(X, "HipR") - P(X, "HipL"))[:, 0]),
               ["HipL", "HipR"], True, "度", "補充", "HipL→HipR 在冠狀面 (x–y) 的傾角")],
    "Es5": [_knee_sag("L"), _knee_sag("R"), _dist("ankle_dist", "AnkleL", "AnkleR", "官方 CF"),
            _f("trunk_area", heron_trunk_area, TRUNK, False, "軀幹長²", "官方 CF", "同 Es1 trunk_area", "照抄官方的邊長組合"),
            _dist("hip_width", "HipL", "HipR", "官方 CF"), _dist("shoulder_width", "ShoulderL", "ShoulderR", "官方 CF"),
            _f("knee_dy", lambda X: P(X, "KneeR")[:, 1] - P(X, "KneeL")[:, 1], ["KneeL", "KneeR"], False, "軀幹長",
               "官方 CF", "KneeR.y − KneeL.y"),
            _knee3d("L", "補充", "暫定清單的三維膝角；官方 Es5 的膝角是矢狀面版本（knee_sag）"),
            _knee3d("R", "補充", "同上"),
            _f("hip_L", lambda X: ang3(X, "ShoulderL", "HipL", "KneeL"), ["ShoulderL", "HipL", "KneeL"], False, "度", "補充",
               "∠(ShoulderL, HipL, KneeL)，三維"),
            _f("hip_R", lambda X: ang3(X, "ShoulderR", "HipR", "KneeR"), ["ShoulderR", "HipR", "KneeR"], False, "度", "補充",
               "∠(ShoulderR, HipR, KneeR)，三維"),
            _f("trunk_forward_lean", lambda X: atan2d(P(X, "ShoulderMid")[:, 2] - P(X, "HipMid")[:, 2],
                                                      P(X, "ShoulderMid")[:, 1] - P(X, "HipMid")[:, 1]),
               TRUNK, True, "度", "補充", "atan2(ShoulderMid.z − HipMid.z, ShoulderMid.y − HipMid.y)（矢狀面；負值 = 朝鏡頭前傾）")],
}

# 列出但不納入的官方特徵：(動作, 官方變數, 內容, 原因)
EXCLUDED = [
    ("Es1", "link_hand", "雙手 Hand 距離", "用到手部點（共同骨架沒有）"),
    ("Es1", "derR / derL", "相鄰峰值間隔（規律性）", "需要找峰值／重複切分（追加 17 A：不使用 findpeaks）"),
    ("Es2", "link_hand", "雙手 Hand 距離", "用到手部點"),
    ("Es2", "assezhipR/L、assexhipR/L", "左右髖的絕對 z、x 座標", "兩邊不可比（追加 7 C：依賴骨盆位移；共同骨架逐幀減 HipMid）"),
    ("Es2", "derR", "規律性", "需要找峰值／重複切分"),
    ("Es3", "assezhipR/L", "左右髖的絕對 z 座標", "兩邊不可比（追加 7 C）"),
    ("Es3", "der1", "規律性", "需要找峰值／重複切分"),
    ("Es4", "variazioneZ、variazioneX（PO）", "SpineBase 的 z、x 相對整段平均的位移", "兩邊不可比（追加 7 C：骨盆位移）"),
    ("Es4", "assezShoulderR/L", "左右肩的絕對 z 座標", "兩邊不可比（追加 7 C）"),
    ("Es4", "derZ", "規律性", "需要找峰值／重複切分"),
    ("Es5", "assezshoulderR/L、assexshoulderR/L", "左右肩的絕對 z、x 座標", "兩邊不可比（追加 7 C）"),
    ("Es5", "link_shoulderhandR/L、link_hand", "肩–手、雙手距離", "用到手部點"),
    ("Es5", "derR / derL", "規律性", "需要找峰值／重複切分"),
]
DROPPED_PROVISIONAL = [
    ("Es1", "shoulder_flex_L/R（暫定）", "官方 Es1 已有肩部主要動作量 shoulder_ext_L/R（矢狀面），不重複保留"),
]

# paired_compare 用：{動作: {名稱: (函式, 關節, 方向角)}}
FEATURES = {ex: {f["name"]: (f["fn"], f["joints"], f["is_dir"]) for f in fs} for ex, fs in FEATURE_SPECS.items()}


def render_markdown():
    L = ["# v7 T7 特徵清單（追加 17 A）", "",
         "來源：KIMORE 官方 MATLAB 腳本 `Script/feat_extract_Ex1.m`～`feat_extract_Ex5.m`（各動作所有副本 md5 相同），"
         "加上第 3.7 節原定義與本專題補充。程式：`features_v7.py`（`FEATURE_SPECS`），由 `paired_compare.py` 使用。", "",
         "## 共同約定", "",
         "- **座標**：共同骨架 17 點，兩邊都用共同參考幀正規化（追加 8 B）。x = 參考幀 HipL→HipR（約 Kinect +X）、"
         "y = 向上、z = x × y（約 Kinect +Z，遠離鏡頭）。官方公式的第 1、2、3 分量對應 x、y、z。",
         "- **單位**：角度 = 度；長度 = 軀幹長（HipMid→ShoulderMid 整段中位數）；面積 = 軀幹長²。官方原始單位為公尺，"
         "這裡因正規化改為軀幹長，Kinect 與 MediaPipe 相同。",
         "- **以共同骨架近似**：官方用 SpineShoulder／SpineBase 的特徵改用 ShoulderMid／HipMid（第 3.2 節）。",
         "- **摘要值**：ROM = P95 − P5、峰值 = P95（追加 17 A），在 Kinect 保留段對應的時段上計算（追加 14 A）。"
         "不使用官方的找峰值（findpeaks）、1 Hz 低通濾波（`filtering.m`）與頭尾裁切。",
         "- **缺漏**：MediaPipe 端 visibility < 0.5 的關節視為缺漏，有效幀 < 50% 的影片不納入該特徵（追加 7 D）。",
         "- **方向角**：標「是」的特徵，逐幀差以 (M − K + 180) mod 360 − 180 計算。",
         "- **來源**：官方 PO = 官方 primary outcome；官方 CF = 官方 control factor；3.7 = 第 3.7 節原定義；"
         "補充 = 本專題補充（追加 17 A 第 4 點）。", ""]
    n = 0
    for ex, fs in FEATURE_SPECS.items():
        L += [f"## {ex}（{len(fs)} 項）", "", "| # | 特徵 | 來源 | 公式 | 關節 | 單位 | 方向角 | 備註 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for i, f in enumerate(fs, 1):
            L.append(f"| {i} | `{f['name']}` | {f['source']} | {f['formula']} | {', '.join(f['joints'])} | {f['unit']} | "
                     f"{'是' if f['is_dir'] else '否'} | {f['note']} |")
            n += 1
        L.append("")
    L += [f"納入合計 **{n}** 項（含左右分開計）。", "",
          "## 列出但不納入的官方特徵", "", "| 動作 | 官方變數 | 內容 | 原因 |", "| --- | --- | --- | --- |"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in EXCLUDED]
    L += ["", "## 暫定清單中移除的項目", "", "| 動作 | 特徵 | 原因 |", "| --- | --- | --- |"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in DROPPED_PROVISIONAL]
    L += ["", "## 與官方腳本不同之處（實作上的選擇）", "",
          "1. 不做官方的 1 Hz Butterworth 低通濾波與頭尾裁切（前 10～30 幀、最後 15 幀）：摘要值改用 P95／P5，"
          "對單幀錯誤已較穩健；MediaPipe 有缺漏幀，`filtfilt` 無法直接套用。兩邊處理相同。",
          "2. ±180° 跳動：官方以「相鄰幀差 > 100° 時反號」處理；這裡 `shoulder_ext` 映射到 [−90°, 270°)、`knee_sag` 取 mod 360，"
          "讓舉臂過頭、深蹲時角度連續。",
          "3. Es3 主要動作量：官方程式是兩肘的 x 距離（`linkelbow = |Elbow_L.x − Elbow_R.x|`），圖例與註解寫 wrist；依程式用肘。",
          "4. 長度、面積改以軀幹長為單位（正規化的結果），不是公尺。", ""]
    return "\n".join(L)


if __name__ == "__main__":
    os.makedirs("runs/v7_gap", exist_ok=True)
    with open("runs/v7_gap/feature_list.md", "w") as f:
        f.write(render_markdown())
    print("→ runs/v7_gap/feature_list.md", sum(len(v) for v in FEATURE_SPECS.values()), "項")
