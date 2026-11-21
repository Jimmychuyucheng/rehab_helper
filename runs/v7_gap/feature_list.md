# v7 T7 特徵清單（追加 17 A）

來源：KIMORE 官方 MATLAB 腳本 `Script/feat_extract_Ex1.m`～`feat_extract_Ex5.m`（各動作所有副本 md5 相同），加上第 3.7 節原定義與本專題補充。程式：`features_v7.py`（`FEATURE_SPECS`），由 `paired_compare.py` 使用。

## 共同約定

- **座標**：共同骨架 17 點，兩邊都用共同參考幀正規化（追加 8 B）。x = 參考幀 HipL→HipR（約 Kinect +X）、y = 向上、z = x × y（約 Kinect +Z，遠離鏡頭）。官方公式的第 1、2、3 分量對應 x、y、z。
- **單位**：角度 = 度；長度 = 軀幹長（HipMid→ShoulderMid 整段中位數）；面積 = 軀幹長²。官方原始單位為公尺，這裡因正規化改為軀幹長，Kinect 與 MediaPipe 相同。
- **以共同骨架近似**：官方用 SpineShoulder／SpineBase 的特徵改用 ShoulderMid／HipMid（第 3.2 節）。
- **摘要值**：ROM = P95 − P5、峰值 = P95（追加 17 A），在 Kinect 保留段對應的時段上計算（追加 14 A）。不使用官方的找峰值（findpeaks）、1 Hz 低通濾波（`filtering.m`）與頭尾裁切。
- **缺漏**：MediaPipe 端 visibility < 0.5 的關節視為缺漏，有效幀 < 50% 的影片不納入該特徵（追加 7 D）。
- **方向角**：標「是」的特徵，逐幀差以 (M − K + 180) mod 360 − 180 計算。
- **來源**：官方 PO = 官方 primary outcome；官方 CF = 官方 control factor；3.7 = 第 3.7 節原定義；補充 = 本專題補充（追加 17 A 第 4 點）。

## Es1（10 項）

| # | 特徵 | 來源 | 公式 | 關節 | 單位 | 方向角 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `shoulder_ext_L` | 官方 PO | atan2(ShoulderMid.z − ElbowL.z, ShoulderMid.y − ElbowL.y)（矢狀面），映射到 [−90°, 270°) | ShoulderL, ShoulderR, ElbowL | 度 | 是 | 以共同骨架近似（SpineShoulder→ShoulderMid）；官方以「相鄰幀跳動 > 100° 時反號」處理 ±180° 跳動，這裡改用連續映射 |
| 2 | `shoulder_ext_R` | 官方 PO | atan2(ShoulderMid.z − ElbowR.z, ShoulderMid.y − ElbowR.y)（矢狀面），映射到 [−90°, 270°) | ShoulderL, ShoulderR, ElbowR | 度 | 是 | 以共同骨架近似（SpineShoulder→ShoulderMid）；官方以「相鄰幀跳動 > 100° 時反號」處理 ±180° 跳動，這裡改用連續映射 |
| 3 | `elbow_L` | 官方 CF | ∠(ShoulderL, ElbowL, WristL)，三維 | ShoulderL, ElbowL, WristL | 度 | 否 |  |
| 4 | `elbow_R` | 官方 CF | ∠(ShoulderR, ElbowR, WristR)，三維 | ShoulderR, ElbowR, WristR | 度 | 否 |  |
| 5 | `knee_L` | 官方 CF | ∠(HipL, KneeL, AnkleL)，三維 | HipL, KneeL, AnkleL | 度 | 否 |  |
| 6 | `knee_R` | 官方 CF | ∠(HipR, KneeR, AnkleR)，三維 | HipR, KneeR, AnkleR | 度 | 否 |  |
| 7 | `angle_c_L` | 官方 CF | atan2(HipMid.x − HipL.x, HipMid.y − HipL.y) | HipL, HipR | 度 | 是 | 以共同骨架近似（SpineBase→HipMid） |
| 8 | `angle_c_R` | 官方 CF | atan2(HipR.x − HipMid.x, HipR.y − HipMid.y) | HipL, HipR | 度 | 是 | 以共同骨架近似（SpineBase→HipMid） |
| 9 | `trunk_area` | 官方 CF | Heron 公式：三角形 (HipL–HipR, ShoulderR–HipL, ShoulderR–HipR) + (ShoulderL–ShoulderR, ShoulderR–HipL, ShoulderL–HipL) | HipL, HipR, ShoulderL, ShoulderR | 軀幹長² | 否 | 照抄官方的邊長組合 |
| 10 | `ankle_dist` | 官方 CF | |AnkleL − AnkleR|，三維 | AnkleL, AnkleR | 軀幹長 | 否 |  |

## Es2（17 項）

| # | 特徵 | 來源 | 公式 | 關節 | 單位 | 方向角 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `trunk_incl_L` | 官方 PO | −atan2(ShoulderL.x − HipL.x, ShoulderL.y − HipL.y)（冠狀面）；官方左側取負號 | ShoulderL, HipL | 度 | 是 |  |
| 2 | `trunk_incl_R` | 官方 PO | atan2(ShoulderR.x − HipR.x, ShoulderR.y − HipR.y)（冠狀面） | ShoulderR, HipR | 度 | 是 |  |
| 3 | `trunk_lateral_tilt` | 3.7 | atan2(ShoulderMid.x − HipMid.x, ShoulderMid.y − HipMid.y)（冠狀面） | HipL, HipR, ShoulderL, ShoulderR | 度 | 是 | 第 3.7 節：Es2 側彎，Q3 對照組 |
| 4 | `elbow_L` | 官方 CF | ∠(ShoulderL, ElbowL, WristL)，三維 | ShoulderL, ElbowL, WristL | 度 | 否 |  |
| 5 | `elbow_R` | 官方 CF | ∠(ShoulderR, ElbowR, WristR)，三維 | ShoulderR, ElbowR, WristR | 度 | 否 |  |
| 6 | `knee_L` | 官方 CF | ∠(HipL, KneeL, AnkleL)，三維 | HipL, KneeL, AnkleL | 度 | 否 |  |
| 7 | `knee_R` | 官方 CF | ∠(HipR, KneeR, AnkleR)，三維 | HipR, KneeR, AnkleR | 度 | 否 |  |
| 8 | `angle_c_L` | 官方 CF | atan2(HipMid.x − HipL.x, HipMid.y − HipL.y) | HipL, HipR | 度 | 是 | 以共同骨架近似（SpineBase→HipMid） |
| 9 | `angle_c_R` | 官方 CF | atan2(HipR.x − HipMid.x, HipR.y − HipMid.y) | HipL, HipR | 度 | 是 | 以共同骨架近似（SpineBase→HipMid） |
| 10 | `axilla_L` | 官方 CF | ∠(ElbowL, ShoulderL, HipR)，三維（官方以對側髖為「腋下角」的一邊） | ElbowL, ShoulderL, HipR | 度 | 否 |  |
| 11 | `axilla_R` | 官方 CF | ∠(ElbowR, ShoulderR, HipL)，三維（官方以對側髖為「腋下角」的一邊） | ElbowR, ShoulderR, HipL | 度 | 否 |  |
| 12 | `wrist_depth_L` | 官方 CF | WristL.z − ShoulderMid.z | WristL, ShoulderL, ShoulderR | 軀幹長 | 否 | 以共同骨架近似（SpineShoulder→ShoulderMid） |
| 13 | `wrist_depth_R` | 官方 CF | WristR.z − ShoulderMid.z | WristR, ShoulderL, ShoulderR | 軀幹長 | 否 | 以共同骨架近似（SpineShoulder→ShoulderMid） |
| 14 | `shoulder_width` | 官方 CF | |ShoulderL − ShoulderR|，三維 | ShoulderL, ShoulderR | 軀幹長 | 否 |  |
| 15 | `hip_width` | 官方 CF | |HipL − HipR|，三維 | HipL, HipR | 軀幹長 | 否 |  |
| 16 | `shoulder_flex_L` | 補充 | 上臂 ShoulderL→ElbowL 與向下軸 (−y) 的夾角，三維 | ShoulderL, ElbowL | 度 | 否 |  |
| 17 | `shoulder_flex_R` | 補充 | 上臂 ShoulderR→ElbowR 與向下軸 (−y) 的夾角，三維 | ShoulderR, ElbowR | 度 | 否 |  |

## Es3（15 項）

| # | 特徵 | 來源 | 公式 | 關節 | 單位 | 方向角 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `trunk_rotation` | 3.7 | 肩連線與骨盆連線在水平面 (x–z) 的方向角差，wrap 到 ±180° | HipL, HipR, ShoulderL, ShoulderR | 度 | 是 | 第 3.7 節：Q3 |
| 2 | `elbow_dist_x` | 官方 PO | |ElbowL.x − ElbowR.x| | ElbowL, ElbowR | 軀幹長 | 否 | 官方程式算的是兩肘（變數 linkelbow），圖例寫 wrist；依程式用肘 |
| 3 | `elbow_L` | 官方 CF | ∠(ShoulderL, ElbowL, WristL)，三維 | ShoulderL, ElbowL, WristL | 度 | 否 |  |
| 4 | `elbow_R` | 官方 CF | ∠(ShoulderR, ElbowR, WristR)，三維 | ShoulderR, ElbowR, WristR | 度 | 否 |  |
| 5 | `knee_L` | 官方 CF | ∠(HipL, KneeL, AnkleL)，三維 | HipL, KneeL, AnkleL | 度 | 否 |  |
| 6 | `knee_R` | 官方 CF | ∠(HipR, KneeR, AnkleR)，三維 | HipR, KneeR, AnkleR | 度 | 否 |  |
| 7 | `angle_c_L` | 官方 CF | atan2(HipMid.x − HipL.x, HipMid.y − HipL.y) | HipL, HipR | 度 | 是 | 以共同骨架近似（SpineBase→HipMid） |
| 8 | `angle_c_R` | 官方 CF | atan2(HipR.x − HipMid.x, HipR.y − HipMid.y) | HipL, HipR | 度 | 是 | 以共同骨架近似（SpineBase→HipMid） |
| 9 | `arm_shoulder_L` | 官方 CF | ∠(ElbowL, ShoulderL, ShoulderMid)，三維 | ElbowL, ShoulderL, ShoulderR | 度 | 否 | 以共同骨架近似（SpineShoulder→ShoulderMid） |
| 10 | `arm_shoulder_R` | 官方 CF | ∠(ElbowR, ShoulderR, ShoulderMid)，三維 | ElbowR, ShoulderR, ShoulderL | 度 | 否 | 以共同骨架近似（SpineShoulder→ShoulderMid） |
| 11 | `axilla_L` | 官方 CF | ∠(ElbowL, ShoulderL, HipR)，三維（官方以對側髖為「腋下角」的一邊） | ElbowL, ShoulderL, HipR | 度 | 否 |  |
| 12 | `axilla_R` | 官方 CF | ∠(ElbowR, ShoulderR, HipL)，三維（官方以對側髖為「腋下角」的一邊） | ElbowR, ShoulderR, HipL | 度 | 否 |  |
| 13 | `wrist_height_L` | 官方 CF | 2·WristL.y − ElbowL.y − ShoulderL.y | WristL, ElbowL, ShoulderL | 軀幹長 | 否 |  |
| 14 | `wrist_height_R` | 官方 CF | 2·WristR.y − ElbowR.y − ShoulderR.y | WristR, ElbowR, ShoulderR | 軀幹長 | 否 |  |
| 15 | `hip_width` | 官方 CF | |HipL − HipR|，三維 | HipL, HipR | 軀幹長 | 否 |  |

## Es4（12 項）

| # | 特徵 | 來源 | 公式 | 關節 | 單位 | 方向角 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `elbow_L` | 官方 CF | ∠(ShoulderL, ElbowL, WristL)，三維 | ShoulderL, ElbowL, WristL | 度 | 否 |  |
| 2 | `elbow_R` | 官方 CF | ∠(ShoulderR, ElbowR, WristR)，三維 | ShoulderR, ElbowR, WristR | 度 | 否 |  |
| 3 | `knee_L` | 官方 CF | ∠(HipL, KneeL, AnkleL)，三維 | HipL, KneeL, AnkleL | 度 | 否 |  |
| 4 | `knee_R` | 官方 CF | ∠(HipR, KneeR, AnkleR)，三維 | HipR, KneeR, AnkleR | 度 | 否 |  |
| 5 | `axilla_L` | 官方 CF | ∠(ElbowL, ShoulderL, HipR)，三維（官方以對側髖為「腋下角」的一邊） | ElbowL, ShoulderL, HipR | 度 | 否 |  |
| 6 | `axilla_R` | 官方 CF | ∠(ElbowR, ShoulderR, HipL)，三維（官方以對側髖為「腋下角」的一邊） | ElbowR, ShoulderR, HipL | 度 | 否 |  |
| 7 | `shoulder_width` | 官方 CF | |ShoulderL − ShoulderR|，三維 | ShoulderL, ShoulderR | 軀幹長 | 否 |  |
| 8 | `hip_dz` | 官方 CF | |HipL.z − HipR.z| | HipL, HipR | 軀幹長 | 否 |  |
| 9 | `hip_dy` | 官方 CF | |HipL.y − HipR.y| | HipL, HipR | 軀幹長 | 否 |  |
| 10 | `shoulder_dz` | 官方 CF | |ShoulderL.z − ShoulderR.z| | ShoulderL, ShoulderR | 軀幹長 | 否 |  |
| 11 | `pelvis_rotation` | 補充 | HipL→HipR 在水平面 (x–z) 的方向角 | HipL, HipR | 度 | 是 |  |
| 12 | `pelvis_lateral_tilt` | 補充 | HipL→HipR 在冠狀面 (x–y) 的傾角 | HipL, HipR | 度 | 是 |  |

## Es5（12 項）

| # | 特徵 | 來源 | 公式 | 關節 | 單位 | 方向角 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `knee_sag_L` | 官方 PO | atan2(HipL.y − KneeL.y, HipL.z − KneeL.z) + atan2(KneeL.y − AnkleL.y, AnkleL.z − KneeL.z)（矢狀面），取 mod 360 | HipL, KneeL, AnkleL | 度 | 是 | 官方以「跳動 > 100° 反號」處理跳動，這裡改用 mod 360（站直 ≈ 180°） |
| 2 | `knee_sag_R` | 官方 PO | atan2(HipR.y − KneeR.y, HipR.z − KneeR.z) + atan2(KneeR.y − AnkleR.y, AnkleR.z − KneeR.z)（矢狀面），取 mod 360 | HipR, KneeR, AnkleR | 度 | 是 | 官方以「跳動 > 100° 反號」處理跳動，這裡改用 mod 360（站直 ≈ 180°） |
| 3 | `ankle_dist` | 官方 CF | |AnkleL − AnkleR|，三維 | AnkleL, AnkleR | 軀幹長 | 否 |  |
| 4 | `trunk_area` | 官方 CF | 同 Es1 trunk_area | HipL, HipR, ShoulderL, ShoulderR | 軀幹長² | 否 | 照抄官方的邊長組合 |
| 5 | `hip_width` | 官方 CF | |HipL − HipR|，三維 | HipL, HipR | 軀幹長 | 否 |  |
| 6 | `shoulder_width` | 官方 CF | |ShoulderL − ShoulderR|，三維 | ShoulderL, ShoulderR | 軀幹長 | 否 |  |
| 7 | `knee_dy` | 官方 CF | KneeR.y − KneeL.y | KneeL, KneeR | 軀幹長 | 否 |  |
| 8 | `knee_L` | 補充 | ∠(HipL, KneeL, AnkleL)，三維 | HipL, KneeL, AnkleL | 度 | 否 | 暫定清單的三維膝角；官方 Es5 的膝角是矢狀面版本（knee_sag） |
| 9 | `knee_R` | 補充 | ∠(HipR, KneeR, AnkleR)，三維 | HipR, KneeR, AnkleR | 度 | 否 | 同上 |
| 10 | `hip_L` | 補充 | ∠(ShoulderL, HipL, KneeL)，三維 | ShoulderL, HipL, KneeL | 度 | 否 |  |
| 11 | `hip_R` | 補充 | ∠(ShoulderR, HipR, KneeR)，三維 | ShoulderR, HipR, KneeR | 度 | 否 |  |
| 12 | `trunk_forward_lean` | 補充 | atan2(ShoulderMid.z − HipMid.z, ShoulderMid.y − HipMid.y)（矢狀面；負值 = 朝鏡頭前傾） | HipL, HipR, ShoulderL, ShoulderR | 度 | 是 |  |

納入合計 **66** 項（含左右分開計）。

## 列出但不納入的官方特徵

| 動作 | 官方變數 | 內容 | 原因 |
| --- | --- | --- | --- |
| Es1 | link_hand | 雙手 Hand 距離 | 用到手部點（共同骨架沒有） |
| Es1 | derR / derL | 相鄰峰值間隔（規律性） | 需要找峰值／重複切分（追加 17 A：不使用 findpeaks） |
| Es2 | link_hand | 雙手 Hand 距離 | 用到手部點 |
| Es2 | assezhipR/L、assexhipR/L | 左右髖的絕對 z、x 座標 | 兩邊不可比（追加 7 C：依賴骨盆位移；共同骨架逐幀減 HipMid） |
| Es2 | derR | 規律性 | 需要找峰值／重複切分 |
| Es3 | assezhipR/L | 左右髖的絕對 z 座標 | 兩邊不可比（追加 7 C） |
| Es3 | der1 | 規律性 | 需要找峰值／重複切分 |
| Es4 | variazioneZ、variazioneX（PO） | SpineBase 的 z、x 相對整段平均的位移 | 兩邊不可比（追加 7 C：骨盆位移） |
| Es4 | assezShoulderR/L | 左右肩的絕對 z 座標 | 兩邊不可比（追加 7 C） |
| Es4 | derZ | 規律性 | 需要找峰值／重複切分 |
| Es5 | assezshoulderR/L、assexshoulderR/L | 左右肩的絕對 z、x 座標 | 兩邊不可比（追加 7 C） |
| Es5 | link_shoulderhandR/L、link_hand | 肩–手、雙手距離 | 用到手部點 |
| Es5 | derR / derL | 規律性 | 需要找峰值／重複切分 |

## 暫定清單中移除的項目

| 動作 | 特徵 | 原因 |
| --- | --- | --- |
| Es1 | shoulder_flex_L/R（暫定） | 官方 Es1 已有肩部主要動作量 shoulder_ext_L/R（矢狀面），不重複保留 |

## 與官方腳本不同之處（實作上的選擇）

1. 不做官方的 1 Hz Butterworth 低通濾波與頭尾裁切（前 10～30 幀、最後 15 幀）：摘要值改用 P95／P5，對單幀錯誤已較穩健；MediaPipe 有缺漏幀，`filtfilt` 無法直接套用。兩邊處理相同。
2. ±180° 跳動：官方以「相鄰幀差 > 100° 時反號」處理；這裡 `shoulder_ext` 映射到 [−90°, 270°)、`knee_sag` 取 mod 360，讓舉臂過頭、深蹲時角度連續。
3. Es3 主要動作量：官方程式是兩肘的 x 距離（`linkelbow = |Elbow_L.x − Elbow_R.x|`），圖例與註解寫 wrist；依程式用肘。
4. 長度、面積改以軀幹長為單位（正規化的結果），不是公尺。
