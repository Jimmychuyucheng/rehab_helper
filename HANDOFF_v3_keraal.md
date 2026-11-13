# HANDOFF v3：KERAAL 可行性檢查 + CTR-GCN 錯誤分類試跑

## 目標
確認 KERAAL 能不能訓練出「錯誤類型（+ 部位）」分類器，作為 demo 的部位建議模組。
這一輪只做 **Kinect 骨架**；BlazePose（手機鏡頭）留到下一輪。

## 背景：KERAAL 重點（已由 Claude 從官方 repo 確認）
- 官方 repo：https://github.com/nguyensmai/KeraalDataset（程式碼為 BSD-2）。
  資料下載頁：http://nguyensmai.free.fr/KeraalDataset.html（**資料授權請使用者在下載頁確認**）。
- 下背痛復健，3 個**坐姿**動作，檔名代碼如下：

| 代碼 | 動作 | Error1 | Error2 | Error3 |
|---|---|---|---|---|
| RTK | 軀幹旋轉 | 手臂沒有抬到水平 | 軀幹旋轉不足 | 屁股離開椅子、旋轉時身體側傾 |
| ELK | 側向伸展 | 沒舉起的那隻手沒放在身側 | 身體沒有側彎去伸展側腰 | 屁股離開椅子、側彎過度 |
| CTK | 呼吸（藏臉） | 手臂抬得不夠高 | 吸氣時手臂張得不夠開 | 手肘沒有彎到 90° |

  （來源：repo 內 annotate_videos/LabellisationDesVideos_fr.pdf）
- 部位標籤：LeftArm / RightArm / BothArms / Spine。嚴重度：SmallError / LargeError。
- 分組與數量（README）：

| 組別 | 內容 | 標註 | 錄影數 |
|---|---|---|---|
| 1A | 病人 | 有（annotatorA / annotatorB，Anvil XML） | 249 |
| 1B | 病人 | 無 | 1631 |
| 2A | 健康人 | 有（Anvil XML） | 51 |
| 2B | 健康人 | 無 | 151 |
| 3 | 健康人**故意做錯** | 錯誤標籤寫在檔名 | 540 |

- 每個檔案是**一次重複**（約 190 幀，30 fps），不是整段多次重複。
- Kinect txt：每列一幀，25 關節 × 7 個值（x y z + 四元數 qx qy qz qw）= 175 個值（實際 split 會得到 176 欄，最後是空欄）。
  單位公尺，相機座標；**關節順序與 KIMORE 完全相同**（SpineBase=0 … ThumbRight=24），可以直接沿用 `preprocess_kimore.py` 的 `JOINT_NAMES` 與 `body_centric_alignment`。
- 標註 XML 的解析可參考 repo 內 `evaluation_mouvements_lstm_python/utils/utils.py` 的 `readAnnotationFile()`：
  - `Global evaluation` track：Correct / Incorrect / Incomplete / Motionless；
  - `Global error` track：type（Error1/2/3/Other）、Evaluation（Small/LargeError）、bodyPart。
- 官方 benchmark 的做法：**用 Group 3（故意做錯）訓練，在病人（1A）上測試**。

## 步驟 0：資料（使用者手動下載）
- 請使用者從下載頁下載，放到 `data/keraal/raw/`，維持官方資料夾結構（group1A/kinect/RTK/... 等）。
- 確認 `data/keraal/` 已在 `.gitignore`。**不要 commit 任何資料。**
- 若下載頁需要填表或資料授權不允許此用途，先停下來回報。

## 步驟 1：盤點（`keraal_inspect.py` → `runs/keraal_inspect.md`）
1. 每個組別 × 動作：檔案數、受試者數、幀數分布（最小 / 中位數 / 最大）、Kinect 有沒有缺幀或全零關節。
2. Group 3 檔名解析：列出所有不重複的 token，確認標籤欄位的寫法（例如 `...-Unknown-C-0`），以及正確/錯誤標籤對應哪個欄位。
3. 1A、2A 的標註統計：Global evaluation 各類數量；Error1/2/3/Other × bodyPart × 嚴重度的交叉表。
4. **兩位標註者一致度**：同一段錄影 annotatorA 與 annotatorB 的 Cohen's kappa（正確/錯誤、錯誤類型各算一次）。這就是模型能達到的合理上限。
5. 檢查錯誤類型和部位的對應關係：例如 RTK 的 Error2 是否幾乎都是 Spine。如果是，部位可以由錯誤類型推得，不需要另外一個輸出頭（左右手除外）。

## 步驟 2：前處理（`preprocess_keraal.py` → `data/keraal/processed/keraal_kinect.pkl`）
- 只取位置（每個關節 7 個值的前 3 個），形狀 (3, T, 25)，沿用 KIMORE 的 `body_centric_alignment`（原點、yaw 對齊、軀幹長正規化）。
- 樣本欄位比照 KIMORE pkl：group, subject, exercise, recording_id, skeleton, num_frames，再加：
  - eval（Correct / Incorrect / Incomplete / Motionless）、error_type（0=Correct, 1/2/3, Other）、body_part、severity；
  - 1A/2A：預設用 annotatorA 的標籤，同時保留 annotatorB 的標籤。
- 無標註的 1B、2B 也轉檔，標籤欄位留 None，之後做預訓練用。

## 步驟 3：CTR-GCN 錯誤分類（`train_keraal.py`，盡量重用 `train_rehab.py` 的結構）
- **任務**：每個動作分開訓練，4 類（Correct / Error1 / Error2 / Error3）。排除 Incomplete、Motionless、Other，但在報告中列出它們的數量。
- **模型**：`model/rehab_ctrgcn.py` 的 Model，`num_class=4`，Cross-Entropy 加類別權重（類別不平衡時）。
- **輸入**：每次重複**整段**重新取樣成 100 幀（不截片段）；測試時不需要多視窗平均。
- **資料增強**：沿用 `SkeletonAugmentor`，但**關掉左右翻轉**（翻轉會讓 LeftArm/RightArm 標籤錯置）。
- **兩種評估方式**（都要做）：
  - **P1（官方）**：訓練 = Group 3 + 2A；測試 = 1A 病人。訓練集內再依受試者切 15% 當驗證集做 early stopping。
  - **P2（輔助）**：所有有標註的資料，依受試者分組做 5 折交叉驗證。報告時把病人和健康人分開列。
- **指標**：macro-F1（主要）、balanced accuracy、各類 recall、混淆矩陣，以及二元（正確 vs 錯誤）的 F1；baseline = 永遠猜最多的類別。
- 其他超參數盡量與 KIMORE baseline 相同（AdamW 1e-3、wd 1e-2、warmup 5 + cosine、patience 30、batch 8）。資料少，epochs 可以先設 100。
- **輸出**：`runs/keraal_P1_<動作>/`、`runs/keraal_P2_<動作>/fold*/`，內容有 metrics.json、test_predictions.csv、confusion matrix，最後產生一份 `runs/keraal_summary.md`。

## 驗證（每一步都要跑過）
1. 前處理：隨機抽 3 個樣本畫骨架（可參考 repo 的 draw_skeleton），確認方向和尺度正確，坐姿看起來合理。
2. overfit 測試：16 個樣本能訓練到接近 100% 正確。
3. smoke test：每個動作跑 3 epochs，確認流程不壞。

## 判斷標準（回報時請明確說明是否達成）
- **可行**：P1 在病人上的 macro-F1 明顯高於多數類 baseline，而且 Error2（幅度不足類）的 recall 不是 0。
- 同時列出標註者一致度。如果模型接近標註者之間的一致度，代表已經接近資料能提供的上限。
- 若三個動作中至少 RTK 或 ELK 可行 → 下一輪做 BlazePose 對照與部位輸出。
- 若都不可行 → 回報各類樣本數和混淆矩陣，不要自行大改模型。

## 不要做的事
- 不要 commit `data/` 或 `runs/`。
- 不要修改 KIMORE 相關程式的預設行為（`train_rehab.py` 要能照常重現 baseline）。
- 這一輪不做 BlazePose、不做多任務、不合併 KIMORE。
