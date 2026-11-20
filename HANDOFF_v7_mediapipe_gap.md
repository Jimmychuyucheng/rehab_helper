# HANDOFF v7：Kinect ↔ MediaPipe 落差測試（事前判斷標準）

撰寫日期：2026-11-20
狀態：**判斷標準凍結**。本文件需在跑任何 MediaPipe 結果之前 commit；commit hash 記在最下方。之後只能「追加」，不能修改第 4 節的數字。

---

## 0. 目的

demo 用手機 MediaPipe，但模型、特徵函式庫門檻與驗證都建立在 Kinect 骨架上。本實驗回答三個問題，每個問題對應一個決策：

| 問題 | 決策 |
| --- | --- |
| Q1 評分落差：同一部影片，MediaPipe 骨架的預測和 Kinect 差多少？ | 正式模型怎麼訓練（Kinect + 模擬抖動／MediaPipe 微調／MediaPipe 從頭訓練） |
| Q2 特徵落差：Es1～Es5 的關節角度差多少？ | 特徵函式庫門檻能不能沿用、要不要重新校準、哪些特徵手機版不能用 |
| Q3 旋轉落差：軀幹旋轉角度撐不撐得住？ | 模組 C 做旋轉 + 側彎，或只做側彎（或規定拍攝角度） |

## 1. 開始前必須完成（沒完成不跑）

- [x] **RGB 影片授權**（2026-11-20 確認）：KIMORE EULA 第 3 條明文允許 "The user may use RGB data from the dataset"；第 1 條限學術研究用途。拿 RGB 跑 MediaPipe 並訓練屬學術研究，在範圍內。執行時遵守：
  - **從 RGB 衍生的骨架也算資料集的一部分**，與原始資料一樣不推上 git、不放進 demo 安裝包（第 2 條禁止散布）。`.gitignore` 必須涵蓋 `data/`。
  - **存取權不可轉讓**（第 3 條）：server 上的 RGB 與衍生資料放在只有本人可讀的目錄（`chmod 700`）。組員要碰資料須各自簽 EULA。
  - 若用於商業性質的比賽或提案，先詢問作者（第 1 條禁止證明商業系統效能）。
  - 不使用 GitHub 上第三方公開的 KIMORE 副本（例如 ulkudilemre/a-deep-learning-framework-...），那些違反 EULA 散布條款；只用作者提供的資料。
- [ ] 本文件 commit 進 git（hash 填在第 9 節）。
- [ ] 確定 353 部影片清單（`data/kimore/rgb_manifest.csv`：受試者、動作、RGB 路徑、對應的 v3 樣本 id、fold）。
- [ ] 共同骨架轉換函式與單元測試完成（第 3.2 節）。

**MediaPipe 擷取（3.1）不依賴共同骨架**，只存原始 33 點，所以 5 部試跑通過後就可以先把 353 部丟上 server 跑，和撰寫轉換函式並行。第 1 節的條件只約束「配對比較與角度分析」之前要完成。

## 2. 和原規劃不同的兩處（及原因）

1. **配對比較不直接用現有的 avg.pt（25 關節）。** avg.pt 吃 Kinect 25 點，MediaPipe 缺 SpineMid、Neck、HandTip、Thumb 等點，硬補進去測到的會是「補點方法的落差」，不是 demo 實際會遇到的落差。所以先做**共同骨架版 Kinect 模型**（K-common），再把 MediaPipe 共同骨架餵進同一個模型。K-common 本來就在待辦清單上（重算 353 筆的 baseline 與門檻），這裡只是提前做。
   - 補充（不參與決策）：avg.pt + MediaPipe 補點版也跑一次，只當參考。
2. **配對預測必須用 out-of-fold 模型**：每部影片只用「它在測試集的那一折」的模型預測。若用看過這部影片的模型，Kinect 預測會偏準，落差會被高估。

## 3. 實驗流程

### 3.1 MediaPipe 擷取（`run_mediapipe.py`）

- 模型：MediaPipe Pose Landmarker **heavy**，`running_mode=VIDEO`，記錄 mediapipe 版本與模型檔 hash。
- 輸出每幀的 `pose_world_landmarks`（33 × 3，公尺、髖部中點為原點）＋ 各點 `visibility`＋ 該幀的影片時間（ms，由 PTS 取得，不假設固定 fps）。
- 同時存 `pose_landmarks`（影像座標），只用來檢查 bug 和左右是否對調。
- 不另外加平滑，用 MediaPipe 內建的設定（之後模擬抖動要的就是原始抖動）。
- 存成 `data/kimore/mediapipe/raw/{subject}_{Es}.npz`，跑完寫 `extract_log.csv`（幀數、偵測失敗幀比例、平均 visibility、ffprobe 的 fps）。

### 3.2 共同骨架與正規化（`common_skeleton.py`）

- 關節子集（17 點，順序即陣列 index，寫成程式常數 `COMMON_JOINTS`，兩邊共用）：

| idx | 共同骨架 | Kinect v2 index | MediaPipe index |
| --- | --- | --- | --- |
| 0 | HipMid | 12、16 的中點（**不用** SpineBase 0） | 23、24 的中點 |
| 1 | ShoulderMid | 4、8 的中點（**不用** SpineShoulder 20） | 11、12 的中點 |
| 2 | Head | 3 Head | 7、8 兩耳中點（比鼻子 0 更接近 Kinect 頭部中心） |
| 3 / 4 | ShoulderL / R | 4 / 8 | 11 / 12 |
| 5 / 6 | ElbowL / R | 5 / 9 | 13 / 14 |
| 7 / 8 | WristL / R | 6 / 10 | 15 / 16 |
| 9 / 10 | HipL / R | 12 / 16 | 23 / 24 |
| 11 / 12 | KneeL / R | 13 / 17 | 25 / 26 |
| 13 / 14 | AnkleL / R | 14 / 18 | 27 / 28 |
| 15 / 16 | FootL / R | 15 / 19 | 31 / 32（foot index） |

  - 兩個中點在兩邊用**同一種算法**：Kinect 的 SpineBase、SpineShoulder 是它自己估的點，和「兩髖中點」位置不同，直接用會造成系統偏移。
  - 不納入：SpineMid、Neck、Hand、HandTip、Thumb（MediaPipe 沒有對應點）。
  - Kinect 與 MediaPipe 的髖關節解剖位置本來就不同，即使轉換正確也會有系統偏移，這是預期的，由第 4.2 節的「系統偏差 b」量化。
- 共同骨架的骨架圖（鄰接矩陣，給 CTR-GCN 用）：HipMid–ShoulderMid、ShoulderMid–Head、ShoulderMid–ShoulderL/R、Shoulder–Elbow–Wrist、HipMid–HipL/R、Hip–Knee–Ankle–Foot；三個子集（自身、向心、離心）的建法與現有 25 關節圖相同，中心點用 HipMid。
- `kinect_to_common(x25) -> (T, 17, 3)`、`mediapipe_to_common(x33, visibility) -> (T, 17, 3)`。中點的 visibility 取兩端點的最小值。
- 座標軸：MediaPipe world 的 y 向下，必須翻轉；z（和 x）的方向**以單元測試確認，不要靠記憶**：Es1 舉臂時手腕 y 要變大；人站在鏡頭前時，鼻子相對髖部的 z 符號要與 Kinect 一致。
- 正規化沿用 Kinect 前處理第二版的做法（只用開頭 10 幀決定原點與方向，整段套用同一個轉換），但**全部改用共同骨架的點**，兩邊完全相同：
  - **原點**：開頭 10 幀 HipMid 的平均。
  - **方向**：開頭 10 幀的 HipL→HipR 向量定左右軸、HipMid→ShoulderMid 向量定垂直軸，正交化後得到旋轉矩陣。
  - **尺度**：整段 HipMid→ShoulderMid 長度的中位數。
  - 共同骨架流程**不使用** SpineBase、SpineShoulder；這兩個點只存在於原本的 25 關節流程（v6 baseline），那邊維持不動。
- 單元測試：左右沒對調、中點位置正確、正規化後專家站姿的軀幹長 = 1。
- 缺漏：visibility < 0.5 的點視為缺漏，短缺漏（≤ 5 幀）線性內插，更長的保留為 NaN 並記錄。

### 3.3 左右與鏡像檢查（每部影片）

對齊後，計算 MediaPipe 左腕軌跡與 Kinect 左腕、右腕軌跡的相關。若「同側」相關低於「對側」，標記為鏡像。多數影片鏡像 → 整批左右翻轉後重跑；少數影片鏡像 → 逐部列出檢查原因，不靜默修正。

### 3.4 時間對齊（`align.py`）

同一次錄影的兩路資料，理論上只差一個固定時間偏移，所以**以固定偏移為主、DTW 為備案**。DTW 會把真實的動作差異扭掉，讓角度誤差看起來比實際小。

1. Kinect 依時間戳、MediaPipe 依影片時間，都重新取樣到 30 Hz 均勻時間軸。
2. 對齊訊號：雙腕、雙肩垂直位置（各自 z-score 後加總），兩邊都量得到、對深度不敏感。
3. 在 ±5 秒範圍內用互相關找固定偏移，記錄峰值相關 ρ。
4. 漂移檢查：前半段、後半段各找一次偏移，相差 > 2 幀 → 改成「偏移 + 線性時間縮放」。
5. ρ < 0.7 → 標記「對齊不可靠」，改用 DTW（Sakoe-Chiba 帶寬 ±0.5 秒）並另外分組。
6. 角度結果分成「固定偏移組」與「DTW 組」報告；**決策只用固定偏移組**，DTW 組當敏感度分析。

### 3.5 K-common baseline（`train_rehab.py --skeleton common`）

- 資料：v3 中有 RGB 的 353 筆，Kinect 轉共同骨架。
- fold：seed 42 的受試者分層 5 折，限縮到 353 筆（受試者不換折）。
- 訓練：v6 設定（平均最後 10 輪權重 + 重算 BN），seed 42、43、44，RTX 3080 Ti。
- 得出：MAE_K（3 次平均 ± std）、σ_seed、新門檻 **T = 2 × σ_seed**、353 筆的盲猜 MAE_blind。

### 3.6 配對評分比較（Q1）

- 每部影片 v、每個 seed s：用 v 所在測試折的 K-common 模型分別預測 Kinect 骨架 K_s(v) 和 MediaPipe 骨架 M_s(v)。測試時一樣取 5 個片段平均。
- MediaPipe 偵測失敗太多的影片（見 3.8）兩邊都排除，兩邊的樣本必須相同。

### 3.7 角度比較（Q2、Q3）

- 用特徵函式庫的公式（三點角度、身體段傾角等），兩邊各算一次 Es1～Es5 用到的角度。
- 每個特徵兩種層級：
  - **逐幀**：角度 MAE（描述用）。
  - **每部錄影的摘要值**（決策用）：特徵函式庫實際拿來比對門檻的量，例如每次重複的最大角度平均、活動範圍（ROM）。
- 軀幹旋轉（Es3）：肩連線相對骨盆連線在水平面上的夾角。側彎（Es2）：軀幹段相對垂直軸的側傾角，作為**對照組**。

### 3.8 排除規則（事前訂定）

- 偵測失敗或缺漏幀 > 20% 的影片從配對分析排除，列入 `excluded.csv` 並註明原因，排除數量寫進報告。
- 只因資料品質排除，不因落差大排除。落差特別大的影片要逐部看 RGB 找原因，結果寫進報告。

## 4. 判斷標準（凍結）

符號：MAE_M = K-common 模型吃 MediaPipe 骨架的 MAE（3 seed 平均）；ΔMAE = MAE_M − MAE_K；Adv = MAE_blind − MAE_K（模型相對盲猜的優勢，v6 約 2.4）。

### 4.1 Q1 評分落差 → 訓練方式

先定「雜訊底線」：同一部影片只換 seed 本來就會差很多（v5 最多差 15 分），所以單部影片的落差要和這個比。

- N = 所有影片、所有 seed 配對 (s, s′) 的 |K_s(v) − K_s′(v)| 中位數（Kinect 自己換 seed 的差異）
- G = 所有影片、所有 seed 的 |M_s(v) − K_s(v)| 中位數（換骨架來源的差異）

| 判定 | 條件 | 做法 |
| --- | --- | --- |
| **落差小** | ΔMAE ≤ T **且** G ≤ 1.5 N | Kinect 資料加模擬 MediaPipe 抖動訓練（抖動統計量取自本實驗的 M − K 殘差），或用 MediaPipe 資料微調；兩者都跑，取較好者 |
| **落差大** | ΔMAE ≥ 0.5 × Adv **或** G ≥ 3 N | 直接用 MediaPipe 共同骨架從頭訓練（3 seed，同一組 fold） |
| **中間** | 其他情況 | 先做 MediaPipe 微調；微調後 ΔMAE 仍 > T，改從頭訓練 |

附加警示（不改變判定，但要寫進報告）：真實 10–25 分組的偏差，MediaPipe 比 Kinect 多 2 分以上 → 標記「重症高估惡化」，之後的訓練方式須特別檢查此組。

### 4.2 Q2 特徵落差 → 特徵函式庫門檻

每個特徵 f 的摘要值（固定偏移組）：

- 系統偏差 b = mean(M − K)：可以靠重新校準修正
- 隨機誤差 r = SD(M − K)：無法靠校準修正
- 專家分布寬度 s_E = Kinect 專家組該特徵的 SD（正常範圍為 ±2 s_E）

| 判定 | 條件 | 做法 |
| --- | --- | --- |
| **可沿用** | \|b\| ≤ 0.5 s_E 且 r ≤ 0.5 s_E | 直接沿用 Kinect 門檻 |
| **需重新校準** | r ≤ 0.5 s_E 但 \|b\| > 0.5 s_E | 用 MediaPipe 版專家角度分布重新定正常範圍 |
| **手機版降級** | 0.5 s_E < r ≤ 1 s_E | 保留，但只做事後報告、不做即時提示，並標「低信心」 |
| **手機版停用** | r > 1 s_E | 雜訊超過正常範圍一半寬，手機版不使用此特徵 |

### 4.3 Q3 旋轉 → 模組 C 範圍

Es3 軀幹旋轉的每部錄影 ROM（固定偏移組）：

| 判定 | 條件 | 做法 |
| --- | --- | --- |
| **撐得住** | Pearson r ≥ 0.8 且隨機誤差 ≤ 0.5 s_E 且逐幀 MAE ≤ 10° | 模組 C 做旋轉 + 側彎 |
| **撐不住** | r < 0.6 或隨機誤差 > 1 s_E | 模組 C 只做側彎 |
| **中間** | 其他情況 | 自錄 30°～45° 斜角影片重測（同時錄手機 + 現有 Kinect／深度鏡頭）；過「撐得住」標準 → 規定斜角拍攝；沒過 → 只做側彎 |

**對照組檢查**：Es2 側彎套用同一套標準。若側彎也「撐不住」，代表問題出在流程（對齊、正規化、座標軸），不是 MediaPipe 的深度推估。必須先除錯，Q3 不判定。

注意：KIMORE 是正面拍攝，對旋轉來說是最不利的角度；這裡的 Es3 是站姿，KERAAL 是坐姿，結果只是代理指標。

## 5. 解讀限制（寫進報告）

- Kinect 不是真值：它本身也有誤差，旋轉尤其明顯。本實驗量的是「兩種裝置的差異」，不是「MediaPipe 的誤差」。
- 落差大不代表 MediaPipe 版一定比較差：從頭訓練後，MAE_M 仍可能接近 MAE_K。最後以 MediaPipe 版模型自己的交叉驗證 MAE 為準。
- 只有 353/380 筆有 RGB，被排除的 27 筆組別分布要列出來，確認沒有集中在某組。

## 6. 輸出

| 檔案 | 內容 |
| --- | --- |
| `data/kimore/mediapipe/raw/*.npz`、`extract_log.csv` | 原始擷取 |
| `data/kimore/kimore_mp_common_v1.pkl`、`kimore_k_common_353.pkl` | 兩邊共同骨架資料（之後訓練直接用） |
| `runs/v7_kcommon_s{42,43,44}/` | K-common baseline |
| `runs/v7_gap/paired_preds.csv` | 每部影片 × seed：K_s、M_s、真實分數、組別、動作 |
| `runs/v7_gap/angles_summary.csv` | 每部影片 × 特徵：K、M 摘要值、對齊方式、ρ |
| `runs/v7_gap/gap_report.md` | 第 4 節三個判定的實際數值與結論，逐條對照標準 |
| 圖 | Bland-Altman（每個特徵）、M vs K 散佈圖（TS、Es3 ROM、Es2 ROM）、各組別／動作的 ΔTS 箱形圖、對齊 ρ 分布 |

## 7. 執行順序（給 Claude Code 的任務清單）

**環境**：全部在實驗室 server（RTX 3080 Ti）上做，透過 VS Code Remote-SSH 連線、在 server 上開 Claude Code；長時間工作用 tmux。RGB 影片用 rsync 傳到 server 上 `chmod 700` 的資料目錄。要互相比較的訓練都在同一張 3080 Ti 上跑。

**給 Claude Code 的規則**：
- 一次做一個任務，每個任務完成「驗收」後停下來回報，不要自動進入下一個。
- 不得修改第 4 節的判斷標準數字；若認為標準有問題，寫在回報裡由使用者決定。
- 不得把 `data/`、`runs/` 加入 git；新增檔案前確認 `.gitignore`。
- 不得因為落差大而排除影片；排除只依第 3.8 節的品質規則。
- 現有程式（`preprocess_kimore.py`、`dataset.py`、`train_rehab.py`、`model/rehab_ctrgcn.py`）以「加參數、不改預設行為」的方式擴充，確保 v6 baseline 仍可重現。

| # | 任務 | 產出 | 驗收 |
| --- | --- | --- | --- |
| T0 | 建 `rgb_manifest.csv`：比對 RGB 檔案與 v3 樣本，標上 fold（seed 42） | `data/kimore/rgb_manifest.csv` | 筆數 = 353；列出 v3 中沒有 RGB 的 27 筆及其組別分布 |
| T1 | `run_mediapipe.py` 試跑 5 部（不同動作、不同組別） | 5 個 npz + `extract_log.csv` | 幀數與影片長度相符；時間戳單調遞增；畫出 1 部的 `pose_landmarks` 疊在 RGB 上目視正確；回報每部執行時間，估算 353 部總時間 |
| T2 | 353 部批次擷取（tmux 背景執行，多行程平行，可續跑：已存在的 npz 跳過） | `data/kimore/mediapipe/raw/` | 353 部都有輸出或在 log 中註明失敗原因 |
| T3 | `common_skeleton.py`：`COMMON_JOINTS`、兩個轉換函式、正規化、共同骨架鄰接矩陣（與 T2 並行） | 程式 + `tests/test_common_skeleton.py` | 單元測試全過（見下方清單），並用 T1 的 5 部真實資料跑過 |
| T4 | 產生共同骨架資料集 | `kimore_k_common_353.pkl`、`kimore_mp_common_v1.pkl` | 兩檔樣本 id 與 fold 完全相同；印出每檔的形狀與缺漏比例 |
| T5 | K-common baseline × 3 seed（`train_rehab.py --skeleton common`） | `runs/v7_kcommon_s{42,43,44}/` | 回報 MAE_K ± std、T、MAE_blind（353 筆）、N |
| T6 | `align.py`：重取樣、互相關偏移、漂移檢查、DTW 備案、鏡像檢查 | 每部影片的偏移、ρ、對齊方式、是否鏡像 | 先抽 5 部畫 K 與 M 的腕部軌跡疊圖給使用者目視確認，通過後再跑全部；回報 ρ 分布與 DTW 組筆數 |
| T7 | 配對評分（out-of-fold）與角度比較 | `paired_preds.csv`、`angles_summary.csv`、圖 | 兩邊樣本相同；排除清單完整 |
| T8 | `gap_report.md`：逐條對照第 4 節標準填入數值與判定 | 報告 | 三個判定（Q1、Q2、Q3）都有明確結論，或明確寫出「Q3 因對照組失敗不判定」 |

**T3 單元測試清單**：
1. 左右沒對調：Kinect 與 MediaPipe 各取一幀「舉左手」的資料（T1 的 Es1 影片中找），轉換後 WristL 的 y 都高於 WristR。
2. 座標軸方向：轉換後 y 向上（MediaPipe 原始 y 向下，須翻轉）；人面向鏡頭時，兩邊 Head 相對 HipMid 的 z 符號一致。z、x 方向以真實資料測，不靠記憶。
3. 中點正確：HipMid 等於 HipL、HipR 的平均（誤差 < 1e-6）。
4. 正規化：專家站姿正規化後，HipMid→ShoulderMid 長度中位數 = 1；開頭 10 幀的 HipMid 平均在原點。
5. 形狀與 NaN：輸出 (T, 17, 3)；visibility < 0.5 的點、短缺漏內插後無 NaN，長缺漏保留 NaN 並回報。

**之後**：依 `gap_report.md` 的判定更新研究歷程紀錄第十二節，進入正式訓練。

## 8. 換成 MediaPipe 的條件（與研究歷程紀錄一致）

① 部位分解第 1 關有結果；② 本實驗 `gap_report.md` 完成。兩者都達成才把主線切到 MediaPipe 骨架。

## 9. 凍結紀錄

- 判斷標準 commit：`________`（日期：____）
- 追加修改（只能追加，需註明日期與原因）：

### 追加 1（2026-11-20，依 T0 回報）

第 4 節判斷標準的數字不變。以下追加內容優先於前文中衝突的敘述。

**A. 樣本數：本實驗的分析集為 347 筆（取代前文的「353 筆」）**

- RGB 共 353 個 mp4，其中 347 個對得上 v3 樣本，`rgb_manifest.csv` 共 347 列。
- 對不上的 6 個：E_ID17 Es1～Es5（v3 缺評分）、NE_ID24 Es2（缺 JointPosition，沒有骨架可配對）。這 6 部仍跑 MediaPipe 擷取（T2），但不進配對分析與 K-common 訓練。
- v3 380 筆中沒有影片的 33 筆（取代前文的「27 筆」），都是資料集本身沒有釋出影片，不是配對失敗。
- 「353」在第 3.5、3.6、4、6、7 節中一律改讀為「347」；輸出檔名 `kimore_k_common_353.pkl` 改為 `kimore_k_common_347.pkl`。
- fold 不重新切分：沿用 seed 42 受試者分層 5 折，限縮後五折各 72、71、67、66、71 筆。
- 只使用 mp4。`B_ID1/Es1/rgb/*.mxf`（同一段影片的中間輸出檔）與 `NE_ID24/Es5/*.lnk`（Windows 捷徑）排除。

**B. 解讀限制（追加到第 5 節）**

- 缺影片集中在 Parkinson（13/80，16.3%）與 Stroke（6/49，12.2%），其他組 ≤ 7.5%。其中包含 3 筆重症樣本（P_ID2 Es2 TS 10、P_ID8 Es2 TS 14、P_ID2 Es3 TS 15）。
- 報告須列出 347 筆中真實 10–25 分的樣本數。若少於 20 筆，第 4.1 節的「重症高估惡化」警示只做描述、不下結論。
- K-common（347 筆）的 MAE 不能和 v6（380 筆）直接比較，因為樣本組成不同（病患組比例較低）。這也是第 3.5 節要在同一批 347 筆上重算 baseline 與門檻的原因。
- RGB 為 960×540，低於一般手機錄影解析度，MediaPipe 的表現可能偏保守。

**C. 時間軸：改以「影片幀 ↔ Kinect 幀」對應，取代第 3.1、3.4 節中「影片時間由 PTS 取得」的做法**

T0 發現影片幀數與 Kinect 骨架幀數幾乎相同（比值中位數 1.009），但影片標示 24／25 fps，Kinect 名義上是 30 Hz。最可能的解釋是一個 Kinect 幀對應一個影片幀，再以 25 fps 重新封裝，所以 PTS 不是真實時間。處理方式：

1. **假設待驗證**：影片第 i 幀 ↔ Kinect 第 i + k 幀（k 為固定幀偏移）。真實時間一律取 Kinect TimeStamp CSV，影片幀繼承對應 Kinect 幀的時間。
2. **驗證所需資料**：把 KIMORE 原始骨架資料夾中的 `JointPosition*.csv`、`TimeStamp*.csv` 與 Label 同步到 server 的 `data/kimore/raw/`（不含 depth 等大檔），同一資料夾結構。
3. **v3 裁切位置**：v3 清洗時有些錄影只取了最長的連續片段（例如 E_ID14 Es2：影片 669 幀，v3 骨架 182 幀），所以 v3 的第 0 幀不一定是原始 CSV 的第 0 幀。需要每筆 v3 樣本在原始 CSV 中的起訖幀號：若 `preprocess_kimore.py` 沒有存，就以「加參數、不改預設輸出」的方式補存，或另寫腳本以原始 CSV 比對 v3 陣列求出。
4. **驗證方式（T6 第一步，先於全面對齊）**：
   - 對 v3 骨架未裁切的錄影，比較「影片幀數」與「原始 CSV 幀數」。若多數只差 0～5 幀，即支持 1:1 假設。
   - 抽 5 部，在幀單位上做互相關，看最佳偏移 k 是否接近 0，且峰值 ρ 高。
   - 1:1 成立 → 第 3.4 節的互相關改以「幀」為單位搜尋偏移；搜尋範圍放寬到整段（以涵蓋 v3 裁切），不再限 ±5 秒。漂移檢查改為比較前後半段的 k。
   - 1:1 不成立 → 回報使用者，暫停 T6，不自行改用其他假設。
5. **MediaPipe 擷取（T1、T2）不受影響，可以先跑**：每幀存「影片幀號（0 起算）」與 PTS 兩者。VIDEO 模式需要的時間戳一律用 `frame_idx × 1000 / 30` ms（模擬真實 30 Hz，讓 MediaPipe 的追蹤平滑行為接近手機即時使用），不用 PTS。

**D. 其他**

- `.gitignore` 目前涵蓋 `data/kimore/`、`data/keraal/`、`runs/`；本實驗所有資料都在 `data/kimore/` 底下，已足夠。為保險起見另加一行 `data/`。
- `scripts/build_rgb_manifest.py` 確認後 commit。
