# HANDOFF v4：PO 標籤修正 + TS/PO/CF 多任務模型

## 背景
- 目前最佳結果：baseline + clip，5 折 MAE **6.23 ± 0.39**（`runs/baseline`，事後 clip）。主要問題是預測往中間擠（斜率約 0.55）。
- KIMORE 作者 Luca Romeo 回信（2026-11-11、11-13）：
  1. 沒有十題逐項分數，建議改做 **PO 與 CF 分開預測的多輸出模型**。
  2. 確認 PO < 3 不符合 EAAQ 計分規則（三題各 1–5 分，最小 3），建議**補成 3，不要刪除**，並寫進方法論。
- 資料檢查：PO 與 CF 相關 0.71；23% 的錄影兩者百分比差超過 20%，所以分開預測有額外資訊。

## 步驟 0：先 commit 目前狀態
- 確認 working tree 乾淨（若有 KERAAL 的未完成修改，先另開 branch 或 commit，不要混進這次）。
- 本次所有修改在新 branch：`git checkout -b v4-po-cf`。

## 步驟 1：前處理加入 PO 修正，產生新版 pkl
1. 在 `preprocess_kimore.py` 加入函式 `fix_scores(sample)`，在組 sample 時呼叫：
   - 若 `po < 3`：記下原值，設 `po = 3`，並重算 `ts = po + cf`。
   - 每筆修正都寫進 `sample["label_fix"] = {"po_orig": ..., "ts_orig": ...}`，沒修正的為 `None`。
   - 其他情況一律不動（例如 Expert/E_ID4 Es5 的 TS 與 PO+CF 差 0.67，維持原樣）。
2. 輸出到**新檔** `data/kimore/processed/kimore_full_pofix.pkl`，不要覆蓋舊的 `kimore_full.pkl`（舊結果要能重現）。
3. `meta` 加入 `"label_fixes"`：列出被修正的 subject_key、exercise、原值與新值，以及說明文字
   "PO<3 set to 3 per KIMORE author (L. Romeo, 2026-11-13); TS recomputed as PO+CF"。
4. `train_rehab.py` 的 `--pkl` 預設值暫時不改（仍指向舊檔），本次實驗用參數明確指定新檔。

**驗證（必須全部通過並在回報中列出）**
- 新舊 pkl 樣本數相同（376），而且順序、subject_key、exercise、骨架完全一致。
- 恰好 3 筆標籤改變：
  | 錄影 | PO | TS |
  |---|---|---|
  | BackPain/B_ID3 Es3 | 0.43 → 3 | 25.18 → 27.75 |
  | BackPain/B_ID3 Es5 | 0.01 → 3 | 16.25 → 19.24 |
  | BackPain/B_ID4 Es5 | 1.91 → 3 | 32.12 → 33.20 |
- 修正後：所有 PO ≥ 3、CF ≥ 7、TS ≥ 10；除了 E_ID4 Es5，其餘 |TS − (PO + CF)| < 0.01。

## 步驟 2：用修正後的標籤重算 baseline 指標（不用重訓）
1. `analyze_runs.py` 加參數 `--pkl`：讀 `test_predictions.csv` 時依 `index` 到 pkl 取**修正後**的真實分數，取代 csv 內的 `y_true`。
2. 執行並存成 `runs/compare_v4_baseline.md`：
   ```
   python analyze_runs.py baseline --clip --pkl data/kimore/processed/kimore_full_pofix.pkl
   ```
3. 同時算出 PO、CF 的**盲猜 baseline**（每折用訓練集各動作的平均分，與 TS 的作法相同；`dataset.mean_baseline` 已有 `target` 參數）。

**預期**：只有 3 筆標籤變動，「baseline + clip」的 MAE 應與 6.23 相差不到 0.05。差異過大代表 index 對應出錯，要先停下來查。

## 步驟 3：TS + PO + CF 多任務模型
### 3.1 模型與 loss
- `model/rehab_ctrgcn.py`：讓輸出層支援 `num_class=3`，`init_bias` 改成可傳入長度 3 的向量（各自是訓練集 TS/50、PO/15、CF/35 的平均）。`num_class=1` 時行為與現在完全相同。
- `dataset.py` 已支援 `targets=("ts","po","cf")`（各自除以滿分正規化到 0–1），直接使用。
- `train_rehab.py` 新增 `--targets`，預設 `ts`（保持 baseline 可重現）。多任務時：
  - loss = 三個輸出各自 Huber（δ = 0.1，正規化尺度）的平均，權重相同；
  - **early stopping 仍依驗證集的 TS MAE**（與 baseline 相同標準，比較才公平）；
  - 評估時 clip 到各自的合理範圍：TS [10, 50]、PO [3, 15]、CF [7, 35]。
- 同一個模型可以得到兩種 TS 估計，兩個都要報告：
  - `ts_head`：TS 輸出頭直接預測；
  - `ts_sum`：PO 預測 + CF 預測（clip 後相加，再 clip 到 [10, 50]）。

### 3.2 輸出格式
- `test_predictions.csv` 欄位：`index, subject_key, subgroup, exercise, y_true, y_pred`（= TS，保持舊工具相容），再加 `y_true_po, y_pred_po, y_true_cf, y_pred_cf, y_pred_ts_sum`。
- `metrics.json`：TS 的所有既有指標，再加 PO、CF 各自的 MAE、RMSE、Pearson、盲猜 baseline，以及 `ts_sum` 的 MAE。
- `summary.md`：再加一張「PO / CF」表（各折 + 平均，含盲猜 baseline）。

### 3.3 驗證（每一步都要跑過）
1. `--overfit --targets ts,po,cf`：16 個樣本三個輸出的 MAE 都能壓到很低（TS < 1）。
2. `--targets ts --epochs 3 --folds 0 --name smoke_v4_ts`：確認單任務行為沒被改壞。跑完刪除。
3. `--targets ts,po,cf --epochs 3 --folds 0 --name smoke_v4_mt`：確認流程不壞。跑完刪除。

### 3.4 正式訓練（背景執行，過程中不要讀 log）
```
caffeinate -i python train_rehab.py --name v4_mt --targets ts,po,cf --pkl data/kimore/processed/kimore_full_pofix.pkl > runs/v4_mt_log.txt 2>&1
```
其他超參數全部與 baseline 相同（clip 開、reweight none、seed、切分、lr、epochs、patience）。預估 1.5–2 小時。

### 3.5（建議）同資料的單任務對照
多任務跑完後再跑一次單任務，讓「唯一的差別就是多任務」：
```
caffeinate -i python train_rehab.py --name v4_ts --targets ts --pkl data/kimore/processed/kimore_full_pofix.pkl > runs/v4_ts_log.txt 2>&1
```
時間不夠可以先跳過，用步驟 2 重算的 baseline 當對照。

## 步驟 4：分析（擴充 `analyze_runs.py`）
1. 三欄對照 `runs/compare_v4.md`：baseline+clip（修正標籤）、v4_ts（若有）、v4_mt（ts_head 與 ts_sum 各一列）。TS 的既有指標全部列出：MAE、平衡 MAE、斜率、10–25 分偏差、以人為單位 MAE。
2. PO / CF 表：MAE、Pearson、盲猜 baseline，以及百分比 MAE（PO/15、CF/35），方便兩者互相比較。
3. **「哪一類問題比較嚴重」判斷力**（最重要）：
   - 取真實值 |PO/15 − CF/35| > 0.20 的錄影（約 86 筆）；
   - 計算模型預測的 sign(PO%pred − CF%pred) 與真實 sign 一致的比例（猜對率）；
   - 同時列出盲猜在這個子集的猜對率。若猜「多數方向」也很高，要特別註明。
4. 一致性：|ts_head − ts_sum| 的平均，以及兩者哪個 MAE 較低。

## 步驟 5：畫圖（`plot_runs.py`，使用者已提供，放在專案根目錄）
- 需要 matplotlib（沒有就 `pip install matplotlib`）。這支程式只讀 `runs/`，不改任何訓練結果。
- `test_predictions.csv` 的多任務欄位名稱必須跟步驟 3.2 一致（`y_true_po`、`y_pred_po`、`y_true_cf`、`y_pred_cf`），圖 07 才會自動畫出。
- 依序執行：
  ```
  python plot_runs.py baseline v2_clip_sqrtinv --clip --pkl data/kimore/processed/kimore_full.pkl
  python plot_runs.py baseline v4_mt --clip --pkl data/kimore/processed/kimore_full_pofix.pkl
  ```
  （有跑 v4_ts 的話，第二行加上 `v4_ts`。第二行會覆蓋第一行產生的 `runs/baseline/plots` 和 `runs/compare_plots`，這是預期行為：之後都以修正後的標籤為準。）
- 若第一行的對照圖要保留，執行第二行前先把 `runs/compare_plots` 改名為 `runs/compare_plots_v2`。
- 每個實驗輸出到 `runs/<name>/plots/`（含 README.md 可一次預覽），多個實驗的對照圖在 `runs/compare_plots/`。
- 回報時附上 `runs/v4_mt/plots/07_po_cf_and_ts_sum.png`、`04_score_bins_mae_bias.png`、`runs/compare_plots/cmp_score_bins.png` 的路徑。

## 判斷標準（回報時逐條說明是否達成）
| # | 標準 | 意義 |
|---|---|---|
| 1 | v4_mt 的 TS MAE ≤ 6.5，而且 5 折都贏 TS 盲猜 | 多任務沒有拖累總分（差異在約 ±0.5 的正常起伏內） |
| 2 | PO、CF 的 MAE 在 5 折都低於各自的盲猜，Pearson ≥ 0.5 | 兩個新輸出真的有學到東西 |
| 3 | 在 PO、CF 差異大的子集，「哪一類比較嚴重」的猜對率 ≥ 65%，且明顯高於盲猜 | 模型能分辨「目標沒達成」和「姿勢代償」，可以接到建議功能 |
| 4（加分） | TS 的平衡 MAE < 7.01，或斜率 > 0.55 | 多任務對重症低估有幫助 |

- 1、2 達成而 3 沒達成：PO/CF 輸出只能當參考，不能用來決定給哪一類建議，要回報子集大小與混淆情況。
- 1 沒達成（TS 變差超過 0.5）：回報即可，不要自行調整 loss 權重；下一步由使用者決定（例如改用 ts_sum，或調整權重）。

## 回報內容
只讀 `runs/v4_mt/summary.md`、`runs/compare_v4_baseline.md`、`runs/compare_v4.md`，回報：
1. 步驟 1 的驗證結果（3 筆修正、其餘不變）；
2. 修正標籤後的 baseline 數字；
3. 判斷標準 1–4 各自是否達成，附關鍵數字。

## 不要做的事
- 不要覆蓋或刪除 `kimore_full.pkl`、`runs/baseline`、`runs/v2_clip_sqrtinv`。
- 不要改 `train_rehab.py` 的預設行為（不加參數時要能重現 baseline）。
- 不要改切分、seed 或其他超參數。
- 不要 commit `data/` 或 `runs/`。完成後在 `v4-po-cf` branch 上 commit 程式碼，訊息例如 "Add PO label fix and TS/PO/CF multi-task training"。
