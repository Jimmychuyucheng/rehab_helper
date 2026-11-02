# HANDOFF v2：輸出範圍限制 + 低分樣本加權，重跑 5 折

## 背景（baseline 結果，runs/baseline）
- 5 折 MAE 6.31 ± 0.53，baseline（依動作平均）8.32，5/5 折勝出，ρ 0.64。
- 問題一：輸出沒有範圍限制。fold 2 有一筆預測 −17.8（TS 最低是 10），另有 5 筆 > 50。
- 問題二：預測往中間擠（pred 對 true 的斜率只有 0.55）。五折合併的 376 筆測試預測：

| 真實 TS | 筆數 | MAE | 平均偏差 (pred − true) |
|---|---|---|---|
| 10–25 | 51 | 10.79 | +9.37 |
| 25–35 | 80 | 7.92 | +3.50 |
| 35–45 | 130 | 4.74 | −0.85 |
| 45–50 | 115 | 4.96 | −4.31 |

  → 平衡 MAE（四個區間 MAE 的平均）= 7.10。以人為單位（同一人五個動作平均後比較）MAE = 4.60。

## 步驟 0：先 commit 目前狀態
- 把 `runs/` 加進 `.gitignore`（模型檔與預測不進 repo）。
- commit `train_rehab.py` 與 `model/rehab_ctrgcn.py`，訊息例如 "Add train_rehab.py baseline (5-fold MAE 6.31)"。

## 步驟 1：輸出限制在 10–50
- `train_rehab.py` 加參數 `--clip`（預設開啟，`--no_clip` 關閉）。
- 只在 `evaluate()` 裡把預測夾在 [10/50, 1.0]（正規化尺度；TS_MAX = 50）。**訓練時的 loss 不要 clamp**，以免梯度在邊界歸零。
- val 與 test 都套用，early stopping 與測試才一致。

## 步驟 2：低分樣本加權（緩解往中間擠）
- 加參數 `--reweight {none,inv,sqrt_inv}`，預設 `none`（保持 baseline 可重現）。
- 做法：只用**訓練集**的 TS，以 5 分為一格分箱（10–15, 15–20, …, 45–50，50 歸入最後一格）。每格權重 = (1/count)^p，inv 時 p = 1，sqrt_inv 時 p = 0.5。再正規化，讓訓練樣本的平均權重 = 1。
- 依 `samples[i]["ts"]` 建一張 index → weight 的查表；train batch 已有 `b["index"]`，可直接查。
- loss 改成逐樣本：`HuberLoss(reduction="none")`（MSE 同理），乘上權重後取平均。`weight=None` 時行為要與原本完全相同。
- `--overfit` 模式不受影響。

## 步驟 3：補評估指標（新舊 run 都要能算）
1. `metrics.json` 新增：
   - `bins`：四個區間（10–25, 25–35, 35–45, 45–50）各自的 n、MAE、bias；
   - `balanced_mae`：有樣本的區間 MAE 平均；
   - `slope`：pred 對 true 的線性迴歸斜率；
   - `subject_mae`：以人為單位的 MAE。
2. `summary.md` 新增一張「分數區間」表（各折 + 合併），並列出 balanced MAE、slope、subject MAE。
3. 新增 `analyze_runs.py`：
   - 讀任意 `runs/<name>/fold*/test_predictions.csv`，五折合併後印出上述所有指標。
   - 支援 `--clip`：事後把預測夾在 10–50 再算。
   - 可以一次比較多個 run，例如 `python analyze_runs.py baseline v2_clip_sqrtinv`，輸出 markdown 對照表，寫到 `runs/compare.md`。

## 步驟 4：驗證（每一步都要跑過）
1. **加權函式**：印出訓練集各分箱的 count 與權重，確認低分箱權重較大，平均權重 = 1。
2. **相容性**：`--reweight none --no_clip`，用 `--epochs 3 --folds 0 --name smoke_v2` 跑一次，確認流程不壞。跑完刪掉 `runs/smoke_v2`。
3. **overfit 測試**：`--overfit` 仍能把 MAE 壓到 < 1。
4. **事後 clamp**：`python analyze_runs.py baseline --clip` 算出「baseline + clip」的數字，作為步驟 1 單獨的效果。

## 步驟 5：重跑 5 折（背景執行）
```
caffeinate -i python train_rehab.py --name v2_clip_sqrtinv --reweight sqrt_inv > runs/v2_log.txt 2>&1
```
（clip 預設開啟，其餘超參數全部與 baseline 相同。）
- 跑的過程不要讀 log。跑完只讀 `runs/v2_clip_sqrtinv/summary.md`，並執行 `python analyze_runs.py baseline v2_clip_sqrtinv`，回報三欄對照：baseline、baseline+clip（事後）、v2。
- 預估 1.5–2 小時（baseline 約 1 小時 40 分）。

## 判斷標準
- **主要目標**：10–25 區間 MAE 明顯低於 10.79，偏差低於 +9.37；平衡 MAE 低於 7.10；斜率高於 0.55。
- **可接受代價**：整體 MAE 不高於 6.6（比 baseline 差不超過約 0.3）。5 折仍全部贏過 baseline。
- **若整體 MAE 變差很多**（> 6.8）：回報即可，先不要自行改超參數。下一步可考慮改用 `inv`，或依 balanced MAE 做 early stopping，由使用者決定。

## 不要做的事
- 不要改切分、seed、epochs、patience、lr 等超參數（為了和 baseline 公平比較）。
- 不要 commit `runs/` 或 `data/`。
