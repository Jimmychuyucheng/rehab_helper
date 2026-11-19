# HANDOFF v5：平均最後幾個 epoch 的權重 + baseline 與多任務各重複 3 次（今晚背景執行）

## 目的
- v4_ts 與 baseline 只差 3 筆標籤，MAE 卻是 6.80 對 6.23。目前「重跑一次」的起伏就有 0.5 以上，挑模型也容易挑到驗證曲線上偶然的低點。
- 今晚做兩件事：
  1. 加入新的挑模型方式：**平均最後 K 個 epoch 的權重**；
  2. baseline 用 3 個不同 seed 各跑一次，量出「同一設定重跑」的起伏有多大。
- 這週接著要深入研究 PO/CF，之後的比較都會用這個起伏當判斷門檻。

## 步驟 0
在目前的 `v4-po-cf` branch 上工作即可，先確認 working tree 乾淨。

## 步驟 1：新增「平均最後 K 個 epoch 權重」（`train_rehab.py`）
1. 新增參數 `--avg_k`，預設 10；`--avg_k 0` 代表關閉，行為與現在完全相同。
2. 訓練時用 `collections.deque(maxlen=K)` 保存最近 K 個 epoch 的權重（`state_dict` 複製到 CPU；模型約 2.6 MB，10 份沒有負擔）。
3. 訓練結束時（early stopping 或到達最大 epoch），把這 K 份權重逐參數取平均，得到平均模型。
4. **一定要重算 BatchNorm 統計量**：平均後的權重配舊的 BN running mean/var 會失準。
   - 用 `torch.optim.swa_utils.update_bn`，或自己在訓練集上跑一次 forward（`model.train()`、`torch.no_grad()`）。
   - **注意**：資料是 dict 格式（`b["x"]`），`update_bn` 不能直接吃 DataLoader，要包一層只回傳 x 的 iterable，或自己寫迴圈；用不做資料增強的訓練集 loader。
   - 整數型 buffer（`num_batches_tracked`）不要取平均，保留最後一份即可。
5. **同一次訓練同時評估兩種模型**，不用多跑：
   - `best`：現有方式（驗證 MAE 最低的 epoch，`best.pt`）；
   - `avg`：平均最後 K 個 epoch（另存 `avg.pt`）。
   - 平均模型也先在驗證集上算一次 MAE，寫進 log 和 metrics。
6. 輸出：
   - `test_predictions.csv` 維持原欄位（`y_pred` = **avg 模型**），另加一欄 `y_pred_best`（best 模型）。多任務時 PO/CF 欄位比照，加 `_best` 版本。
   - `metrics.json`：既有指標改為 avg 模型的結果，另外加 `"best_model": {mae, rmse, pearson, balanced_mae, slope, ...}`，以及 `avg_k`、平均涵蓋的 epoch 範圍、avg 模型的驗證 MAE。
   - `summary.md`：多一欄「best 模型 MAE」，方便並列比較。
7. 確認 `plot_runs.py` 和 `analyze_runs.py` 仍能讀新的 csv（它們讀 `y_true`、`y_pred`，所以畫的是 avg 模型）。

## 步驟 2：驗證（全部通過才進行步驟 3）
1. `--avg_k 0 --epochs 3 --folds 0 --name smoke_v5_off`：確認與原行為一致（沒有 avg 欄位、流程不壞）。
2. `--avg_k 3 --epochs 5 --folds 0 --name smoke_v5_on`：確認 avg.pt 有存、BN 有重算、兩組指標都有寫出。
3. sanity check：在 smoke_v5_on 中，把「只平均 1 份（K=1）且不重算 BN」的結果和最後一個 epoch 的模型比，預測應完全相同（可寫成小測試，不必留檔）。
4. smoke 資料夾跑完後刪除。

## 步驟 3：baseline 重複 3 次（背景依序執行，過程中不要讀 log）
使用修正後的資料、單任務、clip 開、reweight none，其餘超參數與 baseline 完全相同，只改 seed：
```
for s in 42 43 44; do caffeinate -i python train_rehab.py --name v5_base_s$s --seed $s --targets ts --avg_k 10 --pkl data/kimore/processed/kimore_full_pofix.pkl > runs/v5_base_s$s.log 2>&1; done
```
- 三次依序跑，預估共約 5 小時。請在一個背景指令裡完成整個迴圈。

**接著（同一個背景指令內，baseline 三次跑完後）跑多任務 3 次：**
```
for s in 42 43 44; do caffeinate -i python train_rehab.py --name v5_mt_s$s --seed $s --targets ts,po,cf --avg_k 10 --pkl data/kimore/processed/kimore_full_pofix.pkl > runs/v5_mt_s$s.log 2>&1; done
```
- 建議把兩個迴圈寫進同一個 shell script（例如 `runs/run_v5.sh`），用一個背景指令執行，總共約 10 小時。
- 其中某一次失敗時，不要中斷後面的 run（每次用 `|| true` 或在迴圈內記錄錯誤），全部結束後再回報哪一次失敗。
- 注意：MPS 本身不完全可重現，所以 seed 42 這次的結果不一定等於之前的 baseline，這是預期的。

## 步驟 4：彙整（六次都跑完後）
1. `analyze_runs.py` 新增 `--group`：把多個 run 視為同一設定的重複，輸出：
   - 每個 run 的五折平均 MAE、平衡 MAE、斜率、10–25 分偏差、以人為單位 MAE；
   - **三次之間的平均 ± 標準差**（這就是「重跑起伏」）；
   - best 與 avg 兩種挑模型方式各一份。
   寫到 `runs/compare_v5_seeds.md`。
2. 也列出每個 fold 在三次之間的 MAE 範圍（最小～最大），看 fold 4 是否特別不穩。
3. **baseline 組 vs 多任務組**（`runs/compare_v5_seeds.md` 內另一節）：
   - TS（多任務列 ts_head 與 ts_sum 兩種）：兩組各自的「三次平均 ± 標準差」，以及差距是否超過兩組標準差的 2 倍；
   - 多任務組：PO、CF 的 MAE、Pearson 與盲猜，三次平均 ± 標準差，以及各 run 是否五折都贏盲猜；
   - 多任務組的「哪一類比較嚴重」：在 |PO/15 − CF/35| > 0.20 的子集，列出模型猜對率，以及兩種盲猜（一律猜多數方向、依動作種類猜）的猜對率，三次各一列；同時列出「真實 PO 較差」與「真實 CF 較差」各自的猜對數。
4. 畫圖：
   ```
   python plot_runs.py v5_base_s42 v5_base_s43 v5_base_s44 --pkl data/kimore/processed/kimore_full_pofix.pkl
   mv runs/compare_plots runs/compare_plots_v5_base
   python plot_runs.py v5_base_s42 v5_mt_s42 --pkl data/kimore/processed/kimore_full_pofix.pkl
   mv runs/compare_plots runs/compare_plots_v5_base_vs_mt
   python plot_runs.py v5_mt_s43 v5_mt_s44 --pkl data/kimore/processed/kimore_full_pofix.pkl
   ```
   （新 run 已經 clip，不需要 `--clip`。）

## 回報內容
只讀 `runs/compare_v5_seeds.md` 回報：
1. avg 模型：三次的 MAE 平均 ± 標準差；best 模型：同上。
2. avg 是否比 best 穩（三次之間的標準差較小）、是否比較準。
3. 各 fold 在三次之間的起伏範圍。
4. 建議之後的「有意義差距」門檻（例如三次標準差的 2 倍）。
5. 多任務 vs baseline：TS 是否有超過門檻的差異；PO/CF 是否三次都贏盲猜；「哪一類比較嚴重」是否三次都高於依動作種類的盲猜。
6. 是否有任何一次 run 失敗。

## 不要做的事
- 不要改切分、lr、epochs、patience 等超參數；`--avg_k 0` 時要能重現舊行為。
- 不要覆蓋 `runs/baseline`、`runs/v4_*`。
- 不要 commit `data/` 或 `runs/`；完成後 commit 程式碼，訊息例如 "Add last-K weight averaging and seed-repeat analysis"。
