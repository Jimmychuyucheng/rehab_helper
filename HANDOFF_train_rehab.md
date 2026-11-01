# 任務交接：撰寫 `train_rehab.py` 並跑出基準結果

> 給 Claude Code：請先完整讀完本文件再動手。本文件說明專案現況、要做什麼、規格，以及每一步怎麼驗證。
> 有任何與本文件衝突的發現（例如檔案內容和這裡描述不同），先停下來回報，不要自行猜測。

---

## 0. 一句話目標

寫好 `train_rehab.py`，用**受試者 5-fold** 訓練現有的 Rehab CTR-GCN，確認測試 MAE **明顯低於「猜平均」baseline 的 8.32 分**（0–50 分制）。

---

## 1. 專案背景（必讀）

- 大三專題：骨架式復健動作評分。延伸學長的手語辨識 CTR-GCN（7 層，Alter CTRGC + 深度可分離 MSTCN），改成**迴歸**預測物理治療師的臨床總分 TS（0–50）。
- 資料集：**KIMORE**（Kinect v2，25 關節 3D 座標，30 Hz）。5 個動作：Es1 舉臂、Es2 軀幹側彎、Es3 軀幹旋轉、Es4 骨盆水平面旋轉、Es5 深蹲。每段錄影約 5 次重複。
- 分數：TS（0–50）、PO（0–15）、CF（0–35）。本任務**只預測 TS**。
- 硬體：**MacBook Air M3、16 GB、無風扇**。用 PyTorch MPS；長時間訓練會降頻。

---

## 2. 現有檔案（不要改動，除非本文件明確要求）

| 路徑（相對於專案根目錄） | 說明 |
|---|---|
| `preprocess_kimore.py` | v2 前處理，已完成並驗證。輸出 `data/kimore/processed/kimore_full.pkl` |
| `dataset.py` | 受試者分層 5-fold、隨機裁窗、資料增強、baseline 計算。**直接 import 使用，不要改切分邏輯** |
| `model/rehab_ctrgcn.py` | 模型 `Model`，輸入 `(N, 3, T=100, V=25)`，輸出 `(N, 1)` 分數；`forward(x, return_feature=True)` 另回傳 `(N, 128, T')` |
| `model/modules.py` | `MSTCN`、`TemporalConv`（深度可分離）等 |
| `model/graph/kimore_25_graph.py` | 25 點骨架鄰接矩陣，形狀 `(3, 25, 25)` |
| `data/kimore/raw/` | KIMORE 原始資料（唯讀） |

### `kimore_full.pkl` 格式
```python
{"meta": {...}, "samples": [ {
    "group", "subgroup", "subject", "subject_key",   # subject_key = "子組/受試者"，切分用
    "exercise" ("Es1"…), "exercise_id" (0–4),
    "skeleton": float32 (3, T_orig, 25),             # 已校準，保留原始長度
    "num_frames", "ts", "po", "cf", "align_params" }, ... ] }
```
已知數字：**376 筆、77 位受試者**。

### `dataset.py` 的 API（照這個用）
```python
from dataset import (load_kimore, make_split, build_dataloaders, RehabDataset,
                     SkeletonAugmentor, predict_windows, denormalize, mean_baseline, SCORE_MAX)

# 關閉資料增強時，傳入所有機率為 0 的 augmentor（傳 None 的話訓練集會自動套用預設增強）：
no_aug = SkeletonAugmentor(p_flip=0, yaw_deg=0, scale_range=None, noise_std=0, p_joint_mask=0)

loaders, split, samples = build_dataloaders(
    pkl_path, fold, n_folds=5, val_ratio=0.15, seed=42, batch_size=8,
    num_workers=0, out_frames=100, targets=("ts",),
    crop_ratio=(0.4, 0.6), eval_windows=5, eval_crop_ratio=0.5)
# loaders["train"] 每筆 batch：{"x": (B,3,100,25), "y": (B,1) 已除以 50, "exercise_id": (B,), "index": (B,)}
# loaders["val"/"test"]：x 為 (B, K=5, 3, 100, 25)，用 predict_windows 取 5 窗口平均
pred = predict_windows(lambda z: model(z), x)    # → (B, 1)
```
- `split = {"train": [...], "val": [...], "test": [...]}` 是 `samples` 的索引；三者受試者互不重疊（已 assert）。
- 訓練集 DataLoader 是 `drop_last=True`、每次取樣都會隨機裁窗與增強；驗證/測試是固定的。
- `mean_baseline(samples, n_folds=5)` 回傳每折猜平均的 test MAE（原始分數），`["per_fold"]` 形狀 `(5, 2)`：第 0 欄一律猜平均、第 1 欄依動作猜平均。

---

## 3. 要做的事

### 3.1 小幅修改 `model/rehab_ctrgcn.py`（只改這兩處）

1. **fc 初始化**（目前第 194 行是為分類設計的 `normal_(0, sqrt(2/1))`，預測會亂跳）：
   ```python
   nn.init.normal_(self.fc.weight, 0, 0.01)
   nn.init.constant_(self.fc.bias, init_bias)   # init_bias 由建構子參數傳入，預設 0.75
   ```
   訓練時以「該折訓練集 TS/50 的平均」傳入 `init_bias`，讓模型從「猜平均」出發。
2. **加 `gc_type` 參數切換空間圖卷積**（供之後 ablation，本任務用預設值）：`Model(..., gc_type="alter")` → `TCN_GCN_unit` → `unit_ctrgcn`，`"alter"` 用 `Alter_CTRGC`、`"origin"` 用 `CTRGC`。兩個類別都已在同檔案內。

其餘模型程式**不要動**。

### 3.2 新增 `train_rehab.py`

**命令列參數**（用 argparse，皆有預設值）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--pkl` | `data/kimore/processed/kimore_full.pkl` | |
| `--name` | `baseline` | 輸出到 `runs/<name>/` |
| `--folds` | `0,1,2,3,4` | 要跑哪幾折，例如 `--folds 0` 只跑一折 |
| `--epochs` | 150 | 上限 |
| `--patience` | 30 | 驗證 MAE 連續幾個 epoch 沒進步就停 |
| `--batch_size` | 8 | 學長實驗中 8 最好 |
| `--lr` | 1e-3 | |
| `--weight_decay` | 1e-2 | AdamW |
| `--warmup` | 5 | 前幾個 epoch 線性 warmup，之後 cosine 衰減 |
| `--loss` | `huber` | `huber` 或 `mse` |
| `--huber_delta` | 0.1 | 在 0–1 尺度上（約等於 5 分） |
| `--no_aug` | False | 關閉資料增強（`augmentor` 傳一個所有機率為 0 的 `SkeletonAugmentor`） |
| `--gc_type` | `alter` | |
| `--seed` | 42 | 同時設定 torch / numpy / random；**切分的 seed 固定為 42 不要改** |
| `--overfit` | False | 過擬合測試模式（見 4.2） |
| `--device` | `auto` | auto：mps → cuda → cpu |

**訓練流程（每一折）**
1. `build_dataloaders(...)` 取得 loaders；計算該折訓練集 `ts/50` 平均，作為 fc `init_bias`。
2. 優化器 AdamW、warmup + cosine 排程、Huber（或 MSE）損失，都在 0–1 尺度上計算。
3. 每個 epoch：訓練 → 用 `predict_windows` 跑驗證集 → 計算驗證 MAE（原始分數）。
4. 驗證 MAE 創新低就存 `best.pt`；連續 `patience` 個 epoch 沒進步就提早停止。
5. 訓練結束載入 `best.pt`，**只在這時跑一次測試集**。測試集絕不能用於挑模型或調參。
6. 若 `runs/<name>/fold{k}/metrics.json` 已存在就跳過該折（可從中斷處續跑）。

**每折要記錄**
- `log.csv`：epoch、train_loss、train_MAE、val_MAE、val_RMSE、val_ρ、lr、epoch 秒數。
- `test_predictions.csv`：index、subject_key、subgroup、exercise、y_true（0–50）、y_pred（0–50）。
- `metrics.json`：測試集的 MAE、RMSE（0–50 與 0–1 兩種尺度）、Pearson ρ、Spearman ρ、預測值的標準差、各動作 MAE、各子組 MAE、最佳 epoch、總訓練時間；以及**同一折**的猜平均 baseline（全域與依動作）。

**全部跑完後**輸出 `runs/<name>/summary.md`：

| Fold | Baseline（依動作）MAE | 模型 MAE | 模型 RMSE | Pearson ρ | 贏過 baseline？ |
|---|---|---|---|---|---|
| 0 … 4 | | | | | |
| mean ± std | 8.32 ± 0.78 | | | | x / 5 折 |

再加一張各動作的 MAE 表、一張各子組的 MAE 表。

**實作細節**
- Mac：程式開頭 `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`；`num_workers=0`。
- 每個 epoch 印一行：`fold k | ep e | loss | train MAE | val MAE | val ρ | 秒數`。
- Pearson / Spearman 可用 `scipy.stats`；若沒裝，用 numpy 自行計算。
- 程式保持精簡，不要加 TensorBoard、wandb 等額外依賴。

---

## 4. 驗證步驟（依序執行，每一步通過才進下一步）

### 4.1 前置檢查
```bash
python dataset.py
```
預期：載入 **376 筆、77 位受試者**；baseline「依動作猜平均」約 **8.32 ± 0.78**。數字不同就停下來回報。

### 4.2 過擬合測試（確認程式沒有 bug）
`--overfit` 模式：只取 fold 0 訓練集的前 16 筆、關閉資料增強、不裁窗（`crop_ratio=(1.0, 1.0)`），訓練約 300 步，在**同樣 16 筆**上評估。
- **通過條件**：這 16 筆的 MAE < **1.0 分**（0–50 分制）。
- 不通過代表資料與標籤沒對上、梯度沒有回傳、或學習率有問題。先修好再繼續。

### 4.3 單折試跑與計時
```bash
caffeinate -i python train_rehab.py --name smoke --folds 0 --epochs 5
```
- 確認：沒有報錯、train loss 在下降、每個 epoch 的秒數。
- 回報每個 epoch 的耗時，並推估完整 5 折所需時間。

### 4.4 完整基準實驗
```bash
caffeinate -i python train_rehab.py --name baseline
```
長時間執行時提醒使用者：接電源、放硬平面、螢幕打開（Air 無風扇會降頻）。

### 4.5 成功標準（「明顯贏過 8.3」的定義）
- 5 折平均測試 MAE **≤ 7.0**，而且
- **每一折**的模型 MAE 都低於**同一折**的「依動作猜平均」MAE，而且
- Pearson ρ 平均 **≥ 0.5**，而且
- 預測值標準差不能太小（例如 < 2 分代表模型幾乎每個人都給同樣的分數，等於學成猜平均）。

---

## 5. 若沒有達標，依序檢查

1. 4.2 過擬合測試是否真的通過。
2. train MAE 沒下降：調整 `--lr`（試 3e-4、3e-3）。
3. train MAE 很低、val MAE 很高（過擬合）：確認資料增強有開；試 `--weight_decay 5e-2`、提高 dropout。
4. 預測值全部擠在一起：檢查 fc 初始化與學習率。
5. 觀察 `Alter_CTRGC` 的 softmax 注意力（`(B, C, 25, 25)`）是否幾乎是 one-hot。若是，可在 softmax 前除以 `T ** 0.5` 試試，但**先回報再改**，不要直接改預設行為。

每次調整只改一個地方，並記錄在 `runs/<name>/` 下。

---

## 6. 不要做的事

- 不要改 `dataset.py` 的切分邏輯或 seed（42），不要改 `preprocess_kimore.py`。
- 不要用測試集挑模型、調參或決定 early stopping。
- 不要把模型換成其他架構，不要加入 MediaPipe 或 RGB 相關處理（那是之後的任務）。
- 不要預測 PO / CF（之後的 ablation 才做）。

## 7. 已知但本任務不處理的問題

- 前處理在 9 筆「同一動作有多個骨架檔」的資料上，可能選到失敗的重錄（應選與影片時間戳相同者）。之後會修，本任務先忽略。
- E_ID17 的分數在原始資料中是空的，已被前處理排除。

---

## 8. 完成後回報給使用者

1. 4.2 過擬合測試的結果。
2. 每個 epoch 的耗時，以及完整實驗的總時間。
3. `summary.md` 的表格內容。
4. 是否達到 4.5 的成功標準；沒達到的話，依第 5 節檢查到哪一步、發現了什麼。
5. 對下一步的建議（例如哪個 ablation 最值得先做）。
