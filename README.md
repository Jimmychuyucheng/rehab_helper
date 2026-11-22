# CISLR
 CI SLR

Current best wlasl100 model: ./weights/234cf7f2.pth
如果需要產生骨架，請先在conda執行environment.yml來配置環境，並執行執行mm2d_whole_body.py中的load_mmpose_data來取得骨架
若是要訓練模型請修改base_config.yaml並執行python train.py
若是要測試預測結果請執行 python test.py --config_path ./config/test_config.yaml

## Branch 說明（KIMORE 復健評分）

各版本是一條線往下接的，後一版包含前一版的內容。要看哪一版的工作，就切到對應的 branch。

| Branch | 版本 | 內容 | 主要文件 |
|---|---|---|---|
| `main` | v1–v2 | KIMORE 前處理、`train_rehab.py` baseline、輸出截斷與低分加權 | `HANDOFF_train_rehab.md`、`HANDOFF_v2_clip_reweight.md`、`EXPERIMENT_LOG_v2.md` |
| `v4-po-cf` | v4–v5 | PO 標籤修正、TS/PO/CF 多任務訓練；平均最後 K 輪權重、多 seed 重複 | `HANDOFF_v4_po_cf.md`、`HANDOFF_v5_seeds_avg.md` |
| `v6-kimore-v3` | v6 | KIMORE 檔案挑選與多人幀修正（v3 pkl）、去除開頭走位（v3.1 pkl）；各種比較圖、Mac vs server v6 比較 | `HANDOFF_v5_seeds_avg.md`、`plot_*.py`、`run_paths.py` |
| `v7-mediapipe-gap` | v7 | MediaPipe 與 Kinect 骨架落差測試：17 點共同骨架、K-common baseline、對齊、成對比較、落差報告 | `HANDOFF_v7_mediapipe_gap.md`、`docs/v7_gap_report_summary.md`、`docs/v7_feature_list.md` |

注意事項：
- v7 的判斷標準凍結於 commit `4915ae9`（原 `6ccd464`，只存在於 `v7-mediapipe-gap`），這個 branch 的歷史不要再 rebase 或改寫。
- 2026-11-22 曾為移除 commit 署名改寫過一次歷史（只改 commit message，內容與日期不變），新舊 hash 對照見 `docs/commit_hash_map.md`。
- 「Mac vs server v6 比較圖」這個 commit 同時出現在 `v6-kimore-v3`（`a1c218b`）和 `v7-mediapipe-gap`（`8bab625`），兩者內容相同，只是 hash 不同。
