# Commit hash 對照表（2026-11-22 改寫歷史）

2026-11-22 為了移除 commit message 裡的 `Co-Authored-By` 署名，改寫了 `main`、`v4-po-cf`、`v6-kimore-v3`、`v7-mediapipe-gap` 的歷史。
**只改了 commit message**；每個 commit 的檔案內容（tree）、作者、日期都和原本相同，因此 hash 改變。
文件與程式註解中的舊 hash 已換成新 hash；若在其他地方（筆記、討論紀錄）看到舊 hash，請用下表查新 hash。

v7 判斷標準凍結 commit：舊 `6ccd464` → 新 `4915ae9`（內容與日期 2026-11-20 不變）。

| 日期 | 舊 hash | 新 hash | 標題 |
|---|---|---|---|
| 2026-10-07 | `81d8e78` | `81d8e78`（未變） | Initial commit |
| 2026-11-01 | `2ac4de3` | `2ac4de3`（未變） | Add KIMORE preprocessing, dataset pipeline and handoff doc |
| 2026-11-01 | `f0a03f0` | `0fdfb47` | Add train_rehab.py baseline (5-fold MAE 6.31) |
| 2026-11-02 | `18bd5ab` | `5d74eb0` | Add output clipping, low-score reweighting and range metrics |
| 2026-11-13 | `5185237` | `ac54d05` | Add v2 experiment log, v3/v4 handoff docs, plot_runs.py; ignore data/keraal |
| 2026-11-13 | `e610400` | `234ad8b` | Add PO label fix and TS/PO/CF multi-task training |
| 2026-11-14 | `f9d1725` | `da714a4` | Add last-K weight averaging and seed-repeat analysis |
| 2026-11-15 | `1e21047` | `794c76b` | Fix KIMORE file selection, multi-body frames and label files (v3 pkl) |
| 2026-11-15 | `3c36e1c` | `97512fc` | Trim pre-exercise walk-in at recording start (v3.1 pkl) |
| 2026-11-15 | `741889e` | `4820ae7` | Add seed-group plots and v5/v6 fixed-sample comparison |
| 2026-11-19 | `ca689c7` | `b10a0c5` | Add multi-group comparison plots, avg-vs-best and story plots, v5 handoff |
| 2026-11-19 | `52e4e10` | `9d06448` | Add requirements.txt for the rehab training and plotting pipeline |
| 2026-11-20 | `6ccd464` | `4915ae9` | v7: MediaPipe 落差測試規格（判斷標準凍結） |
| 2026-11-20 | `e1f1b3c` | `1082f76` | v7 handoff: append T0 findings (347 samples, frame-index alignment) and common-skeleton normalization |
| 2026-11-20 | `1ef73b6` | `a1c218b` | Support per-version runs/ layout and add Mac vs server v6 comparison plot |
| 2026-11-20 | `600cdd1` | `8bab625` | Support per-version runs/ layout and add Mac vs server v6 comparison plot |
| 2026-11-20 | `609a7da` | `43a7bbf` | v7 handoff: record freeze commit and log the section 3 normalization change |
| 2026-11-20 | `de54358` | `833b7a5` | v7 T0: add RGB manifest builder; ignore data/ contents except tracked .py (追加 D) |
| 2026-11-20 | `0e67d23` | `dc0d2de` | v7: add check that raw/ skeletons reproduce kimore_full_v3.pkl |
| 2026-11-20 | `0e43f31` | `618b303` | v7: recover v3 crop indices in raw CSV (return_index param, default unchanged) |
| 2026-11-20 | `c4289c7` | `8323057` | v7 handoff: append addenda 2-3; add video-frame vs raw CSV row check |
| 2026-11-20 | `5e355f0` | `50b592c` | v7 T1: add run_mediapipe.py; handoff addendum 4 (full as primary model); env adds mediapipe + GL libs |
| 2026-11-21 | `f167f84` | `2b7f013` | v7 handoff: append addendum 6 (scene-stratified analysis sets); freeze scene labels |
| 2026-11-21 | `9287998` | `a555791` | v7 handoff: back-register addendum 5 (heavy as primary model, decided before addendum 6) |
| 2026-11-21 | `93c3ef6` | `8c8ad4b` | v7 T3: common skeleton (17 joints), graph, unit tests; handoff addendum 7 |
| 2026-11-21 | `baee4f9` | `a7f7192` | v7 T4: common-skeleton datasets (Kinect normalized, MediaPipe heavy/full raw); handoff addendum 8 |
| 2026-11-21 | `123a4ca` | `a638508` | v7 T5: K-common baseline (--skeleton common, stored folds); summarize_kcommon.py; handoff addendum 9 |
| 2026-11-21 | `d7ad17d` | `0b692e0` | v7 handoff: append addendum 10 (T6 alignment: absolute image-plane x/y signal, shared mapping, no DTW, mirror rule) |
| 2026-11-21 | `99f9241` | `0bb8917` | v7 T6: align.py (addendum-10 version); handoff addendum 11 (RGB mirrored: flip and rerun MediaPipe, dynamic axis tests, global-scale signal) |
| 2026-11-21 | `f4f2f6f` | `4468d6a` | v7 addenda 11-13: hflip rerun, recomputed QC, L/R-invariant overlap alignment, test video rule |
| 2026-11-21 | `62f480b` | `67c96c4` | v7 handoff: append addendum 14 (MediaPipe restricted to Kinect segment, Kinect coverage); align.py: 100-frame rule note, coverage and segment frame range |
| 2026-11-21 | `fc82f4e` | `a532f5b` | v7 handoff: append addendum 15 (auto mirror-cause check + 6-video spot check; exclude no-ref-frame videos; T6 record) |
| 2026-11-21 | `896c2b8` | `c0e59c7` | v7 addendum 15: mirror-cause auto check (A) and no-ref-frame exclusion (B) scripts |
| 2026-11-21 | `74c4ec7` | `c60b774` | v7 handoff: append addendum 16 (auto rule confirmed; 8 videos manual RGB, 15 cause undeterminable) |
| 2026-11-21 | `2a07f47` | `ffd41b9` | v7 T7: paired_compare.py (Q1 paired predictions, Q2/Q3 per-video angles; addenda 7, 8 B, 10 C, 11 E, 14); provisional feature list |
| 2026-11-21 | `a475de3` | `a90ee88` | v7 handoff: append addendum 17 (Q2 feature list from official KIMORE scripts, P95/P5 summaries; upper/lower-body L/R swap) |
| 2026-11-21 | `a518c79` | `fd1c740` | v7 addendum 17: feature list from official KIMORE scripts (features_v7.py, runs/v7_gap/feature_list.md); upper/lower-body L/R swap |
| 2026-11-21 | `3e413f2` | `d401c3e` | v7 addendum 18 A: freeze feature list (63 items: single angle_c in Es1/Es2/Es3; Es3 elbows; no official filter); move to docs/v7_feature_list.md, untrack runs/ |
| 2026-11-21 | `5141ef7` | `55ea36a` | v7 addendum 18 B: per-pair L/R swap; swap in camera frame before normalization (ref frames may be L/R flipped); ref_lr_flipped flag; --keys all |
| 2026-11-21 | `4550528` | `ba41b63` | v7 T8: gap_report.py (Q1/Q2/Q3 verdicts per section 4 and addenda; sensitivity sets; pipeline debug for failed Q3 control); docs/v7_gap_report_summary.md |
| 2026-11-21 | `9819e41` | `98a0be3` | v7 handoff: append addendum 19 (verdicts unchanged; post-hoc exploratory analyses; constraints on module C and feature library) |
| 2026-11-21 | `e5f5cd7` | `f6a0493` | v7 addendum 19 B: post-hoc exploratory analyses (Q1 underestimation sources, plausible-frame-only Q2/Q3, Q2 by feature type); report section 9; summary updated |
| 2026-11-22 | `62af2e6` | `fa86f8d` | README: add branch guide (v4-po-cf / v6-kimore-v3 / v7-mediapipe-gap) |
| 2026-11-22 | `649675c` | `c099b5f` | README: add branch guide (v4-po-cf / v6-kimore-v3 / v7-mediapipe-gap) |
| 2026-11-22 | `ddcb36a` | `b9996c7` | README: add branch guide (v4-po-cf / v6-kimore-v3 / v7-mediapipe-gap) |
