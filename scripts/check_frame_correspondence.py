"""
v7 追加 C 第 4 點／追加 2 D（不需 MediaPipe 的部分）：檢查「影片第 i 幀 ↔ 原始 CSV 第 i + k 行」。

對 rgb_manifest.csv 的 347 部錄影：
  - 影片：cv2 容器標示的幀數與 fps，並逐幀 grab 實際計數 (不靠 metadata)。
  - 原始 JointPosition / TimeStamp CSV：行數 (= Kinect 幀，0 起算)、空行、TimeStamp (單位 100 ns)。
另對 raw/ 全部 398 組 JointPosition/TimeStamp 檢查空行是否一一對應。

輸出 data/kimore/frame_check.csv (每部一列)，並印出摘要。

    python scripts/check_frame_correspondence.py [--workers 8]
"""
import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

RAW = "data/kimore/raw"
MANIFEST = "data/kimore/rgb_manifest.csv"
OUT = "data/kimore/frame_check.csv"
TICK_MS = 1e-4                  # TimeStamp 單位 100 ns
DROP_MS = 50.0


def read_lines(path):
    with open(path, newline="") as f:
        lines = f.read().replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def cells(line):
    c = line.strip().split(",")
    while c and c[-1] == "":
        c.pop()
    return c


def video_info(path):
    cap = cv2.VideoCapture(path)
    meta_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    last_ms = 0.0
    while cap.grab():
        last_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        n += 1
    cap.release()
    return {"video_frames": n, "video_frames_meta": meta_n, "fps": round(fps, 3), "video_last_pts_ms": last_ms}


def ts_info(jp_path):
    ts_path = os.path.join(os.path.dirname(jp_path), os.path.basename(jp_path).replace("JointPosition", "TimeStamp"))
    jl, tl = read_lines(jp_path), read_lines(ts_path)
    n = len(jl)
    blank_j = np.array([not cells(l) for l in jl])
    tc = [cells(l) for l in tl]
    blank_t = np.array([not c for c in tc])
    multi_ts_diff = 0
    t = np.full(n, np.nan)
    for i, c in enumerate(tc):
        if c:
            v = [int(x) for x in c]
            multi_ts_diff += len(set(v)) > 1
            t[i] = v[0] * TICK_MS
    idx = np.flatnonzero(~blank_t)
    # 相鄰兩行都有時間戳才算幀間隔 (空行沒有時間，跨空行的間隔量不到)
    cons = idx[1:][np.diff(idx) == 1]
    dt = t[cons] - t[cons - 1]
    first, last = idx[0], idx[-1]
    span_ms = t[last] - t[first]
    return {"n_raw": n, "n_ts_lines": len(tl), "blank_mismatch": int((blank_j != blank_t).sum()),
            "n_blank": int(blank_t.sum()), "lead_blank": int(first), "tail_blank": int(n - 1 - last),
            "inner_blank": int(blank_t[first:last + 1].sum()), "multi_ts_diff": multi_ts_diff,
            "ts_span_ms": round(span_ms, 1), "ts_span_lines": int(last - first),
            "dt_n": len(dt), "dt_median": round(float(np.median(dt)), 3) if len(dt) else np.nan,
            "dt_min": round(float(dt.min()), 3), "dt_max": round(float(dt.max()), 3),
            "dt_over50": int((dt > DROP_MS).sum()), "_dt": dt}


def one(row):
    jp = glob.glob(os.path.join(RAW, row["group"], row["subgroup"], row["subject"], row["exercise"], "Raw",
                                row["skeleton_file"].replace("(1)", "")))
    if len(jp) != 1:
        raise RuntimeError(f"{row['subject']} {row['exercise']}: 找到 {jp}")
    out = {"sample_id": row["sample_id"], "subgroup": row["subgroup"], "subject": row["subject"],
           "exercise": row["exercise"]}
    out.update(video_info(row["rgb_path"]))
    out.update(ts_info(jp[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # 全部 398 組：空行一一對應
    mism = []
    for jp in sorted(glob.glob(f"{RAW}/**/JointPosition*.csv", recursive=True)):
        r = ts_info(jp)
        if r["blank_mismatch"] or r["n_raw"] != r["n_ts_lines"]:
            mism.append((jp, r["blank_mismatch"], r["n_raw"], r["n_ts_lines"]))
    print(f"raw/ 全部 JointPosition：空行與 TimeStamp 空行不一致或行數不同的檔 {len(mism)} 個", mism[:10])

    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, rows))
    dts = np.concatenate([r.pop("_dt") for r in res])
    np.save(os.path.join(os.path.dirname(OUT), "frame_check_dt.npy"), dts)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"已寫入 {OUT}（{len(res)} 列），全部幀間隔 {len(dts)} 個")


if __name__ == "__main__":
    sys.exit(main())
