"""
v7 追加 C 第 3 點：求回 kimore_full_v3.pkl 每筆樣本在原始 JointPosition CSV 中的起訖幀號。

原始幀號 = CSV 行號 (0 起算；空行 = 該幀沒追蹤到人，也算一幀)，與同名 TimeStamp CSV 的行一一對應
(398 個檔行數全部相同)。區間含頭不含尾：v3 陣列 = 原始 CSV 第 start..end-1 幀。

輸出 data/kimore/v3_crop_index.csv：
  sample_id       kimore_full_v3.pkl 中 samples 的 index (與 rgb_manifest.csv 相同)
  source_file     raw/ 中實際的檔名 (v3 記為「...(1).csv」的 4 筆，raw/ 只有內容相同的無 (1) 檔)
  n_raw           原始 CSV 幀數 (行數)
  start, end      v3 保留段
  cropped         start > 0 或 end < n_raw (含只去掉頭尾空行的情況)
  lead_blank, tail_blank   原始 CSV 開頭／結尾連續空行數
  dropped_data    保留段以外「有骨架資料」的幀數 (非受試者軌跡、中段斷開後較短的片段、人工裁切)
其他欄位 (subject 等) 方便查閱。

檢查 (每次都跑，也可用 --check_only 只跑)：不經 load_joint_positions，直接取 CSV 第 start..end-1 行，
多人幀依 SpineBase 連續性取受試者、全 0 幀線性內插，再以 body_centric_alignment 正規化，必須等於 v3 陣列。

    python scripts/build_v3_crop_index.py
    python scripts/build_v3_crop_index.py --check_only
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preprocess_kimore as pk  # noqa: E402
from dataset import load_kimore  # noqa: E402

PKL = "data/kimore/processed/kimore_full_v3.pkl"
RAW = "data/kimore/raw"
OUT = "data/kimore/v3_crop_index.csv"
FIELDS = ["sample_id", "subgroup", "subject", "exercise", "source_file", "n_raw", "start", "end",
          "cropped", "lead_blank", "tail_blank", "dropped_data"]
TOL = 1e-5


def raw_path(s, fname=None):
    d = os.path.join(RAW, s["group"], s["subgroup"], s["subject"], s["exercise"], "Raw")
    f = os.path.join(d, fname or s["source_file"])
    if fname is None and not os.path.exists(f):
        f = os.path.join(d, s["source_file"].replace("(1)", ""))
    return f


def read_lines(path):
    with open(path, newline="") as f:
        text = f.read()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":       # 檔尾換行
        lines = lines[:-1]
    return lines


def build(samples):
    rows = []
    for i, s in enumerate(samples):
        path = raw_path(s)
        tag = f"{s['subgroup']}/{s['subject']}/{s['exercise']}"
        max_dropped = 10 ** 9 if tag in pk.VERIFIED_TRIMS else pk.MAX_DROPPED
        _, _, (start, end) = pk.load_joint_positions(path, max_dropped, return_index=True)
        start += s["skeleton_info"].get("start_trim", 0)      # v3 沒有，v3.1 才有；保留以防萬一
        lines = read_lines(path)
        n = len(lines)
        blank = [not l.strip().strip(",") for l in lines]
        lead = next((k for k, b in enumerate(blank) if not b), n)
        tail = next((k for k, b in enumerate(reversed(blank)) if not b), n)
        dropped = sum(1 for k, b in enumerate(blank) if not b and not (start <= k < end))
        if end - start != s["num_frames"]:
            raise RuntimeError(f"{i} {tag}: end-start={end - start} != num_frames {s['num_frames']}")
        rows.append({"sample_id": i, "subgroup": s["subgroup"], "subject": s["subject"],
                     "exercise": s["exercise"], "source_file": os.path.basename(path), "n_raw": n,
                     "start": start, "end": end, "cropped": int(start > 0 or end < n),
                     "lead_blank": lead, "tail_blank": tail, "dropped_data": dropped})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"已寫入 {OUT}（{len(rows)} 列）")


def slice_skeleton(lines, start, end):
    """CSV 第 start..end-1 行 → (T, 25, 3)；多人幀取 SpineBase 水平位置與前一幀最近者。"""
    bodies = [pk._rows_to_bodies(l) for l in lines[start:end]]
    if any(not b for b in bodies):
        raise ValueError("區間內有空行")
    single = [b[0][pk.SPINE_BASE][[0, 2]] for b in bodies if len(b) == 1]
    prev = np.median(single, axis=0) if single else bodies[0][0][pk.SPINE_BASE][[0, 2]]
    out = []
    for b in bodies:
        x = min(b, key=lambda x: np.linalg.norm(x[pk.SPINE_BASE][[0, 2]] - prev))
        out.append(x)
        prev = x[pk.SPINE_BASE][[0, 2]]
    vals = np.stack(out).reshape(len(out), -1)
    zero = np.all(np.abs(vals) < 1e-8, axis=1)
    vals[zero] = np.nan
    vals = (pd.DataFrame(vals).interpolate(method="linear", limit_direction="both", axis=0)
            .bfill().ffill().values)
    return vals.reshape(len(out), 25, 3).astype(np.float32)


def check(samples):
    with open(OUT, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(samples), f"列數 {len(rows)} != 樣本數 {len(samples)}"
    bad = []
    for r in rows:
        i, start, end = int(r["sample_id"]), int(r["start"]), int(r["end"])
        s = samples[i]
        assert (s["subject"], s["exercise"]) == (r["subject"], r["exercise"]), f"sample_id {i} 對不上"
        try:
            skel = slice_skeleton(read_lines(raw_path(s, r["source_file"])), start, end)
            aligned, _ = pk.body_centric_alignment(skel)
            v3 = s["skeleton"].transpose(1, 2, 0)
            err = float(np.abs(aligned - v3).max()) if aligned.shape == v3.shape else np.inf
        except Exception as e:      # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        if not (isinstance(err, float) and err < TOL):
            bad.append((i, r["subject"], r["exercise"], err))
    print(f"檢查：{len(rows) - len(bad)}/{len(rows)} 筆切出後正規化 == v3 陣列 (max abs < {TOL})")
    for b in bad:
        print("  不符", b)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()
    samples, _ = load_kimore(PKL)
    if not args.check_only:
        build(samples)
    ok = check(samples)
    with open(OUT, newline="") as f:
        rows = list(csv.DictReader(f))
    print("cropped:", Counter(r["cropped"] for r in rows), "；dropped_data > 0:",
          sum(int(r["dropped_data"]) > 0 for r in rows))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
