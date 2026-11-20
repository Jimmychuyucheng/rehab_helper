"""
v7 T0：比對 KIMORE RGB 影片與 v3 樣本，輸出 data/kimore/rgb_manifest.csv。

配對規則：路徑 (subgroup, subject, Es) 對到 v3 樣本，再比對檔名時間戳 DDMMYY_HHMMSS
與 v3 所選骨架檔 (source_file) 的時間戳。對不上的不硬湊，只在報告列出。
fold：seed 42 的受試者分層 5 折，用 v3 全部 380 筆建立 (與 v6 訓練相同)，受試者不換折。
sample_id：kimore_full_v3.pkl 中 samples 的 index (v6 run 的 index 以它為準)。

    python scripts/build_rgb_manifest.py            # 只報告
    python scripts/build_rgb_manifest.py --write    # 只寫入能配對的 mp4；對不上的只列出
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import load_kimore, make_subject_folds  # noqa: E402

PKL = "data/kimore/processed/kimore_full_v3.pkl"
RGB_DIR = "data/kimore/rgb"
OUT = "data/kimore/rgb_manifest.csv"
STAMP = re.compile(r"(\d{6}_\d{6})")


def stamp(name):
    m = STAMP.search(os.path.basename(name))
    return m.group(1) if m else None


def to_dt(s):
    return datetime.strptime(s, "%d%m%y_%H%M%S")


def candidates_from_notes(sample):
    """從 notes 取出多骨架檔時的所有候選檔名。"""
    for n in sample["notes"]:
        if n.startswith("多個骨架檔"):
            return sorted(set(re.findall(r"JointPosition\d{6}_\d{6}(?:\(1\))?\.csv", n)))
    return [sample["source_file"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    samples, meta = load_kimore(PKL)
    folds = make_subject_folds(samples, n_folds=5, seed=42)
    fold_of = {k: i for i, f in enumerate(folds) for k in f}
    by_key = {(s["subgroup"], s["subject"], s["exercise"]): i for i, s in enumerate(samples)}

    files = sorted(glob.glob(os.path.join(RGB_DIR, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    others = [f for f in files if not f.endswith(".mp4")]
    mp4s = [f for f in files if f.endswith(".mp4")]

    rows, unmatched, mismatch = [], [], []
    seen = defaultdict(list)
    for f in mp4s:
        rel = os.path.relpath(f, RGB_DIR)
        parts = rel.split(os.sep)  # group/subgroup/subject/EsN/rgb/file
        if len(parts) != 6 or parts[4] != "rgb":
            unmatched.append((rel, "路徑結構不符"))
            continue
        group, subgroup, subject, es = parts[:4]
        es = "Es" + es[-1]
        idx = by_key.get((subgroup, subject, es))
        if idx is None:
            unmatched.append((rel, "v3 無此樣本 (受試者/動作不在 v3)"))
            continue
        s = samples[idx]
        v_st, k_st = stamp(f), stamp(s["source_file"])
        cands = candidates_from_notes(s)
        cand_st = sorted({stamp(c) for c in cands})
        dt = (to_dt(v_st) - to_dt(k_st)).total_seconds() if v_st and k_st else None
        if v_st != k_st:
            mismatch.append((rel, idx, s["source_file"], cands, dt, v_st in cand_st))
        seen[idx].append(rel)
        rows.append({
            "sample_id": idx, "group": group, "subgroup": subgroup, "subject": subject,
            "subject_key": s["subject_key"], "exercise": es, "fold": fold_of[s["subject_key"]],
            "ts": round(float(s["ts"]), 4), "rgb_path": os.path.join(RGB_DIR, rel),
            "rgb_stamp": v_st, "skeleton_file": s["source_file"], "skeleton_stamp": k_st,
            "stamp_match": v_st == k_st, "n_skeleton_candidates": len(cand_st),
            "ambiguous_source": ";".join(s["ambiguous_source"]),
        })

    dups = {i: v for i, v in seen.items() if len(v) > 1}
    missing = [i for i in range(len(samples)) if i not in seen]

    print(f"v3 樣本 {len(samples)}；mp4 {len(mp4s)}；其他檔 {len(others)}")
    print(f"mp4 對到 v3 樣本：{len(rows)}；對不上：{len(unmatched)}；同一樣本多部影片：{len(dups)}")
    for rel, why in unmatched:
        print(f"  [對不上] {rel}：{why}")
    for i, v in dups.items():
        print(f"  [重複] sample {i}: {v}")
    print(f"\n時間戳不一致：{len(mismatch)}")
    for rel, idx, src, cands, dt, in_c in mismatch:
        print(f"  sample {idx} {rel}\n     骨架 {src}；候選 {cands}；差 {dt:+.0f} s；影片時間戳{'是' if in_c else '不是'}其他候選")
    print("\n多骨架檔樣本的影片時間戳：")
    for r in rows:
        if r["n_skeleton_candidates"] > 1:
            print(f"  {r['subject_key']} {r['exercise']}: 影片 {r['rgb_stamp']} / 骨架 {r['skeleton_stamp']} "
                  f"候選 {candidates_from_notes(samples[r['sample_id']])} "
                  f"{'一致' if r['stamp_match'] else '不一致'}"
                  + (f"；ambiguous={r['ambiguous_source']}" if r['ambiguous_source'] else ""))
    # 時間戳一致者的分布 (確認 RGB 與骨架用同一時鐘)
    print(f"\nv3 中沒有影片：{len(missing)} 筆")
    for i in missing:
        s = samples[i]
        print(f"  sample {i:3d} {s['subject_key']:22s} {s['exercise']}  TS {s['ts']:.1f}  fold {fold_of[s['subject_key']]}")
    print("  組別分布：", dict(Counter(samples[i]["subgroup"] for i in missing)))
    print("  v3 全體：", dict(Counter(s["subgroup"] for s in samples)))
    print("  動作分布：", dict(Counter(samples[i]["exercise"] for i in missing)))
    print("\n其他檔：", others)

    if args.write:
        if dups:
            sys.exit("同一樣本有多部影片，不寫檔")
        # 對不上的 mp4 不寫入 (不硬湊)，只在上方列出
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["sample_id"]))
        print(f"\n已寫出 {OUT}（{len(rows)} 筆）")


if __name__ == "__main__":
    main()
