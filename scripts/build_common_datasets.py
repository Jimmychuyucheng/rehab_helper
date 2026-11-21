"""
v7 T4（依追加 7 B、C 調整）：產生共同骨架資料集，三檔樣本 id 與 fold 完全相同（rgb_manifest.csv 的 347 筆）。

  data/kimore/processed/kimore_k_common_347.pkl          Kinect，已正規化（供 T5 訓練）
  data/kimore/processed/kimore_mp_common_heavy_v1.pkl    MediaPipe heavy，未正規化（T6 求出 k 後才正規化）
  data/kimore/processed/kimore_mp_common_full_v1.pkl     MediaPipe full，未正規化

格式同 kimore_full_v3.pkl：{"meta": ..., "samples": [...]}，骨架陣列 layout (C=3, T, V=17)。
- Kinect：原始 CSV 的 v3 保留段 [start, end)（第 j 幀 = 原始第 start + j 行）→ 共同骨架 → 逐幀減 HipMid
  → normalize()：參考幀 = 保留段前 10 幀，原點／方向取中位數，尺度 = 整段軀幹長中位數。
  另存 skeleton_centered（只減 HipMid、未正規化），供 T6 以配對參考幀（追加 7 B）重新正規化。
- MediaPipe：world → 共同骨架（軸向 (x,y,z)→(-x,-y,z)）→ 逐幀減 HipMid；不遮罩、不補齊、不正規化。
  偵測失敗的幀為 NaN；visibility、detected 另存。模型輸入（追加 7 D）由使用端呼叫 fill_undetected()。
- fold：直接取 rgb_manifest.csv 的 fold（追加 1 A：沿用 380 筆時 seed 42 的受試者分層 5 折，不重新切分），
  並檢查與 dataset.make_subject_folds(v3 380 筆) 一致。**T5 須使用樣本內存的 fold，不可對 347 筆重新切分。**

    python scripts/build_common_datasets.py
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_skeleton as cs  # noqa: E402
from dataset import load_kimore, make_subject_folds  # noqa: E402

V3 = "data/kimore/processed/kimore_full_v3.pkl"
MANIFEST = "data/kimore/rgb_manifest.csv"
CROP = "data/kimore/v3_crop_index.csv"
QC = "data/kimore/mediapipe/qc"
MP_RAW = "data/kimore/mediapipe/raw"
OUT = "data/kimore/processed"
KEEP = ["group", "subgroup", "subject", "subject_key", "exercise", "exercise_id", "ts", "po", "cf",
        "source_file", "label_fix"]


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def common_meta(kind):
    return {"version": "v7-common-1", "kind": kind, "layout": "(C=3, T, V=17)", "joint_names": cs.COMMON_JOINTS,
            "flip_pairs": cs.COMMON_FLIP_PAIRS, "inward": cs.COMMON_INWARD, "kinect_src": cs.KINECT_SRC,
            "mediapipe_src": cs.MP_SRC, "mp_to_kinect_axes": cs.MP_TO_KINECT_AXES.tolist(),
            "centered_on_hipmid": True, "git_head": git_head(),
            "fold_source": "rgb_manifest.csv（380 筆時 seed 42 受試者分層 5 折；不可對本檔重新切分）"}


def main():
    v3, _ = load_kimore(V3)
    man = pd.read_csv(MANIFEST)
    crop = pd.read_csv(CROP).set_index("sample_id")
    sc = pd.read_csv(f"{QC}/scene_labels.csv", index_col=0)
    ff = pd.read_csv(f"{QC}/diag_t2/fail_features.csv", index_col=0)
    exc = pd.read_csv(f"{QC}/excluded.csv").set_index("key")

    # fold 檢查：manifest 的 fold = 380 筆 v3 以 seed 42 切分的結果
    folds = make_subject_folds(v3, 5, 42)
    subj_fold = {s: f for f, ss in enumerate(folds) for s in ss}
    bad = [(r.sample_id, r.fold, subj_fold[r.subject_key]) for r in man.itertuples() if subj_fold[r.subject_key] != r.fold]
    assert not bad, f"manifest fold 與 make_subject_folds(v3) 不一致：{bad[:5]}"

    k_samples, mp = [], {"heavy": [], "full": []}
    for r in man.itertuples():
        s, c = v3[r.sample_id], crop.loc[r.sample_id]
        assert (s["subject"], s["exercise"]) == (r.subject, r.exercise) == (c.subject, c.exercise), r.sample_id
        key = f"{r.subject}_{r.exercise}"
        base = {k: s[k] for k in KEEP}
        base.update(sample_id=int(r.sample_id), key=key, fold=int(r.fold), scene=int(sc.loc[key, "scene"]),
                    backlit=bool(sc.loc[key, "backlit_scene_1_4"]),
                    excluded_reason=exc.loc[key, "reason"] if key in exc.index else None)
        base["main_set"] = (not base["backlit"]) and float(ff.loc[key, "fail_heavy"]) <= 0.2   # 追加 6 A

        # Kinect
        path = os.path.join("data/kimore/raw", r.group, r.subgroup, r.subject, r.exercise, "Raw", c.source_file)
        K25 = cs.load_kinect_raw(path, int(c.start), int(c.end))
        seg = K25[int(c.start):int(c.end)]
        assert len(seg) == s["num_frames"] and np.isfinite(seg).all(), key
        kc = cs.kinect_to_common(seg)
        kn, p = cs.normalize(kc)
        ks = dict(base, skeleton=kn.transpose(2, 0, 1).copy(), skeleton_centered=kc.transpose(2, 0, 1).copy(),
                  num_frames=len(kn), raw_start=int(c.start), raw_end=int(c.end), n_raw=int(c.n_raw),
                  norm_params={"origin": p["origin"], "R": p["R"], "scale_m": p["scale"],
                               "ref_idx": p["ref_idx"], "ref_rule": "保留段前 10 幀（單邊）"})
        k_samples.append(ks)

        # MediaPipe
        for m in mp:
            z = np.load(os.path.join(MP_RAW, m, f"{key}.npz"))
            x, vis = cs.mediapipe_to_common(z["world"], z["visibility"], vis_threshold=None)
            det = z["detected"].astype(bool)
            assert np.isfinite(x[det]).all() and np.isnan(x[~det]).all(), (m, key)
            mp[m].append(dict(base, skeleton=x.transpose(2, 0, 1).copy(), visibility=vis, detected=det,
                              frame_idx=z["frame_idx"], num_frames=len(x), fail_frac=float(1 - det.mean()),
                              plaus_frac=float(ff.loc[key, f"plaus_{m}"])))

    meta = common_meta("kinect")
    meta.update(normalized=True, unit="HipMid→ShoulderMid 軀幹長（整段中位數）",
                normalize_rule="追加 7 B：原點/方向 = 參考幀中位數；本檔參考幀 = 保留段前 10 幀（單邊）",
                frame_index="第 j 幀 = 原始 CSV 第 raw_start + j 行（追加 2 C）", source=V3)
    with open(f"{OUT}/kimore_k_common_347.pkl", "wb") as f:
        pickle.dump({"meta": meta, "samples": k_samples}, f)
    for m, ss in mp.items():
        meta = common_meta(f"mediapipe_{m}")
        meta.update(normalized=False, unit="公尺（MediaPipe world），已逐幀減 HipMid",
                    missing="偵測失敗的幀為 NaN；未依 visibility 遮罩（模型輸入用 fill_undetected，角度特徵用 vis < 0.5 遮罩）",
                    frame_index="影片幀號（0 起算）；對應 Kinect 原始 CSV 行由 T6 求出",
                    model_sha256=pd.read_csv(f"{MP_RAW}/{m}/extract_log.csv").model_sha256.iloc[0])
        with open(f"{OUT}/kimore_mp_common_{m}_v1.pkl", "wb") as f:
            pickle.dump({"meta": meta, "samples": ss}, f)
    report(k_samples, mp)


def report(k_samples, mp):
    ids = [s["sample_id"] for s in k_samples]
    folds = [s["fold"] for s in k_samples]
    for m, ss in mp.items():
        assert [s["sample_id"] for s in ss] == ids and [s["fold"] for s in ss] == folds, m
    print(f"三檔樣本 id 與 fold 完全相同：{len(ids)} 筆；fold 筆數 {np.bincount(folds).tolist()}")
    T = np.array([s["num_frames"] for s in k_samples])
    kn = np.concatenate([s["skeleton"].reshape(3, -1) for s in k_samples], axis=1)
    torso = np.array([np.median(np.linalg.norm(s["skeleton"][:, :, 1] - s["skeleton"][:, :, 0], axis=0)) for s in k_samples])
    print(f"\nkimore_k_common_347.pkl：每筆 (3, T, 17)，T {T.min()}–{T.max()}（中位數 {int(np.median(T))}），"
          f"共 {T.sum()} 幀；NaN 比例 {np.isnan(kn).mean():.4%}；各筆軀幹長中位數的中位數 = {np.median(torso):.6f}（應為 1）")
    hip = max(float(np.abs(s["skeleton"][:, :, 0]).max()) for s in k_samples)
    print(f"  逐幀減 HipMid 後 HipMid 最大絕對值 {hip:.2e}；main_set {sum(s['main_set'] for s in k_samples)} 筆")
    for m, ss in mp.items():
        T = np.array([s["num_frames"] for s in ss])
        det = np.concatenate([s["detected"] for s in ss])
        vis = np.concatenate([s["visibility"][s["detected"]] for s in ss])
        low = (vis < 0.5).mean(axis=0)
        main = [s for s in ss if s["main_set"]]
        dmain = np.concatenate([s["detected"] for s in main])
        print(f"\nkimore_mp_common_{m}_v1.pkl：每筆 (3, T, 17)，T {T.min()}–{T.max()}，共 {T.sum()} 幀；"
              f"未偵測幀（NaN）{1 - det.mean():.2%}（主分析集 {1 - dmain.mean():.2%}）")
        print("  偵測到的幀中 visibility < 0.5 的比例（高到低）："
              + "、".join(f"{cs.COMMON_JOINTS[j]} {low[j]:.1%}" for j in np.argsort(-low)))


if __name__ == "__main__":
    main()
