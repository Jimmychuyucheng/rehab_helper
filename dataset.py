"""
================================================================================
檔案名稱: dataset.py
功能說明:
    讀取 preprocess_kimore.py (v2) 輸出的 kimore_full.pkl，提供：
      1. 以「受試者」為單位、依子組分層的 K-fold 切分 (避免同一人同時出現在訓練與測試)
      2. RehabDataset：訓練時隨機裁窗 + 資料增強，驗證/測試時取固定的多個窗口
      3. 平均值 baseline (第一步驗證用)

--------------------------------------------------------------------------------
【切分方式】make_split(samples, fold, n_folds=5, val_ratio=0.15)
    - test  : 第 fold 折的受試者 (約 1/5)，只用來報告最終結果
    - val   : 從其餘受試者中依子組分層抽 val_ratio，用來 early stopping / 選超參數
    - train : 剩下的受試者
    五折都跑完後，報告 test 指標的 mean ± std。

【訓練時 __getitem__ (每個 epoch 都不同)】
    1. 隨機裁窗：窗口長度 = 原始長度 × U(crop_ratio)。
       每段錄影是 5 次重複，crop_ratio=(0.4, 0.6) 約涵蓋 2~3 次重複。
    2. 重採樣到固定 out_frames 幀。
       ※ 因為窗口長度是隨機的、最後又統一縮放到同樣幀數，
         「隨機窗口長度」本身就等同於加速/減速 (tempo) 增強，不需要另外做變速。
    3. 增強：左右鏡像、繞垂直軸小角度旋轉、等比例縮放、關節雜訊、隨機遮關節。
       刻意不做畫面內 (X-Y 平面) 旋轉：那會改變軀幹側傾角，而 Es2 評的就是側傾。

【驗證/測試時】
    從整段等距取 eval_windows 個窗口 (長度 = 原始長度 × eval_crop_ratio)，
    回傳 (K, 3, T, V)；模型對 K 個窗口預測後取平均，見 predict_windows()。
    設 eval_windows=1, eval_crop_ratio=1.0 即為「整段縮放到 out_frames」。

【標籤】
    targets=("ts",) 或 ("ts", "po", "cf")。分數會除以滿分正規化到 0~1：
    TS/50、PO/15、CF/35。用 denormalize() 換回原始分數。
================================================================================
"""

import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SCORE_MAX = {"ts": 50.0, "po": 15.0, "cf": 35.0}
DEFAULT_FLIP_PAIRS = [(4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17),
                      (14, 18), (15, 19), (21, 23), (22, 24)]


# ==================== 讀檔 ====================
def load_kimore(pkl_path="./data/kimore/processed/kimore_full.pkl"):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError("這不是 v2 格式的 pkl，請先用新版 preprocess_kimore.py 重新產生。")
    return payload["samples"], payload["meta"]


# ==================== 受試者切分 ====================
def _stratified_subjects(samples):
    """回傳 {subgroup: [subject_key, ...]}，用於分層。"""
    groups = defaultdict(set)
    for s in samples:
        groups[s["subgroup"]].add(s["subject_key"])
    return {g: sorted(v) for g, v in sorted(groups.items())}


def make_subject_folds(samples, n_folds=5, seed=42):
    """依子組分層，把受試者輪流分配到 n_folds 折。回傳 list[set(subject_key)]。"""
    rng = random.Random(seed)
    folds = [set() for _ in range(n_folds)]
    offset = 0
    for _, subjects in _stratified_subjects(samples).items():
        subjects = subjects[:]
        rng.shuffle(subjects)
        for i, subj in enumerate(subjects):
            folds[(i + offset) % n_folds].add(subj)
        offset += len(subjects)  # 讓各子組的餘數分散到不同折，避免某折人數特別多
    return folds


def make_split(samples, fold, n_folds=5, val_ratio=0.15, seed=42):
    """
    回傳 {"train": [idx...], "val": [idx...], "test": [idx...]}，索引對應 samples。
    同一位受試者的所有動作只會出現在其中一個集合。
    """
    folds = make_subject_folds(samples, n_folds, seed)
    test_subj = folds[fold]

    rng = random.Random(seed + 1000 + fold)
    rest = [s for s in samples if s["subject_key"] not in test_subj]
    val_subj = set()
    for _, subjects in _stratified_subjects(rest).items():
        subjects = subjects[:]
        rng.shuffle(subjects)
        n_val = int(round(len(subjects) * val_ratio))
        val_subj.update(subjects[:n_val])

    split = {"train": [], "val": [], "test": []}
    for i, s in enumerate(samples):
        k = s["subject_key"]
        if k in test_subj:
            split["test"].append(i)
        elif k in val_subj:
            split["val"].append(i)
        else:
            split["train"].append(i)

    # 防呆：確認沒有受試者跨集合
    sets = {name: {samples[i]["subject_key"] for i in idx} for name, idx in split.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), "受試者洩漏！"
    return split


# ==================== 時序處理 ====================
def resample(x, out_frames):
    """線性內插 (C, L, V) → (C, out_frames, V)。"""
    C, L, V = x.shape
    if L == out_frames:
        return x.copy()
    pos = np.linspace(0, L - 1, out_frames)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    w = (pos - lo)[None, :, None].astype(x.dtype)
    return x[:, lo] * (1 - w) + x[:, hi] * w


def crop(x, start, length):
    return x[:, start:start + length]


# ==================== 資料增強 ====================
class SkeletonAugmentor:
    """輸入/輸出皆為 (3, T, 25)，座標已做過身體中心校準 (X 左右、Y 向上、Z 深度)。"""

    def __init__(self, flip_pairs=DEFAULT_FLIP_PAIRS, p_flip=0.5, yaw_deg=15.0,
                 scale_range=(0.9, 1.1), noise_std=0.01, p_joint_mask=0.1, max_masked=2):
        self.flip_pairs = flip_pairs
        self.p_flip = p_flip
        self.yaw_deg = yaw_deg
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.p_joint_mask = p_joint_mask
        self.max_masked = max_masked

    def __call__(self, x, rng=np.random):
        x = x.copy()
        # 1. 左右鏡像：X 取負並交換左右關節
        if rng.rand() < self.p_flip:
            x[0] = -x[0]
            for a, b in self.flip_pairs:
                x[:, :, [a, b]] = x[:, :, [b, a]]
        # 2. 繞垂直 Y 軸旋轉 (模擬受試者沒有正對鏡頭)
        if self.yaw_deg > 0:
            th = np.deg2rad(rng.uniform(-self.yaw_deg, self.yaw_deg))
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=x.dtype)
            x = np.einsum("ij,jtv->itv", R, x)
        # 3. 等比例縮放 (三軸同倍率，不改變角度)
        if self.scale_range is not None:
            x *= rng.uniform(*self.scale_range)
        # 4. 關節雜訊 (單位為軀幹長，0.01 ≈ 0.5 cm)
        if self.noise_std > 0:
            x += rng.normal(0, self.noise_std, x.shape).astype(x.dtype)
        # 5. 隨機遮關節 (模擬追蹤失敗 / 姿態估計漏點)
        if rng.rand() < self.p_joint_mask:
            n = rng.randint(1, self.max_masked + 1)
            joints = rng.choice(x.shape[2], n, replace=False)
            x[:, :, joints] = 0.0
        return x.astype(np.float32)


# ==================== Dataset ====================
class RehabDataset(Dataset):
    """
    回傳 dict:
        x           : train → (3, T, V)；eval → (K, 3, T, V)
        y           : (len(targets),) 正規化到 0~1 的分數
        exercise_id : int (0~4)，可用於動作條件化或分動作迴歸頭
        index       : 在 samples 中的索引 (方便把預測對回受試者)
    """

    def __init__(self, samples, indices, train, out_frames=100, targets=("ts",),
                 crop_ratio=(0.4, 0.6), eval_windows=5, eval_crop_ratio=0.5,
                 augmentor=None, exercises=None):
        self.samples = samples
        self.indices = [i for i in indices
                        if exercises is None or samples[i]["exercise"] in exercises]
        self.train = train
        self.out_frames = out_frames
        self.targets = tuple(targets)
        self.crop_ratio = crop_ratio
        self.eval_windows = eval_windows
        self.eval_crop_ratio = eval_crop_ratio
        self.augmentor = augmentor if augmentor is not None else (SkeletonAugmentor() if train else None)

        for t in self.targets:
            if t not in SCORE_MAX:
                raise ValueError(f"未知的 target: {t}")
            missing = [samples[i]["subject_key"] for i in self.indices if samples[i].get(t) is None]
            if missing:
                raise ValueError(f"{len(missing)} 筆樣本缺少 {t} 分數，例如 {missing[:3]}")

    def __len__(self):
        return len(self.indices)

    def _window_len(self, L, ratio):
        return int(np.clip(round(L * ratio), min(L, 15), L))

    def __getitem__(self, i):
        s = self.samples[self.indices[i]]
        x_full = s["skeleton"]                      # (3, L, V)
        L = x_full.shape[1]

        if self.train:
            win = self._window_len(L, np.random.uniform(*self.crop_ratio))
            start = np.random.randint(0, L - win + 1)
            x = resample(crop(x_full, start, win), self.out_frames)
            if self.augmentor is not None:
                x = self.augmentor(x)
            x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        else:
            win = self._window_len(L, self.eval_crop_ratio)
            starts = np.linspace(0, L - win, self.eval_windows).round().astype(int)
            x = np.stack([resample(crop(x_full, st, win), self.out_frames) for st in starts])
            x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

        y = torch.tensor([s[t] / SCORE_MAX[t] for t in self.targets], dtype=torch.float32)
        return {"x": x, "y": y, "exercise_id": s["exercise_id"], "index": self.indices[i]}


def denormalize(y, targets=("ts",)):
    """把 (..., len(targets)) 的 0~1 分數換回原始分數。"""
    scale = torch.tensor([SCORE_MAX[t] for t in targets], dtype=y.dtype, device=y.device)
    return y * scale


@torch.no_grad()
def predict_windows(model_fn, x):
    """
    eval 用：x 為 (B, K, 3, T, V)，攤平成 (B*K, ...) 丟進 model_fn，再對 K 個窗口取平均。
    model_fn 需回傳 (B*K, out_dim)；若模型回傳 (score, feat)，請包一層 lambda 只取 score。
    """
    B, K = x.shape[:2]
    out = model_fn(x.reshape(B * K, *x.shape[2:]))
    return out.reshape(B, K, -1).mean(dim=1)


# ==================== 一次建好 DataLoader ====================
def build_dataloaders(pkl_path, fold, n_folds=5, val_ratio=0.15, seed=42,
                      batch_size=8, num_workers=0, out_frames=100, targets=("ts",),
                      crop_ratio=(0.4, 0.6), eval_windows=5, eval_crop_ratio=0.5,
                      augmentor=None, exercises=None):
    samples, _ = load_kimore(pkl_path)
    split = make_split(samples, fold, n_folds, val_ratio, seed)
    common = dict(out_frames=out_frames, targets=targets, exercises=exercises)

    train_ds = RehabDataset(samples, split["train"], train=True, crop_ratio=crop_ratio,
                            augmentor=augmentor, **common)
    eval_kw = dict(train=False, eval_windows=eval_windows, eval_crop_ratio=eval_crop_ratio, **common)
    val_ds = RehabDataset(samples, split["val"], **eval_kw)
    test_ds = RehabDataset(samples, split["test"], **eval_kw)

    def _worker_init(wid):
        np.random.seed((torch.initial_seed() + wid) % 2**32)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=num_workers, worker_init_fn=_worker_init),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, split, samples


# ==================== 第一步驗證：平均值 baseline ====================
def mean_baseline(samples, n_folds=5, val_ratio=0.15, seed=42, target="ts"):
    """
    兩種「不看骨架」的猜法，模型必須明顯贏過它們才代表有學到東西：
      global : 一律猜訓練集平均
      per_ex : 依動作猜該動作在訓練集的平均
    回傳每折 test MAE 與整體 mean ± std。
    """
    rows = []
    for k in range(n_folds):
        split = make_split(samples, k, n_folds, val_ratio, seed)
        tr = [samples[i] for i in split["train"] + split["val"]]
        te = [samples[i] for i in split["test"]]
        g = np.mean([s[target] for s in tr])
        ex_mean = defaultdict(list)
        for s in tr:
            ex_mean[s["exercise"]].append(s[target])
        ex_mean = {e: np.mean(v) for e, v in ex_mean.items()}
        y = np.array([s[target] for s in te])
        mae_g = np.abs(y - g).mean()
        mae_e = np.abs(y - np.array([ex_mean.get(s["exercise"], g) for s in te])).mean()
        rows.append((mae_g, mae_e))
    rows = np.array(rows)
    return {"per_fold": rows, "global": (rows[:, 0].mean(), rows[:, 0].std()),
            "per_ex": (rows[:, 1].mean(), rows[:, 1].std())}


# ==================== 自我檢查 ====================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="./data/kimore/processed/kimore_full.pkl")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    samples, meta = load_kimore(args.pkl)
    print(f"載入 {len(samples)} 筆，{len({s['subject_key'] for s in samples})} 位受試者")

    print("\n[切分] 各折人數 (train / val / test 受試者):")
    for k in range(args.folds):
        sp = make_split(samples, k, args.folds)
        n = {name: len({samples[i]['subject_key'] for i in idx}) for name, idx in sp.items()}
        sub = defaultdict(int)
        for i in sp["test"]:
            sub[samples[i]["subgroup"]] += 1
        print(f"   fold {k}: {n['train']:3d} / {n['val']:3d} / {n['test']:3d} 人 | "
              f"test 樣本 {len(sp['test'])} 筆 ({', '.join(f'{g}={c}' for g, c in sorted(sub.items()))})")

    loaders, _, _ = build_dataloaders(args.pkl, fold=0, n_folds=args.folds)
    b = next(iter(loaders["train"]))
    print(f"\n[DataLoader] train x {tuple(b['x'].shape)}, y {tuple(b['y'].shape)}, "
          f"exercise_id {b['exercise_id'].tolist()}")
    b = next(iter(loaders["val"]))
    print(f"[DataLoader] val   x {tuple(b['x'].shape)}  (B, K 個窗口, C, T, V)")

    bl = mean_baseline(samples, args.folds)
    print("\n[Baseline] 不看骨架的 test MAE (TS 原始分數)：")
    print(f"   一律猜平均      : {bl['global'][0]:.2f} ± {bl['global'][1]:.2f}")
    print(f"   依動作猜平均    : {bl['per_ex'][0]:.2f} ± {bl['per_ex'][1]:.2f}")
    print("   → 模型的 test MAE 必須明顯低於這兩個數字。")