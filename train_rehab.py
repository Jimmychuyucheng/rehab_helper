"""
================================================================================
檔案名稱: train_rehab.py
功能說明:
    以受試者 5-fold 訓練 Rehab CTR-GCN，迴歸預測 KIMORE 臨床總分 TS (0~50)。
    每折輸出到 runs/<name>/fold{k}/ (log.csv, best.pt, test_predictions.csv, metrics.json)，
    全部跑完後輸出 runs/<name>/summary.md。

    python train_rehab.py --name baseline                 # 5 折
    python train_rehab.py --name smoke --folds 0 --epochs 5
    python train_rehab.py --overfit                       # 過擬合測試 (16 筆)
================================================================================
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import (build_dataloaders, RehabDataset, SkeletonAugmentor, predict_windows,
                     mean_baseline, make_split, load_kimore, SCORE_MAX)
from model.rehab_ctrgcn import Model

TS_MAX = SCORE_MAX["ts"]
TS_MIN = 10.0
EXERCISES = ["Es1", "Es2", "Es3", "Es4", "Es5"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/kimore/processed/kimore_full.pkl")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber")
    ap.add_argument("--huber_delta", type=float, default=0.1)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--gc_type", choices=["alter", "origin"], default="alter")
    ap.add_argument("--clip", dest="clip", action="store_true", default=True,
                    help="評估時把預測夾在 10~50 (預設開啟)")
    ap.add_argument("--no_clip", dest="clip", action="store_false")
    ap.add_argument("--reweight", choices=["none", "inv", "sqrt_inv"], default="none",
                    help="依訓練集 TS 分箱 (5 分一格) 的樣本加權")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==================== 指標 ====================
def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(x):
    """平均名次 (處理同分)，等同 scipy.stats.rankdata。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    return pearson(_rank(a), _rank(b))


def regression_metrics(y_true, y_pred):
    """輸入為原始分數 (0~50)。"""
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "pearson": pearson(y_true, y_pred)}


SCORE_BINS = [(10, 25), (25, 35), (35, 45), (45, 50)]


def bin_label(lo, hi):
    return f"{lo}-{hi}"


def range_metrics(y_true, y_pred, subjects):
    """分數區間 MAE / bias、平衡 MAE、pred 對 true 的斜率、以人為單位的 MAE。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    bins = {}
    for k, (lo, hi) in enumerate(SCORE_BINS):
        last = k == len(SCORE_BINS) - 1
        m = (y_true >= lo) & ((y_true <= hi) if last else (y_true < hi))
        err = y_pred[m] - y_true[m]
        bins[bin_label(lo, hi)] = {"n": int(m.sum()),
                                   "mae": float(np.abs(err).mean()) if m.any() else None,
                                   "bias": float(err.mean()) if m.any() else None}
    bin_maes = [b["mae"] for b in bins.values() if b["mae"] is not None]
    per_subj = defaultdict(list)
    for subj, t, p in zip(subjects, y_true, y_pred):
        per_subj[subj].append((t, p))
    subj_err = [abs(np.mean([p for _, p in v]) - np.mean([t for t, _ in v])) for v in per_subj.values()]
    return {"bins": bins, "balanced_mae": float(np.mean(bin_maes)),
            "slope": float(np.polyfit(y_true, y_pred, 1)[0]),
            "subject_mae": float(np.mean(subj_err))}


# ==================== 樣本加權 ====================
def reweight_table(samples, train_idx, mode):
    """
    回傳長度 len(samples) 的權重陣列 (只有訓練集索引有意義)，以及各分箱統計。
    以訓練集 TS 5 分一格分箱 (10-15, …, 45-50，50 歸最後一格)，權重 = (1/count)^p，
    再正規化讓訓練樣本平均權重 = 1。mode="none" 回傳 (None, None)。
    """
    if mode == "none":
        return None, None
    p = {"inv": 1.0, "sqrt_inv": 0.5}[mode]
    edges = np.arange(10, 50, 5)  # 每格下界
    ts = np.array([samples[i]["ts"] for i in train_idx])
    b = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, len(edges) - 1)
    counts = np.bincount(b, minlength=len(edges))
    w_bin = np.where(counts > 0, (1.0 / np.maximum(counts, 1)) ** p, 0.0)
    w = w_bin[b]
    scale = 1.0 / w.mean()
    table = np.ones(len(samples), dtype=np.float32)
    table[train_idx] = w * scale
    stats = [{"bin": f"{lo}-{lo + 5}", "count": int(c), "weight": float(wb * scale)}
             for lo, c, wb in zip(edges, counts, w_bin)]
    return table, stats


# ==================== 訓練 / 評估 ====================
def make_scheduler(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_one_epoch(model, loader, opt, sched, criterion, device, weights=None):
    """criterion 為 reduction="none"；weights 為 index → 權重的查表 (None 表示不加權)。"""
    model.train()
    losses, abs_err, n = [], 0.0, 0
    for b in loader:
        x, y = b["x"].to(device), b["y"].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        if weights is not None:
            loss = loss * torch.from_numpy(weights[b["index"].numpy()]).to(device).view(-1, 1)
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        abs_err += (pred.detach() - y).abs().sum().item() * TS_MAX
        n += len(y)
    return float(np.mean(losses)), abs_err / n


@torch.no_grad()
def evaluate(model, loader, device, clip=False):
    """回傳 (indices, y_true, y_pred)，分數為 0~50。x 可為 (B,3,T,V) 或 (B,K,3,T,V)。
    clip=True 時把預測夾在 [10/50, 1] (TS 最低 10 分)；只用於評估，訓練 loss 不夾。"""
    model.eval()
    idx, ys, ps = [], [], []
    for b in loader:
        x = b["x"].to(device)
        pred = predict_windows(lambda z: model(z), x) if x.dim() == 5 else model(x)
        if clip:
            pred = pred.clamp(TS_MIN / TS_MAX, 1.0)
        idx += b["index"].tolist()
        ys.append(b["y"][:, 0].numpy() * TS_MAX)
        ps.append(pred[:, 0].cpu().numpy() * TS_MAX)
    return idx, np.concatenate(ys), np.concatenate(ps)


def build_model(args, init_bias, device):
    return Model(num_class=1, num_point=25, in_channels=3, init_bias=init_bias,
                 gc_type=args.gc_type).to(device)


def make_criterion(args):
    if args.loss == "huber":
        return nn.HuberLoss(delta=args.huber_delta, reduction="none")
    return nn.MSELoss(reduction="none")


def no_augmentor():
    return SkeletonAugmentor(p_flip=0, yaw_deg=0, scale_range=None, noise_std=0, p_joint_mask=0)


# ==================== 過擬合測試 ====================
def run_overfit(args, device, n_samples=16, steps=300):
    samples, _ = load_kimore(args.pkl)
    split = make_split(samples, 0, 5, 0.15, 42)
    idx = split["train"][:n_samples]
    ds = RehabDataset(samples, idx, train=True, crop_ratio=(1.0, 1.0), augmentor=no_augmentor())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_all = np.array([samples[i]["ts"] for i in idx])
    model = build_model(args, float(y_all.mean() / TS_MAX), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = math.ceil(steps / len(loader))
    sched = make_scheduler(opt, args.warmup * len(loader), epochs * len(loader))
    criterion = make_criterion(args)

    print(f"[overfit] {n_samples} 筆 (TS 平均 {y_all.mean():.2f}, std {y_all.std():.2f})，"
          f"{epochs} epochs × {len(loader)} 步 = {epochs * len(loader)} 步")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        loss, tr_mae = train_one_epoch(model, loader, opt, sched, criterion, device)
        if ep % 15 == 0 or ep == epochs:
            _, yt, yp = evaluate(model, eval_loader, device)
            print(f"[overfit] ep {ep:3d} | loss {loss:.5f} | train MAE {tr_mae:.3f} | "
                  f"eval MAE {np.abs(yt - yp).mean():.3f} | {time.time() - t0:.0f}s")
    _, yt, yp = evaluate(model, eval_loader, device)
    mae = float(np.abs(yt - yp).mean())
    print(f"[overfit] 最終 16 筆 MAE = {mae:.3f} 分 → {'通過' if mae < 1.0 else '未通過'} (門檻 < 1.0)")
    for t, p in zip(yt, yp):
        print(f"   y={t:6.2f}  pred={p:6.2f}")
    return mae


# ==================== 單折 ====================
def run_fold(args, fold, device, baselines):
    out_dir = os.path.join("runs", args.name, f"fold{fold}")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"fold {fold}: {metrics_path} 已存在，跳過")
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed + fold)

    loaders, split, samples = build_dataloaders(
        args.pkl, fold, n_folds=5, val_ratio=0.15, seed=42, batch_size=args.batch_size,
        num_workers=0, out_frames=100, targets=("ts",), crop_ratio=(0.4, 0.6),
        eval_windows=5, eval_crop_ratio=0.5, augmentor=no_augmentor() if args.no_aug else None)

    init_bias = float(np.mean([samples[i]["ts"] / TS_MAX for i in split["train"]]))
    weights, w_stats = reweight_table(samples, split["train"], args.reweight)
    if w_stats:
        print(f"fold {fold}: reweight={args.reweight} " +
              ", ".join(f"{r['bin']}:{r['count']}→{r['weight']:.2f}" for r in w_stats))
    model = build_model(args, init_bias, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    sched = make_scheduler(opt, args.warmup * steps_per_epoch, args.epochs * steps_per_epoch)
    criterion = make_criterion(args)
    print(f"fold {fold}: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} 筆，"
          f"init_bias {init_bias:.4f} ({init_bias * TS_MAX:.2f} 分)，device {device}")

    log_path = os.path.join(out_dir, "log.csv")
    best_path = os.path.join(out_dir, "best.pt")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_MAE", "val_MAE", "val_RMSE",
                                "val_rho", "lr", "epoch_sec"])

    best_mae, best_ep, bad = float("inf"), 0, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        loss, tr_mae = train_one_epoch(model, loaders["train"], opt, sched, criterion, device, weights)
        _, yt, yp = evaluate(model, loaders["val"], device, args.clip)
        vm = regression_metrics(yt, yp)
        sec = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{loss:.6f}", f"{tr_mae:.4f}", f"{vm['mae']:.4f}",
                                    f"{vm['rmse']:.4f}", f"{vm['pearson']:.4f}", f"{lr:.6g}", f"{sec:.1f}"])
        mark = ""
        if vm["mae"] < best_mae:
            best_mae, best_ep, bad = vm["mae"], ep, 0
            torch.save(model.state_dict(), best_path)
            mark = " *"
        else:
            bad += 1
        print(f"fold {fold} | ep {ep:3d} | loss {loss:.5f} | train MAE {tr_mae:6.3f} | "
              f"val MAE {vm['mae']:6.3f} | val ρ {vm['pearson']:+.3f} | {sec:5.1f}s{mark}", flush=True)
        if bad >= args.patience:
            print(f"fold {fold}: 驗證 MAE 連續 {args.patience} epoch 沒進步，於 ep {ep} 停止")
            break
    train_time = time.time() - t_start

    # 只在這裡用最佳模型跑一次測試集
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    idx, yt, yp = evaluate(model, loaders["test"], device, args.clip)

    with open(os.path.join(out_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "subject_key", "subgroup", "exercise", "y_true", "y_pred"])
        for i, t, p in zip(idx, yt, yp):
            s = samples[i]
            w.writerow([i, s["subject_key"], s["subgroup"], s["exercise"], f"{t:.4f}", f"{p:.4f}"])

    m = regression_metrics(yt, yp)
    by_ex, by_sub = defaultdict(list), defaultdict(list)
    for i, t, p in zip(idx, yt, yp):
        by_ex[samples[i]["exercise"]].append(abs(p - t))
        by_sub[samples[i]["subgroup"]].append(abs(p - t))
    metrics = {
        "fold": fold,
        "mae": m["mae"], "rmse": m["rmse"],
        "mae_norm": m["mae"] / TS_MAX, "rmse_norm": m["rmse"] / TS_MAX,
        "pearson": m["pearson"], "spearman": spearman(yt, yp),
        "pred_std": float(np.std(yp)), "true_std": float(np.std(yt)),
        "per_exercise_mae": {k: float(np.mean(v)) for k, v in sorted(by_ex.items())},
        "per_subgroup_mae": {k: float(np.mean(v)) for k, v in sorted(by_sub.items())},
        **range_metrics(yt, yp, [samples[i]["subject_key"] for i in idx]),
        "reweight_bins": w_stats,
        "best_epoch": best_ep, "best_val_mae": best_mae, "epochs_run": ep,
        "train_time_sec": train_time,
        "baseline_global_mae": float(baselines[fold, 0]),
        "baseline_per_ex_mae": float(baselines[fold, 1]),
        "n_test": len(yt),
        "args": vars(args),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"fold {fold} test | MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | ρ {m['pearson']:+.3f} | "
          f"pred std {metrics['pred_std']:.2f} | baseline(依動作) {baselines[fold, 1]:.3f} | "
          f"best ep {best_ep} | {train_time / 60:.1f} 分鐘")
    return metrics


# ==================== 總表 ====================
def _ms(v, fmt="{:.2f}"):
    v = np.asarray(v, float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}"


def read_predictions(path, clip=False):
    """讀 test_predictions.csv → (y_true, y_pred, subject_keys)；clip=True 時事後把預測夾在 10~50。"""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    yt = np.array([float(r["y_true"]) for r in rows])
    yp = np.array([float(r["y_pred"]) for r in rows])
    if clip:
        yp = np.clip(yp, TS_MIN, TS_MAX)
    return yt, yp, [r["subject_key"] for r in rows]


def range_table(rows):
    """rows: [(標籤, range_metrics 結果)] → 分數區間 markdown 表格的各行。"""
    labels = [bin_label(lo, hi) for lo, hi in SCORE_BINS]
    lines = ["| | " + " | ".join(f"{l} (n / MAE / bias)" for l in labels) +
             " | 平衡 MAE | 斜率 | 以人為單位 MAE |",
             "|---" * (len(labels) + 4) + "|"]
    for name, m in rows:
        cells = []
        for l in labels:
            b = m["bins"][l]
            cells.append(f"{b['n']} / {b['mae']:.2f} / {b['bias']:+.2f}" if b["n"] else "0 / – / –")
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {m['balanced_mae']:.2f} | {m['slope']:.3f} | {m['subject_mae']:.2f} |")
    return lines


def write_summary(args, results):
    lines = [f"# {args.name}", "",
             "| Fold | Baseline（依動作）MAE | 模型 MAE | 模型 RMSE | Pearson ρ | 預測 std | 贏過 baseline？ |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        win = r["mae"] < r["baseline_per_ex_mae"]
        lines.append(f"| {r['fold']} | {r['baseline_per_ex_mae']:.2f} | {r['mae']:.2f} | {r['rmse']:.2f} | "
                     f"{r['pearson']:.3f} | {r['pred_std']:.2f} | {'✓' if win else '✗'} |")
    wins = sum(r["mae"] < r["baseline_per_ex_mae"] for r in results)
    lines.append(f"| mean ± std | {_ms([r['baseline_per_ex_mae'] for r in results])} | "
                 f"{_ms([r['mae'] for r in results])} | {_ms([r['rmse'] for r in results])} | "
                 f"{_ms([r['pearson'] for r in results], '{:.3f}')} | {_ms([r['pred_std'] for r in results])} | "
                 f"{wins} / {len(results)} 折 |")

    for title, key in [("各動作 MAE", "per_exercise_mae"), ("各子組 MAE", "per_subgroup_mae")]:
        cols = sorted({c for r in results for c in r[key]})
        lines += ["", f"## {title}", "", "| Fold | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        for r in results:
            lines.append(f"| {r['fold']} | " + " | ".join(
                f"{r[key][c]:.2f}" if c in r[key] else "–" for c in cols) + " |")
        lines.append("| mean | " + " | ".join(
            f"{np.mean([r[key][c] for r in results if c in r[key]]):.2f}" for c in cols) + " |")

    # 分數區間：從 test_predictions.csv 重算，舊 run 的 metrics.json 沒有這些欄位也能用
    preds = {r["fold"]: read_predictions(os.path.join("runs", args.name, f"fold{r['fold']}",
                                                      "test_predictions.csv")) for r in results}
    pooled = [np.concatenate([p[i] for p in preds.values()]) for i in range(2)]
    pooled.append([s for p in preds.values() for s in p[2]])
    lines += ["", "## 分數區間（真實 TS）", ""]
    lines += range_table([(f"fold {k}", range_metrics(*p)) for k, p in preds.items()] +
                         [("五折合併", range_metrics(*pooled))])

    path = os.path.join("runs", args.name, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已寫入 {path}\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)
    if args.overfit:
        run_overfit(args, device)
        return

    folds = [int(k) for k in args.folds.split(",")]
    samples, _ = load_kimore(args.pkl)
    baselines = mean_baseline(samples, n_folds=5)["per_fold"]
    os.makedirs(os.path.join("runs", args.name), exist_ok=True)
    results = [run_fold(args, k, device, baselines) for k in folds]
    write_summary(args, results)


if __name__ == "__main__":
    main()
