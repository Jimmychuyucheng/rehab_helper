"""
v7 3.1 + 追加 1 C 第 5 點：MediaPipe Pose Landmarker 擷取（VIDEO 模式，CPU）。

每部影片存 {out}/{subject}_{Es}.npz：
  world      (T, 33, 3) float32  pose_world_landmarks（公尺、髖部中點原點，MediaPipe 原始座標軸）；偵測失敗幀為 NaN
  image      (T, 33, 3) float32  pose_landmarks（x, y 正規化影像座標，z 相對深度）；失敗幀為 NaN
  visibility (T, 33)    float32  失敗幀為 0
  presence   (T, 33)    float32
  detected   (T,)       bool
  frame_idx  (T,)       int32    影片幀號（0 起算）
  pts_ms     (T,)       float64  容器 PTS（cv2 CAP_PROP_POS_MSEC），只供參考，不是真實時間
  ts_ms      (T,)       int64    餵給 MediaPipe 的時間戳 = round(frame_idx × 1000 / 30)
完成後在 {out}/extract_log.csv 追加一列。已存在的 npz 跳過（可續跑）。
--hflip（追加 11 A）：解碼後先水平翻轉再送入 MediaPipe（KIMORE RGB 整批是鏡像畫面），不另存影片；
  image 座標是翻轉後畫面的座標。預設關閉 = 原本 T2 的行為。

    python run_mediapipe.py --model full --out data/kimore/mediapipe/raw/full --workers 14 [--hflip] [--only E_ID14_Es2 ...]

環境（追加 4 A）：rehab env + mediapipe 1.2.3；需 conda 的 libegl、libgl、libgles（CPU 也要 EGL/GLES）。
ffprobe：rehab env 沒有 ffprobe，借用 mediatools env 的，執行前先
    export PATH=$PATH:$HOME/anaconda3/envs/mediatools/bin
找不到 ffprobe 時 log 的 fps_ffprobe 欄為空，不影響擷取。
"""
import argparse
import csv
import glob
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

RGB = "data/kimore/rgb"
MODELS = "data/kimore/mediapipe/models"
KINECT_HZ = 30.0
LOG_FIELDS = ["key", "rgb_path", "model", "model_sha256", "mediapipe", "frames", "frames_meta", "fps_ffprobe",
              "fail_frac", "mean_visibility", "ts_monotonic", "elapsed_s", "status", "hflip"]


def list_videos():
    out = {}
    for p in sorted(glob.glob(f"{RGB}/**/*.mp4", recursive=True)):
        parts = p.split(os.sep)
        key = f"{parts[-4]}_{parts[-3]}"           # .../{subject}/{Es}/rgb/*.mp4
        if key in out:
            raise RuntimeError(f"重複的影片 {key}: {out[key]} / {p}")
        out[key] = p
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ffprobe_fps(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def extract(key, path, model_path, out_dir, hflip=False):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    t0 = time.time()
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(path)
    frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    world, image, vis, pres, det, fidx, pts, tss = [], [], [], [], [], [], [], []
    nan33 = np.full((33, 3), np.nan, np.float32)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            if hflip:
                bgr = cv2.flip(bgr, 1)
            ts = int(round(i * 1000.0 / KINECT_HZ))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, ts)
            if r.pose_world_landmarks:
                w, n = r.pose_world_landmarks[0], r.pose_landmarks[0]
                world.append(np.array([[p.x, p.y, p.z] for p in w], np.float32))
                image.append(np.array([[p.x, p.y, p.z] for p in n], np.float32))
                vis.append(np.array([p.visibility or 0.0 for p in n], np.float32))
                pres.append(np.array([p.presence or 0.0 for p in n], np.float32))
                det.append(True)
            else:
                world.append(nan33); image.append(nan33)
                vis.append(np.zeros(33, np.float32)); pres.append(np.zeros(33, np.float32))
                det.append(False)
            fidx.append(i); tss.append(ts)
            i += 1
    cap.release()
    det = np.array(det)
    vis = np.stack(vis) if vis else np.zeros((0, 33), np.float32)
    tss = np.array(tss, np.int64)
    np.savez_compressed(
        os.path.join(out_dir, f"{key}.npz"), world=np.stack(world), image=np.stack(image), visibility=vis,
        presence=np.stack(pres), detected=det, frame_idx=np.array(fidx, np.int32),
        pts_ms=np.array(pts, np.float64), ts_ms=tss)
    return {"key": key, "rgb_path": path, "frames": len(det), "frames_meta": frames_meta,
            "fps_ffprobe": ffprobe_fps(path), "fail_frac": round(1 - det.mean(), 4) if len(det) else 1.0,
            "mean_visibility": round(float(vis[det].mean()), 4) if det.any() else 0.0,
            "ts_monotonic": bool(np.all(np.diff(tss) > 0)), "elapsed_s": round(time.time() - t0, 1),
            "status": "ok", "mediapipe": mp.__version__, "hflip": bool(hflip)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["heavy", "full", "lite"], default="heavy")
    ap.add_argument("--out", default="data/kimore/mediapipe/raw")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", nargs="*", help="只跑這些 key（{subject}_{Es}）")
    ap.add_argument("--hflip", action="store_true", help="追加 11 A：送入前水平翻轉（預設關閉）")
    args = ap.parse_args()

    model_path = os.path.join(MODELS, f"pose_landmarker_{args.model}.task")
    model_hash = sha256(model_path)
    os.makedirs(args.out, exist_ok=True)
    vids = list_videos()
    if args.only:
        missing = set(args.only) - set(vids)
        if missing:
            sys.exit(f"找不到影片：{sorted(missing)}")
        vids = {k: vids[k] for k in args.only}
    todo = {k: p for k, p in vids.items() if not os.path.exists(os.path.join(args.out, f"{k}.npz"))}
    print(f"{args.model}: {len(vids)} 部，待跑 {len(todo)} 部，workers={args.workers}，hflip={args.hflip}", flush=True)

    log_path = os.path.join(args.out, "extract_log.csv")
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f, ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        futs = {ex.submit(extract, k, p, model_path, args.out, args.hflip): k for k, p in todo.items()}
        for n, fu in enumerate(as_completed(futs), 1):
            k = futs[fu]
            try:
                row = fu.result()
            except Exception as e:                       # 失敗原因寫進 log，不中斷其他影片
                row = {"key": k, "rgb_path": todo[k], "status": f"error: {type(e).__name__}: {e}"}
            row.update(model=args.model, model_sha256=model_hash)
            w.writerow(row); f.flush()
            print(f"[{n}/{len(todo)}] {k} {row.get('frames')} 幀 {row.get('elapsed_s')} s {row['status']}", flush=True)


if __name__ == "__main__":
    main()
