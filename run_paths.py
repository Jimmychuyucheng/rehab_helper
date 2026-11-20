"""
run_paths.py：找 run 資料夾。

runs/ 依實驗版本分資料夾 (runs/v5/v5_base_s42、runs/server_v6/server_v6_base_s42 ...)。
各腳本仍用原本的 run 名稱 (例如 "v6_base_s42")；找不到 runs/<name> 時，往下一層 runs/*/<name> 找。
也可以直接給含版本的名稱 (例如 "v6/v6_base_s42")。
"""
import glob
import os


def run_dir(name, runs_dir="runs"):
    direct = os.path.join(runs_dir, name)
    if os.path.isdir(direct):
        return direct
    hits = sorted(d for d in glob.glob(os.path.join(runs_dir, "*", name)) if os.path.isdir(d))
    if len(hits) > 1:
        raise ValueError(f"run 名稱 {name} 重複出現：{hits}，請改用含版本的名稱 (例如 v6/{name})")
    return hits[0] if hits else direct
