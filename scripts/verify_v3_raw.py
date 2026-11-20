"""
v7：確認 data/kimore/raw/ 的原始骨架就是 v3 用的資料。
對 kimore_full_v3.pkl 每筆樣本，以 preprocess_kimore 的讀檔與正規化從 raw/ 重算，必須與 v3 陣列一致 (max abs < 1e-5)。
v3 記錄為「...(1).csv」的 4 筆，raw/ 只留內容相同的無 (1) 檔，改讀該檔。

    python scripts/verify_v3_raw.py
"""
import sys, os, numpy as np, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preprocess_kimore as pk
from dataset import load_kimore
S, _ = load_kimore('data/kimore/processed/kimore_full_v3.pkl')
R = 'data/kimore/raw'
res = collections.Counter(); bad = []
for s in S:
    d = f"{R}/{s['group']}/{s['subgroup']}/{s['subject']}/{s['exercise']}/Raw"
    f = os.path.join(d, s['source_file'])
    if not os.path.exists(f): f = os.path.join(d, s['source_file'].replace('(1)', ''))
    big = 10**9 if f"{s['subgroup']}/{s['subject']}/{s['exercise']}" in pk.VERIFIED_TRIMS else pk.MAX_DROPPED
    skel, info = pk.load_joint_positions(f, big)
    st = s['skeleton_info'].get('start_trim', 0)
    skel = skel[st:]
    al, _ = pk.body_centric_alignment(skel)
    v3 = s['skeleton'].transpose(1, 2, 0)
    if al.shape != v3.shape:
        res['shape_mismatch'] += 1; bad.append((s['subject'], s['exercise'], al.shape, v3.shape)); continue
    e = float(np.abs(al - v3).max())
    res['match' if e < 1e-5 else 'value_mismatch'] += 1
    if e >= 1e-5: bad.append((s['subject'], s['exercise'], e))
print(res); print(bad[:20])
print('start_trim 樣本:', [(s['subject'], s['exercise'], s['skeleton_info']['start_trim']) for s in S if 'start_trim' in s['skeleton_info']])
