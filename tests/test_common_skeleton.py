"""
v7 T3：common_skeleton.py 的單元測試（第 7 節清單 1～5，另加原始 CSV 行號與骨架圖）。
真實資料用 T1 試跑的 heavy 輸出（data/kimore/mediapipe/trial/heavy/）與對應的 Kinect 原始 CSV；
資料不在時跳過。

    python -m unittest tests.test_common_skeleton -v
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import common_skeleton as cs  # noqa: E402
from common_skeleton import J  # noqa: E402

T1_KEYS = ["E_ID14_Es2", "B_ID1_Es1", "NE_ID6_Es3", "P_ID15_Es4", "S_ID9_Es5"]
CLEAN = ["B_ID1_Es1", "P_ID15_Es4", "S_ID9_Es5"]     # T1 疊圖中 heavy 骨架正常的 3 部
TRIAL = "data/kimore/mediapipe/trial/heavy"
HAVE_DATA = all(os.path.exists(f"{TRIAL}/{k}.npz") for k in T1_KEYS) and os.path.exists("data/kimore/v3_crop_index.csv")
_cache = {}


def real(key):
    """回傳 dict：K25 (n_raw,25,3)、Kc (n_raw,17,3)、Mc (T,17,3)、Mvis、M33、crop row、fps。"""
    if key in _cache:
        return _cache[key]
    import pandas as pd
    ci = pd.read_csv("data/kimore/v3_crop_index.csv")
    man = pd.read_csv("data/kimore/rgb_manifest.csv")
    fc = pd.read_csv("data/kimore/frame_check.csv")
    sub, ex = key.rsplit("_", 1)
    r = ci[(ci.subject == sub) & (ci.exercise == ex)].iloc[0]
    m = man[(man.subject == sub) & (man.exercise == ex)].iloc[0]
    f = fc[(fc.subject == sub) & (fc.exercise == ex)].iloc[0]
    path = f"data/kimore/raw/{m.group}/{m.subgroup}/{sub}/{ex}/Raw/{r.source_file}"
    K25 = cs.load_kinect_raw(path, int(r.start), int(r.end))
    z = np.load(f"{TRIAL}/{key}.npz")
    Mc, Mvis = cs.mediapipe_to_common(z["world"], z["visibility"])
    d = dict(K25=K25, Kc=cs.kinect_to_common(K25), Mc=Mc, Mvis=Mvis, M33=z["world"], vis33=z["visibility"],
             crop=r, sample_id=int(m.sample_id), fps=float(f.fps), diff=int(f.n_raw - f.video_frames))
    _cache[key] = d
    return d


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 50 else -np.inf


def aligned_kinect(key):
    """
    只為測試左右而做的粗對齊：用與左右標記無關的訊號（兩腕平均高度 − HipMid 高度）找
    「影片第 i 幀 ↔ CSV 第 round(k + s·i) 行」。25 fps：s = 1、k ∈ [0, 差值]（追加 3 B）；24 fps：另搜 s。
    回傳 (與 Mc 同長的 Kinect 共同骨架, k, s, ρ)。正式對齊在 T6。
    """
    d = real(key)
    Kc, Mc = d["Kc"], d["Mc"]
    ks = (Kc[:, J["WristL"], 1] + Kc[:, J["WristR"], 1]) / 2 - Kc[:, J["HipMid"], 1]
    ms = (Mc[:, J["WristL"], 1] + Mc[:, J["WristR"], 1]) / 2 - Mc[:, J["HipMid"], 1]
    n = len(Mc)
    scales = [1.0] if d["fps"] == 25 else np.arange(1.0, 1.08, 0.0025)
    best = (-np.inf, 0, 1.0)
    for s in scales:
        for k in range(0, (d["diff"] if s == 1.0 else 60) + 1):
            idx = np.round(k + np.arange(n) * s).astype(int)
            v = idx < len(Kc)
            c = _corr(ks[idx[v]], ms[v])
            if c > best[0]:
                best = (c, k, s)
    c, k, s = best
    idx = np.round(k + np.arange(n) * s).astype(int)
    out = np.full_like(Mc, np.nan)
    v = idx < len(Kc)
    out[v] = Kc[idx[v]]
    return out, k, s, c


class TestMapping(unittest.TestCase):
    """index 對應與軸向（合成資料）。"""

    def test_kinect_indices(self):
        x = np.arange(25 * 3, dtype=np.float32).reshape(1, 25, 3)
        c = cs.kinect_to_common(x, center=False)[0]
        self.assertEqual(c.shape, (17, 3))
        for name, src in [("ShoulderL", 4), ("ShoulderR", 8), ("WristL", 6), ("WristR", 10), ("Head", 3),
                          ("HipL", 12), ("HipR", 16), ("FootL", 15), ("FootR", 19)]:
            np.testing.assert_array_equal(c[J[name]], x[0, src])
        np.testing.assert_allclose(c[J["HipMid"]], (x[0, 12] + x[0, 16]) / 2)
        np.testing.assert_allclose(c[J["ShoulderMid"]], (x[0, 4] + x[0, 8]) / 2)

    def test_mediapipe_indices_axes_visibility(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 33, 3)).astype(np.float32)
        vis = np.ones((2, 33), np.float32)
        vis[1, 24] = 0.3            # 右髖看不到 → HipR 與 HipMid 都缺
        c, cv = cs.mediapipe_to_common(x, vis, center=False)
        np.testing.assert_allclose(c[0, J["WristL"]], x[0, 15] * [-1, -1, 1])
        np.testing.assert_allclose(c[0, J["Head"]], (x[0, 7] + x[0, 8]) / 2 * [-1, -1, 1], rtol=1e-6)
        np.testing.assert_allclose(c[0, J["FootR"]], x[0, 32] * [-1, -1, 1])
        self.assertAlmostEqual(float(cv[1, J["HipMid"]]), 0.3, places=6)
        self.assertTrue(np.isnan(c[1, J["HipR"]]).all() and np.isnan(c[1, J["HipMid"]]).all())
        self.assertTrue(np.isfinite(c[1, J["HipL"]]).all())

    def test_left_raise_synthetic(self):
        """清單 1（合成）：Kinect y 向上、MediaPipe y 向下，各自「舉左手」，轉換後 WristL 都高於 WristR。"""
        k = np.zeros((1, 25, 3), np.float32)
        k[0, 6, 1], k[0, 10, 1] = 0.6, -0.2
        m = np.zeros((1, 33, 3), np.float32)
        m[0, 15, 1], m[0, 16, 1] = -0.6, 0.2
        kc = cs.kinect_to_common(k)[0]
        mc, _ = cs.mediapipe_to_common(m, np.ones((1, 33), np.float32))
        self.assertGreater(kc[J["WristL"], 1], kc[J["WristR"], 1])
        self.assertGreater(mc[0, J["WristL"], 1], mc[0, J["WristR"], 1])


@unittest.skipUnless(HAVE_DATA, "缺 T1 試跑資料或 v3_crop_index.csv")
class TestRealData(unittest.TestCase):

    def test_1_left_right(self):
        """
        清單 1（真實資料）。對齊後：
          (a) B_ID1 Es1（T1 唯一的 Es1）中 Kinect 左腕比右腕高 > 5 cm 的幀，MediaPipe 也是左腕較高；
          (b) 左右肩的 x 符號在 heavy 骨架正常的 3 部中 ≥ 75% 的幀一致（MediaPipe 會逐幀左右跳動，追加 4 B）。
        """
        Ka, k, s, rho = aligned_kinect("B_ID1_Es1")
        Mc = real("B_ID1_Es1")["Mc"]
        kd = Ka[:, J["WristL"], 1] - Ka[:, J["WristR"], 1]
        md = Mc[:, J["WristL"], 1] - Mc[:, J["WristR"], 1]
        sel = np.isfinite(kd) & np.isfinite(md) & (kd > 0.05)
        agree = float(np.mean(md[sel] > 0))
        print(f"\n  [1a] B_ID1_Es1 對齊 k={k} s={s:.4f} ρ={rho:.3f}；Kinect 左腕高 > 5 cm 的幀 {sel.sum()}，"
              f"MediaPipe 也是左腕高 {agree:.0%}")
        self.assertGreaterEqual(sel.sum(), 3)
        self.assertGreater(agree, 0.5)
        for key in T1_KEYS:
            Ka, k, s, rho = aligned_kinect(key)
            Mc = real(key)["Mc"]
            kx = Ka[:, J["ShoulderL"], 0] - Ka[:, J["ShoulderR"], 0]
            mx = Mc[:, J["ShoulderL"], 0] - Mc[:, J["ShoulderR"], 0]
            ok = np.isfinite(kx) & np.isfinite(mx)
            a = float(np.mean(np.sign(kx[ok]) == np.sign(mx[ok])))
            print(f"  [1b] {key}: k={k} s={s:.4f} ρ={rho:.3f}；左右肩 x 符號一致 {a:.0%}（{ok.sum()} 幀）")
            if key in CLEAN:
                self.assertGreaterEqual(a, 0.75, key)

    def test_2_axes(self):
        """
        清單 2：轉換後 y 向上（兩邊 Head 都高於 HipMid）；人面向鏡頭時兩邊 Head 相對 HipMid 的 z 符號一致；
        另以「腳尖在腳踝前方（較靠近鏡頭）」確認 z。z 以多數決（≥ 4/5 部）判定。
        """
        agree_head = agree_foot = 0
        for key in T1_KEYS:
            d = real(key)
            Kc, Mc = d["Kc"], d["Mc"]
            for name, X in [("Kinect", Kc), ("MediaPipe", Mc)]:
                self.assertGreater(np.nanmedian(X[:, J["Head"], 1] - X[:, J["HipMid"], 1]), 0, f"{key} {name} y")
            kv = np.flatnonzero(np.isfinite(Kc[:, J["Head"], 2]))[:cs.REF_FRAMES]
            mv = np.flatnonzero(np.isfinite(Mc[:, J["Head"], 2]) & np.isfinite(Mc[:, J["HipMid"], 2]))[:cs.REF_FRAMES]
            kz = np.median(Kc[kv, J["Head"], 2] - Kc[kv, J["HipMid"], 2])
            mz = np.median(Mc[mv, J["Head"], 2] - Mc[mv, J["HipMid"], 2])
            kf = np.nanmedian(Kc[:, J["FootL"], 2] - Kc[:, J["AnkleL"], 2])
            mf = np.nanmedian(Mc[:, J["FootL"], 2] - Mc[:, J["AnkleL"], 2])
            agree_head += np.sign(kz) == np.sign(mz)
            agree_foot += np.sign(kf) == np.sign(mf)
            print(f"\n  [2] {key}: 站姿 Head−HipMid z  Kinect {kz:+.3f}  MP {mz:+.3f} | "
                  f"FootL−AnkleL z  Kinect {kf:+.3f}  MP {mf:+.3f}", end="")
            self.assertLess(kf, 0, f"{key} Kinect 腳尖應較靠近鏡頭")
        print(f"\n  [2] Head z 符號一致 {agree_head}/5，腳尖 z 符號一致 {agree_foot}/5")
        self.assertGreaterEqual(agree_head, 4)
        self.assertGreaterEqual(agree_foot, 4)

    def test_3_midpoints(self):
        """清單 3：HipMid = (HipL + HipR)/2、ShoulderMid 同理，誤差 < 1e-6（兩邊）。"""
        for key in T1_KEYS:
            d = real(key)
            for X in (d["Kc"], d["Mc"]):
                ok = np.all(np.isfinite(X[:, [J["HipMid"], J["HipL"], J["HipR"], J["ShoulderMid"]], 0]), axis=1)
                self.assertTrue(ok.any())
                for mid, l, r in [("HipMid", "HipL", "HipR"), ("ShoulderMid", "ShoulderL", "ShoulderR")]:
                    err = np.abs(X[ok, J[mid]] - (X[ok, J[l]] + X[ok, J[r]]) / 2).max()
                    self.assertLess(err, 1e-6, f"{key} {mid}")

    def test_4_normalize_expert_standing(self):
        """清單 4：專家（E_ID14 Es2）正規化後 HipMid→ShoulderMid 中位數 = 1、參考幀 HipMid 中位數在原點；兩邊都測。"""
        d = real("E_ID14_Es2")
        for name, X in [("Kinect", d["Kc"]), ("MediaPipe", cs.fill_short_gaps(d["Mc"])[0])]:
            N, p = cs.normalize(X)
            torso = np.linalg.norm(N[:, J["ShoulderMid"]] - N[:, J["HipMid"]], axis=-1)
            self.assertAlmostEqual(float(np.nanmedian(torso)), 1.0, places=5, msg=name)
            np.testing.assert_allclose(np.median(N[p["ref_idx"], J["HipMid"]], axis=0), 0, atol=1e-5, err_msg=name)
            R = p["R"].astype(np.float64)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=5)
            # 中位數逐分量計算、與旋轉不可交換，所以在正規化前的座標檢查：R 把參考幀向量的中位數轉到 +y / xy 平面
            raw = np.asarray(X, dtype=np.float64)[p["ref_idx"]]
            up = R @ np.median(raw[:, J["ShoulderMid"]] - raw[:, J["HipMid"]], axis=0)
            lat = R @ np.median(raw[:, J["HipR"]] - raw[:, J["HipL"]], axis=0)
            self.assertLess(np.abs(up[[0, 2]]).max(), 1e-5)
            self.assertGreater(up[1], 0)
            # 垂直軸精確；左右軸經 Gram-Schmidt 投影，所以髖連線只保證 z 分量為 0（y 分量 = 髖與軀幹不垂直的程度）
            self.assertLess(abs(lat[2]), 1e-5)
            self.assertGreater(lat[0], 0)
            print(f"\n  [4] {name}: scale = {p['scale']:.3f} m，ref 幀 {p['ref_idx'][0]}–{p['ref_idx'][-1]}，"
                  f"髖連線傾斜 {np.degrees(np.arctan2(lat[1], lat[0])):+.1f}°", end="")

    def test_5_shape_nan(self):
        """清單 5：輸出 (T, 17, 3)；visibility < 0.5 → NaN；短缺漏內插後沒有 ≤ 5 幀的內部缺漏，長缺漏保留並回報。"""
        for key in T1_KEYS:
            d = real(key)
            Mc, vis = d["Mc"], d["Mvis"]
            self.assertEqual(Mc.shape, (len(d["M33"]), 17, 3))
            self.assertEqual(d["Kc"].shape, (len(d["K25"]), 17, 3))
            self.assertTrue(np.isnan(Mc[vis < 0.5]).all())
            self.assertTrue(np.isfinite(Mc[vis >= 0.5]).all())
            F, info = cs.fill_short_gaps(Mc)
            for v in range(17):
                miss = ~np.isfinite(F[:, v, 0])
                dd = np.diff(np.r_[0, miss.astype(int), 0])
                for s, e in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
                    if 0 < s and e < len(F):
                        self.assertGreater(e - s, cs.MAX_SHORT_GAP, f"{key} {cs.COMMON_JOINTS[v]} {s}")
            print(f"\n  [5] {key}: vis<0.5 點 {int((vis < 0.5).sum())}，內插 {info['filled_points']}，"
                  f"長缺漏 {info['long_gap_points']} 點 / {len(info['long_gaps'])} 段（最長 {info['longest_gap']} 幀），"
                  f"頭尾缺漏 {info['edge_gap_points']} 點", end="")

    def test_raw_row_index(self):
        """追加 2 C：Kinect 依原始 CSV 行號排列；[start, end) 以外為 NaN；保留段以第二版正規化後 = v3 陣列。"""
        import preprocess_kimore as pk
        from dataset import load_kimore
        samples, _ = load_kimore("data/kimore/processed/kimore_full_v3.pkl")
        for key in T1_KEYS:
            d = real(key)
            r, K = d["crop"], d["K25"]
            self.assertEqual(len(K), int(r.n_raw))
            seg = K[int(r.start):int(r.end)]
            self.assertTrue(np.isfinite(seg).all())
            self.assertTrue(np.isnan(K[:int(r.start)]).all() and np.isnan(K[int(r.end):]).all())
            aligned, _ = pk.body_centric_alignment(seg)
            v3 = samples[d["sample_id"]]["skeleton"].transpose(1, 2, 0)
            self.assertLess(float(np.abs(aligned - v3).max()), 1e-5, key)


class TestAddendum7(unittest.TestCase):
    """追加 7 B（參考幀、中位數）、C（逐幀減 HipMid）、D（未偵測幀補齊）。"""

    @staticmethod
    def _standing(T=30, seed=0):
        """合成站姿：HipL/R 左右各 0.1 m、ShoulderMid 在上方 0.5 m，加一點雜訊與整體平移。"""
        rng = np.random.default_rng(seed)
        x = np.zeros((T, 17, 3))
        x[:, J["HipL"]] = [-0.1, 0, 0]
        x[:, J["HipR"]] = [0.1, 0, 0]
        x[:, J["ShoulderL"]] = [-0.18, 0.5, 0]
        x[:, J["ShoulderR"]] = [0.18, 0.5, 0]
        x[:, J["HipMid"]] = (x[:, J["HipL"]] + x[:, J["HipR"]]) / 2
        x[:, J["ShoulderMid"]] = (x[:, J["ShoulderL"]] + x[:, J["ShoulderR"]]) / 2
        x += rng.normal(scale=0.002, size=x.shape) + np.linspace(0, 0.3, T)[:, None, None] * [1, 0, 0.5]
        return x.astype(np.float32)

    def test_center_kinect(self):
        x = np.random.default_rng(1).normal(size=(5, 25, 3)).astype(np.float32)
        c = cs.kinect_to_common(x)
        np.testing.assert_allclose(c[:, J["HipMid"]], 0, atol=1e-7)
        np.testing.assert_allclose(c[:, J["HipL"]], -c[:, J["HipR"]], atol=1e-6)
        raw = cs.kinect_to_common(x, center=False)
        np.testing.assert_allclose(c, raw - raw[:, [J["HipMid"]]], atol=1e-6)

    def test_center_before_mask_mediapipe(self):
        """MediaPipe 先用輸出的髖座標置中再遮罩：髖 visibility 低時其他點仍保留；vis_threshold=None 不遮罩。"""
        x = np.random.default_rng(2).normal(size=(3, 33, 3)).astype(np.float32)
        vis = np.ones((3, 33), np.float32)
        vis[1, 23] = 0.2
        c, _ = cs.mediapipe_to_common(x, vis)
        self.assertTrue(np.isnan(c[1, J["HipMid"]]).all() and np.isnan(c[1, J["HipL"]]).all())
        self.assertTrue(np.isfinite(c[1, J["WristL"]]).all())
        mid = (x[1, 23] + x[1, 24]) / 2 * [-1, -1, 1]
        np.testing.assert_allclose(c[1, J["WristL"]], x[1, 15] * [-1, -1, 1] - mid, atol=1e-6)
        c2, _ = cs.mediapipe_to_common(x, vis, vis_threshold=None)
        self.assertTrue(np.isfinite(c2).all())
        np.testing.assert_allclose(c2[:, J["HipMid"]], 0, atol=1e-7)

    def test_fill_undetected(self):
        x = np.arange(6 * 17 * 3, dtype=np.float32).reshape(6, 17, 3)
        det = np.array([False, True, False, False, True, False])
        y = x.copy()
        y[~det] = np.nan
        f = cs.fill_undetected(y, det)
        np.testing.assert_array_equal(f[det], x[det])
        np.testing.assert_array_equal(f[0], x[1])                       # 頭：最近的有效幀
        np.testing.assert_array_equal(f[5], x[4])                       # 尾
        np.testing.assert_allclose(f[2], x[1] + (x[4] - x[1]) / 3, rtol=1e-6)
        with self.assertRaises(ValueError):
            cs.fill_undetected(y, np.zeros(6, bool))

    def test_normalize_ref_idx_and_median(self):
        x = self._standing()
        N_auto, p_auto = cs.normalize(x)
        self.assertEqual(p_auto["ref_idx"], list(range(10)))
        ref = list(range(5, 15))
        N, p = cs.normalize(x, ref_idx=ref)
        self.assertEqual(p["ref_idx"], ref)
        np.testing.assert_allclose(np.median(N[ref, J["HipMid"]], axis=0), 0, atol=1e-5)
        # 中位數：參考幀中 1 幀壞掉（髖連線轉 90°），方向幾乎不變
        bad = x.copy()
        bad[7, J["HipL"]], bad[7, J["HipR"]] = [0, 0, -0.1], [0, 0, 0.1]
        bad[7, J["HipMid"]] = 0
        _, pb = cs.normalize(bad, ref_idx=ref)
        ang = np.degrees(np.arccos(np.clip(np.dot(pb["R"][0], p["R"][0]), -1, 1)))
        self.assertLess(ang, 2.0)
        # 參考幀不合格 → ValueError（「無有效參考幀」由呼叫端記錄）
        nan = x.copy()
        nan[6, J["ShoulderMid"]] = np.nan
        with self.assertRaises(ValueError):
            cs.normalize(nan, ref_idx=ref)
        with self.assertRaises(ValueError):
            cs.normalize(x, ref_idx=ref[:9])

    @unittest.skipUnless(HAVE_DATA, "缺 T1 試跑資料或 v3_crop_index.csv")
    def test_real_center_fill(self):
        for key in T1_KEYS:
            d = real(key)
            Kc = d["Kc"]
            ok = np.isfinite(Kc[:, 0, 0])
            np.testing.assert_allclose(Kc[ok, J["HipMid"]], 0, atol=1e-6)
            z = np.load(f"{TRIAL}/{key}.npz")
            Mraw, _ = cs.mediapipe_to_common(z["world"], z["visibility"], vis_threshold=None)
            self.assertTrue(np.isfinite(Mraw[z["detected"]]).all(), key)
            F = cs.fill_undetected(Mraw, z["detected"])
            self.assertTrue(np.isfinite(F).all(), key)
            np.testing.assert_array_equal(F[z["detected"]], Mraw[z["detected"]])


class TestGraph(unittest.TestCase):

    def test_adjacency(self):
        A = cs.common_adjacency()
        self.assertEqual(A.shape, (3, 17, 17))
        self.assertEqual(len(cs.COMMON_INWARD), 16)
        np.testing.assert_array_equal(A[0], np.eye(17))
        # 樹：從 HipMid 沿向心邊反向可到達所有點
        children = {}
        for c, p in cs.COMMON_INWARD:
            children.setdefault(p, []).append(c)
        seen, stack = set(), [J["HipMid"]]
        while stack:
            n = stack.pop()
            seen.add(n)
            stack += children.get(n, [])
        self.assertEqual(seen, set(range(17)))
        for k in (1, 2):
            rs = A[k].sum(axis=1)
            self.assertTrue(np.allclose(rs[rs > 0], 1))
        self.assertTrue(np.allclose(A[1].sum(axis=1), [0] + [1] * 16))      # 只有 HipMid 沒有父節點

    def test_graph_class(self):
        from model.graph.common_17_graph import Graph
        g = Graph()
        self.assertEqual(g.num_node, 17)
        np.testing.assert_array_equal(g.A, cs.common_adjacency())


if __name__ == "__main__":
    unittest.main(verbosity=2)
