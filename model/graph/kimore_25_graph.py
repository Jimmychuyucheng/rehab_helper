# 檔案路徑: model/graph/kimore_25_graph.py
import numpy as np

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 25
        self.self_link = [(i, i) for i in range(self.num_node)]
        # 依據 Kinect v2 / KIMORE 骨骼樹狀結構定義相鄰骨骼連線
        self.inward = [
            (1, 0), (20, 1), (2, 20), (3, 2),          # 脊椎基底至頭部
            (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 6),  # 左手臂與左手
            (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 10), # 右手臂與右手
            (12, 0), (13, 12), (14, 13), (15, 14),     # 左腿至左腳
            (16, 0), (17, 16), (18, 17), (19, 18)      # 右腿至右腳
        ]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode):
        if labeling_mode == 'spatial':
            # 建立形狀為 (3, 25, 25) 的相鄰矩陣：自身、向心、離心
            A = np.zeros((3, self.num_node, self.num_node))
            for i, j in self.self_link:
                A[0, i, j] = 1
            for i, j in self.inward:
                A[1, i, j] = 1
            for i, j in self.outward:
                A[2, i, j] = 1
            # 歸一化相鄰矩陣 (D^-1 * A)
            for k in range(3):
                deg = np.sum(A[k], axis=1, keepdims=True)
                deg[deg == 0] = 1
                A[k] = A[k] / deg
            return A
        else:
            raise ValueError("Unsupported labeling mode")