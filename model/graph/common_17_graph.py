# 檔案路徑: model/graph/common_17_graph.py
# v7 第 3.2 節：Kinect / MediaPipe 共同骨架（17 點）的圖，介面同 kimore_25_graph.Graph。
from common_skeleton import COMMON_INWARD, NUM_COMMON, common_adjacency


class Graph:
    def __init__(self, labeling_mode='spatial'):
        if labeling_mode != 'spatial':
            raise ValueError("Unsupported labeling mode")
        self.num_node = NUM_COMMON
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = list(COMMON_INWARD)
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = common_adjacency()
