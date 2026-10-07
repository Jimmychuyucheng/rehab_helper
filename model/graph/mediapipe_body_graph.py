import sys
import numpy as np

sys.path.extend(['..'])
from . import tools

num_node = 7
self_link = [(i, i) for i in range(num_node)]
# need to edit to mediapipe holistic landmarks
inward= [(0,1), (0,2), (2,4), (1,3), (3,5)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A