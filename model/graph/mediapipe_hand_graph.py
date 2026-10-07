import sys
import numpy as np

sys.path.extend(['..'])
from . import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
# need to edit to mediapipe holistic landmarks
inward= [(1,0),(2,1),(3,2),(4,3),
                           (5,0),(6,5),(7,6),(8,7),
                           (9,0),(10,9),(11,10),(12,11),
                           (13,0),(14,13),(15,14),(16,15),
                           (17,0),(18,17),(19,18),(20,19)]
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