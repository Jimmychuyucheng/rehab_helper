import sys
import numpy as np

sys.path.extend(['..'])
from . import tools

num_node = 57
self_link = [(i, i) for i in range(num_node)]
hand = [(1,0),(2,1),(3,2),(4,3),
        (5,0),(6,5),(7,6),(8,7),
        (9,0),(10,9),(11,10),(12,11),
        (13,0),(14,13),(15,14),(16,15),
        (17,0),(18,17),(19,18),(20,19)]
finger_connect = [(4,8),(8,12),(12,16),(16,20)]
hand += finger_connect
left_hand = [(i+16, j+16) for (i, j) in hand] 
right_hand = [(i+37, j+37) for (i, j) in hand] 

inward= [(0,1),(0,2),(1,3),(2,4),(5,6),(0,5),(0,6),(0,14),(0,15),#face
         (0,13),(13,7),(13,8),(7,9),(8,10),(9,11),(10,12),#body
        ]
inward += left_hand + right_hand
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
            A = tools.get_multiscale_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A