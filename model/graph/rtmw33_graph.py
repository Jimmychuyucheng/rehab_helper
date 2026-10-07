import sys
import numpy as np

sys.path.extend(['..'])
from . import tools

num_node = 34
self_link = [(i, i) for i in range(num_node)]
hand = [(1,0),(2,1), (3,0),(4,3), (5,0),(6,5), (7,0),(8,7), (9,0),(10,9)]

left_hand = [(j+11, i+11) for (i, j) in hand] 
right_hand = [(j+23, i+23) for (i, j) in hand] 

inward= [(1,0),(2,0),(3,1),(4,2),(5,0),(6,5),(7,5),(9,7),(8,6),(10,0),(11,9),(23,8)] 
inward += (left_hand + right_hand)
# inward += finger_to_body
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = tools.get_multiadj_spatial_graph(num_node, self_link, inward, outward,1)