import sys
import numpy as np

sys.path.extend(['..'])
from . import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
hand = [(1,0),(2,1),(3,2),(4,3),
        (5,0),(6,5),(7,6),(8,7),
        (9,0),(10,9),(11,10),(12,11),
        (13,0),(14,13),(15,14),(16,15),
        (17,0),(18,17),(19,18),(20,19)]
# finger_connect = [(4,8),(8,12),(12,16),(16,20),(4,0),(8,0),(12,0),(16,0),(20,0)]
# hand+=finger_connect
hand = [(i, j) for (i, j) in hand] 

inward= hand
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor + self_link
        self.A = tools.get_multiadj_spatial_graph(num_node, self_link, inward, outward,1)
        self.A_adjacency = tools.get_adjacency_matrix_undigraph(num_node, self_link, inward, outward)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)