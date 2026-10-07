import sys
import numpy as np

sys.path.extend(['..'])
from . import tools

num_node = 60
self_link = [(i, i) for i in range(num_node)]
hand = [(1,0),(2,1),(3,2),(4,3),
        (5,0),(6,5),(7,6),(8,7),
        (9,0),(10,9),(11,10),(12,11),
        (13,0),(14,13),(15,14),(16,15),
        (17,0),(18,17),(19,18),(20,19)]
# finger_connect = [(4,8),(8,12),(12,16),(4,0),(8,0),(12,0),(16,0)]
# hand+=finger_connect
left_hand = [(j+18, i+18) for (i, j) in hand] 
right_hand = [(j+39, i+39) for (i, j) in hand] 
finger_to_head = [(22, 0), (26, 0), (30, 0), (34, 0), (38, 0), (43, 0), (47, 0), (51, 0), (55, 0), (59, 0)]
# finger_to_body = [(22, 5), (26, 5), (30, 5), (34, 5), (38, 5), (43, 5), (47, 5), (51, 5), (55, 5), (59, 5)]
inward= [(1,0),(2,0),(3,1),(4,2),(5,0),(6,5),(5,7),(6,8),(7,9),(8,10),(9,11),(0,15),(15,14),(15,16),(15,17),(10,18),(11,39)]

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
        self.A_adjacency = tools.get_adjacency_matrix_undigraph(num_node, self_link, inward, outward)
        self.A = tools.get_multiadj_spatial_graph(num_node, self_link, inward, outward,1)