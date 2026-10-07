import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from model.net.utils.tgcn import ConvTemporalGraphical
# from model.net.utils.graph import Graph

from model.net.st_gcn import Model as ST_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V).zero_(),
                        x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                        torch.cuda.FloatTensor(N, C, 1, V).zero_()), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res