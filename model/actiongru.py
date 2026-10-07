"""
    reference:
        https://github.com/Maghoumi/DeepNAG/tree/master
    using simple GRU to classify action

    input shape: (batch_size, seq_len, input_size)
    input size is 2*2*21(2d hand joints) + 6*3(3d body joints) = 54 (might need to add other information )
"""

import torch
import torch.nn as nn


class ActionGRU(nn.Module):
    def __init__(self, input_shape, num_classes, device = 'cuda', num_layers = 2 , *args, **kwargs):
        super(ActionGRU,self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.device = device
        self.multi_gru = nn.ModuleList(
            [nn.GRU(input_shape, 256, 2, batch_first=True),
            nn.GRU(256, 128, 2, batch_first=True)]
        )
        
        self.fcn = nn.Sequential(
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        h0 = torch.randn(2, x.size(0), 256).to(self.device)
        for i, gru in enumerate(self.multi_gru):
            if i == 0:
                x, hn = gru(x, h0)
            else:
                x, hn = gru(x)
        
        out = self.fcn(hn[-1])
        return out
