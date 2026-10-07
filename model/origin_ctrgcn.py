import math
import pdb

import numpy as np
import torch
from torch import nn, einsum

from model.modules import conv_init,bn_init,import_class,MSTCN,TemporalConv

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os


class Alter_CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(Alter_CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2 = self.conv1(x), self.conv2(x)
        x3 = self.conv3(x)
        x1 = self.softmax(x1.permute(0,1,3,2)@x2)
        x1 = x1 * alpha  + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3) 
        return x1 


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha  + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_ctrgcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, layer_num):
        super(unit_ctrgcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.A = nn.Parameter(A.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act =  nn.ReLU(inplace=True)
        self.layer_num = layer_num

    def forward(self, x, id):
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.act(y)

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,  kernel_size=5, dilations=[1,2], layer_num=0, num_point=21, t=50):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A, layer_num= layer_num)
        self.tcn1 = MSTCN(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=residual,t=t)
        self.bn = nn.BatchNorm2d(out_channels)
        # print(dilations)
        self.act =  nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x,id=None):
        y = self.tcn1(self.gcn1(x,id))
        # y = self.gcn1(self.tcn1(x),id)
        y = self.act(y + self.residual(x))
        return y

# graph module position model.graph
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0.2, adaptive=True , bone= False,k=0):
        super(Model, self).__init__()

        if not bone:
            A = np.stack([np.eye(num_point)] * 3, axis=0)
        else:
            self.graph = import_class(graph)(**graph_args)
            A = self.graph.A
        A = torch.tensor(A,dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('A', A)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A.clone(), residual=False,  layer_num= 1, num_point=num_point)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A.clone(), layer_num= 2, num_point=num_point)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A.clone(), layer_num= 3, num_point=num_point)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A.clone(),layer_num= 4, num_point=num_point)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A.clone(), stride=2, layer_num= 5, num_point=num_point)
        # self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A.clone(), layer_num= 6, num_point=num_point)
        # self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A.clone(), layer_num= 7, num_point=num_point)
        # self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A.clone(),stride=2)
        # self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A.clone())
        # self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A.clone())
        self.relu = nn.ReLU(inplace=True)
        # self.pattern = nn.Conv1d(in_channels, base_channel*2)
        self.fc = nn.Linear(base_channel*2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
    
    def forward(self, x, id=''):
        N, C, T, V = x.size()
        # p = x.mean(-2)
        x = x.permute(0, 3, 1, 2).contiguous().view(N,  V * C, T)
        x = self.data_bn(x)
        x= x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        # origin_x = x
        x = self.l1(x, id+'_l1')
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x, id+'_l4')
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x, id+'_l7')
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)
        # N,C,T,V
        # o_x = self.forward_origin(origin_x)
        # x = self.relu(x+o_x)
        extra = x
        c_new = x.size(1)
        
        x = x.contiguous().view(N, c_new, -1)
        x = x.mean(2)
        
        # p = self.pattern(p)

        x = self.drop_out(x)
        return self.fc(x), extra