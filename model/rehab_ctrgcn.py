from model.graph.kimore_25_graph import Graph as KimoreGraph

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

class TDGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1): # r = 4 r = 16
        super(TDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # This convolution (self.conv2) is redundant, 
        # but when you want to use the weight files we provide for action recognition, you have to uncomment it!
        # self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1) 
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1, gamma=0.1):

        x1, x3 = self.conv1(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x3.mean(-3).unsqueeze(-1) - x3.mean(-3).unsqueeze(-2))
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)
        x1 = x1 * beta  + x5 * gamma

        return x1

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
        self.act = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2 = self.conv1(x), self.conv2(x)
        x3 = self.conv3(x)
        x1 = self.act(x1.permute(0,1,3,2)@x2)
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
        # print(f"x shape {x.size()}, x1 shape {self.conv1(x).size()}, x2 shape {x1.size()}, x3 shape {x3.size()}")
        # print(x1.unsqueeze(-1).size(),' ',x2.unsqueeze(-2).size())
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha  + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        # print(x1.size())
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
            self.convs.append(Alter_CTRGC(in_channels, out_channels))
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
        self.act =  nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x,id=None):
        y = self.tcn1(self.gcn1(x,id))
        y = self.act(y + self.residual(x))
        return y


# graph module position model.graph
class Model(nn.Module):
    def __init__(self, num_class=1, num_point=25, num_person=1, in_channels=3, drop_out=0.2, **kwargs):
        super(Model, self).__init__()
        
        # 載入 25 節點相鄰矩陣 (形狀: 3, 25, 25)
        graph_obj = KimoreGraph()
        A = torch.tensor(graph_obj.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.num_class = num_class
        self.num_point = num_point

        base_channel = 64
        # num_person 設為 1 (復健為單人)，關節數為 25
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # 7 層 TCN_GCN_unit 保持不變，內部傳入 num_point=25
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A.clone(), residual=False, layer_num=1, num_point=num_point)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A.clone(), layer_num=2, num_point=num_point)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A.clone(), layer_num=3, num_point=num_point)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A.clone(), layer_num=4, num_point=num_point)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A.clone(), stride=2, layer_num=5, num_point=num_point)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A.clone(), layer_num=6, num_point=num_point)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A.clone(), layer_num=7, num_point=num_point)
        
        # 將全連接層輸出改成 1 (預測臨床連續評分)
        self.fc = nn.Linear(base_channel * 2, 1)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 1))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else (lambda x: x)

    def forward(self, x, return_feature=False):
        # x 輸入維度: (N, C=3, T=100, V=25)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x) # 輸出特徵 (N, 128, T', V)
        
        # 空間平均池化 (對關節維度平均)
        feat_pooled = x.mean(dim=3) # (N, 128, T')，可供時序 DTW 比對使用
        
        # 時序全域平均池化 (對動作整段進行綜合評分)
        global_feat = feat_pooled.mean(dim=2) # (N, 128)
        score = self.fc(self.drop_out(global_feat)) # (N, 1) 臨床量化評分
        
        if return_feature:
            return score, feat_pooled
        return score
    