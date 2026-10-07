import torch
import torch.nn as nn

from model.ctrgcn import Model as ctr_gcn
from model.modules import import_class
from model.net.st_gcn import Model as st_gcn
from model.origin_ctrgcn import Model as origin_ctrgcn
# from model.gcn import GCN as gcn
import math
import os 
# used gru to handle body data
"""
    if input is concat dataset, 
    then input shape wil be (batch, frame, (21+21+7)left hand and right hand and body, channel)
"""

class Model(nn.Module):

    def __init__(self, num_class, hand_arg, body_arg, dropout_rate = 0.3, **kwargs):
        super().__init__()
        self.hand_stream = st_gcn(3, hand_arg, **kwargs)
        self.body_stream = st_gcn(3, body_arg, **kwargs)
        self.body_motion_stream = st_gcn(3, body_arg, **kwargs)
        self.fcn = nn.Sequential(
            nn.Linear(256*3,num_class)
        )

    def forward(self, hand, body):

        left_hand_emb = self.hand_stream(hand[:,:,:,:21])
        right_hand_emb = self.hand_stream(hand[:,:,:,21:])
        body_emb = self.body_stream(body)
        N,C,T,V  = body.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V).zero_(),
                        body[:, :, 1:-1] - 0.5 * body[:, :, 2:] - 0.5 * body[:, :, :-2],
                        torch.cuda.FloatTensor(N, C, 1, V).zero_()), 2)
        body_motion = self.body_motion_stream(m)
        body = body_motion + body_emb
        out = torch.cat([left_hand_emb, right_hand_emb, body],dim=-1)
        
        # print(torch.max(out,1))
        out = self.fcn(out)
        return out
"""
    add gesture stream with (F,42,2) flatten as input
"""    
def weight_init(t):
    for m in t.modules():
        if isinstance(m, nn.Linear):
            dim = m.out_features
            std = 1/ math.sqrt(dim)
            nn.init.uniform_(m.weight,-std,std)

class single_hand_model(nn.Module):
    def __init__(self, num_class, graph, dropout_rate = 0.0, **kwargs):
        super().__init__()
        graph_args = {'labeling_mode': 'spatial'}
        self.left_hand_stream = ctr_gcn(num_class,21,1,'model.graph.rtmw_single_hand_graph.Graph',graph_args,2,dropout_rate, bone= True, k=5)
    def forward(self,data, id=''):
        left_hand = data[...,-21:]
        left_hand, extra = self.left_hand_stream(left_hand,os.path.join(id,'left_hand'))
        return left_hand,extra
class both_hand_model(nn.Module):
    def __init__(self, num_class, graph, dropout_rate = 0.0, **kwargs):
        super().__init__()
        graph_args = {'labeling_mode': 'spatial'}
        self.both_hand_stream = ctr_gcn(num_class,42,1,'model.graph.rtmw_both_hand_graph.Graph',graph_args,2,dropout_rate, bone= True,k=5)
    def forward(self,data, id=''):
        both_hand = data[...,-42:]
        both_hand, extra = self.both_hand_stream(both_hand,os.path.join(id,'both_hand'))
        return both_hand,extra
class whole_body_model(nn.Module):
    def __init__(self, num_class, graph, dropout_rate = 0.3, **kwargs):
        super().__init__()
        graph_args = {'labeling_mode': 'spatial'}
        self.whloe_body_stream = ctr_gcn(num_class,60,1,'model.graph.rtmw_whole_body60_graph.Graph',graph_args,2,dropout_rate, bone= True,k=8)
    def forward(self,data, body,id=''):
        whole_body,extra = self.whloe_body_stream(data,os.path.join(id,'whole_body'))
        return whole_body

class ctrgcn_model(nn.Module): 
    def __init__(self, num_class, graph, dropout_rate = 0.3, **kwargs):
        super().__init__()
        graph_args = {'labeling_mode': 'spatial'}
        self.left_hand_stream = ctr_gcn(num_class,21,1,'model.graph.rtmw_single_hand_graph.Graph',graph_args,2,dropout_rate, bone= True, k=5)
        self.both_hand_stream = ctr_gcn(num_class,42,1,'model.graph.rtmw_both_hand_graph.Graph',graph_args,2,dropout_rate, bone= True,k=5)
        self.whloe_body_stream = ctr_gcn(num_class,60,1,'model.graph.rtmw_whole_body60_graph.Graph',graph_args,2,dropout_rate, bone= True,k=8)
        # self.graph = import_class('model.graph.rtmw_whole_body60_graph.Graph')(**graph_args)
        # A = self.graph.neighbor
        # self.A = torch.IntTensor(A).permute(1,0).to('cuda:0')
        # self.fix_stream = gcn(2,256, num_class)

    def forward(self,data, id=''):
        left_hand = data[...,-21:]
        both_hand = data[...,-42:]
        # indices = torch.randperm(data.size(1))[:1].to(data.device)
        # sampled_data = data.index_select(2, indices)
        # fixed_hand = self.fix_stream(sampled_data.squeeze(-2).permute(0,2,1),self.A)
        
        left_hand, last_single_hand = self.left_hand_stream(left_hand,os.path.join(id,'left_hand'))
        both_hand, last_both_hand = self.both_hand_stream(both_hand,os.path.join(id,'both_hand'))
        whole_body, last_whole_body = self.whloe_body_stream(data,os.path.join(id,'whole_body'))
        return  both_hand, left_hand, whole_body  # ,(last_single_hand,last_both_hand,last_whole_body),fc_weight