import argparse
import yaml
import uuid
import torch
import numpy as np
import random
# from calflops import calculate_flops
import torch.optim as optim
from model.S3D import S3D
from model.SLR_STGCN import ctrgcn_model, whole_body_model
def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        string = file.read()
        dict = yaml.safe_load(string)
    return dict

# def count_parameter(model):
#     x = torch.ones(1, 2, 50,60)
#     y = torch.ones(1, 2, 50,42)
#     z = torch.ones(1, 2, 50,21)
#     x_shape = (1, 2, 50,60)
#     y_shape = (1, 2, 50,42)
#     z_shape = (1, 2, 50,21)
#     flops, macs, params = calculate_flops(model=model, 
#                                     input_shape=x_shape,
#                                     #   kwargs = {'data':x,'single_hand':z,'both_hand':y},
#                                     output_as_string=True,
#                                     print_detailed=True,
#                                     output_precision=4)
#     print("net FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#     # print(flops.total())
#     print( f'Number of paremeter ={sum(p.numel() for p in model.parameters() if p.requires_grad)/1000**2}M')

def get_parser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/base_config.yaml' )
    parser.add_argument('--data_info_path', type=str, default="./WLASL_v0.3.json")
    parser.add_argument('--dir_path', type=str, default="./landmarks/wlasl_rtmw_1")
    parser.add_argument('--save_path', type=str, default="./weights")
    parser.add_argument('--input_format',type=str,default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_class',type=int, default=100)
    parser.add_argument('--graph',type=str)
    parser.add_argument('--weight_path',type=str,default='weights/test100_2.pth')
    parser.add_argument('--max_frame',type=int,default=60)
    parser.add_argument('--patience',type= int, default= 6)
    parser.add_argument('--comment',type=str,default='')
    parser.add_argument('--seed',type=int,default=33)
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    parser.add_argument('--momentum',type=float, default= 0.9)
    parser.add_argument('--nesterov',type=bool, default= True)
    parser.add_argument('--pad',type=str,default='zero')
    parser.add_argument('--aug',type=list,default= [])
    parser.add_argument('--wandb_tags',type=list,default= [])
    parser.add_argument('--fig_path',type= str,default= './heatmap')
    parser.add_argument('--model_name',type=str,default='ctrgcn')
    parser.add_argument('--wandb_mode',type=str,default='disabled', help= 'Set to online to enable wandb')
    parser.add_argument('--warmup_steps',type=int,default= 5)
    parser.add_argument('--if_video',type=bool,default= False) 
    return parser

def load_args():
    parser = get_parser()
    arg = parser.parse_args()
    if arg.config_path is not None:
        default_arg = read_yaml(arg.config_path)
        key = vars(arg).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
    parser.set_defaults(**default_arg)
    return parser.parse_args()

def generate_rd_name():
    uuid_str = str(uuid.uuid4())
    uuid_str = uuid_str.replace('-', '')[:8]
    return uuid_str

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

def load_optimizer(model,arg):
    if arg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=arg.lr,
            momentum=arg.momentum,
            nesterov=arg.nesterov,
            weight_decay=arg.weight_decay)
    elif arg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=arg.lr,
            weight_decay=arg.weight_decay)
    elif arg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=arg.lr,
            weight_decay=arg.weight_decay)
    else:
        raise ValueError()
    return optimizer
def load_model(model_name, num_class, graph):
    print(f'Loading {model_name} model.')
    if model_name == 'ctrgcn':
        print('using ctrgcn')
        model = ctrgcn_model(num_class, graph)
    elif model_name == 'origin':
        model = whole_body_model(num_class, graph)
    elif model_name == 'S3D':
        model = S3D(num_class)
    else:
        raise ValueError('Input model format not exist.')
    # count_parameter(model)
    return model

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math


    
class LabelSmoothCE(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.2, reduction='mean', ignore_index=-100, word_emb_tab=None, norm_type='softmax', temp=0.5, variant='word_sim'):
        super(LabelSmoothCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.word_emb_sim = None
        if word_emb_tab is not None:
            self.word_emb_sim = t.matmul(F.normalize(word_emb_tab, dim=-1), F.normalize(word_emb_tab, dim=-1).T)
            self.norm_type = norm_type
            self.temp = temp
        self.variant = variant


    def forward(self, logits, label, topk_idx=None, mixup_lam=None, y_a=None, y_b=None, **kwargs):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with t.no_grad():
            # print(self.variant)
            if self.variant is None:
                num_classes = logits.size(1)
                label = label.clone().detach()
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                label[ignore] = 0
                lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / (num_classes-1)
                lb_one_hot = t.empty_like(logits).fill_(
                    lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

            elif 'dual' in self.variant:
                # assert topk_idx is not None
                B,K,N = logits.shape
                label = label.clone().detach()

                lb_one_hot = t.zeros(B*K,N).to(logits.device)
                label_exp = label.unsqueeze(1).expand(-1,K).reshape(-1)
                topk_idx = topk_idx.view(-1)
                idx = t.arange(lb_one_hot.shape[0])

                if mixup_lam is None:
                    lb_one_hot[idx, label_exp] = 0.5
                    lb_one_hot[idx, topk_idx] += 0.5
                else:
                    lb_one_hot[idx, topk_idx] += 0.5
                    y_a_exp = y_a.unsqueeze(1).expand(-1,K).reshape(-1)
                    y_b_exp = y_b.unsqueeze(1).expand(-1,K).reshape(-1)
                    lb_one_hot[idx, y_a_exp] += mixup_lam * 0.5
                    lb_one_hot[idx, y_b_exp] += (1.-mixup_lam) * 0.5

                lb_one_hot = lb_one_hot.detach().reshape(B,K,N)
                n_valid = B*K
            
            elif 'word_sim' in self.variant:
                assert self.word_emb_sim is not None
                lb_one_hot = self.word_emb_sim[label]
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                idx = t.arange(label.shape[0])
                if self.norm_type == 'l1':
                    lb_one_hot[idx, label] = 0.0
                    lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                elif self.norm_type == 'softmax':
                    lb_one_hot[idx, label] = float('-inf')
                    lb_one_hot /= self.temp
                    lb_one_hot = F.softmax(lb_one_hot, dim=-1)
                lb_one_hot *= self.lb_smooth
                lb_one_hot[idx, label] = 1.0-self.lb_smooth
                lb_one_hot = lb_one_hot.detach()
            
            else:
                raise ValueError
        
        logs = self.log_softmax(logits)
        loss = -t.sum(logs * lb_one_hot, dim=-1)
        
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss