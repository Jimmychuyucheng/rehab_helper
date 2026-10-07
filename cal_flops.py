import torch
from model.SLR_STGCN import ctrgcn_model
from fvcore.nn import parameter_count_table,FlopCountAnalysis

def cal_flops():
    tensor = torch.randn(1, 2, 50, 60)
    model = ctrgcn_model(100, None)
    print(parameter_count_table(model))
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total())
if __name__ == '__main__':
    cal_flops()
