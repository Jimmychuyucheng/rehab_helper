import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.rehab_ctrgcn import Model

# 初始化適用於 KIMORE 25 節點的模型
model = Model(num_class=1, num_point=25, num_person=1, in_channels=3)
model.eval()

# 模擬 2 筆 batch、3 座標軸、100 幀、25 關節的輸入張量
dummy_input = torch.randn(2, 3, 100, 25)

with torch.no_grad():
    score, feat = model(dummy_input, return_feature=True)


print("測試成功！")
print("模型輸出臨床分數維度:", score.shape)      # 預期: torch.Size([2, 1])
print("提取之時序特徵維度 (供 DTW 使用):", feat.shape) # 預期: torch.Size([2, 128, 50])