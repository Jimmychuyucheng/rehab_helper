import numpy as np 
from tqdm import tqdm
from torch.nn import Softmax

from utils.metric import compute_top_n_accuracy

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
# left_hand = np.load('left_hand_rtmw60_result.npy')
# both_hand = np.load('both_hand_rtmw60_result.npy')
# full_body = np.load('full_body_rtmw60_result.npy')
ctrgcn_stream = np.load('ctrgcn_WLASL100.npy')
S3D_stream = np.load('S3D_WLASL100.npy')
label = np.load('label.npy')
label_dict = np.load('label_dict.npy',allow_pickle= True)

label_dict = {v: k for k, v in label_dict.item().items()}
# print(f'{full_body.shape}, {both_hand.shape}, {left_hand.shape}')
alpha = 1
beta = 0.5
count= 0
top_5 = 0
# left_hand = softmax(left_hand)
# both_hand = softmax(both_hand)
# full_body = softmax(full_body)

for i in range(ctrgcn_stream.shape[0]):
    p1 = ctrgcn_stream[i,0]
    p2 = S3D_stream[i,0]
    p = p1*alpha + p2*beta
    p_result = np.argmax(p)
    p1_top5 = dict(zip([label_dict[i]for i in np.argsort(p1)[-5:]],np.sort(p1)[-5:]))
    p2_top5 = dict(zip([label_dict[i]for i in np.argsort(p2)[-5:]],np.sort(p2)[-5:]))
    l = np.argmax(label[i,0])
    best_n = np.argsort(p)[-5:]
    top_5 += (l in best_n)
    # if np.argmax(p) != l:
    #     print(f'left hand {p1_top5},\n both hand {p2_top5},\n full body {p3_top5},\n ensemble {label_dict[p_result]}, true {label_dict[l]}.')
    if np.argmax(p) != l:
        print(f'ctrgcn {p1_top5},\n S3D {p2_top5},\n ensemble {label_dict[p_result]}, true {label_dict[l]}.')
    count+=(l == p_result)
print(f'Ensemble accuracy {count/ctrgcn_stream.shape[0]*100}% top 5 accuracy {top_5/ctrgcn_stream.shape[0]*100}%.')
