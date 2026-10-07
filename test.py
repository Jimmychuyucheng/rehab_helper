from matplotlib import animation
import torch
import torchvision
from torch.utils.data import DataLoader
from data.dataset import WLASL_Define_Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
from utils.metric import accuracy,compute_top_n_accuracy
from utils.util import load_args,init_seed, load_model
from model.SLR_STGCN import ctrgcn_model
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


device = 'cuda:0'

# hand = [(1,0),(2,1),(3,2),(4,3),
#         (5,0),(6,5),(7,6),(8,7),
#         (9,0),(10,9),(11,10),(12,11),
#         (13,0),(14,13),(15,14),(16,15),
#         (17,0),(18,17),(19,18),(20,19)]
# left_hand = [(j+18, i+18) for (i, j) in hand] 
# right_hand = [(j+39, i+39) for (i, j) in hand] 
# BODY_CONNECT = [(12,0),(13,0),(1,0),(2,0),(3,1),(4,2),(5,0),(6,5),(5,7),(6,8),(7,9),(8,10),(9,11),(0,15),(15,14),(15,16),(15,17),(10,18),(11,39)] + left_hand + right_hand

hand = [(1,0),(2,1), (3,0),(4,3), (5,0),(6,5), (7,0),(8,7), (9,0),(10,9)]

left_hand = [(j+11, i+11) for (i, j) in hand] 
right_hand = [(j+23, i+23) for (i, j) in hand] 

inward= [(1,0),(2,0),(3,1),(4,2),(5,0),(6,5),(7,5),(9,7),(8,6),(10,0),(11,9),(23,8)] 
inward += (left_hand + right_hand)

def vis_upsample(path,data):
    fig, ax = plt.subplots()
    data = data.transpose(1,2,0)
    def animtate(i):
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.invert_yaxis()
        ax.scatter(data[i,:,0],data[i,:,1],s=3)
        for idx, (x, y) in enumerate(data[i]):
            ax.text(x, y, str(idx), fontsize=8)
        for line in inward:
            ax.plot([data[i,line[0],0],data[i,line[1],0]],[data[i,line[0],1],data[i,line[1],1]],c='r')
    ani = animation.FuncAnimation(fig,animtate,frames=data.shape[0],interval=100,repeat=True) 
    ani.save(path,writer='pillow')
    plt.close()
def vis_heatmap(id_path, extra , name, fc_weight=None):
    _,C,T,V = extra.shape
    extra = np.squeeze(extra).mean(0)
    extra = extra.clip(min=0)
    extra = 2*(extra-extra.min())/(extra.max()-extra.min())
    extra = extra.transpose(1,0)
    sns.heatmap(extra, cbar=True)
    plt.savefig(id_path+'_'+name+'_heatmap.png')
    plt.close()

#wandb.init(project='my-slr-test')

vis_dir = './result_vis'
os.makedirs(vis_dir,exist_ok=True)
def main():
    args = load_args()
    init_seed(args.seed)
    print(args)
    print('Using {} config.'.format(args.config_path))
    print('Load weight path : {}'.format(args.weight_path))
    test_data =  WLASL_Define_Dataset(args.data_info_path,args.dir_path,'test',input_format= args.input_format,max_frame=args.max_frame,if_video = args.if_video)  
    label_dict = test_data.label_dict
    label_dict = {value: key for key, value in label_dict.items()}
    print("Using device {}.".format(device))
    print(f"Test data shape {len(test_data)}.")
    test_dataloader = DataLoader(test_data,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

    model = load_model(args.model_name,args.num_class, args.graph).to(device)
    # if os.path.isfile(args.weight_path):
    #     print ('loading weight file')
    #     weight_dict = torch.load(args.weight_path)
    #     model_dict = model.state_dict()
    #     for name, param in weight_dict.items():
    #         if 'module' in name:
    #             name = '.'.join(name.split('.')[1:])
    #         if name in model_dict:
    #             if param.size() == model_dict[name].size():
    #                 model_dict[name].copy_(param)
    #             else:
    #                 print (' size? ' + name, param.size(), model_dict[name].size())
    #         else:
    #             print (' name? ' + name)

    #     print (' loaded')
    # else:
    #     print ('weight file?')
    model.load_state_dict(torch.load(args.weight_path, map_location=torch.device(args.device)))
    model.eval()
    save_result = []
    save_result_vector = []
    acc = 0
    acc1 = 0
    acc2 = 0
    acc3 = 0
    top5_acc = 0
    y_pred = []
    y_label = []
    false_ins = []
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            input_data = data['data'].to(device)
            label = data['label'].to(device)
            # hand_heatmap = data['hand_heatmap'].to(device)
            id = data['id']
            id_path = os.path.join(vis_dir,id[0])
            # output = model(input_data)
            output1,output2, output3= model(input_data,id_path) # output1: single hand, output2: both hand, output3: whole body
            output = output1+output2+output3
            acc+=accuracy(output,label)
            # acc1+=accuracy(output1+output2,label)
            # acc2+=accuracy(output1+output3,label)
            # acc3+=accuracy(output2+output3,label)
            top5_acc+=compute_top_n_accuracy(output,label,5)
            save_result_vector.append(output.detach().cpu().numpy())
            save_result.append(label.detach().cpu().numpy())
            true_label = label.max(1)[1].detach().cpu().numpy().tolist()[0]
            predict_label = output.max(1)[1].detach().cpu().numpy().tolist()[0]
            
            if true_label != predict_label:
               false_ins.append([label_dict[true_label],label_dict[predict_label],id[0]])
            # fc_wegiht = fc_wegiht.detach().cpu().numpy()[predict_label,:]
            # single_hand,both_hand,whole_body = vis
            # single_hand = single_hand.detach().cpu().numpy()
            # both_hand = both_hand.detach().cpu().numpy()
            # whole_body = whole_body.detach().cpu().numpy()
            # whole_body = (whole_body-whole_body.min())/(whole_body.max()-whole_body.min())
            # print(id)
            # vis_upsample('./whole_body_'+str(id[0])+'.gif',whole_body)
            # vis_heatmap(id_path,single_hand,'single_hand')
            # vis_heatmap(id_path,both_hand,'both_hand')
            # vis_heatmap(id_path,whole_body,'whole_body',fc_wegiht)
            y_pred.append(true_label)
            y_label.append(predict_label)
            # p = output.detach().cpu().numpy()[0]
            # p1 = output1.detach().cpu().numpy()[0]
            # p2 = output2.detach().cpu().numpy()[0]
            # p3 = output3.detach().cpu().numpy()[0]
            # p1_top5 = dict(zip([label_dict[i]for i in np.argsort(p1)[-5:]],np.sort(p1)[-5:]))
            # p2_top5 = dict(zip([label_dict[i]for i in np.argsort(p2)[-5:]],np.sort(p2)[-5:]))
            # p3_top5 = dict(zip([label_dict[i]for i in np.argsort(p3)[-5:]],np.sort(p3)[-5:]))
            # p_top5 = dict(zip([label_dict[i]for i in np.argsort(p)[-5:]],np.sort(p)[-5:]))
            # with open('muti_stream_result.txt', 'a') as f:
            #     f.write(f'{id[0]}:\n {p_top5} {label_dict[true_label]}\n single hand {p1_top5}\n both hand {p2_top5}\n whole body {p3_top5}\n')

    element, cot= np.unique([l[0] for l in false_ins],return_counts=True)
    sort_index = np.argsort(cot)
    ue = element[sort_index]
    uc = cot[sort_index]
    print(dict(zip(ue,uc)))
    print(false_ins)
    if args.num_class == 100:
        conf_matrix = confusion_matrix(y_label,y_pred)
        f,ax = plt.subplots(figsize = (20,20))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',ax=ax)
        plt.xlabel('predict label')
        plt.ylabel('true label')
        plt.title('confusion matrix')
        #wandb.log ({"plot":wandb.Image(f)})
    save_result_vector = np.asarray(save_result_vector,dtype=np.float32)
    np.save('S3D_WLASL100.npy',save_result_vector)
    
    print(f'accuracy {acc/len(test_dataloader)*100:.3f}%, top 5 accuracy {top5_acc/len(test_dataloader)*100:.3f}%')
    print(f'acc1 {acc1/len(test_dataloader)*100:.3f}% acc2 {acc2/len(test_dataloader)*100:.3f}% acc3 {acc3/len(test_dataloader)*100:.3f}%')
if __name__ == '__main__':
    main()