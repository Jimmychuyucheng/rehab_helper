import torch
import torchvision
from torch.utils.data import DataLoader
from data.dataset import WLASL_Define_Dataset
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import torch.optim as optim
from model.S3D import S3D
from utils.metric import accuracy,EarlyStopping
from utils.util import load_args,generate_rd_name,init_seed,load_optimizer,load_model
import os
from torchsummary import summary
import wandb
import pickle



def main():
    args = load_args()
    device = args.device
    init_seed(args.seed)
    run = wandb.init(project='my-slr',config=args,tags=args.wandb_tags,mode=args.wandb_mode)
    if args.save_path == "./weights":
        args.save_path=os.path.join(args.save_path,generate_rd_name()+'.pth')
    print(args)
    file_weight = r'pretrained_models/S3D_kinetics400.pt'
    train_data = WLASL_Define_Dataset(args.data_info_path,args.dir_path,'train',args.input_format,max_frame=args.max_frame,aug =args.aug,if_video = True)   
    val_data = WLASL_Define_Dataset(args.data_info_path,args.dir_path,'val',args.input_format,max_frame=args.max_frame,if_video = True)
    train_dataloader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_dataloader = DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    print(args.comment)
    print("Using config {}.".format(args.config_path))
    print("Train data shape {}, Val data shape {}.".format(len(train_data),len(val_data)))
    print(f"Using device {device}.")
    model = S3D(num_class = args.num_class).to(device)
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')
    # summary(model,(1,2,50,60))
    model = nn.DataParallel(model, device_ids=[0,1]).to("cuda")
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = load_optimizer(model,args)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = args.epoch, eta_min = 1e-8,last_epoch=-1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.3,patience = args.patience, min_lr= 1e-7)
    # warm_up_scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps)
    torch.backends.cudnn.benchmark = False
    model.train()
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}, Trainable params: {total_params_trainable}.")
    best_epoch = 0
    best_acc = 0
    best_loss = float('inf')
    train_loss = []
    train_acc  = []
    val_loss = [] 
    val_acc = []
    for e in tqdm(range(args.epoch)):
        tmp_loss = 0
        tmp_acc = 0
        for batch, data in enumerate(train_dataloader):
            input_data = data['data'].to(device)
            label = data['label'].to(device)
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output,label)
            tmp_loss += (loss.item()*input_data.size()[0])
            tmp_acc += accuracy(output,label)
            loss.backward()
            optimizer.step()

        train_loss.append(tmp_loss/len(train_data))
        train_acc.append(tmp_acc/len(train_data))

        model.eval()
        with torch.no_grad():
            tmp_loss = 0
            tmp_acc = 0
            for batch, data in enumerate(val_dataloader):
                input_data = data['data'].to(device)
                label = data['label'].to(device)
                output = model(input_data)
                loss = criterion(output,label)
                tmp_loss += (loss.item()*input_data.size()[0])
                tmp_acc += accuracy(output,label)
        val_loss.append(tmp_loss/len(train_data))
        val_acc.append(tmp_acc/len(val_data))
        if val_acc[e] >= best_acc:
            best_epoch = e
            best_loss = val_loss[e]
            best_acc = val_acc[e]
            
            torch.save(model.state_dict(),os.path.join(args.save_path))
        # if e < args.warmup_steps:
        #     warm_up_scheduler.step()
        # else:
        scheduler.step()
        # scheduler.step()
        wandb.log({'train_acc':train_acc[e]*100,'val_acc':val_acc[e]*100})
        wandb.log({'train_loss':train_loss[e],'val_loss':val_loss[e]})
        # current_lr = scheduler._last_lr[0]
        current_lr = optimizer.param_groups[0]['lr']
        # current_lr = scheduler.get_last_lr()[0]
        
        print(" epoch {}, acc : {:.3f}%, loss : {:.3f}, val_acc : {:.3f}%, val_loss : {:.3f}, lr {:.6f}.".format(e,train_acc[e]*100,train_loss[e],val_acc[e]*100,val_loss[e],current_lr))
    print(f" save epoch {best_epoch} val loss {best_loss} val acc {best_acc*100:.3f}% at {args.save_path}.")
    print(args.comment)
    # torch.save(model.state_dict(),os.path.join(args.save_path))
    # artifact = wandb.Artifact('model', type='model')
    # artifact.add_file(args.save_path)
    # run.log_artifact(artifact)
    # run.finish()


if __name__ == '__main__':
    main()