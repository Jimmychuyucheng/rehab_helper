from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
from data.dataset import WLASL_ML_Dataset
from torchvision.models.resnet import resnet50 as resnet
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data.visualization import show_body_3d,show_track_3d

data_json_path  = r"C:\Users\Alex\Desktop\Sign_dataset\asl100_select.json"
dir_path = r"C:\Users\Alex\Desktop\CISLR\landmarks\wlasl_mediapipe_holistic_raw"
mm2d_dir_path = r"C:\Users\Alex\Desktop\CISLR\landmarks\wlasl_mmpose2d_54"
pose_dir_path = r"C:\Users\Alex\Desktop\CISLR\landmarks\wlasl_mediapipe_pose_hand"
select_trajetory = [1,4,7,8,9,12,15,17,22,23,24]
RANDOM_NUM = 33

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc =nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

def load_data(dataset):
    data = []
    label = []
    for i in range(len(dataset)):
        d,l = dataset[i]
        data.append(d)
        label.append(l)
    data = np.array(data)
    data = data.reshape(data.shape[0],-1)
    label = np.array(label)
    return data,label

def mlp_train(data,label):
    input_dim = data.shape[1]
    epoch = 500
    hidden_dim = 256
    output_dim = 12
    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for ep in range(epoch):
        inputs = torch.tensor(data).type(torch.FloatTensor)
        targets = torch.tensor(label).type(torch.LongTensor)
        outputs = model(inputs)
        outputs = nn.functional.softmax(outputs,dim=1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(ep, 100, loss.item()))
    return model

def mlp_test(model, data, label):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data)
        targets = torch.tensor(label)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        cf = confusion_matrix(targets,predicted)
        sns.heatmap(cf,annot=True)
        plt.show()
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))

def select_model(select):  
    if select == 'SVM':
        clf = svm.SVC()
    elif select == 'RandomForest':
        clf = RandomForestClassifier(max_depth=2, random_state=RANDOM_NUM)
    elif select == 'AdaBoost':
        clf = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_NUM)
    elif select == 'SGD':   
        clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=RANDOM_NUM, max_iter=1000, tol=None)
    elif select == 'GDBT':
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,max_depth=2, random_state=RANDOM_NUM)
    elif select == 'RBF':
        clf = svm.SVC(kernel='rbf',class_weight='balanced')
    elif select == 'XGB':
        clf = XGBClassifier(tree_method="hist", early_stopping_rounds=3)
    elif select == 'DecisionTree':
        clf = DecisionTreeClassifier(max_depth=5)
    else:
        raise ValueError('Invalid model selection')
    return clf

def ml_train_test(data,label,test_data,test_label,test_dataset,select, five_fold = False):
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_NUM)
    clf = select_model(select)
    best_score = 0
    best_model = None
    if five_fold:
        for i, (train_data, val_data) in enumerate(skf.split(data,label)):
            train_data, train_label = data[train_data], label[train_data]
            val_data, val_label = data[val_data], label[val_data]
            clf.fit(train_data,train_label)
            score = clf.score(val_data,val_label)
            if score > best_score:
                best_score = score
                best_model = clf
            print('Fold: {} , Score: {}'.format(i,score))
    else:
        clf.fit(data,label)
        best_model = clf
        best_score = clf.score(data,label)
    print('Best score: ',best_score)
    clf = best_model
    result = clf.predict(test_data)
    for i,(l,r) in enumerate(zip(test_label,result)):
       if l!=r:
           print(f'Index: {i}, Label: {l}, Predict: {r}')
           print(test_dataset.get_index_data(i))
    cf = confusion_matrix(test_label,result)
    ax = plt.subplot()
    sns.heatmap(cf,ax =ax, annot=True)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.show()
    print('Accuracy of {} on the test data: {} %'.format(select, 100 * np.trace(cf) / np.sum(cf)))

## Need to add shuffle to the dataloader
def main():
    train_dataset = WLASL_ML_Dataset(data_json_path,dir_path,select_trajectory=select_trajetory,phase='train')
    val_dataset = WLASL_ML_Dataset(data_json_path,dir_path,select_trajectory=select_trajetory,phase='val')
    test_dataset = WLASL_ML_Dataset(data_json_path,dir_path,select_trajectory=select_trajetory,phase='test')
    train_data, train_label = load_data(train_dataset)
    val_data, val_label = load_data(val_dataset)
    test_data, test_label = load_data(test_dataset)
    
    data = np.concatenate((train_data,val_data),axis=0)
    label = np.concatenate((train_label,val_label),axis=0)
    data,label = shuffle(data,label,random_state=RANDOM_NUM)
    # data = train_data
    # label = train_label
    # test_data = np.concatenate((val_data,test_data),axis=0)
    # test_label = np.concatenate((val_label,test_label),axis=0)
    print(np.unique(label,return_counts=True),np.unique(test_label,return_counts=True))
    # model = mlp_train(data,label)
    # mlp_test(model, test_data, test_label)
    norm = Normalizer()
    data = norm.fit_transform(data)
    test_data = norm.transform(test_data)
    print(f"train data size: {np.shape(data)}, test data size: {np.shape(test_data)}")
    ml_train_test(data,label,test_data,test_label,test_dataset,'SVM')

def test_data():
    dataset = WLASL_ML_Dataset(data_json_path,pose_dir_path,input_type='pose_hand',select_trajectory=select_trajetory,phase='test')
    index = 3
    d,l = dataset[index]
    d_info = dataset.get_index_data(index)
    d_data = dataset.get_orgin_data(index)
    show_track_3d(d_data[:,[4,5],:])
    print(d)
    print(d_info)
    print(np.shape(d))
    figs,axs = plt.subplots(d.shape[1],d.shape[0],figsize=(15,15))
    for i in range(d.shape[1]):
        for j in range(d.shape[0]):
            axs[i, j].stairs(d[j, i, :],np.arange(d.shape[2]+1),fill = True, alpha = 0.5, color = 'b')
            axs[i, j].set_xticks(np.arange(d.shape[2]))
            axs[i, j].set_yticks(np.arange(2))
            axs[i, j].set_xticklabels(np.arange(d.shape[2]))
            axs[i, j].set_title(f'Histogram ({i}, {j})')
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()

    # fig = plt.figure(figsize=(5,5),layout='constrained')
    # ax = plt.subplot(projection='3d')
    # def plot_two_hand(i,ax,d):
    #     ax.clear()
    #     # ax.view_init(elev = -90 , azim = 90)
    #     ax.set_xlim(-6,6)
    #     ax.set_ylim(-6,6)
    #     ax.set_zlim(0,6)
    #     ax.invert_yaxis() 
    #     ax.scatter(d[i][:,0],d[i][:,1],d[i][:,2])
    # def plot_vec_hand(i,ax,d):
    #     ax.clear()
    #     ax.set_xlim(-1,1)
    #     ax.set_ylim(-1,1)
    #     ax.set_zlim(0,2)
    #     ax.invert_yaxis() 
    #     ax.quiver(0,0,0,d[i][0,0],d[i][0,1],d[i][0,2],color='b',length=1)
    #     ax.quiver(0,0,0,d[i][1,0],d[i][1,1],d[i][1,2],color='r',length=1)
    # def animate(i):
    #     plot_two_hand(i,ax,d_data)
    # ani = FuncAnimation(fig, animate, frames=len(d), interval=250, repeat=True)
    # plt.show()  

if __name__ == '__main__':
    main()
    