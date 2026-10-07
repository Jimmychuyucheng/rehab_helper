import os
import pickle
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.interpolate as interpolate
import shutil
import cv2
from .preprocess import  down_sample, fix_data_length, interpolated_landmarks, get_hands_heatmap, normalize_hand
from .tools import random_shift,random_move,random_rot,random_crop_right_hand,extract_frame 
# from .visualization import *
from .body_joint_norm import normalize_body_joints

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21


"""
    數據處理策略:
        使用mediapipe抓取身體3d空間的座標，形狀為(6,3)
        使用mmpose 中 rtmw for whole body抓取手部形狀
"""

def get_missing_id(missing_txt):
    missing = []
    with open(missing_txt,'r') as file:
        missing = [line.rstrip() for line in file]
    return missing

# label num is the number of video per label
# label_dict is the mapping from label to number
def load_data_info(json_file,need,phase = None):
    label_list = []
    label_num = []
    data_info = [] 
    with open(json_file) as f:
        dic = json.load(f)
        for gesture in dic :
            gloss = gesture["gloss"]
            label_list.append(gloss)
            cot=0
            for inst in gesture["instances"]:
                tmp = {}
                tmp["gloss"] = gloss
                if phase == None or inst['split'] == phase:
                    cot+=1
                    for n in need :
                        tmp[n] = inst[n]
                    data_info.append(tmp)
            label_num.append(cot)
    num_list = np.arange(len(label_list))
    return data_info,dict(zip(label_list,num_list)),label_num

def create_gloss_dir(json_file, source_path, target_path):
    with open(json_file) as f:
        dic = json.load(f)
        for gesture in dic :
            gloss = gesture["gloss"]
            if not os.path.exists(os.path.join(target_path,gloss)):
                os.mkdir(os.path.join(target_path,gloss))
            for inst in gesture["instances"]:
                id = inst['video_id']
                if not os.path.exists(os.path.join(target_path,gloss,id+'.mp4')):
                    shutil.copy(os.path.join(source_path,id+'.mp4'),os.path.join(target_path,gloss,id+'.mp4'))

def load_pickle_data(id,path):
    with open(os.path.join(path,id+'.pickle'),'rb') as file:
        data = pickle.load(file)
    return data

# if input contain numpy array 
def numpy_to_list(arr):
    nparr = np.array(arr)
    if np.all(nparr == -1):
        nparr = np.where(nparr == -1, 0, nparr)
    return nparr.tolist()


"""
    for each video id, load data with specific define joint set and return with 
    only body have conf
"""

class keypoint_select:
    
    def __init__(self,input_format,threshold = 0.5):
        self.input_format = input_format
        self.threshold = threshold
        # print(input_format)
        # print(input_format == 'None')
        if input_format != 'None':
            self.get_idx()
    def get_idx(self):
        ## for face we select 152 is chin, 10 is forhead
        # for pose we select 0 nose, 2,5  eyes 7,8 ears 9,10 mouth
        # 17 is the midele of shoulder 11,12 is left and right shoulder
        if self.input_format == 'mediapipe_whole_body':
            self.hand_idx = np.arange(21).tolist()
            self.face_idx = [152,10]
            self.pose_idx = [0,2,5,7,8,9,10,11,12,13,14,15,16]
        else:
            raise ValueError('input format none exist')

    def filter_body_landmark(self,body):
        body_data = np.asarray([landmark['landmark'] for landmark in body])
        conf = np.asarray([landmark['conf'] for landmark in body])
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                if conf[i,j] < self.threshold:
                    body_data[i,j] = np.zeros(3)
        data = body_data[:,self.pose_idx]
        shoulder_middle = (data[:,8]+data[:,9])/2
        data = np.concatenate([data,shoulder_middle.reshape(-1,1,3)],axis=1)
        self.pose_idx.append(17)
        return body_data[:,self.pose_idx]
    

    def get_holistic_joints(self,data):
        body_data = self.filter_body_landmark(data['body'])
        face_data = data['face'][:,self.face_idx]
        left_hand_data = data['left_hand'][:,self.hand_idx]
        right_hand_data = data['right_hand'][:,self.hand_idx]
        data = np.concatenate([body_data,face_data,left_hand_data,right_hand_data],axis=1)
        return data
                   
    def __call__(self,data):
        return self.get_holistic_joints(data)



def get_slope(point1, point2):
        tmp = []
        
        v = point2 - point1
        if np.all(v == 0):
            return np.zeros(point1.shape[0])
        tmp.append(v[:,1]/(v[:,0]+1e-8))
        return np.asarray(tmp).flatten()

# input shape is (frame, 2, 3)
def get_slope_var_sequence(body, gap = 5):
    slope = []
    for i in range(0,body.shape[0]-gap,gap):
        slope.append(get_slope(body[i],body[i+gap]))
    slope_variation = []
    for i in range(len(slope)-1):
        slope_variation.append(slope[i]-slope[i+1])
    return np.asarray(slope_variation)

def get_vector_data(body, gap = 5):
    vector = []
    for i in range(0,body.shape[0]-gap,gap):
        v = body[i+gap]-body[i]
        v = v / (np.linalg.norm(v)+1e-8)
        vector.append(v)
    return np.asarray(vector)



def ema(body, alpha = 0.8):
    v = np.zeros(body[0].shape)
    for i in range(len(body)):
        v = alpha * body[i] + (1-alpha) * v
        body[i] = v
    return body

# if hand vis is getting smaller and gap is bigger then 0.1 or hand is getting larger and vis is >0.3
def check_vis(body, vis):
    s = body.shape
    for i in range(s[0]):
        for j in range(s[1]):
            if vis[i,j]<0.5:
                body[i,j] = np.zeros(body[i,j].shape)
    return body

def joints_to_vector(body_position,step = 2):
    vector = []
    for i in range(0,body_position.shape[0]-step,step):
        if np.all(body_position[i+step] == 0) or np.all(body_position[i] == 0):
            v = np.zeros(body_position[i].shape)
        else:
            v = body_position[i+step] - body_position[i]
        vector.append(v)
    return np.array(vector)

"""
    input shape is (batch, input channel, frame count, number of node)
    haven't move to orgin yet
    current norm strategy
"""
class WLASL_Define_Dataset(Dataset):
    def __init__(self, data_info_json, pickle_dir, phase, hand=0,input_format = None,pad = 'zero', max_frame = 50,aug = [],if_video = None):
        print(f'{phase} Using aug {aug}.')
        self.data_info,self.label_dict,self.label_num = load_data_info(data_info_json,["video_id","split","frame_start","frame_end","bbox","variation_id"],phase)
        self.pickle_dir = pickle_dir
        self.max_frame = max_frame
        self.phase = phase
        self.pad = pad
        self.aug = aug
        self.aug_rate = 0.5
        self.hand = hand
        self.if_video = if_video
        self.video_dir = r'/home/video/'
        if input_format is not None:
            self.kp_select = keypoint_select(input_format)
        else:
            self.kp_select = None
    def __getitem__(self, index):
        if self.if_video:
            data = self.get_video_data(index)
        else:
            data = self.get_whole_body_data(index)
        return data
    def __len__(self):
        return len(self.data_info)
    def get_whole_body_data(self,index):
        l = self.data_info[index]['gloss'] 
        # if self.phase == 'test':
        #     label = np.zeros(len(self.label_dict))
        #     label[self.label_dict[l]] = 1
        # else:
        label = np.full(len(self.label_dict), 0.1 / (len(self.label_dict) - 1))  # 初始化為平滑值
        label[self.label_dict[l]] = 0.9
        id = self.data_info[index]['video_id']
        data = load_pickle_data(id,self.pickle_dir)
        # norm to -1~1
        landmarks = data['landmarks'].astype(np.float32)
        select_keypoint = [0,1,2,3,4,5,6,7,8,9,17,39,40,41,43,44,47,48,51,52,55,56,59,18,20,22,23,26,27,30,31,34,35,38]
        small_body = landmarks[:,select_keypoint]
        zero_idex = np.all(landmarks == 0,axis=2)
        landmarks = 2*(landmarks-0.5)
        # shoulder_middle = landmarks[:,5]
        # for i in range(len(landmarks)):
        #     landmarks[i] = landmarks[i]-shoulder_middle[i]
        landmarks[zero_idex] = 0
        if self.kp_select is not None:
            landmarks = keypoint_select(landmarks)        
        landmarks = fix_data_length(landmarks,self.max_frame,landmarks.shape[1:],strategy=self.pad,if_video = self.if_video).transpose(2,0,1)
        small_body = fix_data_length(small_body,self.max_frame,small_body.shape[1:],strategy=self.pad).transpose(2,0,1)
        if 'random_shift' in self.aug and np.random.rand() < self.aug_rate:
            landmarks = random_shift(landmarks)
        if 'random_move' in self.aug : 
            landmarks = random_move(landmarks)
        if 'random_rot' in self.aug and np.random.rand() < self.aug_rate:
            landmarks = random_rot(landmarks)
        if 'random_drop_right_hand' in self.aug and np.random.rand() < self.aug_rate:
            landmarks = random_crop_right_hand(landmarks)
        return {'data':torch.from_numpy(landmarks).type(torch.float32),'body':torch.from_numpy(small_body).type(torch.float32),'label':torch.from_numpy(label).type(torch.float32),'id':id} 
    def get_video_data(self,index):
        l = self.data_info[index]['gloss']
        label = np.zeros(len(self.label_dict))
        label[self.label_dict[l]] = 1
        id = self.data_info[index]['video_id']
        video = extract_frame(os.path.join(self.video_dir,id+'.mp4'))
        if 'random_flip' in self.aug and np.random.rand() < self.aug_rate:
            video = np.flip(video, axis=2)
        video = fix_data_length(video,self.max_frame,video.shape[1:],if_video=True)
        video = transform(video)
        return {'data':video, 'label':torch.from_numpy(label).type(torch.float32),'id':id}
    # def load_heatmap_data(self,landmarks,confs = None , frame_shape = (56,56)):
    #     hand_landmarks = landmarks[:,-42:]
    #     if confs is not None:
    #         left_hand_heatmap = get_hands_heatmap(hand_landmarks[:,:21],confs[:,:21])
    #         right_hand_heatmap = get_hands_heatmap(hand_landmarks[:,21:],confs[:,21:])
    #     else:
    #         left_hand_heatmap = get_hands_heatmap(hand_landmarks[:,:21])
    #         right_hand_heatmap = get_hands_heatmap(hand_landmarks[:,21:])
    #     return left_hand_heatmap.transpose(1,0,2,3),right_hand_heatmap.transpose(1,0,2,3)

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().type(torch.float32)
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)
"""
    reference :https://www.ijcai.org/Proceedings/13/Papers/203.pdf
    implement the algorithm HOD (Histogram of Oriented Displacement) to get the trajectory
    input should be the 3d landmarks
"""
class HOD(object):
    def __init__(self, bins = 8, level = 3):
        self.bins = bins
        self.level = level

    def vector2bin_num(self,v):
        radians = np.arctan2(v[:,1],v[:,0])
        angle = radians * 180 / np.pi
        angle[angle<0] = angle[angle<0] + 360
        return np.floor(angle*self.bins / 360).astype(np.uint8)
    
    def split2twodim(self,landmark):
        xy = landmark[...,[0,1]]
        yz = landmark[...,[1,2]]
        xz = landmark[...,[0,2]]
        return xy,yz,xz

    # input landmark shape is (frame, 2, 3)
    # output shape is (frame-1, 2 ,bins)
    def get_hod_hist(self, landmark):
        xy,yz,xz = self.split2twodim(landmark)
        result = []
        for joints in [xy,yz,xz]:
            v = joints_to_vector(joints,step=1)
            hist = np.zeros((v.shape[1],self.bins))
            record_result = np.zeros((v.shape[0] + 1, v.shape[1], self.bins))
            record_result[0] = hist
            for i in range(v.shape[0]):
                if not np.all(v[i] == 0):
                    bin_num = self.vector2bin_num(v[i])
                    length = np.linalg.norm(v[i],axis=1)
                    hist[np.arange(v.shape[1]), bin_num] += length
                record_result[i + 1] = hist
            result.append(record_result)
        return result 
    
    def __call__(self,landmark):
        hist = self.get_hod_hist(landmark)
        ret = []
        for dim_data in hist:
            for i in range(self.level):
                num = 2**i
                split = len(dim_data) // num
                index = dim_data[::split-1]
                for j in range(len(index)-1):
                    ret.append(index[j+1]-index[j])
        return np.array(ret)
            

""" 
    select json attribute: [{gloss:str , instances:List[Dict]}]
    instances attribute: [{video_id:str, split:str, variation_id : int, trajectory:int}]
    current subjcet : 
        get raw skeleton data of two hands for trajectory prediction
        write a function about select data that is in select_label
"""

class WLASL_ML_Dataset():
    def __init__(self,data_info_json, dir_path, phase, select_trajectory, input_type='holistic',select_index= [15,16], threshhold=0.0, max_frame = 40):
        self.data_info,self.label_dict,self.label_num = load_data_info(data_info_json,["video_id","split","frame_start","frame_end","bbox","trajectory"],phase)
        self.dir_path = dir_path
        self.max_frame = max_frame
        self.select_index = select_index
        self.select_trajectory = select_trajectory
        self.threshhold = threshhold
        self.input_type = input_type
        self.label_dict = dict(zip(select_trajectory,range(len(select_trajectory))))
        self.transform2hist = HOD(bins=8,level=2)
        self.remove_unneed_data()
    
    def remove_unneed_data(self):
        ret = []
        for info in self.data_info:
            if info['trajectory'] in self.select_trajectory:
                ret.append(info)
        self.data_info = ret  
    
    
    def norm_data(self,body_landmarks):
        body_position = np.array(body_landmarks)
        body_position[:,:,2] = body_position[:,:,2]*-1
        body_position[:,:,2] = (body_position[:,:,2] - body_position[:,:,2].min()) / (body_position[:,:,2].max() - body_position[:,:,2].min())
        # body_position = body_position - 0.5
        # for i in range(len(body_position)):
        #     body_position[i,11:17] = normalize_body_joints(body_position[i,11:17].tolist())
        
        return body_position

    def cal_avg_frame(self):
        avg = 0
        for info in self.data_info:
            id = info['video_id']
            data = load_pickle_data(id,self.dir_path)
            avg += len(data['body'])
        return avg / len(self.data_info)

    
    
    def get_label_trajectory(self):
        ret = {}
        for info in self.data_info:
            label = info['gloss']
            trajectory = info['trajectory']
            if trajectory not in ret:
                ret[trajectory] = []
            if label not in ret[trajectory]:
                ret[trajectory].append(label)
        return ret
    
    def smooth_data(self,body_position):
        for i in range(len(body_position)):
            body_position[i] = ema(body_position[i])
        return body_position
    
    def get_index_data(self,index):
        return self.data_info[index]

    def get_diff_type_data(self,data):
        if self.input_type == 'holistic':
            body_landmarks = [ d['landmark'] for d in data['body']]
            body_vis = [ d['conf'] for d in data['body']]
        elif self.input_type == 'pose_hand':
            body_landmarks = [ d['Body'] for d in data]
            body_vis = [ d['conf'] for d in data]
        return body_landmarks, body_vis

    def get_orgin_data(self,index):
        info = self.data_info[index]
        id = info['video_id']
        trajectory = info['trajectory']
        data = load_pickle_data(id,self.dir_path)
        body_landmarks,body_vis = self.get_diff_type_data(data)
        body_position = self.norm_data(body_landmarks)
        body_vis = np.array(body_vis)
        body_position[body_vis < self.threshhold] = np.zeros(body_position.shape[2])
        body_position = body_position[:,11:17]
        return body_position
    
    def  __getitem__(self,index):
        info = self.data_info[index]
        id = info['video_id']
        trajectory = info['trajectory']
        data = load_pickle_data(id,self.dir_path)
        body_landmarks,body_vis = self.get_diff_type_data(data)
        body_position = self.norm_data(body_landmarks)
        # body_vis = np.array(body_vis)
        # body_position[body_vis < self.threshhold] = np.zeros(body_position.shape[2])

        body_position = self.smooth_data(body_position)
        body_position = body_position[:,self.select_index]
        body_position = self.transform2hist(body_position)
        # body_position = fix_data_length(body_position,self.max_frame,body_position.shape[1:],strategy='zero')

        return body_position, self.label_dict[trajectory]
    

    def __len__(self):
        return len(self.data_info)
    


if __name__ == '__main__':
    data_info_json = "/home/CISLR/data_info/wlasl/asl100.json"# "/home/SLR_Code/WLASL_v0.3.json"
    pickle_dir_path = "/home/CISLR/landmarks/wlasl_mmpose2d_54"
    # train_data = WLASL_Dataset(data_info_json,hand_dir_path,body_dir_path,'train',top_k = 100)
    # val_data = WLASL_Dataset(data_info_json,hand_dir_path,body_dir_path,'val',top_k=100)
    # with open('output.txt', 'w') as f:
    #     numpy_array = train_data[0]['hand'].cpu().numpy().tolist()
    #     f.write(str(numpy_array))
    # label_dict = train_data.label_dict
    # print("Train data shape {}, Val data shape {}.".format(len(train_data),len(val_data)))
    # data = WLASL_Define_Dataset(data_info_json, pickle_dir_path, 'train',input_format=None)
    # fig = plt.figure(figsize=(5,5), layout='constrained')
    # ax = plt.subplot()
    # def animtate(i):
    #     ax.clear()
    #     ax.set_xlim(0,1)
    #     ax.set_ylim(0,1)
    #     ax.invert_yaxis()
    #     ax.scatter(test_data[i,:,0],test_data[i,:,1])
    # ani = animation.FuncAnimation(fig,animtate,frames=test_data.shape[0],interval=250,repeat=False) 
    # plt.show()
