from typing import Any
import numpy as np
import random
import math


class PreNormalize2D:
    """
        source: https://github.com/kennymckormick/pyskl/blob/main/pyskl/datasets/pipelines/pose_related.py
        Normalize the range of keypoint values. 
    """

    def __init__(self, img_shape=(1080, 1920), threshold=0.01, mode='fix'):
        self.threshold = threshold
        # Will skip points with score less than threshold
        self.img_shape = img_shape
        self.mode = mode
        assert mode in ['fix', 'auto']

    # mask means joints that are need to remove
    def __call__(self, results):
        mask, maskout, keypoint = None, None, results['keypoint'].astype(np.float32)
        if 'keypoint_score' in results:
            keypoint_score = results.pop('keypoint_score').astype(np.float32)
            keypoint = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)

        if keypoint.shape[-1] == 3:
            mask = keypoint[..., 2] > self.threshold
            maskout = keypoint[..., 2] <= self.threshold

        if self.mode == 'auto':
            if mask is not None:
                if np.sum(mask):
                    x_max, x_min = np.max(keypoint[mask, 0]), np.min(keypoint[mask, 0])
                    y_max, y_min = np.max(keypoint[mask, 1]), np.min(keypoint[mask, 1])
                else:
                    x_max, x_min, y_max, y_min = 0, 0, 0, 0
            else:
                x_max, x_min = np.max(keypoint[..., 0]), np.min(keypoint[..., 0])
                y_max, y_min = np.max(keypoint[..., 1]), np.min(keypoint[..., 1])
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                keypoint[..., 0] = (keypoint[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                keypoint[..., 1] = (keypoint[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
        else:
            h, w = results.get('img_shape', self.img_shape)
            keypoint[..., 0] = (keypoint[..., 0] - (w / 2)) / (w / 2)
            keypoint[..., 1] = (keypoint[..., 1] - (h / 2)) / (h / 2)

        if maskout is not None:
            keypoint[..., 0][maskout] = 0
            keypoint[..., 1][maskout] = 0
        results['keypoint'] = keypoint
        return results
"""
    source : https://github.com/dxli94/WLASL/blob/master/code/TGCN/sign_dataset.py
    sample method 
"""
def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    # 
    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample

def down_sample(landmarks_data, max_frame):
    length = len(landmarks_data)
    increment = max_frame / length
    if increment > 1.0:
        increment = 1.0
    curr_increment = 0
    curr_frame = 0
    new_landmarks_data = []
    for l in landmarks_data:
        curr_increment += increment
        if curr_increment > curr_frame:
            curr_frame += 1
            new_landmarks_data.append(l)
    if len(new_landmarks_data) > max_frame:
        new_landmarks_data = new_landmarks_data[:max_frame]
    return new_landmarks_data    

def uniform_sample_np(data_numpy, size):
    T,V,C = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [round(i * interval) for i in range(size)]
    return data_numpy[uniform_list]

def random_sample_np(data_numpy, size):
    T,V,C = data_numpy.shape

    print(data_numpy.shape)
    if T == size:
        return data_numpy
    interval = int(np.ceil(size / T))
    
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data_numpy[random_list]
    
def padding(data,frmae_len, max_frame, input_shape, strategy ='zero'):
    need_frame = max_frame - frmae_len 
    if strategy == 'zero':
        begin = random.randint(0, max_frame - frmae_len)
        tmp1 = np.zeros((max_frame,) + input_shape)
        tmp1[begin:begin+frmae_len] = data[:frmae_len]
        data = tmp1
    elif strategy == 'last_frame':
        tmp1 = [data[-1]]*need_frame
        data.extend(tmp1)
    elif strategy == 'loop':
        tmp1 = data[:need_frame]
        while len(data) < max_frame:
            data.extend(tmp1)
        if len(data) > max_frame:
            data = data[:max_frame]
    else:
        raise ValueError('Input padding strategy dont exist')
    return data
"""
    source : https://github.com/hulianyuyy/DSTA-SLR/blob/main/feeders/tools.py
"""

def fix_data_length(data, max_frame, input_shape, strategy ='zero',if_video = False):
    frmae_len = len(data)
    if frmae_len > max_frame:
        data = down_sample(data,max_frame)
    if frmae_len < max_frame:
        data = padding(data,frmae_len,max_frame,input_shape, 'zero')
    if if_video:
        data = np.array(data,dtype=np.uint8)
    else:
        data = np.array(data,dtype=np.float32)
    return data



# input landmarks shape (f,x,3) f is frame number, x is joint number 3 is channel
def interpolated_landmarks(landmarks, p:float=0.5):
    """
    landmarks: list of landmark
    p: interplot rate
    """
    new_landmarks = []
    new_landmarks.append(landmarks[0])
    for i in range(1,len(landmarks)-1):
        prev_landmark = np.array(landmarks[i-1])
        post_landmark = np.array(landmarks[i+1])
        cur_landmark = np.array(landmarks[i])
        if np.all(cur_landmark == 0) and not np.all(prev_landmark == 0) and not np.all(post_landmark == 0):
            new_landmark = prev_landmark * p + post_landmark * (1-p)
            new_landmarks.append(new_landmark.tolist())
        else:
            new_landmarks.append(landmarks[i])
    new_landmarks.append(landmarks[-1])
    return np.asarray(new_landmarks)

def get_bound(landmarks):
    x_min = max(min(landmarks[:,0])-0.05, 0)
    x_max = min(max(landmarks[:,0])+0.05, 1)
    y_min = max(min(landmarks[:,1])-0.05, 0)
    y_max = min(max(landmarks[:,1])+0.05, 1)
    return [x_min,x_max,y_min,y_max]
"""
    source :https://github.com/kennymckormick/pyskl/blob/main/pyskl/datasets/pipelines/heatmap_related.py
    input hand_landmarks shape (f,42,2) f is frame number, 42 is joint number 3 is channel
    input hand_confs shape (f,42) 
    sigma: size of circle

    return:
        image_left: heatmap of left hand (f,150,150,3)
        image_right: heatmap of right hand (f,150,150,3)
"""
def get_hands_heatmap(hand_landmarks, hand_confs = None, frame_shape = (56, 56), sigma=0.6): 
    # print(np.shape(hand_landmarks))
    image = np.zeros((hand_landmarks.shape[1], hand_landmarks.shape[0],frame_shape[0], frame_shape[1]), dtype=np.float32) 
    if hand_confs is None:
        hand_confs = np.ones((hand_landmarks.shape[0],21),dtype=np.float32)
    for j in range(hand_landmarks.shape[1]):
        video_joint_pos = hand_landmarks[:,j]
        video_joint_conf = hand_confs[:,j]
        for f in range(hand_landmarks.shape[0]):
            hand_bound = get_bound(hand_landmarks[f])
            joint_pos = video_joint_pos[f]
            joint_conf = video_joint_conf[f]
            # x = int(joint_pos[0]*frame_shape[0])
            x = int((joint_pos[0]-hand_bound[0])/(hand_bound[1]-hand_bound[0])*frame_shape[0])
            # y = int(joint_pos[1]*frame_shape[1])
            y = int((joint_pos[1]-hand_bound[2])/(hand_bound[3]-hand_bound[2])*frame_shape[1])
            st_x = max(int(x - 3 * sigma), 0)
            ed_x = min(int(x + 3 * sigma) + 1, frame_shape[0])
            st_y = max(int(y - 3 * sigma), 0)
            ed_y = min(int(y + 3 * sigma) + 1, frame_shape[1])
            o_x = np.arange(st_x, ed_x, 1, np.float32)
            o_y = np.arange(st_y, ed_y, 1, np.float32)
            if not (len(o_x) == len(o_y)):
                continue
            o_y = o_y[:,None]
            patch = np.exp(-((o_x - x) ** 2 + (o_y - y) ** 2) / (2 * sigma ** 2))
            patch = patch * joint_conf
            image[j,f,st_y:ed_y, st_x:ed_x] = np.maximum(image[j,f,st_y:ed_y, st_x:ed_x], patch)
    # for i in range(hand_landmarks.shape[0]):
    #     hand = hand_landmarks[i]
    #     hand_conf = hand_confs[i]
    #     hand_bound = hand_bounds[i]
    #     for landmark,conf in zip(hand,hand_conf):

    #         if np.all(hand_bound == 0):
    #             continue
            
    #         x = int(landmark[0]*frame_shape[0])# int((landmark[0]-left_hand_bound[0])/(left_hand_bound[1]-left_hand_bound[0])*frame_shape[0])
    #         y = int(landmark[1]*frame_shape[1])#int((landmark[1]-left_hand_bound[2])/(left_hand_bound[3]-left_hand_bound[2])*frame_shape[1])
    #         st_x = max(int(x - 3 * sigma), 0)
    #         ed_x = min(int(x + 3 * sigma) + 1, frame_shape[0])
    #         st_y = max(int(y - 3 * sigma), 0)
    #         ed_y = min(int(y + 3 * sigma) + 1, frame_shape[1])
    #         o_x = np.arange(st_x, ed_x, 1, np.float32)
    #         o_y = np.arange(st_y, ed_y, 1, np.float32)

    #         if not (len(o_x) == len(o_y)):
    #             continue
    #         o_y = o_y[:,None]
    #         patch = np.exp(-((o_x - x) ** 2 + (o_y - y) ** 2) / (2 * sigma ** 2))
    #         patch = patch * conf
    #         image[i,st_y:ed_y, st_x:ed_x] = np.maximum(image[i,st_y:ed_y, st_x:ed_x], patch)
    return image

def random_mirror(data_numpy):
    C, T, V = data_numpy.shape
    data_numpy = data_numpy.transpose(1,2,0)
    if np.all(np.isclose(data_numpy, 0)):
        return data_numpy
    if np.random.rand() < 0.5:
        data_numpy = data_numpy[:,::-1]
    return data_numpy.transpose(2,0,1)

def normalize_hand(hand_data_numpy):
    # 找到 x 和 y 軸的最小和最大值
    min_x = np.min(hand_data_numpy[:, :, 0])
    max_x = np.max(hand_data_numpy[:, :, 0])
    min_y = np.min(hand_data_numpy[:, :, 1])
    max_y = np.max(hand_data_numpy[:, :, 1])
    if max_x - min_x == 0 and  max_y - min_y == 0 :
       return hand_data_numpy

    # 正規化 x 和 y 軸的數據
    hand_data_numpy[:, :, 0] = (hand_data_numpy[:, :, 0] - min_x) / (max_x - min_x)
    hand_data_numpy[:, :, 1] = (hand_data_numpy[:, :, 1] - min_y) / (max_y - min_y)

    return hand_data_numpy