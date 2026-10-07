import math 
import numpy as np
import random
import torch
import cv2

"""
    source: https://github.com/Uason-Chen/CTR-GCN/blob/main/feeders/tools.py
"""

def random_shift(data_numpy):
    # input: C,T,V
    C, T, V = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape, dtype=np.float32)
    valid_frame = (data_numpy != 0).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :] = data_numpy[:, begin:end, :]
    return data_shift

"""
    平移和縮放
"""

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[1.0, 1.1, 1.1, 1.3],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V
    C, T, V = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    # A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    # a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        # a[node[i]:node[i + 1]] = np.linspace(
        #     A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    # theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
    #                   [np.sin(a) * s, np.cos(a) * s]])
    theta = np.array([[s, np.zeros_like(s)], [np.zeros_like(s), s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :] = new_xy.reshape(2, V)

    return data_numpy

def _rot2d(theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

def _rot3d(theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

def random_rot(data_numpy, theta=0.3):

    C, T, V = data_numpy.shape
    data_numpy = data_numpy.transpose(1,2,0)
    if np.all(np.isclose(data_numpy, 0)):
        return data_numpy   
    theta = np.random.uniform(-theta)
    rot_mat = _rot2d(theta)
    data_numpy = np.einsum('ab,tvb->tva', rot_mat, data_numpy)

    return data_numpy.transpose(2,0,1)

def random_crop_right_hand(data_numpy):
    C, T, V = data_numpy.shape
    data_numpy = data_numpy.transpose(1,2,0)
    data_numpy[:,-42:-21,:] = 0
    return data_numpy.transpose(2,0,1)

"""
    自創資料增強，將左右手稍微旋轉
"""
def random_hand_rot(data_numpy):
    # input: C,T,V
    C, T, V = data_numpy.shape
    data_mirror = np.zeros(data_numpy.shape, dtype=np.float32)
    data_mirror = data_numpy[:, ::-1, :]
    data_mirror[0, :, :] *= -1
    return data_mirror

class kalman_filter(object):
    def __init__(self):
        self.cot = 0
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix =  np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) *1e-1
    
    def process(self,input):
        if self.cot == 0:
            self.kalman.statePost = np.array([[input[0]],[input[1]],[0],[0]], dtype=np.float32)
        predict = self.kalman.predict()
        self.kalman.correct(input)
        self.cot += 1
        return  predict[:2].squeeze()
    
class kalman_filter3D(object):
    def __init__(self):
        self.cot = 0
        self.kalman = cv2.KalmanFilter(6,3)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1., 0., 0., 1., 0., 0.],
                                            [0., 1., 0., 0., 1., 0.],
                                            [0., 0., 1., 0., 0., 1.],
                                            [0., 0., 0., 1., 0., 0.],
                                            [0., 0., 0., 0., 1., 0.],
                                            [0., 0., 0., 0., 0., 1.]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
    
    def process(self,input):
        if self.cot == 0:
            self.kalman.statePost = np.array([[input[0]],[input[1]],[input[2]],[0],[0],[0]], dtype=np.float32)
        predict = self.kalman.predict()
        self.kalman.correct(input)
        self.cot += 1
        return  predict[:3].reshape(-1)
    

def extract_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 25)
    vid=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224,224))[...,::-1]
        vid.append(frame)
    return np.asarray(vid).astype(np.uint8)