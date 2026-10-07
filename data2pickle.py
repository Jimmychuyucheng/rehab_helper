import pickle
from data.process_backbone import *
# from data.mm2d_whole_body import load_mmpose_data
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import time
data_dir = "C:\\Users\\Alex\\Desktop\\Sign_dataset\\WLASL2000\\asl2000.json"
video_dir = 'C:\\Users\\Alex\\Desktop\\Sign_dataset\\WLASL2000\\video'
landmarks_dir = 'C:\\Users\\Alex\\Desktop\\CISLR\\landmarks\\wlasl_mediapipe_pose_hand'
l = []
def process_data(data):
    id = data['video_id']
    if os.path.exists(os.path.join(landmarks_dir,id+'.pickle')):
        return
    landmarks,_ = video2landmark(id,video_dir)
    data = landmarks
    with open(os.path.join(landmarks_dir,id+'.pickle'),'wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def main():
    data = load_data(data_dir)
    print(len(data))
    with Pool(8) as p:
        p.map(process_data,data)
if __name__ == '__main__':
    main()
