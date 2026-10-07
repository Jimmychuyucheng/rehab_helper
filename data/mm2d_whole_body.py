

import numpy as np
import decord
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time 
from data.tools import extract_frame
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmdet.utils.setup_env import register_all_modules
from mmdet.apis import DetInferencer
from data.preprocess import get_hands_heatmap
import matplotlib.patches as patches

register_all_modules(True)
base_path = r'mmpose' #自行設定路徑

det_model_cfg = os.path.join(base_path,'rtmdet_m_640-8xb32_coco-person.py')
det_ckpt =os.path.join(base_path,'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth')
model_cfg = os.path.join(base_path,'rtmw-x_8xb320-270e_cocktail13-384x288.py')
ckpt = os.path.join(base_path,'rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.pth')
# model_cfg = os.path.join(base_path,'td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256.py')
# ckpt = os.path.join(base_path,'hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth')

device = 'cuda'
det_inferencer = DetInferencer(model=det_model_cfg, weights=det_ckpt)
model = init_model(model_cfg, ckpt, device=device)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta)
# hand keypoint 91~132(91~111 left hand 112~132 right hand)
# upprt body 5~10

# SELECT_KEYPOINT = [0,1,2,3,4,5,6,7,8,80,91,93,95,96,99,100,103,104,107,108,111,112,114,116,117,120,121,124,125,128,129,132]
# ONLY_BODY_KEYPOINT = [0,1,2,3,4,5,6,7,8,9,10]
SELECT_KEYPOINT = [0,1,2,3,4,5,6,7,8,9,10,42,47,71,74,77,80,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]
ONLY_BODY_KEYPOINT = [0,1,2,3,4,5,6,7,8,9,10]
# hand = [(1,0),(2,1), (3,0),(4,3), (5,0),(6,5), (7,0),(8,7), (9,0),(10,9)]
# left_hand = [(j+11, i+11) for (i, j) in hand] 
# right_hand = [(j+22, i+22) for (i, j) in hand] 
# BODY_CONNECT = [(1,0),(2,0),(3,1),(4,2),(5,0),(6,5),(7,5),(9,7),(8,6),(10,0),(11,8),(22,9)] + left_hand + right_hand
def get_person_bbox(frames):
    tmp=[]
    # print('Frames',len(frames))
    shape = frames[0].shape
    for frame in frames:
        det_result = det_inferencer(frame)
        # print(det_result)
        if det_result['predictions'][0]['labels']:
            tmp.append(det_result['predictions'][0]['bboxes'][0])
    tmp = np.array(tmp)
    max_val = np.amax(tmp,axis=0)
    min_val = np.amin(tmp,axis=0)
    bbox = np.array([np.concatenate((min_val[:2],max_val[2:]))])
    bbox[:,0]= max(0,np.round(bbox[:,0]-5))
    bbox[:,1]= max(0,np.round(bbox[:,1]-5))
    bbox[:,2]= min(shape[1],np.round(bbox[:,2]+5))
    bbox[:,3]= min(shape[0],np.round(bbox[:,3]+5))
    return bbox.astype(np.int32)

    

def set_person_frame(frames):
    person_frame = []
    bbox = get_person_bbox(frames)
    #print('Person bounding box is ',bbox)
    #print('Before shape', frames[0].shape)
    for i,frame in enumerate(frames):
        frame = frame[bbox[0,1]:bbox[0,3],bbox[0,0]:bbox[0,2]]
        # frame = cv2.resize(frame,dsize=(288,384),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        person_frame.append(frame)
    #print('After shape', person_frame[0].shape)
    return person_frame

def ema(body, alpha = 0.8):
    v = np.zeros(body[0].shape)
    for i in range(len(body)):
        v = alpha * body[i] + (1-alpha) * v
        body[i] = v
    return body


def show_pose_result(frames,pose_results,out=None,save_video=False):
    # visulize video
    for idx in range(len(frames)):
        results = merge_data_samples(pose_results[idx])
        print('Results',results)
        frame = visualizer.add_datasample( 'result', frames[idx], data_sample=results)
        cv2.imshow('Ouput',frame)
        if save_video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25)& 0xFF == ord('q'):
            break
def show_webcam_result():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        bbox = get_person_bbox([frame])
        frame = frame[bbox[0,1]:bbox[0,3],bbox[0,0]:bbox[0,2]]
        result = inference_topdown(model, frame)
        result = merge_data_samples(result)
        frame = visualizer.add_datasample( 'result', frame, data_sample=result)

        cv2.imshow('Ouput',frame)
        if cv2.waitKey(25)& 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def pose_inference(video_path):
    frames = extract_frame(video_path)
    person_frames = set_person_frame(frames)
    pose_results = []
    for frame in person_frames:
        result = inference_topdown(model, frame)
        pose_results.append(result)
    return pose_results,person_frames

def shift_to_shoulder_middle(landmark):
    shoulder_middle = landmark[5]
    landmark = landmark-shoulder_middle
    return landmark

# hand keypoint 91~132(91~111 left hand 112~132 right hand) body kepoint 1~11 and add middle of shoulder
# normalize landmark by frame shape to -1~1
# total 54 2d keypoints
def get_whole_body_landmark(pose_result,person_frames,conf_threshold = 0.6 ,vis_threshold=0.6):
    landmarks = []
    confs = []
    for result,frame in zip(pose_result,person_frames):
        data = result[0].pred_instances
        landmark = data.keypoints[:,SELECT_KEYPOINT].reshape(-1,2)
        conf = data.keypoint_scores[0,SELECT_KEYPOINT]
        vis = data.keypoints_visible[0,SELECT_KEYPOINT]
        for i in range(len(landmark)):
            if conf[i] < conf_threshold or vis[i] < vis_threshold:
                landmark[i] = np.array([0,0])
        shoulder_middle = (landmark[5]+landmark[6])/2
        shoulder_middle_conf = (conf[5]+conf[6])/2
        landmark = np.insert(landmark,5,shoulder_middle,axis=0)
        conf = np.insert(conf,5,shoulder_middle_conf,axis=0)
        landmark = landmark/np.array([frame.shape[1],frame.shape[0]])
        landmarks.append(landmark)
        confs.append(conf)
    return np.array(landmarks), np.array(confs)

# hand keypoint 91~132(91~111 left hand 112~132 right hand) body kepoint 1~11
# adding new landmark :
def get_landmark(pose_result,shape,threshold=0.8):
    landmarks = []
    if not isinstance(shape,np.ndarray):
        shape =  np.asarray(shape)[:2]
    for result in pose_result:
        landmark = {'Left':np.full((21,2),0),'Right':np.full((21,2),0),'Body':np.full((6,2),0)}
        data = result[0].pred_instances
        left_score = np.mean(data.keypoint_scores[:,91:112])
        right_score = np.mean(data.keypoint_scores[:,112:133])
        # print('Left Score',left_score)
        # print('Right Score', right_score)
        # print('='*100)
        if left_score > threshold:
            left_keypoint = data.keypoints[:,91:112]
            landmark['Left'] = left_keypoint.reshape(-1,2)/shape
        if right_score > threshold:
            right_keypoint = data.keypoints[:,112:133]
            landmark['Right'] = right_keypoint.reshape(-1,2)/shape
        landmark['Body'] = data.keypoints[:,5:11].reshape(-1,2)/shape
        landmarks.append(landmark)
    return landmarks

def load_mmpose_data(video_path):
    pose_results,person_frames = pose_inference(video_path)
    landmarks,confs = get_whole_body_landmark(pose_results,person_frames)
    frames_shape = [frame.shape for frame in person_frames]
    return {'landmarks':landmarks, 'confs':confs, 'frames_shape': frames_shape}  

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


def main():
    
    # ret = load_mmpose_data('12326',r"C:\Users\Alex\Desktop\Sign_dataset\WLASL2000\video")
    start = time.time()
    ret = load_mmpose_data('e720d834',r"C:\Users\Alex\Desktop\Sign_dataset\demo_video")
    print('Time',time.time()-start)
    #　get_hands_heatmap(landmarks,conf)
    # print(ret['landmarks'])
    # hands_landmarks = ret['landmarks'][:,18:]
    # hands_conf = ret['confs'][:,18:]
    # left_heatmap,right_heatmap = get_hands_heatmap(hands_landmarks,hands_conf)
    # for left_image, right_image in zip(left_heatmap,right_heatmap):
    #     image = cv2.hconcat([left_image,right_image])
    #     cv2.imshow('Ouput',image)
    #     if cv2.waitKey(25)& 0xFF == ord('q'):
    #         break
    landmarks = ret['landmarks']
    
    # 
    from tools import random_rot,random_move
    

    zero_idex = np.all(landmarks == 0,axis=2)
    # landmarks = ema(landmarks)

    # single_hand = landmarks[:,-21:]
    # pos = landmarks[:,-21:-20]
    # single_hand = single_hand - pos
    # single_hand = single_hand *1.3
    # landmarks[:,-21:] = single_hand + pos

    landmarks = 2*(landmarks-0.5)
    # shoulder_middle = landmarks[:,5]
    # for i in range(len(landmarks)):
    #     landmarks[i] = landmarks[i]-shoulder_middle[i]
    
    landmarks[zero_idex] = 0
    print(landmarks.shape)
    # landmarks = random_move(landmarks.transpose(2,0,1)).transpose(1,2,0)
    # landmarks = random_rot(landmarks)
    print(landmarks.shape)
    np.save('../self_film_computer.npy',landmarks)
    fig = plt.figure(figsize=(10,10), layout='constrained')
    ax = plt.subplot()
    def animtate(i):
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.invert_yaxis()
        ax.scatter(landmarks[i,:,0],landmarks[i,:,1],s=3)
        for idx, (x, y) in enumerate(landmarks[i]):
            ax.text(x, y, str(idx), fontsize=8)
        # for line in BODY_CONNECT:
        #     ax.plot([landmarks[i,line[0],0],landmarks[i,line[1],0]],[landmarks[i,line[0],1],landmarks[i,line[1],1]],c='r')
        # left_rects = patches.Rectangle((left_hands_bbox[i,0],left_hands_bbox[i,2]),left_hands_bbox[i,1]-left_hands_bbox[i,0],left_hands_bbox[i,3]-left_hands_bbox[i,2],linewidth=1,edgecolor='r',facecolor='none')
        # right_rects = patches.Rectangle((right_hands_bbox[i,0],right_hands_bbox[i,2]),right_hands_bbox[i,1]-right_hands_bbox[i,0],right_hands_bbox[i,3]-right_hands_bbox[i,2],linewidth=1,edgecolor='r',facecolor='none')
        # ax.add_patch(left_rects)
        # ax.add_patch(right_rects)
    ani = animation.FuncAnimation(fig,animtate,frames=landmarks.shape[0],interval=100,repeat=True) 
    plt.show()
    print(landmarks.shape)
    print(np.max(landmarks),np.min(landmarks))
def show_video_result():
    id = '50036'
    video_dir = r"C:\Users\Alex\Desktop\Sign_dataset\WLASL2000\video"
    # output_dir = os.path.join(r"C:\Users\Alex\Desktop\Sign_dataset",'rtmpose_'+id)
    output_dir = r"C:\Users\Alex\Desktop\Sign_dataset\67578"
    os.makedirs(output_dir, exist_ok=True)
    ret = load_mmpose_data(id,video_dir)
    video_path = os.path.join(video_dir,id+'.mp4')
    frames = extract_frame(video_path)
    contrast = 0
    brightness = -50
    for i in range(len(frames)):
        blur_img = cv2.GaussianBlur(frames[i], (0, 0), 5)
        frames[i] = cv2.addWeighted(frames[i], 1.5, blur_img, -0.5, 0)
    # frames = [cv2.convertScaleAbs(frame, alpha=contrast/127 + 1, beta=brightness-contrast) for frame in frames]
    # frames = frames * (contrast/127 + 1) - contrast + brightness
    person_frames = set_person_frame(frames)
    landmarks = ret['landmarks']
    confs = ret['confs']
    frames_shape = ret['frames_shape']
    for i,f in enumerate(person_frames):
        landmark = landmarks[i]
        landmark = landmark*np.array([frames_shape[i][1]*2,frames_shape[i][0]*2])
        landmark = landmark.astype(np.int32)
        f = cv2.resize(f, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        for line in BODY_CONNECT:
            if np.all(landmark[line[0]]) == 0 or np.all(landmark[line[1]]) == 0:
                continue
            cv2.line(f,(landmark[line[0],0],landmark[line[0],1]),(landmark[line[1],0],landmark[line[1],1]),(255,145,0),1)
        for idx, (x, y) in enumerate(landmark):
            cv2.circle(f,(x,y),3,(0,0,255),-1)
            # cv2.putText(f,str(idx),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
       
        
        cv2.imshow('Ouput',f)
        if cv2.waitKey(100)& 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
   main()