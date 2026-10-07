# -*- coding: utf-8 -*-

import json
import  cv2
import numpy as np
import pandas as pd
import pickle

import mediapipe as mp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import os
from collections import Counter



#from mm2d_whole_body import *
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from body_joint_norm import *
from utils import *

WLASL_VIDEOS = r"C:\Users\Alex\Desktop\Sign_dataset\WLASL2000\video"
WLASL_JSON = "C:\\Users\\Alex\\Desktop\\CISLR\\WLASL_v0.3.json"
FRAME_SIZE = (224,224)
BG_COLOR = (192, 192, 192) # gray


hand_ckpt = 'mediapipe_ckpt\hand_landmarker.task'
with open(hand_ckpt,'rb') as f:
        hand_ckpt = f.read()
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
hands = mp_hands.Hands(model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.8,
    static_image_mode = False)
pose = mp_pose.Pose()

writergif = animation.PillowWriter(fps=100)
writermp4 = animation.FFMpegWriter(fps=25,extra_args=['-vcodec', 'libx264'])
print('load complete')

hand_connections = [[0,1],[1,2],[2,3],[3,4],
                   [0,5],[5,6],[6,7],[7,8],
                   [0,9],[9,10],[10,11],[11,12],
                   [0,13],[13,14],[14,15],[15,16],
                   [0,17],[17,18],[18,19],[19,20]]
joints_list = [[1,0,5],[1,0,9],[1,0,13],[1,0,17],
                [0,1,2],[1,2,3],[2,3,4],
                [0,5,6],[5,6,7],[6,7,8],
                [0,9,10],[9,10,11],[10,11,12],
                [0,13,14],[13,14,15],[14,15,16],
                [0,17,18],[17,18,19],[18,19,20]]
body_connection = [[0,1],[0,2],[2,4],[1,3],[3,5]]

coco_connection = [[]]
    

def vedio2frames(video_path,bbox = None,interval= None,resize = False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print('FPS: ', fps)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # print('After FPS: ', fps)
    frames = []    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # using bbox might effect landmark accuracy
        if bbox:
            print(bbox)
            frame=frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        if resize:
            frame = cv2.resize(frame,dsize=(768,768),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames if not interval else frames[interval[0]-1:interval[1]]

def get_missing_id(missing_txt):
    missing = []
    with open(missing_txt,'r') as file:
        missing = [line.rstrip() for line in file]
    return missing

# need data fps, vedio_id, split(train or test), frames, landmark,bounding box 
def load_data(file):
    print('Start load data.')
    need = ["video_id","split","frame_start","frame_end","bbox"]
    data = [] 
    # missing = get_missing_id(MISSING_FILE) 
    with open(file) as f:
        dic = json.load(f)
        for gesture in dic :
            gloss = gesture["gloss"]
            for inst in gesture["instances"]:
                tmp = {}
                tmp["gloss"] = gloss
                for n in need :
                    tmp[n] = inst[n]
                data.append(tmp)
    return data



## only need 11~17 from left wrist to right wrist 
## return shape : [60,{Left:[21,3],Right:[21,3]},Body:{6,3}] [21,3] is mediapipe landmark shape 
def video2landmark(id,video_dic = WLASL_VIDEOS):
    video_path = os.path.join(video_dic,id+".mp4")
    frames = vedio2frames(video_path, resize=True)
    landmarks = []
    for frame in frames:
        landmark = {'Left':[[0]*3]*21,'Right':[[0]*3]*21,'Body':[[0]*3]*6}
        size=frame.shape
        hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        body_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # if hand_results.hand_rects:
        #     for rect in hand_results.hand_rects:
        #       # print(rect)
        #       w = size[1]
        #       h = size[0]
        #       x_center = rect.x_center * size[1]
        #       y_center = rect.y_center * size[0]

        #       left_top = (max(0, int(x_center - rect.width*w/2)), max(0, int(y_center - rect.height*h/2)))
        #       right_down = (min(int(x_center + rect.width*w/2), w), min(int(y_center + rect.height*h/2), h))

        #       # print('left:', left_top)
        #       # print('right:', right_down)
        #       cv2.rectangle(frame, left_top, right_down, (255, 0, 0), 2)
        # if hand_results.palm_detections:
        #     for palm_detection in hand_results.palm_detections:

        #         bbox = palm_detection.location_data.relative_bounding_box
        #         print(normalized_to_pixel_coordinates(bbox.xmin,bbox.ymin,size[1],size[0]))
        #         print(normalized_to_pixel_coordinates(bbox.xmin+bbox.width,bbox.ymin+bbox.height,size[1],size[0]))
        #         cv2.rectangle(frame,normalized_to_pixel_coordinates(bbox.xmin,bbox.ymin,size[1],size[0]),normalized_to_pixel_coordinates(bbox.xmin+bbox.width,bbox.ymin+bbox.height,size[1],size[0]),(0,0,0),1)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks,handedness in  zip(hand_results.multi_hand_landmarks,hand_results.multi_handedness):
                    tmp = []
                    for i,h in enumerate(hand_landmarks.landmark):
                        tmp.append([h.x,h.y,h.z])
                    # tmp= normalize_landmark_length(tmp,frame.shape)
                    landmark[handedness.classification[0].label]=tmp
                    
                    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if body_results.pose_landmarks:
            tmp = [[landmark.x,landmark.y,landmark.z] for landmark in body_results.pose_landmarks.landmark]
            conf = [landmark.visibility for landmark in body_results.pose_landmarks.landmark]
            # mp_drawing.draw_landmarks( frame, body_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())        
            landmark['Body'] = tmp
            landmark['conf'] = conf
        else:
            landmark['Body'] = [[0]*3]*33
            landmark['conf'] = [0]*33
        landmarks.append(landmark)
    #     cv2.imshow('Landmarks Detection', frame)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    return landmarks, len(frames)

def get_body_landmark(frames):
    ret = []
    for idx, image in enumerate(frames):
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            upper_body = results.pose_landmarks.landmark[11:17]
            output_position = np.array([[landmark.x,landmark.y,landmark.z] for landmark in upper_body])
            visibility = np.array([landmark.visibility for landmark in upper_body])
            ret.append({'landmark':output_position,'visibility':visibility})
    return ret


"""
    source : https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
             https://github.com/AI4Bharat/OpenHands/tree/main
    pose_landmarks
    z: Should be discarded as currently the model is not fully trained to predict depth, but this is something on the roadmap.(pose的z座標不準)
    left_hand_landmarks :
    A list of 21 hand landmarks on the left hand. 
    Each landmark consists of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively.
    z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera. 
    The magnitude of z uses roughly the same scale as x.
    right_hand_landmarks :
    A list of 21 hand landmarks on the right hand, in the same representation as left_hand_landmarks.
    adding nose landmark into body landmark

    haven't shift all hand to orgin
"""
def get_frames(id, video_dic = WLASL_VIDEOS):
    video_path = os.path.join(video_dic,id+".mp4")
    frames = vedio2frames(video_path)
    return frames

def process_body_landmarks(component, n_points):
        kps = np.zeros((n_points, 3))
        conf = np.zeros(n_points)
        if component is not None:
            landmarks = component.landmark
            kps = np.array([[p.x, p.y, p.z] for p in landmarks])
            conf = np.array([p.visibility for p in landmarks])
        return kps, conf
    
def process_other_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
    return kps
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    return image
def get_orgin_holistic(id, video_dic = WLASL_VIDEOS):
    video_path = os.path.join(video_dic,id+".mp4")
    output_dir = os.path.join(r"C:\Users\Alex\Desktop\Sign_dataset",'holistic_'+id)
    os.makedirs(output_dir, exist_ok=True)
    frames = vedio2frames(video_path,resize=True)
    body = []
    face = []
    left_hand = []
    right_hand = []
    with mp_holistic.Holistic(
    min_tracking_confidence = 0.5,
    min_detection_confidence = 0.5,
    model_complexity=1,
    static_image_mode = False) as holistic:
        for idx, image in enumerate(frames):
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            body_data, body_conf = process_body_landmarks(
                results.pose_landmarks, 33
            )
            face_data = process_other_landmarks(
                results.face_landmarks, 468
            )
            lh_data = process_other_landmarks(
                results.left_hand_landmarks, 21
            )
            rh_data = process_other_landmarks(
                results.right_hand_landmarks, 21
            )
            body.append({'landmark':body_data,'conf':body_conf})
            face.append(face_data)
            left_hand.append(lh_data)
            right_hand.append(rh_data)
            image = draw_landmarks(image, results)
            cv2.imshow('Landmarks Detection', image)
            cv2.imwrite(os.path.join(output_dir, f'{idx}.jpg'), image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    return {'body':np.array(body),'face':np.array(face),'left_hand':np.array(left_hand),'right_hand':np.array(right_hand)},frames

def get_holistic_landmark(id, video_dic = WLASL_VIDEOS):
    video_path = os.path.join(video_dic,id+".mp4")
    frames = vedio2frames(video_path)
    body = []
    hand = []
    nose = []
    with mp_holistic.Holistic(
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as holistic:
        for idx, image in enumerate(frames):
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                upper_body = results.pose_landmarks.landmark[11:17]
                n = results.pose_landmarks.landmark[0]
                output_position = [[landmark.x,landmark.y,landmark.z] for landmark in upper_body]
                output_position.append([n.x,n.y,n.z]) # add nose and shift to shoulder middle
                visibility = np.array([landmark.visibility for landmark in upper_body])
                norm_output_position = normalize_body_joints(output_position)   
                # norm_output_position = shift_body2orgin(output_position)
                nose.append(norm_output_position[-1])
                norm_output_position = norm_output_position[:-1]
                
                body.append({'landmark':np.asarray(norm_output_position, dtype=np.float32),'visibility':visibility})
            else:
                body.append({'landmark':np.zeros((6,3)),'visibility':np.zeros(6)})
                nose.append([0,0,0])
            if results.left_hand_landmarks:
                left_hand = results.left_hand_landmarks.landmark
                left_hand_output_position = np.array([[landmark.x,landmark.y,landmark.z] for landmark in left_hand])
            else:
                left_hand_output_position = np.zeros((21,3))
            if results.right_hand_landmarks:
                right_hand = results.right_hand_landmarks.landmark
                right_hand_output_position = np.array([[landmark.x,landmark.y,landmark.z] for landmark in right_hand])
            else:
                right_hand_output_position = np.zeros((21,3))
            hand.append({'Left':left_hand_output_position,'Right':right_hand_output_position})

    # hand_kf = [kalman_filter() for _ in range(42)]
    body_kf = [kalman_filter() for _ in range(6)]
    for idx,frame in enumerate(frames):
        # for i in range(21):
        #     hand[idx]['Left'][i][:2] = hand_kf[i].process(hand[idx]['Left'][i][:2])
        #     hand[idx]['Right'][i][:2] = hand_kf[i].process(hand[idx]['Right'][i][:2])
        if np.all(body[idx]['landmark'] == 0):
            continue
        for i in range(6):
            body[idx]['landmark'][i][:2] = body_kf[i].process(body[idx]['landmark'][i][:2]).flatten()
    ret = {'body':body,'hand':hand,'nose':np.asarray(nose)}
    return ret



def mediapipe_webcam_result():
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer= hand_ckpt),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        i=0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detect_result= landmarker.detect_for_video(mp_image, i)
            if detect_result.hand_landmarks:
                for hand_landmark in detect_result.hand_landmarks:
                    joints = [[joint.x,joint.y,joint.z] for joint in hand_landmark]
                    for j in joints:
                        x,y = int(j[0]*frame.shape[1]),int(j[1]*frame.shape[0])
                        cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)
            # Our operations on the frame come here
            # Display the resulting frame
            cv2.imshow('frame', frame)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    

def load_from_pickle(id,dir_name):
    with open(os.path.join(dir_name,id+'.pickle'),'rb') as f:
        data = pickle.load(f)
    return data


def get_topk_label_list(k,data):
    l = [d['gloss'] for d in data]
    counts = Counter(l)
    top__k = counts.most_common(k)
    print(sum([count for e,count in top__k]))

def get_label_index_dict(path):
    ret = {}
    with open(path,'r') as file:
        for line in file:
            l = line.strip().split('\t')
            ret[l[1]] = l[0]
    return ret
def get_vedio_and_label(data,dic,output_file):
    with open(output_file,'w') as file:
        for d in data:
            id = d['video_id']
            print(d['gloss'])
            label = dic[d['gloss']]
            video_path = os.path.join(WLASL_VIDEOS,id+'.mp4')
            file.write(f"{video_path} {label}\n")
# 07933 07943
# 03238 03246
if __name__ == '__main__':
    # video_path = r"C:\Users\Alex\Desktop\kaggle_WLASL\videos\03235.mp4"
    # data = load_data(WLASL_JSON)
    # # label_dict = get_label_index_dict(LABEL_FILE)

    # #landmarks = load_from_pickle('03120',r'C:\Users\Alex\Desktop\SLR_Code\landmarks\wlasl_rtmw')
    # landmarks  = video2landmark('03120')
    # landmarks1 = video2landmark('03238')
    # landmarks2 = video2landmark('03246')
    # print(landmarks)
    ret,frames = get_orgin_holistic('28207')
    # body = np.asarray([l['landmark']  for l in ret['body']])[...,:2]
    # left_hand = ret['left_hand'][...,:2]
    # right_hand = ret['right_hand'][...,:2]
    # print(np.shape(body),ret['face'].shape,ret['left_hand'].shape,ret['right_hand'].shape)
    # for i,f in enumerate(frames):
    #     s = f.shape
    #     body[i] = body[i]*np.array([s[1],s[0]])
    #     left_hand[i] = (left_hand[i]*np.array([s[1],s[0]]))
    #     right_hand[i] = (right_hand[i]*np.array([s[1],s[0]]))
    #     body = body.astype(int)
    #     left_hand = left_hand.astype(int)
    #     right_hand = right_hand.astype(int)
    #     for j in range(33):
    #         cv2.circle(f,(body[i][j][0],body[i][j][1]),3,(0,0,255),-1)
    #         cv2.putText(f,str(i),(body[i][j][0],body[i][j][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    #     for j in range(21):
    #         cv2.circle(f,(left_hand[i][j][0],left_hand[i][j][1]),3,(0,0,255),-1)
    #         cv2.circle(f,(right_hand[i][j][0],right_hand[i][j][1]),3,(0,0,255),-1)
    #     cv2.imshow('frame',f)
    #     if cv2.waitKey(100) & 0xFF == ord('q'):
    #         break