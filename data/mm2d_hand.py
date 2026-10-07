# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/kennymckormick/pyskl/blob/main/tools/data/custom_2d_skeleton.py

import numpy as np
import decord
import cv2
import os
try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector, DetInferencer
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose  # noqa: F401
    from mmpose.apis import inference_topdown, init_model
    from mmpose.apis import visualize
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

root = 'C:\\Users\\Alex\\Desktop\\SLR_Code\\MMpose'
det_config = r'C:\Users\Alex\Desktop\mmpose\demo\mmdetection_cfg\rtmdet_nano_320-8xb32_hand.py'
det_ckpt = r'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth'
pose_config  = os.path.join(root,'td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288.py')
pose_ckpt =  os.path.join(root,'hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')
metainfo = os.path.join(root,'coco_wholebody.py')
pose_model = init_model(pose_config, pose_ckpt, 'cuda')
det_model = init_detector(det_config,det_ckpt,'cuda')
# hand keypoint 92~133
# upprt body 6~9

def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def show_det_result(frames,det_results):
    for frame,pred in zip(frames,det_results):
        frame = frame[:,:,::-1].copy()
        frame = np.ascontiguousarray(frame)
        # print(frame.shape)
        x0, y0, x1, y1 = pred.pred_instances['bboxes'][0]
        # print((x0, y0), (x1, y1))  
        cv2.rectangle(frame, (int(x0+0.5), int(y0+0.5)), (int(x1+0.5), int(y1+0.5)), (0, 255, 0), 2) 
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25)& 0xFF == ord('q'):
            break

def show_pose_result(frames,pose_results):
    # visulize video
    for idx in range(len(frames)):
        img = frames[idx][:,:, ::-1].astype(np.uint8)
        # print('**'*50)
        # print(pose_results[idx])
        img = visualize(img=img,keypoints=np.array([pose_results[idx]]),keypoint_score=None,metainfo=metainfo)
        cv2.imwrite('temp.png', img)
        img = cv2.imread('temp.png')
        f = cv2.imread('temp.png')
        cv2.imshow('Ouput',f)
        if cv2.waitKey(100)& 0xFF == ord('q'):
            break

def pose_inference(video_path):
    garbage_frame = 0
    frames = extract_frame(video_path)
    det_results = inference_detector(det_model,frames)['predictions']
    print(len(frames))
    print(len(det_results))
    if det_results is not None:
        assert len(frames) == len(det_results)
        total_frames = len(frames)
        kp = np.zeros((total_frames, 133, 2), dtype=np.float32)
        # bb = np.zeros((total_frames, 5), dtype=np.float32)

        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            #d = [dict(bbox=x['bboxes']) for x in list(d)]
            pose = inference_topdown(pose_model, f, d['bboxes'])[0]
            # print('=='*50)
            pred = pose.pred_instances
            print(pred)
            keypoints = pred['keypoints']
            bbox = pred['bboxes']
            keypoints_scores = pred['keypoint_scores']
            # print(keypoints_scores)
            # print('keypoints ', keypoints)
            # print('bbox ', bbox)
            # print('=='*50)
            if pose == []:
                # not detect person
                garbage_frame += 1
                continue
            #pose = sorted(pose, key=lambda x:x['bbox'][-1])
            #keypoints, bbox = pose[-1]['keypoints'], pose[-1]['bbox']
            kp[i] = keypoints
            # bb[i] = bbox

    else:
        print(frames.shape)
        d = [{'bbox': np.array([0, 0, frames.shape[2]-1, frames.shape[1]-1])}]
        pose = inference_topdown(pose_model, frames[0], None, format='xyxy')[0]

    return kp, garbage_frame


def main():
    video_path = r'C:\Users\Alex\Desktop\SLR_Code\00414.mp4'
    frames = extract_frame(video_path)
    shape = frames[0].shape
    #print(det_results['predictions'])
    #pose_result,garbage_frame = pose_inference(video_path)
    det_results = inference_detector(det_model,frames)
    #print(pose_result.shape)
    #show_det_result(frames,det_results)
    #print('garbage_frame ',garbage_frame)
    print(det_results[0].pred_instances)
    show_det_result(frames,det_results)
if __name__ == '__main__':
    main()