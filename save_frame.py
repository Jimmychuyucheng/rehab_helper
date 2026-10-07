import cv2
from matplotlib import animation, pyplot as plt
import numpy as np
import os
from data.dataset import WLASL_Define_Dataset
import pytube
video_path = r"C:\Users\Alex\Desktop\Sign_dataset\WLASL100\short\51224.mp4"
output_dir = r"C:\Users\Alex\Desktop\Sign_dataset\51224"
os.makedirs(output_dir, exist_ok=True)

def idk():
    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        # frame = frame[:,100:-140]
        frame_count += 1
        cv2.imwrite(os.path.join(output_dir, f'{frame_count}.jpg'), frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames: {frame_count}")

def download_youtube_video(url, output_path):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    video.download(output_path)

if __name__ == "__main__":
    idk()