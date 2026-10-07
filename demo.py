import torch
import numpy as np
from data.mm2d_whole_body import load_mmpose_data
from data.preprocess import fix_data_length
from utils.util import load_model 
import time
weight_file = './weights/5aa55624.pth' # current best skelton base model\
def demo_output():
    start = time.time()
    ret = load_mmpose_data('e720d834',r"C:\Users\Alex\Desktop\Sign_dataset\demo_video")
    det_time = time.time()-start
    landmarks = ret['landmarks']
    zero_idex = np.all(landmarks == 0,axis=2)
    landmarks = 2*(landmarks-0.5)
    landmarks[zero_idex] = 0
    model = load_model('ctrgcn',100, 'model.graph.rtmw_whole_body60_graph.Graph').to('cuda')
    model.load_state_dict(torch.load(weight_file, map_location=torch.device('cuda')))
    model.eval()
    data = landmarks
    label_dict = np.load('label_dict.npy',allow_pickle= True)
    label_dict = {v: k for k, v in label_dict.item().items()}
    data = fix_data_length(data.astype(np.float32), 50, data.shape[1:])
    data = torch.tensor(data).unsqueeze(0).to('cuda').permute(0,3,1,2)
    start = time.time()
    output1,output2,output3 = model(data)
    inf_time = time.time()-start
    output = output1 + output2 + output3
    predict_label = output.max(1)[1].detach().cpu().numpy().tolist()[0]
    output  = output.detach().cpu().numpy()[0]
    output_top_5 = dict(zip([label_dict[i]for i in np.argsort(output)[-5:]],np.sort(output)[-5:]))
    print('Top 5:', output_top_5)
    print('Predicted label:', label_dict[predict_label])
    return det_time,inf_time,output_top_5,label_dict[predict_label]
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class VideoClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("影片分類器")
        self.master.geometry("600x400")
        
        # 設置主題
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 自定義顏色
        self.style.configure('TButton', 
                             background='#4CAF50', 
                             foreground='white', 
                             font=('Arial', 12, 'bold'),
                             padding=10)
        self.style.map('TButton', 
                       background=[('active', '#45a049')])
        
        self.style.configure('TLabel', 
                             font=('Arial', 14),
                             padding=10)
        
        self.bg_image = Image.open("background.jpg")  # 請確保你有一張名為 background.jpg 的圖片
        self.bg_image = self.bg_image.resize((600, 400), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # 設置背景
        self.master.configure(bg='#f0f0f0')
        
        self.start_page()
        
    def start_page(self):
        self.clear_window()
        
        start_button = tk.Button(self.master, text="開始", command=self.selection_page)
        start_button.pack(expand=True)
        
    def selection_page(self):
        self.clear_window()
        
        for i in range(10):
            button = tk.Button(self.master, text=f"選項 {i+1}", command=lambda x=i: self.upload_video(x))
            button.pack()
        
    def upload_video(self, option):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            result = self.process_video(file_path, option)
            self.show_result(result)
    
    def process_video(self, file_path, option):
        # 這裡應該是你的深度學習模型處理邏輯
        # 這只是一個示例，你需要根據你的實際模型來修改
        ret = load_mmpose_data(file_path)
        landmarks = ret['landmarks']
        zero_idex = np.all(landmarks == 0,axis=2)
        landmarks = 2*(landmarks-0.5)
        landmarks[zero_idex] = 0
        model = load_model('ctrgcn',100, 'model.graph.rtmw_whole_body60_graph.Graph').to('cuda')
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cuda')))
        model.eval()
        data = landmarks
        label_dict = np.load('label_dict.npy',allow_pickle= True)
        label_dict = {v: k for k, v in label_dict.item().items()}
        data = fix_data_length(data.astype(np.float32), 50, data.shape[1:])
        data = torch.tensor(data).unsqueeze(0).to('cuda').permute(0,3,1,2)
        start = time.time()
        output1,output2,output3 = model(data)
        inf_time = time.time()-start
        output = output1 + output2 + output3
        predict_label = output.max(1)[1].detach().cpu().numpy().tolist()[0]
        output  = output.detach().cpu().numpy()[0]
        output_top_5 = dict(zip([label_dict[i]for i in np.argsort(output)[-5:]],np.sort(output)[-5:]))
        print('Top 5:', output_top_5)
        print('Predicted label:', label_dict[predict_label])
        
        return f"選項 {option+1} 的結果\nTop 5: {output_top_5}\nPredicted label: {label_dict[predict_label]}"
    
    def show_result(self, result):
        self.clear_window()
        
        result_label = tk.Label(self.master, text=result)
        result_label.pack(expand=True)
        
        back_button = tk.Button(self.master, text="返回", command=self.selection_page)
        back_button.pack()
    
    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()

root = tk.Tk()
app = VideoClassifierApp(root)
root.mainloop()