import tkinter as tk
from tkinter import filedialog
from moviepy.editor import VideoFileClip
from demo import demo_output

def upload_video():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        video = VideoFileClip(file_path)
        duration = video.duration
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"影片已上傳!\n檔案路徑: {file_path}\n影片長度: {duration:.2f} 秒")
        video.close()
        inference_button.config(state=tk.NORMAL)  # 啟用推論按鈕

def run_inference():
    if 'file_path' in globals():
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "\n\n開始進行模型推論...")
        root.update()
        det_time, inf_time, output_top_5, output = demo_output()
        result_text.insert(tk.END, f"\n偵測時間: {det_time:.2f} 秒")
        result_text.insert(tk.END, f"\n推論時間: {inf_time:.2f} 秒")
        result_text.insert(tk.END, f"\nTop 5: {output_top_5}")
        result_text.insert(tk.END, f"\n預測結果: {output}")
    else:
        result_text.insert(tk.END, "\n\n請先上傳影片!")

# 創建主窗口
root = tk.Tk()
root.title("Demo")

# 創建並放置上傳按鈕
upload_button = tk.Button(root, text="上傳影片", command=upload_video)
upload_button.pack(pady=20)

# 創建並放置推論按鈕
inference_button = tk.Button(root, text="執行推論", command=run_inference, state=tk.DISABLED)
inference_button.pack(pady=10)

# 創建文本框來顯示結果
result_text = tk.Text(root, height=10, width=100)
result_text.pack(pady=100)

# 運行應用程序
root.mainloop()