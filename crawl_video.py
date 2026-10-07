
import requests
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time 
from selenium.webdriver.support.ui import WebDriverWait
import urllib.parse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import ssl
import os
from pytube import YouTube
from tqdm import tqdm
import random
import json
from multiprocessing import cpu_count, Process
from pytube.exceptions import VideoUnavailable
from http.client import IncompleteRead
import uuid
"""
    crwal steps:
        1. enter url 
        2. use selenium to click the 筆畫查詢 button
        3. iterate each 筆畫 button and click

"""


url = 'https://special.moe.gov.tw/signlanguage/vocabulary/category'
twsl_url = 'https://www.twsl.cc/TradChinese/Type/WordsType'

# headers = {
#    "Referer": "https://special.moe.gov.tw/signlanguage/vocabulary/category",
# }

save_dir = r"C:\Users\Alex\Desktop\Sign_dataset\tw_sign_language"

options = Options()
ssl._create_default_https_context = ssl._create_stdlib_context
# options.add_argument("headless")
options.page_load_strategy = 'normal'

# 常用手語詞彙
def crawl_gov_website(url):
    label_list = []
    with webdriver.Chrome(options=options) as driver:
        driver.set_page_load_timeout(100)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        sign_link_box = soup.find('div',class_ = 'box')
        sign_links = sign_link_box.find_all('a')[3:]
        # print(type(sign_links))
        for link in tqdm(sign_links, desc='Processing category'):
            link_url = urllib.parse.urljoin(url,link['href'])
            driver.get(link_url)
            wait = WebDriverWait(driver, timeout=300)
            wait.until(lambda driver: driver.find_element(By.CLASS_NAME,'idx-tab'))
            link_soup = BeautifulSoup(driver.page_source,'html.parser')
            link_box = link_soup.find('div',class_ = 'box')
            tabindexes = link_box.find_all('a')
            click_objects = driver.find_elements(By.CLASS_NAME,'dot')
            i = 0
            for tabindex,click_object in tqdm(zip(tabindexes,click_objects),desc='Processing link'):
                chinese_label = tabindex.find('span',class_ = 'dot')
                english_label = tabindex.find('span',class_ = 'text')
                # print(english_label.text,' ',chinese_label.text)
                js = 'document.getElementsByClassName("dot")[{}].click()'.format(i)
                try :
                
                    driver.execute_script(js)
                    time.sleep(1)
                except TimeoutException:
                    print("TimeoutException: The open pop up window command did not complete in enough time.")
                soup = BeautifulSoup(driver.page_source,'html.parser')
                youtube_link = soup.find('iframe')['src']
                # print('youtube ',youtube_link)
                # yt = YouTube(youtube_link)
                # streams = yt.streams.filter(file_extension='mp4').first()
                # f_name = streams.default_filename
                # streams.download(save_dir)
                label_list.append({'label':english_label.text,'chineese':chinese_label.text,'youtube':youtube_link})
                js = 'document.getElementsByClassName("popup-close tab-index-no popupClose")[0].click()'
                try:
                    driver.execute_script(js)
                    time.sleep(1)
                except:
                    print("TimeoutException: The close pop up window command did not complete in enough time.")
                
                i+=1
    with open('label.json','w') as f:    
        json.dump(label_list,f,ensure_ascii=False, indent=2)
# 幼兒園常用手語詞彙
def crawl_ece_website(url):
    label_list = []
    with webdriver.Chrome(options=options) as driver:
        driver.set_page_load_timeout(100)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        link_table = soup.find('div',class_ = 'containerBlk')
        a_tags = link_table.find_all('a')
        hrefs = [a.get('href') for a in a_tags if a.get('href').startswith('https://') ]
        print(len(hrefs))
        for href in hrefs:
            driver.get(href)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            link_table = soup.find('div',class_ = 'containerBlk')
            video_urls = link_table.find_all('a')
            labels = [a.text.split('：')[1] for a in video_urls if a.text and a.text.startswith('詞彙')]
            hrefs = [a.get('href') for a in video_urls if a.get('href').startswith('https://www.youtube.com/watch?v=')]
            for label,href in zip(labels,hrefs):
                label_list.append({'label':label,'youtube':href})
    with open('label.json','w', encoding='utf-8') as f:    
        json.dump(label_list,f,ensure_ascii=False, indent=2)
# 學齡前2至6歲教保服務人員手語手冊
def crawl_ece_website2(url):
    label_list = []
    with webdriver.Chrome(options=options) as driver:
        driver.set_page_load_timeout(100)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        link_table = soup.find('div',class_ = 'annexBlk link')
        links = link_table.find_all('a')
        hrefs = ['https://www.ece.moe.edu.tw/'+a.get('href') for a in links if not a.get('href').endswith('.pdf') ]
        print(hrefs)
        for href in hrefs:
            driver.get(href)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            table = soup.find("table")
            tbody = table.find('tbody')
            video_urls = tbody.find_all('a')
            labels = [a.text.split('：')[1] for a in video_urls if a.text and a.text.startswith('詞彙')]
            hrefs = [a.get('href') for a in video_urls if a.get('href').startswith('https://youtu.be/')]
            for label,href in zip(labels,hrefs):
                print(label,'   ',href)
                label_list.append({'label':label,'youtube':href})
    with open('label.json','w', encoding='utf-8') as f:    
        json.dump(label_list,f,ensure_ascii=False, indent=2)
def download_video():
    with open(r"C:\Users\Alex\Desktop\Sign_dataset\tw_sign_language\全國教保資訊網\chinese_label.json",'r',encoding='utf-8') as f:
        data = json.load(f)
        base_url = "https://www.youtube.com/watch?v="
        target_dir = r'C:\Users\Alex\Desktop\Sign_dataset\tw_sign_language\全國教保資訊網\videos'
        print(len(data))
        for d in tqdm(data):
            link = base_url + d['youtube_id']
            try:
                yt = YouTube(link)
                streams = yt.streams.filter(file_extension='mp4').first()
                f_name = streams.default_filename
                d['filename'] = f_name
                if not os.path.exists(os.path.join(target_dir,f_name)):

                    streams.download(target_dir,filename=d['youtube_id']+'.mp4')
                else:
                    print('File{} already exists'.format(f_name))
            except:
                print('Error ',link)
    with open('label.json','w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False, indent=2)

def crawl_twsl_website():
    url = 'https://www.twsl.cc/TradChinese/Words/Detail/86'
    with webdriver.Chrome(options=options) as driver:
        driver.set_page_load_timeout(100)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        video_urls = soup.find_all('video')
        url = video_urls[0]['src']
        for video_url in video_urls:
            print(video_url.prettify())
def generate_rd_name():
    uuid_str = str(uuid.uuid4())
    uuid_str = uuid_str.replace('-', '')[:8]
    return uuid_str
def dl_wb_video(url,save_dir):
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the content length of the video
        content_length = response.headers.get('content-length')

        # If the content length is available, start downloading the video
        if content_length is not None:
            print(f"Downloading video ({content_length} bytes)...")
            rd_name = generate_rd_name()
            with open(os.path.join(save_dir,rd_name+".mp4"), "wb") as video_file:
                # Write the video data to the file in chunks
                
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        video_file.write(chunk)
            print(f"Video{os.path.join(save_dir,rd_name+'.mp4')}downloaded successfully!")
            return os.path.join(save_dir,rd_name+".mp4")
        else:
            print("Unable to get the content length of the video.")
    else:
        print(f"Failed to download the video (status code: {response.status_code})")

def dl_yt_video(url):
    yt = YouTube(url)
    streams = yt.streams.get_highest_resolution()
    f_name = streams.default_filename
    
    streams.download(r"C:\Users\Alex\Desktop\Sign_dataset\MS-ASL",filename='tmp.mp4')
    print(f"Video {f_name} downloaded successfully!")
   

def clip_video(url,start_time,end_time,save_dir,replace = False):
    video = VideoFileClip(url)
    clipped_video = video.subclip(start_time, end_time)
    rd_name = generate_rd_name()    
    if replace:
        clipped_video.write_videofile(url)
    else:
        clipped_video.write_videofile(os.path.join(save_dir,rd_name+".mp4"))
    video.close()
    clipped_video.close()

def download_yt_video_segment(url, start_time, end_time, id, output_path, retries=5):
    for _ in range(retries):
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            download_path = stream.download(output_path=output_path,filename=str(id)+'.mp4')
            video = VideoFileClip(download_path)
            clipped_video = video.subclip(start_time, end_time)
            clipped_video.write_videofile(os.path.join(output_path, "clipped_video.mp4"))
            
            # Close the video files
            video.close()
            clipped_video.close()
            print(f"Downloaded video segment from {start_time} to {end_time} seconds.")
            return True
        except VideoUnavailable:
            print(f"Error downloading video segment: {url}")
        except IncompleteRead:
            print(f"Incomplete read while downloading video segment: {url}. Retrying...")
    return False

def MSASL_downloader(josn_path,save_dir):
    with open(josn_path,'r',encoding='utf-8') as f:
        data_info = json.load(f)
    start = [i['start_time'] for i in data_info]
    end = [i['end_time'] for i in data_info]
    url = [i['url'] for i in data_info]
    id = [i['id'] for i in data_info]
    print(len(data_info))
    fail_id = []
    for i in range(len(data_info)):
        if os.path.exists(os.path.join(save_dir,str(id[i]))):
            print('File{} already exists'.format(id[i]))
            continue
        if not download_yt_video_segment(url[i],start[i],end[i],id[i],save_dir):
            fail_id.append(id[i])
    print("Number of failed video: ",len(fail_id))
    print(fail_id)
    with open('fail_id.json','w') as f:
        json.dump(fail_id,f,ensure_ascii=False, indent=2)
        
def cut_video_to_image(video_pth, save_dir):
    video = VideoFileClip(video_pth)
    for i, frame in enumerate(video.iter_frames()):
        frame.save(os.path.join(save_dir, f"frame_{i}.png"))
    video.close()

def tmp():
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    data = np.load(r"C:\Users\Alex\Desktop\Sign_dataset\MS-ASL\MSASL100\train_data_joint.npy",allow_pickle=True)
    data = data.squeeze()
    print(data.shape)
    sample = data[0]
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Define the update function for the animation
    def update(frame):
        ax.clear()
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Create a figure and axis for the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=data, interval=200)

    # Display the animation
    plt.show()


"""
    First store link and label then download
"""
if __name__ == '__main__':
    # save_dir = r"C:\Users\Alex\Desktop\Sign_dataset\MS-ASL\masl_train"
    # json_path = r'C:\Users\Alex\Desktop\Sign_dataset\MS-ASL\MSASL_train.json'   
    # # MSASL_downloader(json_path,save_dir)
    # input_soucre = r"https://www.youtube.com/watch?v=knu2YforVYw"
    # output_dir =r'C:\Users\Alex\Desktop\Sign_dataset\tw_sign_language\第一冊單詞\一'
    # # output_video = dl_wb_video(input_soucre,output_dir)
    # # clip_video(input_soucre,143,146,output_dir)
    # download_video()
    # clip_video(r"C:\Users\Alex\Desktop\Sign_dataset\台灣手語南北差異.mp4",0,8,r"C:\Users\Alex\Desktop\Sign_dataset\南北差異")
    clip_video(r"C:\Users\Alex\Pictures\Camera Roll\WIN_20240701_14_26_24_Pro.mp4",1,6,r"C:\Users\Alex\Desktop\Sign_dataset\demo_video")




            


