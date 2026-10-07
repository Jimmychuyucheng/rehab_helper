import cv2
import matplotlib.pyplot as plt
import matplotlib as maplotlib
import matplotlib.animation as animation
import numpy as np
from .body_joint_norm import *
import time
import os

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

ffmpegpath = os.path.abspath(r'C:\Users\Alex\PYTHON\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2.exe')
maplotlib.rcParams['animation.ffmpeg_path'] = ffmpegpath

writergif = animation.PillowWriter(fps=30)
writermp4 = animation.FFMpegWriter(fps=25,extra_args=['-vcodec', 'libx264'])


def normalize_landmark_length(landmark,image_size):
    normalize_joint = normalize_all_joints(np.array(landmark).astype('float64'))
    return normalize_joint

## not test yet
def landmark2angle(landmark):
    angle_list = []
    for joint in joints_list:
        a = np.array([landmark[joint[0]][0], landmark[joint[0]][1]]) # First coord
        b = np.array([landmark[joint[1]][0], landmark[joint[1]][1]]) # Second coord
        c = np.array([landmark[joint[2]][0], landmark[joint[2]][1]]) # Third coord
        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = radians*180.0/np.pi#np.abs(radians*180.0/np.pi)
        # if angle > 180.0:
        #     angle = 360-angle
        angle_list.append(angle)
    return angle_list

def body2angle(landmark):
    angle_list = []
    for joint in body_connection:
        a = np.array([landmark[joint[0]][0], landmark[joint[0]][1]]) # First coord
        b = np.array([landmark[joint[1]][0], landmark[joint[1]][1]]) # Second coord
        radians = np.arctan2(b[1] - a[1], b[0]-a[0])
        angle = radians*180.0/np.pi#np.abs(radians*180.0/np.pi)
        # if angle > 180.0:
        #     angle = 360-angle
        angle_list.append(angle)
    return angle_list
    
def angle_visualize(angle_list,ax=None,reference_point=[[0,0],[1,0]]):
    ax.set_title('Angle Visualize')
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,5)
    ax.invert_yaxis()
    point_list = np.zeros((21,2), dtype=np.float64)
    point_list[0]=reference_point[0][:2]
    point_list[1]=reference_point[1][:2]
    for a,j in zip(angle_list,joints_list):
        theta = np.deg2rad(a)
        rotation_matrix = np.array([ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        output = np.dot(rotation_matrix,point_list[j[0]]-point_list[j[1]])
        point_list[j[2]] = point_list[j[1]] + output
    for c in hand_connections:
        ax.plot((point_list[c[0]][0],point_list[c[1]][0]),(point_list[c[0]][1],point_list[c[1]][1]),c = 'g')
    ax.scatter(point_list[:,0],point_list[:,1],c='r')

    for i,p in enumerate(point_list):
        ax.annotate(i,(p[0],p[1]))

def track_hand_visable(ax,landamrk,visibility):
    anotate_landmark = landamrk[4:6]
    anotate_vis = visibility[4:6]
    for i in range(len(anotate_landmark)):
        ax.text(anotate_landmark[i][0],anotate_landmark[i][1],anotate_landmark[i][2],str(anotate_vis[i]))

def cmp_orgin_and_angle_landmark(landmarks):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    
    def animate(i):
        for ax in axs:
            ax.clear()
        draw_landmark_2d(landmarks[i]['Left'],axs[0],angle=True)
        angle = landmark2angle(landmarks[i]['Left'])
        angle_visualize(angle,axs[1])

    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=25,repeat = False)
    plt.show()
    ani.save(filename="output/Angle.gif",writer=writergif)


# input shape [frame ,{left:[21,3],right[21,3]}]   
def show_landmark_2d(landmarks):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    def animate(i):
        for ax in axs:
            ax.clear()
        draw_landmark_2d(landmarks[i],axs)

    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=25,repeat = False)
    plt.show()
    #ani.save(filename="output/03120_mediapipe.gif",writer=writergif)
    

def draw_landmark_2d(landmark,axs,angle=False,set_lim=True):  
    if 'Right' in landmark:
        axs[1].set_title('Right Hand')
        if set_lim:
            # axs[1].set_xlim(-2,1)
            # axs[1].set_ylim(-2,1)
            axs[1].set_xlim(0,1)
            axs[1].set_ylim(0,1)
        axs[1].invert_yaxis()

        x = [round(l[0],5) for l in list(landmark['Right'])]
        y = [round(l[1],5) for l in list(landmark['Right'])]
        for c in hand_connections:
           axs[1].plot((x[c[0]],x[c[1]]),(y[c[0]],y[c[1]]),c='g')
        axs[1].scatter(x,y,c='r')
    if 'Left' in landmark:
        axs[0].set_title('Left Hand')
        if set_lim:
            # axs[0].set_xlim(-1,2)
            # axs[0].set_ylim(-1,2)
            axs[0].set_xlim(0,1)
            axs[0].set_ylim(0,1)
        axs[0].invert_yaxis()
        x = [round(l[0],5) for l in list(landmark['Left'])]
        y = [round(l[1],5) for l in list(landmark['Left'])]
        for c in hand_connections:
            axs[0].plot((x[c[0]],x[c[1]]),(y[c[0]],y[c[1]]),c='g')
        axs[0].scatter(x,y,c='b')
    if angle:
        axs.set_title('Origin')
        if set_lim:
            axs.set_xlim(-1,5)
            axs.set_ylim(-1,5)
        axs.invert_yaxis()
        x = [l[0] for l in landmark]
        y = [l[1] for l in landmark]
        for c in hand_connections:
            axs.plot((x[c[0]],x[c[1]]),(y[c[0]],y[c[1]]),c='yellowgreen')
        axs.scatter(x,y,c='r')

def show_landmark_3d(landmarks):
    fig = plt.figure(figsize=(10,5),layout='constrained')
    ax1 = plt.subplot(121,projection='3d')
    ax2 = plt.subplot(122,projection='3d')
    axs = [ax1,ax2]
    def animate(i):
        for ax in axs:
            ax.clear()
        draw_landmark_3d(landmarks[i],axs)

    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=25,repeat = False)
    plt.show()
    #ani.save(filename="output/diff_3dnorm.gif",writer=writergif)

def draw_landmark_3d(landmark,axs,angle=False):
    if 'Right' in landmark:
        axs[1].view_init(elev = -90 , azim = 90)
        axs[1].set_title('Right Hand')
        axs[1].set_xlim(0,1)
        axs[1].set_ylim(0,1)
        axs[1].set_zlim(-1,1)
        # axs[1].set_xlim(0.3,0.8)
        # axs[1].set_ylim(0.5,1.0)
        # axs[1].set_zlim(-0.2,0)
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_zlabel('Z')
        axs[1].invert_yaxis()
        xs = [l[0] for l in landmark['Right']]
        ys = [l[1] for l in landmark['Right']]
        zs = [l[2] for l in landmark['Right']]
        for c in hand_connections:
            axs[1].plot([xs[c[0]],xs[c[1]]],[ys[c[0]],ys[c[1]]],zs =[zs[c[0]],zs[c[1]]],c = 'violet')
        axs[1].scatter(xs,ys,zs = zs,color ='aquamarine')
    if 'Left' in landmark:
        axs[0].view_init(elev = -90 , azim = 90)
        axs[0].set_title('Left Hand')
        axs[0].set_xlim(0,1)
        axs[0].set_ylim(0,1)
        axs[0].set_zlim(-1,1)
        # axs[0].set_xlim(0.3,0.8)
        # axs[0].set_ylim(0.5,1.0)
        # axs[0].set_zlim(-0.2,0)
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_zlabel('Z')
        axs[0].invert_yaxis()
        xs = [l[0] for l in landmark['Left']]
        ys = [l[1] for l in landmark['Left']]
        zs = [l[2] for l in landmark['Left']]
        for c in hand_connections:
            axs[0].plot([xs[c[0]],xs[c[1]]],[ys[c[0]],ys[c[1]]],zs =[zs[c[0]],zs[c[1]]],c = 'teal')
        axs[0].scatter(xs,ys,zs = zs,color ='lime')
    if angle:
        xs = [l[0] for l in landmark]
        ys = [l[1] for l in landmark]
        zs = [l[2] for l in landmark]
        for c in hand_connections:
            plt.plot([xs[c[0]],xs[c[1]]],[ys[c[0]],ys[c[1]]],zs =[zs[c[0]],zs[c[1]]],c = 'teal')
        plt.scatter(xs,ys,zs = zs,color ='lime')
            
def show_body_3d(landmarks, visbility = None, id = None):
    fig = plt.figure(figsize=(10,5),layout='constrained')
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    def animate(i):
        ax1.clear()
        ax2.clear()
        draw_body_3d(landmarks[i],ax1)
        draw_body_3d(landmarks[i],ax2,elev=0,azim=90)
        if visbility is not None:
            track_hand_visable(ax1,landmarks[i],visbility[i])
    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=10,repeat = True)
    plt.show()
    if id:
        filename = "results/body_3d_" + id + ".gif"
    else:
        filename = "results/body_3d.gif"
    # ani.save(filename=filename,writer=writergif)

def cmp_body_landmarks(landmarks1,landmarks2,title1,title2,name):
    fig = plt.figure(figsize=(10,5),layout='constrained')
    ax1 = plt.subplot(121,projection='3d')
    ax2 = plt.subplot(122,projection='3d')
    def animate(i):
        print("frame: ",i)
        if i < len(landmarks1):
            ax1.clear()
            draw_body_3d(landmarks1[i], ax1, title1)
        if i < len(landmarks2):
            ax2.clear()
            draw_body_3d(landmarks2[i], ax2, title2)
    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks1), interval=25,repeat = False)
    plt.show()
    # ani.save(filename='output/cmp_kmean_'+name+'.gif',writer=writergif)
        


def draw_body_3d(landmark, ax, title = None, elev = -90 , azim = 90):
    ax.view_init(elev = elev , azim = azim)
    if title:
        ax.set_title(title)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-3,3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.invert_xaxis()
    xs = [float(l[0]) for l in landmark]
    ys = [float(l[1]) for l in landmark]
    zs = [float(l[2]) for l in landmark]
    for c in body_connection:
        ax.plot([xs[c[0]],xs[c[1]]],[ys[c[0]],ys[c[1]]],zs =[zs[c[0]],zs[c[1]]],c = 'violet')
        v1 = np.array([xs[c[0]],ys[c[0]],zs[c[0]]])
        v2 = np.array([xs[c[1]],ys[c[1]],zs[c[1]]])
        # ax.text((xs[c[0]]+xs[c[1]])/2,(ys[c[0]]+ys[c[1]])/2,(zs[c[0]]+zs[c[1]])/2, s = str(np.round(np.linalg.norm(v1-v2),4)) ,size=10, zorder=1, color='k')
    ax.scatter(xs,ys,zs=zs,color ='aqua')

def show_track_3d(landmarks,title = 'Track 3D', store = True):
    fig = plt.figure(figsize=(10,5), layout='constrained', num=title)
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    text = ax1.text(1.0, -2.0,0.0 , "frame : ")
    def animate(i):
        # ax.clear()
        text.set_text("frame : "+str(i))
        if i == 0:
            draw_track_3d(landmarks[i], ax1, elev=-90, azim=90, title='front view')
            draw_track_3d(landmarks[i], ax2, elev=0, azim=90, title='above view')
        else:
            draw_track_3d(landmarks[i],ax1,landmarks[i-1], elev=-90, azim=90, title='front view')
            draw_track_3d(landmarks[i],ax2,landmarks[i-1], elev=0, azim=90, title='above view')
        if i == len(landmarks) - 1:  # If this is the last frame
            plt.close()
    if store == False:
        ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=0.25,repeat = True)
        plt.show()
    else:
        ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=1,repeat = False)
        plt.show(block = False)
        ani.save(filename="results/classification_test/" + title + ".gif",writer=writergif)

def show_track_3d_result(landmarks,title = 'Track 3D Result'):
    fig = plt.figure(figsize=(10,5), layout='constrained', num=title)
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    def animate(i):
        
        if i == 0:
            
            draw_track_3d(landmarks[i], ax1, elev=-90, azim=90, title='front view')
            draw_track_3d(landmarks[i], ax2, elev=0, azim=90, title='above view')
        else:
            draw_track_3d(landmarks[i],ax1,landmarks[i-1], elev=-90, azim=90, title='front view')
            draw_track_3d(landmarks[i],ax2,landmarks[i-1], elev=0, azim=90, title='above view')
        
    for i in range(len(landmarks)):
        animate(i)
    # ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=0.1,repeat = False)
    plt.show(block = False)
    plt.pause(2)
    plt.close('all')

def cmp_track_landmarks(landmarks1,landmarks2,title1,title2,name):
    fig = plt.figure(figsize=(10,5),layout='constrained')
    ax1 = plt.subplot(121,projection='3d')
    ax2 = plt.subplot(122,projection='3d')
    def animate(i):
        if i < len(landmarks1):
            ax1.clear()
            draw_track_3d(landmarks1[i], ax1, title1)
        if i < len(landmarks2):
            ax2.clear()
            draw_track_3d(landmarks2[i], ax2, title2)
    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks1), interval=25,repeat = False)
    plt.show()
    # ani.save(filename='output/cmp_kmean_'+name+'.gif',writer=writergif)
# if >0.8 then the hand is behind camera not visible
def draw_track_3d(landmark, ax, previous = None, title = None, elev = -90, azim = 90):
    ax.view_init(elev = int(elev) , azim = int(azim)) 
    if title:
        ax.set_title(title)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.invert_xaxis()
    xs = [l[0] for l in landmark]
    ys = [l[1] for l in landmark]
    zs = [l[2] for l in landmark]
    if previous is not None: 
        pxs = [l[0] for l in previous]
        pys = [l[1] for l in previous]
        pzs = [l[2] for l in previous]
        for i in range(len(xs)):
            ax.plot([pxs[i],xs[i]],[pys[i],ys[i]],zs =[pzs[i],zs[i]],c = 'violet')
    ax.scatter(xs,ys,zs=zs,color ='aqua', s=5, alpha=0.5)
    # ax.scatter(nxs,nys,zs=nzs,color ='lightcoral', s=10, alpha=0.5)

def show_track_2d(landmarks,title = 'Track 2D'):
    fig = plt.figure(figsize=(10,5), layout='constrained', num=title)
    ax1 = plt.subplot()
    def animate(i):
        # ax.clear()
        if i == 0:
            draw_track_2d(landmarks[i], ax1, title='front view')
        else:
            draw_track_2d(landmarks[i],ax1,landmarks[i-1], title='front view')
        if i == len(landmarks) - 1:  # If this is the last frame
            plt.close()
    ani = animation.FuncAnimation(fig, animate, frames=len(landmarks), interval=1,repeat = False)
    plt.show(block = False)
    ani.save(filename="results/2d_" + title + ".gif",writer=writergif)

def draw_track_2d(landmark, ax, previous = None, title = None):
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.invert_yaxis()
    xs = [l[0] for l in landmark]
    ys = [l[1] for l in landmark]
    if previous != None: 
        pxs = [l[0] for l in previous]
        pys = [l[1] for l in previous]
        for i in range(len(xs)):
            ax.plot([pxs[i],xs[i]],[pys[i],ys[i]],c = 'violet')
    ax.scatter(xs,ys,color ='aqua', s=10, alpha=0.5)

def draw_on_frame(frames,landmarks):
    for frame in frames:
        for j in landmarks:
            x,y = int(j[0]*frame.shape[1]),int(j[1]*frame.shape[0])
            cv2.circle(frame['frame'], (x,y), 5, (0, 0, 255), -1)
        cv2.imshow('frame', frame['frame'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break