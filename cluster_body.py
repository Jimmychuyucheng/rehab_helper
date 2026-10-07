from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
import time
import numpy as np 
import os
import pickle
import argparse


from data.dataset import WLASL_cluster_data as Dataset
from data.dataset import get_topk_id, create_gloss_dir
from data.visualization import show_track_3d, show_track_2d, show_body_3d, show_track_3d_result,show_landmark_3d
import matplotlib.pyplot as plt
from data.process_backbone import video2landmark, get_holistic_landmark


data_info_json = r"C:\Users\Alex\Desktop\CISLR\WLASL_v0.3.json"
hand_dir_path =  r"C:\Users\Alex\Desktop\CISLR\landmarks\wlasl_rtmw_1"
body_dir_path = r"C:\Users\Alex\Desktop\CISLR\landmarks\wlasl_mediapipe_holistic"
save_path = 'C:\\Users\\Alex\\Desktop\\CISLR\\landmarks'


num_clusters = 10 # 50 

# test using 2d input data
dataset = Dataset(data_info_json, body_dir_path,'train')
twod_dataset = Dataset(data_info_json, hand_dir_path,'train')
val_dataset = Dataset(data_info_json, body_dir_path,'val')
test_dataset = Dataset(data_info_json, body_dir_path,'test')
scaler = StandardScaler()
print('Train {}, Val {}, Test {}'.format(len(dataset.data_info),len(val_dataset.data_info),len(test_dataset.data_info)))
print(len(dataset.data_info)+len(val_dataset.data_info)+len(test_dataset.data_info))

def dataset2pickle(ids = None):
    ret_dict = dataset.get_video_data(ids)
    print('data shape ',ret_dict['data'].shape)
    for k,v in ret_dict.items():
        with open(os.path.join(save_path,k+'.pickle'),'wb') as f:
            pickle.dump(v,f)

def load_data(name):
    with(open(os.path.join(save_path,name+'.pickle'),'rb')) as f:
        data = pickle.load(f)
    return data


def combine_data(data, slope):
    flat_data = data.reshape(-1,120*2*3)
    flat_slop = slope.reshape(-1,22*2)
    return np.concatenate((flat_data, flat_slop), axis=1)

def detach_data(combine):
    return combine[:,:-44].reshape(-1,2,3), combine[:,-44:]

def find_elbow(init, max, step, data):
    tmp_i = []
    tmp = []
    for i in range(init,max,step):
        # data = scaler.fit_transform(data)
        kmeans  = KMedoids(n_clusters= i,max_iter= 100,
                            random_state= 30).fit(data)
        tmp_i.append(i)
        tmp.append(int(kmeans.inertia_))
    plt.plot(tmp_i,tmp)
    plt.show()

def show_2d_result(id, kmeans):
    for d in id:
        data = dataset.get_single_video_data(d)
        ret = kmeans.predict(data.reshape(-1, 90*2*2))
        print('Output ', d,' ',ret)
        ret = kmeans.cluster_centers_[ret]
        show_track_2d(data.reshape(-1,2,2).tolist(),'Original'+str(d))
        show_track_2d(ret.reshape(-1,2,2).tolist(), 'Cluster'+str(d))


def dim_reduction(center,dim = 2):
    pca = PCA(n_components=dim).fit(center)
    center = pca.transform(center)
    return center

def cluster_data(data,norm=True, strategy='kmeans' ):
    print('input data shape ',data.shape)
    print('Using {} as cluster strategy and num_clusters is {}.'.format(strategy,num_clusters))
    if norm:
        data = scaler.fit_transform(data)
    if strategy == 'kmeans':
        kmeans = KMeans(n_clusters= num_clusters,
                            random_state= 33,
                            n_init= 'auto').fit(data)
        center = kmeans.cluster_centers_
        print(kmeans.inertia_)
        return kmeans, center
    elif strategy == 'dbscan':
        dbscan = DBSCAN(eps=0.05, min_samples=4).fit(data)
        label = dbscan.labels_
        return dbscan, label
    elif strategy ==  'kmedoids':
        kmedoids = KMedoids(n_clusters=num_clusters,
                            random_state=33).fit(data)
        center = kmedoids.cluster_centers_
        print(kmedoids.inertia_)
        return kmedoids, center
    elif strategy == 'kmedoids_dtw':
        start = time.time()
        km = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw",n_jobs= 8,
                               max_iter=50, random_state=33,max_iter_barycenter=50, verbose= True).fit(data)
        end = time.time()
        print('Time: ', end-start)
        center = km.cluster_centers_
        print(km.inertia_)
        return km, center
    elif strategy == 'cuda_kmedoids_soft_dtw':
        km = KMedoids(n_clusters=num_clusters,
                            random_state=33,
                            metric= 'precomputed')
        loss = cdist_dtw(data, be= 'pytorch', verbose= 1)
        km.fit(loss)
        center = km.cluster_centers_
        print(km.inertia_)
        return kmedoids, center
    elif strategy == 'optics':
        optics = OPTICS(min_samples=num_clusters,n_jobs = 4).fit(data)
        label = optics.labels_
        return optics, label
    elif strategy == 'GMM':
        gmm = GMM(n_components=num_clusters, covariance_type='full', random_state=33).fit(data)
        return gmm
    else:
        raise ValueError('No such strategy')
    
def draw_center(center):
    for i,c in enumerate(center):
        # landmark = scaler.inverse_transform(c).reshape(-1,2,3).tolist()
        # landmark = scaler.inverse_transform(c)[:,:-22].reshape(-1,2,3).tolist()
        # if i in [7,10,28,30]:
        # show_track_3d(landmark, title= 'Cluster '+str(i))
        show_track_3d(c.reshape(-1,2).tolist(), title= 'Cluster '+str(i))

def predict_video(kmeans, center, data, ids):
    result = []
    result_dict = {}
    for id in ids:
        index = id_list.index(id)
        d = data[index]
        # combine = combine_data(data, slope)
        output = kmeans.predict(d.reshape(1,-1))    
        # print(id)
        # print('Output ', output)
        result.append(output[0])
        # output_landmark = scaler.inverse_transform(center[output])
        # output_data, output_slope = detach_data(output_landmark)
        # show_track_3d_result(data.reshape(-1,2,3).tolist(), 'Original'+str(id))
        if output[0] in result_dict:
            result_dict[output[0]].append(id)
        else:
            result_dict[output[0]] = [id]
        # show_track_3d(data.reshape(-1,2,3).tolist(), 'Original'+str(id)) 
        # show_track_3d(output_landmark.reshape(-1,2,3).tolist(), 'Cluster'+str(id))   
        

    unique, counts = np.unique(result, return_counts=True)
    print(dict(zip(unique, counts)))
    print(result_dict)
    for v in unique:
        landmark = center[v].reshape(1, -1)
        # landmark = scaler.inverse_transform(center[v].reshape(1, -1))
        show_track_3d(landmark.reshape(-1,2,3).tolist(), title= 'Cluster '+str(v),store= True)
    return max(counts)

def validate_result(cluster, center):
    label_list = dataset.label_dict.keys()
    pos_num = 0
    for label in label_list:
        ids = [ d['video_id'] for d in dataset.data_info if d['gloss'] == label]
        pos_num+=predict_video(cluster, center, ids)
    print('Accuracy: ', pos_num/len(dataset.data_info))
    return pos_num/len(dataset.data_info)

def show_result(output, id_list, ids):
    tmp = {}
    ret_id = []
    for id in ids:
        index = id_list.index(id)
        if output[index] not in tmp :
            tmp[output[index]] = [id]
            ret_id.append(id)
        else:
            tmp[output[index]].append(id)
    print(tmp)
    return ret_id
"""
    02611 & 02620 same person
    02607 & 02612 same gloss
    49643 & 49645 & 67172 scientist
    50331 & 50332 send
    50551 & 50552 & 50553 service
    03328 & 69215 arm
    53363 & 53366 sorry
    53363 & 05963 
    01561 depth is a piece of shit
    38997 under elbow
    02105 weird movemen
    
"""

## now input data is only data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type = str, default='kmeans', help = 'cluster strategy')
    parser.add_argument('--padding_strategy', type= str, default='zero', help = 'data padding strategy')
    args = parser.parse_args()

    # landamrk = get_holistic_landmark(r"07967",r'C:\Users\Alex\Desktop\Sign_dataset\WLASL2000\video')
    # body = [ l['landmark'] for l in landamrk['body']]
    # visbility = [ l['visibility']for l in landamrk['body']]
    # body = [ b for b,v in zip(body,visbility) ]
    # hand = landamrk['hand']
    # show_body_3d(body, visbility=visbility)


    brown = [ d['video_id'] for d in dataset.data_info if d['gloss'] == 'brown' ]
    woman = [ d['video_id'] for d in dataset.data_info if d['gloss'] == 'woman' ]
    now = [ d['video_id'] for d in dataset.data_info if d['gloss'] == 'now' ]
    meet = [ d['video_id'] for d in dataset.data_info if d['gloss'] == 'meet' ]
    tall = [ d['video_id'] for d in dataset.data_info if d['gloss'] == 'tall' ]
    sorry = [d['video_id'] for d in dataset.data_info if d['gloss'] == 'sorry' ]
    test_data = ['53370']


    ids = get_topk_id(dataset.data_info,dataset.label_dict,dataset.label_num,100)

    # seg_data = dataset.get_seg_data()

    # dataset2pickle()
    data = load_data('data')
    id_list = load_data('id')
    direct = load_data('direct')
    length = load_data('length')
    angle = load_data('angle')

    # print('Direct shape ',direct.shape)
    # seg_direct_data = dataset.get_seg_data()
    # seg_direct = seg_direct_data['direct']
    # print('Seg direct shape ',seg_direct.shape)
    # seg_cluster, seg_center = cluster_data(seg_direct.reshape(seg_direct.shape[0],-1),norm=False,strategy=args.strategy)
    # for i,center in enumerate(seg_center):
    #     center = center.reshape(-1,2)
    #     plt.plot(np.arange(center.shape[0]),center)
    #     plt.title('Seg Direct '+str(i))
    #     plt.show()
    # for b in brown:
    #     seg_b = dataset.get_seg_data(b)
    #     seg_b = seg_b['direct']
    #     seg_result =seg_cluster.predict(seg_b.reshape(seg_b.shape[0],-1))
    #     print('Seg {} result : '.format(b),seg_result)

    for id in sorry:
        index = id_list.index(id)
        print(direct[index].shape)
        print('=='*100)
        plt.plot(np.arange(direct[index].shape[0]),direct[index])
        plt.title('Direct '+str(id))
        plt.show()
        show_track_3d(data[index].reshape(-1,2,3).tolist(),str(id),store= False)

    twod_data = data[:,:,:,:2] # 2d data
    # data = data.reshape(len(data),3,-1,2)
    point_data = data.reshape(-1,3)
    ## testig split direct cluster
    print(direct.shape)
    cluster, center = cluster_data(direct.reshape(direct.shape[0],-1),norm=False,strategy=args.strategy)
    
    output = cluster.predict(direct.reshape(direct.shape[0],-1))

    print(np.unique(output,return_counts=True))

    # for c in center:
    #     c = c.reshape(-1,2)
    #     plt.plot(np.arange(c.shape[0]),c[:,0])
    #     plt.plot(np.arange(c.shape[0]),c[:,1])
    #     plt.show()

    print('Brown')
    show_result(output, id_list, brown)
    print('Woman')
    show_result(output, id_list, woman)
    print('Sorry')
    show_result(output, id_list, sorry)