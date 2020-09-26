import os
import pickle
import numpy as np
from scipy.io import loadmat, savemat
import sys
from itertools import permutations
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
import cv2
import pickle as pkl

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d

use_old_vd = True
use_3d = False
tableau10_extended = list(mcolors.TABLEAU_COLORS) + ['k', 'b', 'lime', 'cyan', 'yellow', 'brown', 'plum', 'navy']
tableau10 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
            (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
            (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
            (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
            (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)][::2]

if use_old_vd:
    sys.path.insert(1, '/media/sda/fangyu/Projects/single-view-reconstruction/')
    from vehicle_dataset import *
    from utils import generate_spline_points_world, homo, cart, find_rt_c2w
    dataset_names_list = ['IMG_0121', 'jackson_hole']
else:
    sys.path.insert(1, '/media/sda/fangyu/Projects/traffic4d/')
    from dataset.vehicle import *
    from utils.utils import generate_spline_points_world, homo 
    from config.inputs import dataset_names_list, result_dir

def eval_mat(use_old_vd, use_3d, dataset_idx):
    if use_old_vd:
        dataset_name = dataset_names_list[dataset_idx]
        if use_3d:
            result_name = dataset_name + "3DResults.mat"
        else:
            result_name = dataset_name + "2DResults.mat"
    else:
        result_dir = './'
        dataset_name = dataset_names_list[dataset_idx]
        if use_3d:
            result_name = result_dir + '/' + dataset_name + '3DResults.mat'
        else:
            result_name = result_dir + '/' + dataset_name + '2DResults.mat'
    mat_data = loadmat(result_name)

    # show gt
    result_name = dataset_name + "Data.mat"
    mat_data_before = loadmat(result_name)
    print(mat_data_before.keys())
    method_name = 'gt'
    traj_name = 'DataSplines' if use_3d else 'DataBboxes'
    spline_points_before = np.array([np.array(d['data']).flatten() for d in mat_data_before[traj_name][0]])
    cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data_before[traj_name][0]])-2
    #draw_traj_proj_all(spline_points_before, cluster_ids_gt, method_name, dataset_idx, use_3d, thick=False)
    #exit(0)


    # evaluate and show each method
    for (traj_name, method_name) in zip(['Traj1', 'Traj2', 'Traj3', 'Traj4'], ['MS', 'MBMS', 'AMKS noreg', 'AMKS']):
        spline_points = np.array([np.array(d['data']).flatten() for d in mat_data[traj_name][0]])
        cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data[traj_name][0]])-2
        n_clusters = max(cluster_ids_gt)+1
        # gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='diag').fit(spline_points)
        # cluster_ids = gmm.predict(spline_points)
        splines,spline_points = np_spline_fit(spline_points)
        kmeans =  SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
         random_state=0).fit(splines)#spline_points.reshape((len(spline_points),-1)))
        cluster_ids = kmeans.labels_
        # gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='diag').fit(splines)
        # cluster_ids = gmm.predict(splines)
        # print(splines.astype(int), cluster_ids_gt, cluster_ids)

        ### below four lines used for evaluation
        _, cluster_ids, cluster_ids_gt = filter_out_outlier_clusters(None, cluster_ids, cluster_ids_gt)
        print(str("3D results" if use_3d else "2D results") + ' ' + method_name)
        eval_cluster_accuracy(cluster_ids, cluster_ids_gt)
        draw_traj_all_id_spline_with_cluster(spline_points, cluster_ids)
        #draw_traj_proj_all(spline_points, cluster_ids, method_name, dataset_idx, use_3d)
        #draw_traj_proj_all(spline_points_before, cluster_ids, method_name+'before_', dataset_idx, use_3d, thick=False)


def eval_cluster_accuracy(cluster_ids0, cluster_ids1):
    """
    For two cluster results, find their best match and computer the accuracy
    """
    assert min(cluster_ids0)>=0 and min(cluster_ids1)>=0
    n_clusters0 = max(cluster_ids0)+1
    n_clusters1 = max(cluster_ids1)+1
    # always make sure cluster0 has fewer cluster indices than cluster1
    if n_clusters0 > n_clusters1:
        cluster_ids0, cluster_ids1 = cluster_ids1, cluster_ids0
        n_clusters0, n_clusters1 = n_clusters1, n_clusters0

    n_total = len(cluster_ids1)
    n_correct_best = 0
    n_correct_cluster_best = []
    n_correct_perm_best = []

    # permutes cluster1 (the longer one)
    all_permus = list(permutations(range(n_clusters1)))
    for permu in all_permus:
        cluster_ids1_permu = [permu[i] for i in cluster_ids1]
        n_correct = np.sum(np.array(cluster_ids0) == np.array(cluster_ids1_permu))
        if n_correct > n_correct_best:
            n_correct_best = n_correct
            n_correct_perm_best = cluster_ids1_permu
            # n_correct_cluster_best = [np.sum(
            #     np.logical_and(
            #     np.array(index_global_permu)==np.array(index_global_gt),
            #     np.array(index_global_gt)==cluster)
            #     ) for cluster in range(n_clusters)]
    print("best cluster matches: %d/%d - %.4f"%(n_correct_best, n_total, n_correct_best/n_total))


def filter_out_outlier_clusters(spline_points, cluster_ids, cluster_ids_gt):
    cluster_ids = np.array(cluster_ids)
    cluster_ids_gt = np.array(cluster_ids_gt).flatten()
    index = np.where(cluster_ids_gt>=0)[0]
    return spline_points[index] if spline_points is not None else None, cluster_ids[index], cluster_ids_gt[index]


def np_spline_fit(spline_points):
    spline_points = spline_points.reshape((spline_points.shape[0], spline_points.shape[1]//2,2))
    splines = []
    for n in range(len(spline_points)):
        x = spline_points[n,:,0]
        y = spline_points[n,:,1]
        t = np.arange(len(spline_points[n]))
        spline = np.concatenate([np.polyfit(t, x, 3), np.polyfit(t, y, 3)])
        splines.append(spline)
    # print(np.array(splines).shape)
    return np.array(splines), spline_points



def draw_traj_all_id_spline_with_cluster(spline_points, cluster_ids):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for spline, cluster_id in zip(spline_points, cluster_ids):
        ax.plot(spline[:, 0], spline[:, 1], np.zeros_like(spline[:, 1]), color=list(tableau10_extended)[cluster_id])


    #ax.set_xlim3d(-25, 25)
    #ax.set_ylim3d(-10, 10)
    #ax.set_zlim3d(0, 50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # axisEqual3D(ax)
    ax.view_init(elev=-60, azim=-90)
    # plt.savefig(output_img_name)
    plt.show()
    plt.close()
    return


def find_landmark_cam(plane, landmark_perspective):
    height, width = 1080, 1920
    focal = 2100.0
    cx = width/2.0
    cy = height/2.0
    landmark_cam = []
    for i in range(len(landmark_perspective)):
        u, v = landmark_perspective[i]
        a = np.array([[focal, 0.0, cx-u],[0.0, focal, cy-v],[plane[0], plane[1], plane[2]]])
        b = np.array([0.0, 0.0, -plane[3]])
        result = np.linalg.solve(a, b)
        landmark_cam.append(result)
    landmark_cam = np.array(landmark_cam)
    return landmark_cam


def find_affine_w2s():
    with open("./results/IMG_0121_with_spline_clusters.vd", 'rb') as f:
         vd = pkl.load(f)
    plane = vd.plane
    rt_c2w = find_rt_c2w(vd)

    landmark_sate_path = "../visual_opengl/IMG_0121/bird_view.txt"
    with open(landmark_sate_path, 'r') as f:
        lines = f.readlines()
    landmark_sate = []
    for line in lines:
        line_split = line.split(",")
        landmark_2d = [float(line_split[0]), float(line_split[1])]
        landmark_sate.append(landmark_2d)
    landmark_sate = np.array(landmark_sate)

    # landmark in cam coordinate
    landmark_perspective_path = "../visual_opengl/IMG_0121/cam_view.pklcor"
    with open(landmark_perspective_path, 'rb') as f:
        landmark_perspective = pkl.load(f)
    landmark_cam = find_landmark_cam(plane, landmark_perspective)
    landmark_world = cart((rt_c2w @ ((homo(landmark_cam)).T)).T)

    affine_w2s, _ = cv2.estimateAffine2D(landmark_world[:,0:2], landmark_sate)
    print(affine_w2s)
    return affine_w2s


def draw_traj_proj_all(spline_points, cluster_ids, method_name, dataset_idx, use_3d=False, thick=True):
    dataset_name = dataset_names_list[dataset_idx]
    cluster_id_color_mapping = [6,5,1,0,4,3]
    if use_3d:
        img = cv2.imread(dataset_name+'_bird_view.jpg')
        affine_w2s = find_affine_w2s()
    else:
        img = cv2.imread(dataset_name+'.jpg')
        affine_w2s = None
    for spline, cluster_id in zip(spline_points, cluster_ids):
        if len(spline.shape) != 2:
            spline = spline.reshape((-1,2))
        if use_3d:
            spline_ = homo(spline)
            spline = (affine_w2s @ (spline_.T)).T
        if cluster_id < 0:
            continue
        for i in range(len(spline)-1):
            img = cv2.line(img, tuple(spline[i,0:2].astype(np.int32)), tuple(spline[i+1,0:2].astype(np.int32)), tableau10[cluster_id_color_mapping[cluster_id]][::-1], 8 if thick else 2)
    out_img_name = dataset_name+'_'+method_name+'_bird_view.jpg' if use_3d else dataset_name+'_'+method_name+'_proj.jpg'
    cv2.imwrite(out_img_name, img)
    print(out_img_name + ' saved')


np.set_printoptions(threshold=sys.maxsize, suppress=True)
if len(sys.argv) == 1:
    print("python post_process [use_3d] [dataset_idx in traffic4d]")
    exit(0)
eval_mat(use_old_vd, use_3d=bool(int(sys.argv[1])), dataset_idx=0 if use_old_vd else int(sys.argv[2]))
