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
from plot import draw_traj_all_id_spline_with_cluster, draw_traj_proj_all, tableau10, tableau10_extended

"""
if use_old_vd:
    sys.path.insert(1, '/media/sda/fangyu/Projects/single-view-reconstruction/')
    from vehicle_dataset import *
    from utils import generate_spline_points_world, homo, cart, find_rt_c2w
    dataset_names_list = ['IMG_0121', 'jackson_hole']
else:
    sys.path.insert(1, '/media/sda/fangyu/Projects/traffic4d/')
    from dataset.vehicle import *
    from utils.utils import generate_spline_points_world, homo, cart, find_rt_c2w
    from config.inputs import dataset_names_list, result_dir
"""

from utils_eval import *


def find_best_affine(use_old_vd, dataset_idx):
    use_3d = True
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

    # show gt
    result_name = dataset_name + "Data.mat"
    mat_data_before = loadmat(result_name)
    print(mat_data_before.keys())
    method_name = 'gt'
    traj_name = 'DataSplines' if use_3d else 'DataBboxes'
    spline_points_before = np.array([np.array(d['data']).flatten() for d in mat_data_before[traj_name][0]])
    cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data_before[traj_name][0]])-2

    n_random = 100
    line = ""
    # generate random offset
    result_dir = 'random_affine/'+dataset_name
    for ni in range(n_random):
        affine_w2s = find_affine_w2s_random(dataset_name)
        print(ni, affine_w2s)
        line += (str(ni) + '\n' + np.array_str(affine_w2s) + '\n')
        draw_traj_proj_all(spline_points_before, cluster_ids_gt, str(ni), dataset_name, use_3d=True, thick=False, affine_w2s=affine_w2s, result_dir='random_affine/'+dataset_name)
    with open(result_dir + '/records.txt', 'w') as f:
        f.write(line)


def draw_vd(use_old_vd, dataset_idx):
    if use_old_vd:
        dataset_name = dataset_names_list[dataset_idx]
    else:
        result_dir = './'
        dataset_name = dataset_names_list[dataset_idx]

    # show gt
    result_name = dataset_name + "vd.mat"
    mat_data_before = loadmat(result_name)
    print(mat_data_before.keys())
    affine_w2s = find_affine_w2s(dataset_name)

    use_3d = True
    method_name = 'vd_3d' if use_3d else 'vd_2d'
    traj_name = 'DataSplines' if use_3d else 'DataBboxes'
    spline_points_before = np.array([np.array(d['data']).flatten() for d in mat_data_before[traj_name][0]])
    cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data_before[traj_name][0]])-2
    draw_traj_proj_all(spline_points_before, cluster_ids_gt, method_name, dataset_name, use_3d, thick=False, affine_w2s=affine_w2s, result_dir='visual_clusters_goin_paper')

    ### temp for first level cluster
    # method_name = 'vd_3d_first' # temp for first level cluster
    # draw_traj_proj_all(spline_points_before, cluster_ids_gt, method_name, dataset_name, use_3d, thick=False, affine_w2s=affine_w2s, result_dir='visual_clusters_goin_paper')
    # exit(0)

    use_3d = False
    method_name = 'vd_3d' if use_3d else 'vd_2d'
    traj_name = 'DataSplines' if use_3d else 'DataBboxes'
    spline_points_before = np.array([np.array(d['data']).flatten() for d in mat_data_before[traj_name][0]])
    draw_traj_proj_all(spline_points_before, cluster_ids_gt, method_name, dataset_name, use_3d, thick=False, result_dir='visual_clusters_goin_paper')
    draw_traj_all_id_spline_with_cluster(spline_points_before, cluster_ids_gt)


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
    spline_points_before_3d = np.array([np.array(d['data']).flatten() for d in mat_data_before['DataSplines'][0]])
    spline_points_before_2d = np.array([np.array(d['data']).flatten() for d in mat_data_before['DataBboxes'][0]])
    cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data_before[traj_name][0]])-2
    #draw_traj_all_id_spline_with_cluster(spline_points_before, cluster_ids_gt)
    #draw_traj_proj_all(spline_points_before, cluster_ids_gt, method_name, dataset_idx, use_3d, thick=False)
    #exit(0)

    affine_w2s = find_affine_w2s(dataset_name)

    # evaluate and show each method
    for (traj_name, method_name) in zip(['Traj1', 'Traj2', 'Traj3', 'Traj4'], ['MS', 'MBMS', 'AMKS noreg', 'AMKS']):
        spline_points = np.array([np.array(d['data']).flatten() for d in mat_data[traj_name][0]])
        cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data[traj_name][0]])-2
        cluster_ids_set = set()
        for cluster_id in cluster_ids_gt:
            cluster_ids_set.add(cluster_id)
        # n_clusters = max(cluster_ids_gt)+1
        cluster_ids_set = list(cluster_ids_set)
        n_clusters = len(cluster_ids_set)-1
        for i in range(len(cluster_ids_gt)):
            if cluster_ids_gt[i] >= 0:
                cluster_ids_gt[i] = cluster_ids_set.index(cluster_ids_gt[i])
        # gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='diag').fit(spline_points)
        # cluster_ids = gmm.predict(spline_points)
        spline_points, spline_points_before_2d, spline_points_before_3d, cluster_ids_gt = filter_out_outlier_clusters([spline_points, spline_points_before_2d, spline_points_before_3d, cluster_ids_gt])
        #print(cluster_ids_gt)
        splines,spline_points = np_spline_fit(spline_points)
        kmeans =  SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
         random_state=0).fit(splines)#spline_points.reshape((len(spline_points),-1)))
        # kmeans =  SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
        #  random_state=0).fit(spline_points.reshape((len(spline_points),-1)))
        cluster_ids = kmeans.labels_
        # gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='diag').fit(splines)
        # cluster_ids = gmm.predict(splines)
        # print(splines.astype(int), cluster_ids_gt, cluster_ids)

        ### below two lines used for evaluation
        print(str("3D results" if use_3d else "2D results") + ' ' + method_name)
        #eval_cluster_accuracy(cluster_ids, cluster_ids_gt)
        #draw_traj_all_id_spline_with_cluster(spline_points_before_2d, cluster_ids)
        draw_traj_proj_all(spline_points_before_3d, cluster_ids, method_name+'_3d_before_use3d_'+str(use_3d), dataset_name, use_3d=True, thick=False, affine_w2s=affine_w2s, result_dir='visual_clusters_goin_paper')
        draw_traj_proj_all(spline_points_before_2d, cluster_ids, method_name+'_2d_before_use3d_'+str(use_3d), dataset_name, use_3d=False, thick=False, affine_w2s=affine_w2s, result_dir='visual_clusters_goin_paper')


def eval_cluster_accuracy(cluster_ids0, cluster_ids1):
    """
    For two cluster results, find their best match and computer the accuracy
    """
    assert min(cluster_ids0)>=0 and min(cluster_ids1)>=0
    n_clusters0 = max(cluster_ids0)+1
    n_clusters1 = max(cluster_ids1)+1
    print(n_clusters1)
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


def filter_out_outlier_clusters(data_list):
    # assume data_list the last is cluster_ids_gt
    cluster_ids_gt = np.array(data_list[-1]).flatten()
    index = np.where(cluster_ids_gt>=0)[0]
    data_list_filtered = []
    for data in data_list:
        data = np.array(data)
        data_list_filtered.append(data[index])
    return data_list_filtered

    # return spline_points[index] if spline_points is not None else None, cluster_ids[index], cluster_ids_gt[index]


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


np.set_printoptions(threshold=sys.maxsize, suppress=True)
if len(sys.argv) < 3:
    print("python post_process [use_3d] [dataset_idx in traffic4d]")
    print("python post_process best-affine [dataset_idx in traffic4d]")
    print("python post_process draw-vd [dataset_idx in traffic4d]")
    exit(0)
if sys.argv[1] == 'best-affine':
    find_best_affine(use_old_vd, dataset_idx=0 if use_old_vd else int(sys.argv[2]))
elif sys.argv[1] == 'draw-vd':
    draw_vd(use_old_vd, dataset_idx=0 if use_old_vd else int(sys.argv[2]))
else:
    eval_mat(use_old_vd, use_3d=bool(int(sys.argv[1])), dataset_idx=0 if use_old_vd else int(sys.argv[2]))
