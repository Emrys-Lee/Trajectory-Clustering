import os
import pickle
import numpy as np
from scipy.io import loadmat, savemat
import sys
from itertools import permutations
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d

use_old_vd = False
use_3d = False
tableau10_extended = list(mcolors.TABLEAU_COLORS) + ['k', 'b', 'lime', 'cyan', 'yellow', 'brown', 'plum', 'navy']

if use_old_vd:
    sys.path.insert(1, '/media/sda/fangyu/Projects/single-view-reconstruction/')
    from vehicle_dataset import *
    from utils import generate_spline_points_world
else:
    sys.path.insert(1, '/media/sda/fangyu/Projects/traffic4d/')
    from dataset.vehicle import *
    from utils.utils import generate_spline_points_world
    from config.inputs import dataset_names_list, result_dir

def eval_mat(use_old_vd, use_3d, dataset_idx):
    if use_old_vd:
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
    print(mat_data.keys())

    for (traj_name, method_name) in zip(['Traj1', 'Traj2', 'Traj3', 'Traj4'], ['MS', 'MBMS', 'AMKS noreg', 'AMKS']):
        spline_points = np.array([np.array(d['data']).flatten() for d in mat_data[traj_name][0]])
        cluster_ids_gt = np.array([int(d['label'][0][0]) for d in mat_data[traj_name][0]])-2
        n_clusters = max(cluster_ids_gt)+1
        # gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='diag').fit(spline_points)
        # cluster_ids = gmm.predict(spline_points)
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(spline_points)
        splines,spline_points = np_spline_fit(spline_points)
        kmeans =  SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
         random_state=0).fit(splines)
        cluster_ids = kmeans.labels_
        # gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='diag').fit(splines)
        # cluster_ids = gmm.predict(splines)
        # print(splines.astype(int), cluster_ids_gt, cluster_ids)
        cluster_ids, cluster_ids_gt = filter_out_outlier_clusters(cluster_ids, cluster_ids_gt)
        print(str("3D results" if use_3d else "2D results") + ' ' + method_name)
        eval_cluster_accuracy(cluster_ids, cluster_ids_gt)
        draw_traj_all_id_spline_with_cluster(spline_points, cluster_ids)


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


def filter_out_outlier_clusters(cluster_ids, cluster_ids_gt):
    cluster_ids = np.array(cluster_ids)
    cluster_ids_gt = np.array(cluster_ids_gt).flatten()
    index = np.where(cluster_ids_gt>=0)[0]
    return cluster_ids[index], cluster_ids_gt[index]


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



np.set_printoptions(threshold=sys.maxsize)
eval_mat(use_old_vd, use_3d=bool(int(sys.argv[1])), dataset_idx=0 if use_old_vd else int(sys.argv[2]))
