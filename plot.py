import os
import pickle
import numpy as np
from scipy.io import loadmat, savemat
import sys
import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d

tableau10_extended = list(mcolors.TABLEAU_COLORS) + ['k', 'b', 'lime', 'cyan', 'yellow', 'brown', 'plum', 'navy']

tableau10 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
            (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
            (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
            (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
            (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

selected_out_index_dict = {
        '00002_00005':[117,120,58,62,42],
        '00005_00002':[92,105], #[35,89]#[108, 18]# for vd
        'aic-test-S02-c007':[42,45]#[35,89]#[108, 18]# for vd
        }

def draw_traj_all_id_spline_with_cluster(spline_points, cluster_ids, veh_ids=None):
    """plot in matplotlib"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # if veh_ids is none, then these from local data, need to reindex cluster ids, otherwise come from vd, keep the original
    if veh_ids is None:
        spline_points_temp = []
        cluster_ids_temp = []
        for i in range(max(cluster_ids)+1):
            index = np.where(cluster_ids==i)[0]
            spline_points_temp.append(spline_points[index])
            cluster_ids_temp.append(cluster_ids[index])
        spline_points = np.concatenate(spline_points_temp, axis=0)
        cluster_ids = np.concatenate(cluster_ids_temp, axis=0)

    for i, (spline, cluster_id) in enumerate(zip(spline_points, cluster_ids)):
        if len(spline.shape) != 2:
            spline = spline.reshape((-1,2))
        if cluster_id < 0:
            continue
        ax.plot(spline[:, 0], spline[:, 1], np.zeros_like(spline[:, 1]), color=list(tableau10_extended)[cluster_id])
        if veh_ids is not None:
            ax.text(spline[1, 0]+np.random.rand(),spline[1, 1]+np.random.rand(), 0, str(veh_ids[i]), color=list(tableau10_extended)[cluster_id])
            ax.text(spline[-1, 0]+np.random.rand(),spline[-1, 1]+np.random.rand(), 0, str(veh_ids[i]), color=list(tableau10_extended)[cluster_id])
        else:
            ax.text(spline[1, 0]+np.random.rand(),spline[1, 1]+np.random.rand(), 0, str(i), color=list(tableau10_extended)[cluster_id])
            ax.text(spline[-1, 0]+np.random.rand(),spline[-1, 1]+np.random.rand(), 0, str(i), color=list(tableau10_extended)[cluster_id])


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


def draw_traj_proj_all(spline_points, cluster_ids, method_name, dataset_name, use_3d=False, thick=True, affine_w2s=None, result_dir = 'visual_clusters/'):
    """trajectory projection, support both cam and sate view"""
    if dataset_name=='IMG_0121':
        cluster_id_color_mapping = [6,5,1,0,4,3]
    elif dataset_name=='00005_00002':
        cluster_id_color_mapping = [6,0,5,3,1,2,4]
    elif dataset_name=='00002_00005':
        # cluster_id_color_mapping = [4,6,8,2] #use_3d=True
        cluster_id_color_mapping = [9,2,4,6]#use_3d=False
    elif dataset_name=='aic-test-S02-c007':
        cluster_id_color_mapping = [5,1,9,0,2,6,3,7,8,4]
    else:
        cluster_id_color_mapping = range(20)

    if dataset_name in ['00016_00001', '00016_00002', '00016_00003', '00016_00004']:
        spline_points = spline_points[::2]
        cluster_ids = cluster_ids[::2]
    spline_points_temp = []
    cluster_ids_temp = []
    for i in range(max(cluster_ids)+1):
        index = np.where(cluster_ids==i)[0]
        spline_points_temp.append(spline_points[index])
        cluster_ids_temp.append(cluster_ids[index])
    spline_points = np.concatenate(spline_points_temp, axis=0)
    cluster_ids = np.concatenate(cluster_ids_temp, axis=0)

    if use_3d:
        img = cv2.imread('../visual_opengl/%s/bird_view.png'%(dataset_name))
    else:
        img = cv2.imread('../visual_opengl/%s/cam_view.jpg'%(dataset_name))
    # hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # hsvImg[...,2] = hsvImg[...,2]*0.4
    # img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    for i, (spline, cluster_id) in enumerate(zip(spline_points, cluster_ids)):
        if dataset_name in selected_out_index_dict and i in selected_out_index_dict[dataset_name]:
            continue
        if len(spline.shape) != 2:
            spline = spline.reshape((-1,2))
        if use_3d:
            spline_ = np.concatenate([spline, np.ones((len(spline), 1))], axis=1)
            spline = (affine_w2s @ (spline_.T)).T
        if cluster_id < 0:
            continue
        for i in range(len(spline)-1):
        # for i in range(5,len(spline)-6):
            img = cv2.line(img, tuple(spline[i,0:2].astype(np.int32)), tuple(spline[i+1,0:2].astype(np.int32)), tableau10[cluster_id_color_mapping[cluster_id]][::-1], 8 if thick else 2)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    out_img_name = dataset_name+'_'+method_name+'_bird_view.jpg' if use_3d else dataset_name+'_'+method_name+'_proj.jpg'
    cv2.imwrite(result_dir + '/' + out_img_name, img)
    print(result_dir + out_img_name + ' saved')

