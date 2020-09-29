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


use_old_vd = False

if use_old_vd:
    sys.path.insert(1, '/media/sda/fangyu/Projects/single-view-reconstruction/')
    from vehicle_dataset import *
    from utils import generate_spline_points_world, homo, cart, find_rt_c2w
    from utils import rt_seperate_to_mat, generate_spline_points_cam
    from config import focal_dict
    dataset_names_list = ['IMG_0121', 'jackson_hole']
else:
    sys.path.insert(1, '/media/sda/fangyu/Projects/traffic4d/')
    from dataset.vehicle import *
    from utils.utils import generate_spline_points_world, homo, cart, find_rt_c2w
    from utils.utils import rt_seperate_to_mat, generate_spline_points_cam
    from config.inputs import dataset_names_list, result_dir
    from config.params import focal_dict

affine_w2s_dict = {'00002_00005': [[  5.66808489,  10.30150004, 251.96978511],
 [  7.07443229,  -5.01999157, 356.51345173]],
 '00005_00002':[[  2.41294175,   3.37841096, 313.18111755],
 [  0.75287752,  -5.88317909, 314.00104097]],
 'aic-test-S02-c007':[[  0.97757371,   7.81633461, 525.80017998],
 [  6.5063183,   -1.29127501, 264.48059294]]
        }

def find_landmark_cam(plane, landmark_perspective, focal):
    height, width = 1080, 1920
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


def find_affine_w2s(dataset_name):
    if dataset_name in affine_w2s_dict:
        return np.array(affine_w2s_dict[dataset_name])
    with open("./results/%s_with_spline_clusters.vd"%(dataset_name), 'rb') as f:
         vd = pkl.load(f)
    plane = vd.plane
    rt_c2w = find_rt_c2w(vd)

    landmark_sate_path = "../visual_opengl/%s/bird_view.txt"%(dataset_name)
    with open(landmark_sate_path, 'r') as f:
        lines = f.readlines()
    landmark_sate = []
    for line in lines:
        line_split = line.split(",")
        landmark_2d = [float(line_split[0]), float(line_split[1])]
        landmark_sate.append(landmark_2d)
    landmark_sate = np.array(landmark_sate)

    # landmark in cam coordinate
    landmark_perspective_path = "../visual_opengl/%s/cam_view.pklcor"%(dataset_name)
    with open(landmark_perspective_path, 'rb') as f:
        landmark_perspective = pkl.load(f)
    landmark_cam = find_landmark_cam(plane, landmark_perspective, focal_dict[dataset_name])
    landmark_world = cart((rt_c2w @ ((homo(landmark_cam)).T)).T)

    affine_w2s, _ = cv2.estimateAffine2D(landmark_world[:,0:2], landmark_sate)
    print(affine_w2s)
    return affine_w2s


def find_affine_w2s_random(dataset_name):
    with open("./results/%s_with_spline_clusters.vd"%(dataset_name), 'rb') as f:
         vd = pkl.load(f)
    plane = vd.plane
    rt_c2w = find_rt_c2w(vd)

    landmark_sate_path = "../visual_opengl/%s/bird_view.txt"%(dataset_name)
    with open(landmark_sate_path, 'r') as f:
        lines = f.readlines()
    landmark_sate = []
    for line in lines:
        line_split = line.split(",")
        landmark_2d = [float(line_split[0]), float(line_split[1])]
        landmark_sate.append(landmark_2d)
    landmark_sate = np.array(landmark_sate)

    # landmark in cam coordinate
    landmark_perspective_path = "../visual_opengl/%s/cam_view.pklcor"%(dataset_name)
    with open(landmark_perspective_path, 'rb') as f:
        landmark_perspective = pkl.load(f)
    offset = np.random.randint(-40, 40, tuple(landmark_perspective.shape)) # best for IMG_0121
    landmark_perspective += offset
    landmark_cam = find_landmark_cam(plane, landmark_perspective, focal_dict[dataset_name])
    landmark_world = cart((rt_c2w @ ((homo(landmark_cam)).T)).T)

    affine_w2s, _ = cv2.estimateAffine2D(landmark_world[:,0:2], landmark_sate)
    print(affine_w2s)
    return affine_w2s
