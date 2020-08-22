import os
import pickle
import numpy as np
from scipy.io import loadmat, savemat
import sys

use_old_vd = False
if use_old_vd:
    sys.path.insert(1, '/media/sda/fangyu/Projects/single-view-reconstruction/')
    from vehicle_dataset import *
    from utils import generate_spline_points_world
else:
    sys.path.insert(1, '/media/sda/fangyu/Projects/traffic4d/')
    from dataset.vehicle import *
    from utils.utils import generate_spline_points_world
    from config.inputs import dataset_names_list, result_dir

correct_cluster_id_dict = {10:5,81:5,145:5,150:1,200:4,222:4,235:4,249:4,250:5,304:1,370:5,410:4,414:4,456:1,477:0,482:4,520:5,38:-1,55:-1,52:-1,120:-1,121:-1,178:-1,308:-1}

def load_demo_data_mat():
    demo_data = loadmat("TrajData.mat")
    print(demo_data.keys())
    lin_data = demo_data['DataLin']
    print(lin_data)


def zero_to_mat():
    dt = [('data', type(list)), ('label', 'uint8')]
    points = np.zeros((16,2), dtype=np.float32).tolist()
    zero_data = [(points, 1) for i in range(32)]
    zero_data = np.array(zero_data, dtype=dt)
    print(zero_data)
    savemat('ZeroData.mat', {'DataZero':zero_data})
    print("mat saved")


def vd_to_mat(use_old_vd, dataset_idx=0):
    if use_old_vd:
        result_name = "IMG_0121_with_spline_clusters.vd"
    else:
        dataset_name = dataset_names_list[dataset_idx]
        result_name = result_dir + '/' + dataset_name + '_with_spline_clusters.vd'
    with open(result_name, 'rb') as f:
        vd = pickle.load(f)
        print(vd)

    splines, spline_points, bbox_points, index_global_gt, index_global = vd_to_list(vd, use_old_vd)
    dt = [('data', type(list)), ('label', 'uint8')]

    num_vehicles = len(splines)
    # construct 2d bbox data
    bboxes_data = [(bbox_points[i], index_global_gt[i]+2) for i in range(num_vehicles)]
    bboxes_data = np.array(bboxes_data, dtype=dt)
    print(bboxes_data[0:10])

    splines_data = []
    splines_data = [(spline_points[i][:,0:2], index_global_gt[i]+2) for i in range(num_vehicles)]
    splines_data = np.array(splines_data, dtype=dt)
    print(splines_data[0:10])
    savemat(dataset_name+'Data.mat', {'DataBboxes': bboxes_data, 'DataSplines': splines_data})
    print(dataset_name+"Data.mat saved")


def vd_to_list(vd, use_old_vd):
    splines = []
    spline_points = [] # in world coord
    bbox_points = []
    index_global_gt = []
    index_global = []
    for veh_id, vehicle in sorted(vd.vehicles.items(), key=lambda item:item[0]):
        if vehicle.spline is None:
            continue

        # get bbox center and t in each frame
        t_max = None
        bbox_point = []
        for obj_id, bbox in vehicle.bboxs.items():
            bbox_center = [(bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0]
            bbox_point.append(bbox_center)
            if vehicle.rt[obj_id] is not None:
                t = vehicle.image_ids[obj_id] - vehicle.first_appearance_frame_id
                t_max = t if t_max is None else max(t, t_max)

        bbox_points.append(np.array(bbox_point))
        splines.append(vehicle.spline)
        spline_points.append(generate_spline_points_world(vehicle.spline, t_max, extend=False))
        index_global.append(vehicle.traj_cluster_id )
        if use_old_vd and veh_id in correct_cluster_id_dict:
            index_global_gt.append(correct_cluster_id_dict[veh_id])
        else:
            index_global_gt.append(vehicle.traj_cluster_id)

    splines = np.array(splines)
    index_global_gt = np.array(index_global_gt)
    print(splines.shape)
    return splines, spline_points, bbox_points, index_global_gt, index_global





# load_demo_data_mat()
vd_to_mat(use_old_vd, 0 if use_old_vd else int(sys.argv[1]))
