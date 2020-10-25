import os
import pickle
import numpy as np
from scipy.io import loadmat, savemat
import sys
from plot import draw_traj_all_id_spline_with_cluster

use_old_vd = False
if use_old_vd:
    sys.path.insert(1, '/media/sda/fangyu/Projects/single-view-reconstruction/')
    from vehicle_dataset import *
    from utils import generate_spline_points_world
    dataset_names_list = ['IMG_0121']
else:
    sys.path.insert(1, '/media/sda/fangyu/Projects/traffic4d/')
    from dataset.vehicle import *
    from utils.utils import generate_spline_points_world
    from config.inputs import dataset_names_list

correct_cluster_id_dict = {'IMG_0121':{10:5,81:5,145:5,150:1,200:4,222:4,235:4,249:4,250:5,304:1,370:5,410:4,414:4,456:1,477:0,482:4,520:5,38:-1,55:-1,52:-1,120:-1,121:-1,178:-1,308:-1}, # old
        '00016_00009':{362:3,252:-1,463:-1,148:-1,97:-1,261:-1,431:-1,217:2,417:2,419:2,280:4,88:4,405:2,82:4,201:3,355:3,440:3,86:3,29:3,288:3,354:4,10:4,137:3,345:3,426:3,430:3,181:3,252:3,163:4,252:-1,163:-1,354:-1,363:-1},
        '00005_00002':{253:-1,132:-1,139:-1,367:3,442:3,397:3,344:-1,368:-1,219:-1,383:-1,250:-1,140:-1,86:-1,456:3,354:4,189:4,332:-1,418:-1,441:3},
        '00002_00005':{325:-1,318:-1,213:2,36:7,333:7,230:7,250:4,12:4,390:-1,115:-1,375:4,436:4,69:4,406:4,41:-1,417:8,2:-1,95:-1,8:-1,187:-1,123:-1,195:-1,4:-1,289:-1,331:-1,363:-1,364:-1,
            #62:-1,149:-1,384:-1,3:-1,194:-1,42:-1,301:-1,234:-1,409:-1,12:-1,73:-1,192:-1,279:-1,174:-1,286:-1,254:-1
            },
        '00016_00001':{622:6,1188:6,1141:-1,1128:-1,1654:-1,1681:-1,1223:-1,1212:-1,24:-1,1443:-1,411:9,1711:9,29:9,1702:9,1728:9,406:9,459:-1,700:-1,924:-1,197:-1,1094:-1,1460:-1,700:6,459:-1,924:-1,439:-1,205:-1,601:-1,1016:-1,346:-1,770:-1,1262:-1,691:6,608:6,626:5,158:10,974:10,978:10,154:-1,1524:-1,1368:-1,199:-1, 399:-1,744:-1,872:-1,602:-1
            },
        '00016_00002':{350:-1,352:-1,486:-1,374:-1,161:-1,867:-1,167:-1,449:0,250:0,459:0,938:0,251:11,789:11,263:-1,81:10,267:10,55:-1,344:10,408:6,399:6,743:-1,412:-1,171:5,391:5,655:5,385:6,654:6,660:6,675:6,132:-1
            },
        '00016_00003':{124:3,210:-1,445:-1,528:-1,606:6,59:-1,151:9,889:9,460:6,604:9,541:-1,333:-1,95:-1,596:6,240:-1,964:-1,786:-1,466:7,458:6,689:7,169:7,45:-1,0:6,457:6,101:4,768:4,332:4,759:4,173:6,760:4,466:-1,591:6,611:6,577:5,32:6,477:6,31:6,232:6
            },
        '00016_00004':{534:-1,751:-1,481:-1,356:-1,156:-1,306:-1,321:-1,713:-1,614:-1,323:-1,676:8,696:8,672:8,301:8,392:-1,764:-1,164:-1,200:-1,123:-1,536:-1,881:-1,286:-1,580:2,767:-1,612:-1
            },
        '00016_00007':{109:-1,93:8,177:8,46:8,20:8,18:-1,93:8,49:8,20:8,103:0,162:0,145:0,26:7,94:7,150:7,15:7,47:7,156:7,94:7,101:7,124:7,63:6,165:6,106:6,89:6,58:-1,90:-1,151:-1,61:-1,97:-1
            },
        'aic-test-S02-c007':{22:-1,59:-1,9:-1,95:3,19:3,123:-1,58:-1,54:6,118:-1,124:-1,26:-1,2:-1,44:-1,41:-1,32:-1,125:1,131:1,54:-1,55:-1}
                           }
correct_whole_cluster_id_dict = {
        '00002_00005':{0:-1}, '00016_00001':{2:-1,11:-1,7:-1},
        '00016_00002':{1:-1,2:-1,9:-1}, '00016_00003':{10:9,2:-1,0:8},
        '00016_00004':{7:-1,9:-1,4:-1},
        '00016_00007':{4:-1,5:-1,1:-1},
        'aic-test-S02-c007':{2:-1,7:-1},
        }

def load_demo_data_mat():
    demo_data = loadmat("TrajData.mat")
    print(demo_data.keys())
    lin_data = demo_data['DataLin']
    print(lin_data)


def zero_to_mat():
    dtype = [('data', type(list)), ('label', 'uint8')]
    points = np.zeros((16,2), dtype=np.float32).tolist()
    zero_data = [(points, 1) for i in range(32)]
    zero_data = np.array(zero_data, dtype=dtype)
    print(zero_data)
    savemat('ZeroData.mat', {'DataZero':zero_data})
    print("mat saved")


def vd_to_mat(use_old_vd, dataset_idx=0):
    result_dir = 'results/'
    dataset_name = dataset_names_list[dataset_idx]
    result_name = result_dir + '/' + dataset_name + '_with_spline_clusters.vd'
    with open(result_name, 'rb') as f:
        vd = pickle.load(f)
        print(vd)

    splines, spline_points, bbox_points, index_global_gt, index_global = vd_to_list(vd, use_old_vd, dataset_idx)
    dtype = [('data', type(list)), ('label', 'uint8')]
    num_correct_vd = len(np.where(np.logical_and(index_global==index_global_gt, index_global>=0))[0])
    num_valid_vd = len(np.where(index_global>=0)[0])
    print("vd accuracy = %.2f", num_correct_vd/num_valid_vd) 

    num_vehicles = len(splines)
    # construct 2d bbox data
    bboxes_data = [(bbox_points[i], index_global_gt[i]+2) for i in range(num_vehicles)]
    bboxes_data = np.array(bboxes_data, dtype=dtype)
    # print(bboxes_data[0:2])

    splines_data = [(spline_points[i][:,0:2], index_global_gt[i]+2) for i in range(num_vehicles)]
    splines_data = np.array(splines_data, dtype=dtype)
    # print(splines_data[0:2])

    savemat(dataset_name+'Data.mat', {'DataBboxes': bboxes_data, 'DataSplines': splines_data})
    print(dataset_name+"Data.mat saved")
    print("vd accuracy = %.2f"%(num_correct_vd/num_valid_vd*100))


def vd_to_mat_as_vd(use_old_vd, dataset_idx=0):
    result_dir = 'results/'
    dataset_name = dataset_names_list[dataset_idx]
    result_name = result_dir + '/' + dataset_name + '_with_spline_clusters.vd'
    with open(result_name, 'rb') as f:
        vd = pickle.load(f)
        print(vd)

    splines, spline_points, bbox_points, index_global_gt, index_global = vd_to_list(vd, use_old_vd, dataset_idx)
    dtype = [('data', type(list)), ('label', 'uint8')]
    num_correct_vd = len(np.where(np.logical_and(index_global==index_global_gt, index_global>=0))[0])
    num_valid_vd = len(np.where(index_global>=0)[0])
    print("vd accuracy = %.2f", num_correct_vd/num_valid_vd) 

    num_vehicles = len(splines)
    # construct 2d bbox data
    bboxes_data = [(bbox_points[i], index_global[i]+2) for i in range(num_vehicles)]
    bboxes_data = np.array(bboxes_data, dtype=dtype)
    # print(bboxes_data[0:2])

    splines_data = [(spline_points[i][:,0:2], index_global[i]+2) for i in range(num_vehicles)]
    splines_data = np.array(splines_data, dtype=dtype)
    # print(splines_data[0:2])
    savemat(dataset_name+'vd.mat', {'DataBboxes': bboxes_data, 'DataSplines': splines_data})
    print(dataset_name+"vd.mat saved")


def vd_to_list(vd, use_old_vd, dataset_idx):
    splines = []
    spline_points = [] # in world coord
    bbox_points = []
    index_global_gt = []
    index_global = []
    veh_ids = []
    dataset_name = dataset_names_list[dataset_idx]
    correct_cluster_ids = correct_cluster_id_dict[dataset_name]
    correct_whole_cluster_ids = correct_whole_cluster_id_dict[dataset_name] if dataset_name in correct_whole_cluster_id_dict else dict()
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

        if veh_id in correct_cluster_ids:
            index_gt = correct_cluster_ids[veh_id]
        elif vehicle.traj_cluster_id in correct_whole_cluster_ids:
            index_gt = correct_whole_cluster_ids[vehicle.traj_cluster_id]
        else:
            index_gt = vehicle.traj_cluster_id
        if index_gt < 0:
            continue
        index_global_gt.append(index_gt)

        bbox_points.append(np.array(bbox_point))
        splines.append(vehicle.spline)
        spline_points.append(generate_spline_points_world(vehicle.spline, t_max, extend=False))
        index_global.append(vehicle.traj_cluster_id )
        veh_ids.append(veh_id)
        # if use_old_vd and veh_id in correct_cluster_ids:

    splines = np.array(splines)
    index_global_gt = np.array(index_global_gt)
    index_global = np.array(index_global)
    draw_traj_all_id_spline_with_cluster(np.array(spline_points), index_global_gt, veh_ids=veh_ids)
    draw_traj_all_id_spline_with_cluster(np.array(bbox_points), index_global, veh_ids=veh_ids)
    print(splines.shape)
    return splines, spline_points, bbox_points, index_global_gt, index_global





# load_demo_data_mat()
if len(sys.argv) == 1:
    print("python input.py [dataset_idx] as-vd")
    print("python input.py [dataset_idx]")
    exit(0)
if len(sys.argv)>=3 and sys.argv[2]=='as-vd':
    vd_to_mat_as_vd(use_old_vd, 0 if use_old_vd else int(sys.argv[1])) # vd cluster into mat
else:
    vd_to_mat(use_old_vd, 0 if use_old_vd else int(sys.argv[1])) # correct vd cluster into mat
