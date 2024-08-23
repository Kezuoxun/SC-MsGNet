import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning, RuntimeWarning))
import os
import sys
import numpy as np
import argparse
import random
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'util'))
sys.path.append('/home/dsp/DCM_Grasp/6DCM')

from models.graspnet import GraspNet, pred_decode, GraspNet_Self_Attention_Fuse
from dataset.graspnet_dataset import minkowski_collate_fn
from util.collision_detector import ModelFreeCollisionDetector
from util.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, read_diff
from util.sscm2pointcloud import *
from util.sscm_prediction import sscm_prediction_v2, ssc_prediction_v2
from core.res2net_v2 import SSC_net, SSD_net, SSD_net_AFF

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/ntfs/graspnet')
# parser.add_argument('--checkpoint_path', default='/home/dsp/6DCM_Grasp/graspness_v2/logs/log_ssv/minkuresunet_epoch21.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='./logs/log')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=True)
parser.add_argument('--vis', action='store_true', default=True)
parser.add_argument('--scene', type=str, default='0101')
parser.add_argument('--index', type=str, default='0049')
cfgs = parser.parse_args()

# cfgs.scene = '%04d' % random.randint(100, 130)    #  valid
# cfgs.index = '%04d' % random.randint(1, 255)
cfgs.scene = '%04d' %80
cfgs.index ='%04d' % 121
print('visualize scene: %s, index: %s' % (cfgs.scene, cfgs.index))

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)

def data_process():  # SSD_shape completion
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    '''Chose 6DCM weight'''
    # checkpoints = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSV_v2/SSV_v2_checkpoint_0011.pt"
    # checkpoints = "/home/dsp/Meng/6DCM_Grasp/6DCM/checkpoints/SSD_AFF/SSD_AFF_checkpoint_0030.pt"
    # checkpoints = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSD_v1_checkpoint_0021.pt"  # Meng lod ssd AFF
    checkpoints  = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSD/SSD_checkpoint_0027.pt"  # lod ssd
    # checkpoints = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSC/SSC_checkpoint_0018.pt"  # lod ssd AFF
    # checkpoints = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSC/SSC_checkpoint_0018.pt"  # lod ssd AFF
    # checkpoints = "/home/dsp/Meng/6DCM_Grasp/6DCM/checkpoints/SSC_full/SSC_full_checkpoint_0018.pt"

    '''NOTE SSCM net chose'''
    sscm_net = SSD_net()
    checkpoint = torch.load(checkpoints)
    sscm_net = sscm_net.cuda().eval()
    sscm_net.load_state_dict(checkpoint["state_dict"])
    start_epoch = checkpoint['epoch']
    # print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint, start_epoch))

    rgb = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')))
    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')))
    seg = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')))
    meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat'))
    diff = read_diff(os.path.join(root, 'comap', scene_id, camera_type, 'diff', str(index).zfill(4) + '.npz'))  # for ssd


    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    r'''Chose point cloud generation  method '''
    cloud_rear = ssd2pointcloud(cloud, seg, diff)  # for ssd
    # cloud_rear = ssc2pointcloud(seg, cloud)  # for ssc ssv
    # cloud_rear = ssd2UVmap(cloud, seg, diff)


    seg_mask = np.array(seg > 0)
    cloud_obj = cloud*np.expand_dims(seg_mask, axis=2)
    cloud_obj_rear = cloud_rear * np.expand_dims(seg_mask, axis=2)

    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]
    cloud_masked_rear = cloud_rear[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    cloud_sampled_rear = cloud_masked_rear[idxs]

    cloud_sampled = np.append(cloud_sampled, cloud_sampled_rear, axis=0)

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                'cloud_obj': cloud_obj.reshape(-1, 3),
                'cloud_obj_rear': cloud_obj_rear.reshape(-1, 3),
                }
    return ret_dict


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(data_input):

    # net.eval()
    tic = time.time()

    toc = time.time()
    print('inference time: %fs' % (toc - tic))


if __name__ == '__main__':
    scene_id = 'scene_' + cfgs.scene
    index = cfgs.index
    data_dict = data_process()

    if cfgs.infer:
        inference(data_dict)
    if cfgs.vis:
        '''front'''
        pc_obj = data_dict['cloud_obj']
        nonzero_mask = np.any(pc_obj != 0, axis=1)
        pc_obj = pc_obj[nonzero_mask]
        cloud_obj = o3d.geometry.PointCloud()
        cloud_obj.points = o3d.utility.Vector3dVector(pc_obj.astype(np.float32))
        o3d.visualization.draw_geometries([cloud_obj])

        '''rear'''
        pc_obj_r = data_dict['cloud_obj_rear']
        nonzero_mask = np.any(pc_obj_r != 0, axis=1)
        pc_obj_r = pc_obj_r[nonzero_mask]
        cloud_obj_rear = o3d.geometry.PointCloud()
        cloud_obj_rear.points = o3d.utility.Vector3dVector(pc_obj_r.astype(np.float32))
        o3d.visualization.draw_geometries([cloud_obj_rear])

        '''merage'''
        merged_cloud = o3d.geometry.PointCloud()
        merged_cloud.points = o3d.utility.Vector3dVector(np.vstack((pc_obj, pc_obj_r)).astype(np.float32))
        o3d.visualization.draw_geometries([cloud_obj, merged_cloud])