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
from graspnetAPI.graspnet_eval import GraspGroup   # have 7 dof grasp parameter


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'util'))
sys.path.append('/home/dsp/DCM_Grasp/6DCM')

from models.graspnet import GraspNet, pred_decode, GraspNet_Self_Attention_Fuse, GraspNet_MSCG_context_seed_global, \
                                                                GraspNet_MSCG_context_fusion
from dataset.graspnet_dataset import minkowski_collate_fn
from util.collision_detector import ModelFreeCollisionDetector
from util.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, read_diff
from util.sscm2pointcloud import *
from util.sscm_prediction import sscm_prediction_v2, ssc_prediction_v2
from core.res2net_v2 import SSC_net, SSD_net, SSD_net_AFF

REAR = True
# REAR = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/ntfs/graspnet')
'''modify weight of GsNet'''
# parser.add_argument('--checkpoint_path', default='/home/dsp/6DCM_Grasp/graspness_v2/logs/log_ssv/minkuresunet_epoch21.tar')
parser.add_argument('--checkpoint_path', default='/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_ssd_AFF_MSCG_context_no_qkv_pt10/minkuresunet_epoch26.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='./logs/log')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default= 0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=True)
parser.add_argument('--vis', action='store_true', default=True)
parser.add_argument('--scene', type=str, default='0101')
parser.add_argument('--index', type=str, default='0049')
cfgs = parser.parse_args()

# cfgs.scene = '%04d' % random.randint(0, 130)
# cfgs.index = '%04d' % random.randint(1, 255)
cfgs.scene = '%04d' % 125
cfgs.index ='%04d' % 000
print('visualize scene: %s, index: %s' % (cfgs.scene, cfgs.index))

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


def data_process():  # SSD_shape completion
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    checkpoints  = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSD_v1_checkpoint_0021.pt"  # lod ssd Meng

    sscm_net = SSD_net()
    checkpoint = torch.load(checkpoints)
    sscm_net = sscm_net.cuda().eval()
    sscm_net.load_state_dict(checkpoint["state_dict"])

    rgb = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')))
    color = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'rgb',index + '.png')), dtype=np.float32) / 255.0
    # graspness_full = np.load(os.path.join(root, 'graspness', scene_id, camera_type, index + '.npy')).squeeze()

    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')))
    seg = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')))
    meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat'))
    # diff = read_diff(os.path.join(root, 'comap', scene_id, camera_type, 'diff', str(index).zfill(4) + '.npz'))
    seg_pre, diff = sscm_prediction_v2(sscm_net, rgb, depth, pred_depth=True, pred_cloud=False)
    # seg_pre, cloud_rear = ssc_prediction_v2(sscm_net, rgb, depth)

    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_rear = ssd2pointcloud(cloud, seg, diff)


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

    cloud_rear_demo, indices = np.unique(cloud_masked_rear, axis=0, return_index=True)
    color = color[mask]
    color_masked_rear = color[indices]

    if REAR:
        rgb_point_cloud = np.append(cloud_masked, cloud_rear_demo, axis=0)
        color = np.append(color, color_masked_rear, axis=0)
    else:
         rgb_point_cloud = cloud_masked

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

    ret_dict = {  'rgb_pcd':   rgb_point_cloud,
                            'color': color,
                            'point_clouds': cloud_sampled.astype(np.float32),
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
    batch_data = minkowski_collate_fn([data_input])
    # net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    # net = GraspNet_Self_Attention_Fuse(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    net = GraspNet_MSCG_context_fusion(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    # print(net)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    net.eval()
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    # collision detection
    print('cd:', cfgs.collision_thresh)
    if cfgs.collision_thresh > 0:
        cloud = data_input['rgb_pcd']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    # save grasps
    save_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
    save_path = os.path.join(save_dir, cfgs.index + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)

    toc = time.time()
    print('inference time: %fs' % (toc - tic))


if __name__ == '__main__':
    scene_id = 'scene_' + cfgs.scene
    index = cfgs.index
    data_dict = data_process()

    if cfgs.infer:
        inference(data_dict)
    if cfgs.vis:
        pc = data_dict['point_clouds']
        nonzero_mask = np.any(pc != 0, axis=1)
        pc = pc[nonzero_mask]
        gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index + '.npy'))
        gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()
        pc_obj = data_dict['cloud_obj']
        nonzero_mask = np.any(pc_obj != 0, axis=1)
        pc_obj = pc_obj[nonzero_mask]
        pc_obj_r = data_dict['cloud_obj_rear']
        nonzero_mask = np.any(pc_obj_r != 0, axis=1)
        pc_obj_r = pc_obj_r[nonzero_mask]
        if gg.__len__() > 30:
            gg = gg[:30]
        grippers = gg.to_open3d_geometry_list()

        '''DEMO point cloud(color pcd) grasp pose'''
        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        # cloud_obj = o3d.geometry.PointCloud()
        # cloud_obj.points = o3d.utility.Vector3dVector(pc_obj.astype(np.float32))
        # cloud_obj_rear = o3d.geometry.PointCloud()
        # cloud_obj_rear.points = o3d.utility.Vector3dVector(pc_obj_r.astype(np.float32))
        # # o3d.visualization.draw_geometries([cloud_obj])
        # o3d.visualization.draw_geometries([cloud_obj, cloud_obj_rear])
        # o3d.visualization.draw_geometries([cloud, *grippers])
        # o3d.visualization.draw_geometries([cloud_obj, cloud_obj_rear, *grippers])

        '''DEMO point cloud(Scene color pcd) grasp pose'''
        cloud = o3d.geometry.PointCloud()
        point_cloud = data_dict['rgb_pcd']
        color = data_dict['color']
        cloud.points = o3d.utility.Vector3dVector( point_cloud.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])


