import open3d as o3d
import scipy.io as scio
from PIL import Image
import os
import random
import numpy as np
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'util'))
from util.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from util.data_utils import read_diff
from util.sscm2pointcloud import sscm2pointcloud_v2, ssd2pointcloud, ssc2pointcloud
from core.res2net_v2 import SSC_net, SSD_net, SSD_net_AFF
from util.sscm_prediction import sscm_prediction_v2, ssc_prediction_v2
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval   # have 7 dof grasp parameter
import parser
import torch
REAR = True
# REAR = False

data_path = '/media/ntfs/graspnet'
# scene_id = 'scene_%04d' % random.randint(0, 100)
# scene_id = 'scene_%04d' % random.randint(100, 129)
# scene_id = 'scene_%04d' % random.randint(130, 159)
# ann_id = '%04d' % random.randint(1, 255)
# scene_id = 'scene_0125'
scene_id = 'scene_0111'
scene_id_int = int(scene_id.split('_')[1])
ann_id = '0000'
camera_type = 'realsense'
print('visualize scene: %s, index: %s' % (scene_id, ann_id))

net = SSD_net()
checkpoints  = "/home/dsp/6DCM_Grasp/6DCM/checkpoints/SSD_v1_checkpoint_0021.pt"  # lod ssd Meng
checkpoint = torch.load(checkpoints)
net = net.cuda().eval()
net.load_state_dict(checkpoint["state_dict"])

color = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'rgb', ann_id + '.png')), dtype=np.float32) / 255.0
rgb = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'rgb', ann_id + '.png')))
depth = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'depth', ann_id + '.png')))
seg = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'label', ann_id + '.png')))
if scene_id_int < 130:
    cmp = read_diff(os.path.join(data_path, 'comap', scene_id, camera_type, 'diff', ann_id + '.npz'))
else:
    seg_pre, diff = sscm_prediction_v2(net, rgb, depth, pred_depth=True)
meta = scio.loadmat(os.path.join(data_path, 'scenes', scene_id, camera_type, 'meta', ann_id + '.mat'))


intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
if scene_id_int < 130:
    cloud_rear = ssd2pointcloud(point_cloud, seg, cmp)  # Rear cloud
else:
    cloud_rear = ssd2pointcloud(point_cloud, seg, diff)  # Rear cloud

depth_mask = (depth > 0)
camera_poses = np.load(os.path.join(data_path, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
align_mat = np.load(os.path.join(data_path, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
trans = np.dot(align_mat, camera_poses[int(ann_id)])
workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=True, outlier=0.02)
mask = (depth_mask & workspace_mask)

point_cloud = point_cloud[mask]
cloud_masked_rear = cloud_rear[mask]
cloud_masked_rear, indices = np.unique(cloud_masked_rear, axis=0, return_index=True)

color = color[mask]
color_masked_rear = color[indices]

seg = seg[mask]
seg_masked_rear = seg[indices]

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
o3d.visualization.draw_geometries([cloud])

'''Graspness 假如使用similar or noval sense need to command'''
if REAR:
    point_cloud = np.append(point_cloud, cloud_masked_rear, axis=0)
    seg = np.append(seg, seg_masked_rear, axis=0)
    color = np.append(color, color_masked_rear, axis=0)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
o3d.visualization.draw_geometries([cloud])

if REAR:    
    graspness_full = np.load(os.path.join(data_path, 'graspness', scene_id, camera_type, ann_id + '.npy')).squeeze()
else:
    graspness_full = np.load(os.path.join(data_path, 'graspness_old', scene_id, camera_type, ann_id + '.npy')).squeeze()

graspness_full[seg == 0] = 0.
print('graspness full scene: ', graspness_full.shape, (graspness_full > 0.1).sum())
color[graspness_full > 0.1] = [0., 1., 0.]  # B G R 

# 根据 graspness 的范围设置颜色
bright_green_indices = graspness_full > 0.6  # 分数高于0.8的点为亮绿色
light_green_indices = (graspness_full > 0.4) & (graspness_full <= 0.6)  # 分数介于0.5和0.8之间的点为青绿色
medium_green_indices = (graspness_full > 0.2) & (graspness_full <= 0.4)  # 分数介于0.2和0.5之间的点为稍深的绿色
dark_green_indices = (graspness_full > 0.1) & (graspness_full <= 0.2)  # 分数低于0.2的点为暗绿色

# 设置亮绿色和暗绿色的颜色
color[bright_green_indices] = [0., 1., 0.]  # 亮绿色
color[light_green_indices] = [0., 0.8,0]  # 青绿色q
color[medium_green_indices] = [0., 0.6, 0]  # 稍微深的绿色
color[dark_green_indices] = [0., 0.4, 0.]  # 暗绿色
'''Graspness 假如使用similar or noval sense need to command'''

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
o3d.visualization.draw_geometries([cloud])


# dump_dir = '/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_ssd_AFF_MSCG_context_no_qkv_pt10/dump_26_test_CD/'
# dump_dir = '/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_ssd_AFF_test_Meng/dump_28_test_CD/'
# dump_dir = '/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_GraspNet_MSCG_GFF_LA_MS_CAM/dump_29_test_CD/'
dump_dir = '/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_ssd_AFF_MSCG_seed_global_diff_R_High/dump_22_test_CD/'

gg = np.load(os.path.join(dump_dir, scene_id, camera_type, ann_id + '.npy'))
gg = GraspGroup(gg)
gg = gg.nms()
gg = gg.sort_by_score()


if gg.__len__() > 300:
    gg = gg[:300]
grippers = gg.to_open3d_geometry_list()

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
o3d.visualization.draw_geometries([cloud, *grippers])
