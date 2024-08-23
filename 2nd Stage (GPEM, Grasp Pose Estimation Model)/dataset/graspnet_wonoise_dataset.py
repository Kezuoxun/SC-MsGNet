""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from util.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points
from util.data_utils import read_diff
from util.sscm_prediction import sscm_prediction_v2, ssc_prediction_v2, ssv_prediction_v2
from util.sscm2pointcloud import ssd2pointcloud, ssd2UVmap, ssc2pointcloud
from core.res2net_v2 import SSC_net, SSD_net, SSD_net_AFF

import open3d as o3d
from graspnetAPI.utils.utils import *
from graspnetAPI.utils.eval_utils import create_table_points,transform_points
import multiprocessing

def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, align=False, camera='realsense'):
    if align:
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = np.matmul(align_mat,camera_pose)
    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml'%anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    model_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)
    for obj_idx, pose in zip(obj_list, mat_list):
        plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
        model = o3d.io.read_point_cloud(plyfile)
        points = np.array(model.points)
        if align:
            pose = np.dot(camera_pose, pose)
        points = transform_points(points, pose)
        model.points = o3d.utility.Vector3dVector(points)
        model_list.append(model)
        pose_list.append(pose)
    if return_poses:
        return model_list, obj_list, pose_list
    else:
        return model_list

class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True, mode = "pure", val = False):
        assert(num_points<=50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.mode = mode
        self.val = val
        # self.step = 10

        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,190) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        elif split == 'all':
            self.sceneIds = list(range(0,190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.pcdpath = []
        self.segpath = []

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []

        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.pcdpath.append(os.path.join(root, 'clean_scenes', x, camera, 'points', str(img_num).zfill(4)+'.npy'))
                self.segpath.append(os.path.join(root, 'clean_scenes', x, camera, 'seg', str(img_num).zfill(4)+'.npy'))
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]


    def scene_list(self):
        return self.scenename

    def __len__(self):
        # return int(len(self.pcdpath) / self.step)
        return len(self.pcdpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans,flip_mat.T)


        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans,rot_mat.T)

        return point_clouds, object_poses_list, aug_trans

    def __getitem__(self, index):
        if self.load_label:
            if self.mode == "mix":
                flag = np.random.randint(0,2)
                if flag == 0:
                    return self.get_data_label(index)
                else:
                    return self.get_data_label_noise(index)
            else:
                return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index):

        cloud_masked = np.load(self.pcdpath[index])
        seg_masked = np.load(self.segpath[index])
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        #ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        return ret_dict

    def get_data_label(self, index):
        cloud_masked = np.load(self.pcdpath[index])
        seg_masked = np.load(self.segpath[index])
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1

        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        #collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))

        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            #collision = collision_list[i]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)
            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        # ret_dict['instance_mask'] = seg_sampled
        return ret_dict

    def get_data_label_noise(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1
        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        #collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            #collision = collision_list[i]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans

        return ret_dict

    def create_table_points(self, lx, ly, lz, dx=0, dy=0, dz=0, grid_size=[0.01,0.01,0.01]):
        '''
        **Input:**
        - lx:
        - ly:
        - lz:
        **Output:**
        - numpy array of the points with shape (-1, 3).
        '''
        xmap = np.linspace(0, lx, int(lx / grid_size[0]))
        ymap = np.linspace(0, ly, int(ly / grid_size[1]))
        zmap = np.linspace(0, lz, int(lz / grid_size[2]))
        xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
        xmap += dx
        ymap += dy
        zmap += dz
        points = np.stack([xmap, ymap, zmap], axis=-1)
        points = points.reshape([-1, 3])
        return points

    def project_cad_to_camera_pcd(self,index,camera_pose,align_mat,scene_points):
        model_list, obj_list, pose_list = generate_scene_model(self.root, self.scenename[index], self.frameid[index], return_poses=True,
                                          align=False, camera="realsense")
        table = self.create_table_points(1.0, 1.0, 0.01, dx=-0.5, dy=-0.5, dz=0, grid_size=[0.002,0.002,0.008])
        table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        t = o3d.geometry.PointCloud()
        t.points = o3d.utility.Vector3dVector(table_trans)
        pcd_combined = o3d.geometry.PointCloud()
        seg_id_list = []
        for i in range(len(model_list)):
            model = model_list[i].voxel_down_sample(0.002)
            pcd_combined += model
            seg_id_list.append(np.ones(len(model.points))*obj_list[i])
        pcd_combined += t
        seg_id_list.append(np.zeros(len(t.points)))
        seg_mask = np.concatenate(seg_id_list,axis=0)
        scene_w_noise = o3d.geometry.PointCloud()
        scene_w_noise.points = o3d.utility.Vector3dVector(scene_points)
        dists = pcd_combined.compute_point_cloud_distance(scene_w_noise)
        dists = np.asarray(dists)
        ind = np.where(dists < 0.008)[0]
        pcd_combined_crop = pcd_combined.select_by_index(ind)
        seg_mask = seg_mask[ind]
        # color_mask = get_color_mask(seg_mask,nc=len(obj_list)+1)/255
        #pcd_combined_crop.colors = o3d.utility.Vector3dVector(color_mask)
        # o3d.visualization.draw_geometries([pcd_combined_crop])
        # o3d.visualization.draw_geometries([scene_w_noise])
        return np.asarray(pcd_combined_crop.points),seg_mask



class GraspNetDataset_Align(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))
            # self.sceneIds = list(range(1))

        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'valid_seen':
            self.sceneIds = list(range(100, 130))
            # self.sceneIds = list(range(1))
            self.sceneIds = self.sceneIds[1::5]
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        # SSD prediction init (select weight file)
        self.checkpoints = "/home/dsp520/grasp/6DCM_Grasp/6DCM/checkpoints/SSD_v1(Meng)/SSD_v1_checkpoint_0021.pt"

        # SSD prediction init (select model)
        self.net = SSD_net()

        checkpoint = torch.load(self.checkpoints)
        self.net = self.net.cuda().eval()
        self.net.load_state_dict(checkpoint["state_dict"])

        self.pcdpath = []
        self.segpath = []

        self.rgbpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.diff_path = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.pcdpath.append(os.path.join(root, 'clean_scenes', x, camera, 'points', str(img_num).zfill(4)+'.npy'))
                self.segpath.append(os.path.join(root, 'clean_scenes', x, camera, 'seg', str(img_num).zfill(4)+'.npy'))
                self.rgbpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness_old', x, camera, str(img_num).zfill(4) + '.npy'))
                self.diff_path.append(os.path.join(root, 'comap', x, camera, 'diff', str(img_num).zfill(4) + '.npz'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data_NcM(self, point_clouds,point_clouds_wonoise, object_poses_list, rear_pt):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            point_clouds_wonoise = transform_point_cloud(point_clouds_wonoise, flip_mat, '3x3')
            rear_pt = transform_point_cloud(rear_pt, flip_mat, '3x3')   

            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans, flip_mat.T)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        point_clouds_wonoise = transform_point_cloud(point_clouds_wonoise, rot_mat, '3x3')
        rear_pt = transform_point_cloud(rear_pt, rot_mat, '3x3')

        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds, point_clouds_wonoise, object_poses_list, aug_trans, rear_pt

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label_SSDNet(index)
        else:
            return self.get_data_SSDNet(index)

    def get_data_SSDNet(self, index, return_raw_cloud=False):
        rgb = np.array(Image.open(self.rgbpath[index]))
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        seg_pre, diff = sscm_prediction_v2(self.net, rgb, depth, pred_depth=True)
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud_rear = ssd2pointcloud(cloud, seg_pre, diff)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        cloud_masked_rear = cloud_rear[mask]
        color_masked = rgb[mask]
        clear_cloud_masked = np.load(self.pcdpath[index])
        seg_masked = seg[mask]
        if return_raw_cloud:
            return np.append(cloud_masked, cloud_masked_rear, axis=0)

        # sample points

        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # sample clear points

        if len(clear_cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(clear_cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(clear_cloud_masked))
            idxs2 = np.random.choice(len(clear_cloud_masked), self.num_points-len(clear_cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        clear_cloud_sampled = clear_cloud_masked[idxs]
        cloud_sampled_rear = cloud_masked_rear[idxs]
        cloud_sampled = np.append(cloud_sampled, cloud_sampled_rear, axis=0)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['clear_point_clouds'] = clear_cloud_sampled.astype(np.float32)

        return ret_dict

    def get_data_label_SSDNet(self, index):
        rgb = np.array(Image.open(self.rgbpath[index]))
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        seg_pre, diff = sscm_prediction_v2(self.net, rgb, depth, pred_depth=True)
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud_rear = ssd2pointcloud(cloud, seg, diff)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        cloud_masked_rear = cloud_rear[mask]
        color_masked = rgb[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        cloud_sampled_rear = cloud_masked_rear[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()

        objectness_label[objectness_label > 1] = 1

        '''sample wonoise points   (clear point cloud just only front-view)''' 
        clear_cloud_masked = np.load(self.pcdpath[index])
        clear_seg_masked = np.load(self.segpath[index])
        if len(clear_cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(clear_cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(clear_cloud_masked))
            idxs2 = np.random.choice(len(clear_cloud_masked), self.num_points-len(clear_cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        clear_cloud_sampled = clear_cloud_masked[idxs]
        clear_seg_sampled = clear_seg_masked[idxs]


        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []

        # collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)


        ret_dict = {}
        if self.augment:
            cloud_sampled,clear_cloud_sampled, object_poses_list, aug_trans, cloud_sampled_rear = self.augment_data_NcM(cloud_sampled,clear_cloud_sampled, object_poses_list, cloud_sampled_rear)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)
        
        cloud_sampled = np.append(cloud_sampled, cloud_sampled_rear, axis=0)
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32)/ self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        ret_dict['clear_point_clouds'] = clear_cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_widths_list'] = grasp_widths_list
        ret_dict['grasp_scores_list'] = grasp_scores_list
        ret_dict['trans'] = trans
        return ret_dict

class GraspNetDataset_mix(GraspNetDataset_Align):
    ''' NOTE NcM '''
    def augment_data(self, point_clouds, npcd, cpcd, object_poses_list, rear_pt):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            npcd = transform_point_cloud(npcd, flip_mat, '3x3')   # noice
            cpcd = transform_point_cloud(cpcd, flip_mat, '3x3')   # clean
            rear_pt = transform_point_cloud(rear_pt, flip_mat, '3x3')  # rear

            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans, flip_mat.T)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        npcd = transform_point_cloud(npcd, rot_mat, '3x3')
        cpcd = transform_point_cloud(cpcd, rot_mat, '3x3')
        rear_pt = transform_point_cloud(rear_pt, rot_mat, '3x3')

        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds,npcd,cpcd, object_poses_list, aug_trans, rear_pt

    def get_data_label_SSDNet(self, index):
        rgb = np.array(Image.open(self.rgbpath[index]))
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        seg_pre, diff = sscm_prediction_v2(self.net, rgb, depth, pred_depth=True)
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud_rear = ssd2pointcloud(cloud, seg_pre, diff)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        cloud_masked_rear = cloud_rear[mask]
        color_masked = rgb[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        noise_cloud_sampled = cloud_masked[idxs]
        noise_cloud_sampled_rear = cloud_masked_rear[idxs]
 

        color_sampled = color_masked[idxs]
        noise_seg_sampled = seg_masked[idxs]

        # sample wonoise points   just only front-view
        clear_cloud_masked = np.load(self.pcdpath[index])
        clear_seg_masked = np.load(self.segpath[index])
        if len(clear_cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(clear_cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(clear_cloud_masked))
            idxs2 = np.random.choice(len(clear_cloud_masked), self.num_points-len(clear_cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        clear_cloud_sampled = clear_cloud_masked[idxs]
        clear_seg_sampled = clear_seg_masked[idxs]

        '''NOTE mix method'''
        # print('------------------mix------------------')
        mix_cloud_sampled, mix_seg_sampled = self.mix(noise_cloud_sampled,noise_seg_sampled,clear_cloud_sampled,clear_seg_sampled)
 

        if len(mix_cloud_sampled) >= self.num_points:
            idxs = np.random.choice(len(mix_cloud_sampled), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(mix_cloud_sampled))
            idxs2 = np.random.choice(len(mix_cloud_sampled), self.num_points - len(mix_cloud_sampled), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = mix_cloud_sampled[idxs]  # have clean front view
        seg_sampled = mix_seg_sampled[idxs]
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()

        objectness_label[objectness_label > 1] = 1

        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []

        # collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]

            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)


        ret_dict = {}
        if self.augment:
            cloud_sampled,noise_cloud_sampled, clear_cloud_sampled, object_poses_list, aug_trans, cloud_sampled_rear = self.augment_data(cloud_sampled,noise_cloud_sampled,clear_cloud_sampled, object_poses_list, noise_cloud_sampled_rear)
            ret_dict['aug_trans'] = aug_trans

        '''noise fro + rear point clouds'''
        cloud_sampled = np.append(cloud_sampled, cloud_sampled_rear, axis=0)
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        ret_dict['noise_point_clouds'] = noise_cloud_sampled.astype(np.float32)
        ret_dict['clear_point_clouds'] = clear_cloud_sampled.astype(np.float32)
        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_widths_list'] = grasp_widths_list
        ret_dict['grasp_scores_list'] = grasp_scores_list
        ret_dict['trans'] = trans
        ret_dict['instance_mask'] = seg_sampled
        return ret_dict

    def mix(self,pcd,pcd_seg,cpcd,cpcd_seg):
        object_idxs = np.unique(pcd_seg)
        mix_pcd = []
        mix_pcd_seg = []
        for i,object_id in enumerate(object_idxs):
            if np.random.random() > 0.25:
                mix_pcd.append(pcd[pcd_seg == object_id])
                mix_pcd_seg.append(pcd_seg[pcd_seg == object_id])
            else:
                mix_pcd.append(cpcd[cpcd_seg == object_id])
                mix_pcd_seg.append(cpcd_seg[cpcd_seg == object_id])
        mix_pcd = np.concatenate(mix_pcd)
        mix_pcd_seg = np.concatenate(mix_pcd_seg)
        return mix_pcd,mix_pcd_seg


def load_grasp_labels(root):
    obj_names = list(range(1, 89))
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return  grasp_labels


def minkowski_collate_fn(list_data):
    '''將數據樣本中的坐標和特徵進行了組合'''
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    '''坐標批次和特徵批次進行了量化'''
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        '''統一格式化'''
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
        
    res = collate_fn_(list_data)

    return res
