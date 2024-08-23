import os
import pickle
import numpy as np
import cv2
import pptk
import matplotlib.pyplot as plt
from graspnetAPI import GraspNet
from PIL import Image, ImageFile
import open3d as o3d
from utils.utils import parse_posevector
from utils.xmlhandler import xmlReader


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def read_rgb_np(rgb_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img, np.uint8)
    return img

# the value of depth image is in mm(0.001 m).
def read_depth_np(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.uint16)
    return depth

def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int16)
    return mask_seg


def read_comap_np(comap_path, frameId, merge=True):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cmpf_path = os.path.join(comap_path, "cmpf", "%04d.png" % frameId)
    cmpb_path = os.path.join(comap_path, "cmpb", "%04d.png" % frameId)
    cmpf = np.asarray(Image.open(cmpf_path).convert('RGB'), np.float64)/255
    cmpb = np.asarray(Image.open(cmpb_path).convert('RGB'), np.float64)/255
    if merge:
        return np.append(cmpf, cmpb, axis=2)
    else:
        return cmpf, cmpb


def sscm2pointcloud(pose_list,pt,rgb, depth, k, segMask, sscm, obj_list, model_list, format='numpy'):
    rgb = rgb.astype(np.float32) / 255.0
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    s = 1000.0

    xmap, ymap = np.arange(rgb.shape[1]), np.arange(rgb.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    filter = (points_z > 0)
    scene_points = np.stack([points_x, points_y, points_z], axis=-1)  # scene point cloud in shape (720, 1280)

    for i in range(len(obj_list)):
        posef = np.asarray(pose_list[i][:3])
        obj_model = np.asarray(model_list[i].points).T
        mask = np.expand_dims(np.asarray(segMask == obj_list[i] + 1, np.int32), axis=2)
        masked_scene_pt = scene_points * mask
        masked_cmpf = sscm[..., :3] * mask
        masked_cmpf = (masked_cmpf[..., :].reshape(-1, 3).T * (obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(720, 1280, 3) * mask
        cmpf_pt = masked_cmpf.reshape(-1,3)
        original_tran = np.concatenate((cmpf_pt.T[:3, :], np.ones((1, cmpf_pt.T.shape[1]))), axis=0)
        tran_cmpf=posef @ original_tran
        centers = (masked_scene_pt - masked_cmpf).reshape(-1,3)
        centers = np.unique(centers, axis=0)
        # The center would not be a const because the noise, so we still need to find a TRUE center

        print('ok')

    points = scene_points[filter]
    colors = rgb[filter]

    if format == 'open3d':
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud
    elif format == 'numpy':
        return points, colors
    else:
        raise ValueError('Format must be either "open3d" or "numpy".')


if __name__ == '__main__':
    gn_root = "/media/dsp/JTsai/graspnet"
    camera = 'realsense'
    scene, frame = 0, 0

    gn = GraspNet(gn_root, camera=camera, split='train')

    rgb = gn.loadRGB(scene, camera, frame)
    depth = gn.loadDepth(scene, camera, frame)
    obj_list = gn.getObjIds(scene)
    model_list = gn.loadObjModels(obj_list, simplify=True)
    k = np.load(os.path.join(gn_root, 'scenes', 'scene_' + str(scene).zfill(4), camera, 'camK.npy'))
    sscm = read_comap_np(os.path.join(gn_root, 'comap', 'scene_%04d' % scene, camera), frame)
    segMask = read_mask_np(os.path.join(gn_root, 'scenes', 'scene_' + str(scene).zfill(4), camera, 'label', '%04d.png' % frame))
    pt,_=gn.loadScenePointCloud(scene,camera,frame,format='numpy')

    scene_reader = xmlReader(
        os.path.join(gn.root, 'scenes', 'scene_%04d' % scene, camera, 'annotations', '%04d.xml' % frame))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_id, pose = parse_posevector(posevector)
        obj_list.append(obj_id)
        pose_list.append(pose)
    model_list = gn.loadObjModels(obj_list, simplify=True)

    points = sscm2pointcloud(pose_list, pt,rgb, depth, k, segMask, sscm, obj_list, model_list)

    print('ok')

    # xyz1=np.append(comap[...,:3].reshape(-1,3),comap[...,3:].reshape(-1,3), axis=0)
    # xyz2= (xyz1.T*(original.max(1)[:, None] - original.min(1)[:, None])+original.min(1)[:, None]).transpose(1,0)
    # xyz3=xyz2[...,:]+center