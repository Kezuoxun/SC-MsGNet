import os
import cv2
import pickle
import math as m
import random
import numpy as np
import open3d as o3d
from PIL import Image, ImageFile
from graspnetAPI import GraspNet

# debug
import pptk
import matplotlib.pyplot as plt

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
    cmpf = np.asarray(Image.open(cmpf_path).convert('RGB'), np.float32)/255
    cmpb = np.asarray(Image.open(cmpb_path).convert('RGB'), np.float32)/255
    if merge:
        return np.append(cmpf, cmpb, axis=2)
    else:
        return cmpf, cmpb


def find_translation(a, b):
    # 物體坐標系中的三個點
    A = np.ones((4,4))
    A[:3,:3] = a
    B = np.ones((4, 4))
    B[:3, :3] = b

    R = np.dot(np.linalg.inv(A), B)

    return R[:3,:3]


def sscm2pointcloud(rgb, depth, k, segMask, sscm, obj_list, model_list, format='numpy'):
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
    mask_f = np.expand_dims(filter.astype(np.int32), axis=2)
    scene_points = np.stack([points_x, points_y, points_z], axis=-1)  # scene point cloud in shape (720, 1280)
    final_pt = None

    for i in range(len(obj_list)):
        obj_model = np.asarray(model_list[i].points).T
        mask = np.asarray(segMask == obj_list[i] + 1, np.int16)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.erode(mask, kernel)
        mask = np.expand_dims(mask, axis=2)
        masked_scene_pt = scene_points * mask
        masked_cmpf = sscm[..., :3] * mask
        masked_cmpb = sscm[..., 3:] * mask
        masked_cmpf = (masked_cmpf[..., :].reshape(-1, 3).T * (obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(720, 1280, 3) * mask
        masked_cmpb = (masked_cmpb[..., :].reshape(-1, 3).T * (obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(720, 1280, 3) * mask

        # Find distance between front and back view
        fr_d = masked_cmpb - masked_cmpf
        fr_d = np.sqrt(np.power(fr_d[..., 0], 2) + np.power(fr_d[..., 1], 2) + np.power(fr_d[..., 2], 2))
        # Find unit vector of front view pt
        uv = np.sqrt(np.power(masked_scene_pt[..., 0], 2) + np.power(masked_scene_pt[..., 1], 2) + np.power(masked_scene_pt[..., 2], 2))
        uv = masked_scene_pt[..., :] / np.expand_dims(uv, axis=2)
        uv[np.isnan(uv)] = 0
        fr = uv[..., :] * np.expand_dims(fr_d, axis=2)
        cr = (masked_scene_pt + fr) * mask_f
        obj_pt = np.append(cr.reshape(-1,3), masked_scene_pt.reshape(-1,3), axis=0)
        if final_pt is None:
            final_pt = obj_pt
            back_image = cr
        else:
            final_pt = np.append(final_pt, obj_pt, axis=0)
            back_image = back_image + cr

    back_pt = back_image.reshape(-1, 3)
    points = scene_points[filter]
    obj_filter = np.asarray(segMask > 0)
    background_pt = scene_points[~obj_filter]
    bad_pt_filter = np.array(background_pt[..., 2].mean() > back_pt[..., 2])
    points = np.append(points, back_pt, axis=0)
    colors = rgb[filter]

    if format == 'open3d':
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud
    elif format == 'numpy':
        return points, colors
    elif format == "img_rear":
        return back_image
    else:
        raise ValueError('Format must be either "open3d" or "numpy".')


if __name__ == '__main__':
    gn_root = "/media/ntfs/graspnet"
    camera = 'realsense'
    scene = random.randint(0, 99)
    frame = random.randint(0, 255)
    # scene, frame = 0,0

    gn = GraspNet(gn_root, camera=camera, split='train')

    rgb = gn.loadRGB(scene, camera, frame)
    depth = gn.loadDepth(scene, camera, frame)
    obj_list = gn.getObjIds(scene)
    model_list = gn.loadObjModels(obj_list)
    k = np.load(os.path.join(gn_root, 'scenes', 'scene_' + str(scene).zfill(4), camera, 'camK.npy'))
    sscm = read_comap_np(os.path.join(gn_root, 'comap', 'scene_%04d' % scene, camera), frame)
    segMask = read_mask_np(os.path.join(gn_root, 'scenes', 'scene_' + str(scene).zfill(4), camera, 'label', '%04d.png' % frame))
    pt,_=gn.loadScenePointCloud(scene,camera,frame,format='numpy', use_workspace=True)

    points, _ = sscm2pointcloud(rgb, depth, k, segMask, sscm, obj_list, model_list, format='numpy')

    print('ok')

    # xyz1=np.append(comap[...,:3].reshape(-1,3),comap[...,3:].reshape(-1,3), axis=0)
    # xyz2= (xyz1.T*(original.max(1)[:, None] - original.min(1)[:, None])+original.min(1)[:, None]).transpose(1,0)
    # xyz3=xyz2[...,:]+center