import pickle
import numpy as np
import open3d as o3d
import cv2
import os
from PIL import Image, ImageFile

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
    depth = depth.astype(np.int16)
    return depth


def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int16)
    return mask_seg


def read_comap_np_old(comap_path, frameId, merge=True):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cmpf_path = os.path.join(comap_path, "cmpf", "%04d.png" % frameId)
    cmpb_path = os.path.join(comap_path, "cmpb", "%04d.png" % frameId)
    cmpf = np.asarray(Image.open(cmpf_path).convert('RGB'), np.float32)/255
    cmpb = np.asarray(Image.open(cmpb_path).convert('RGB'), np.float32)/255
    if merge:
        return np.append(cmpf, cmpb, axis=2)
    else:
        return cmpf, cmpb


def read_comap_np(cmpf_path, cmpb_path, merge=True):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cmpf = np.asarray(Image.open(cmpf_path).convert('RGB'), np.float32)/255
    cmpb = np.asarray(Image.open(cmpb_path).convert('RGB'), np.float32)/255
    if merge:
        return np.append(cmpf, cmpb, axis=2)
    else:
        return cmpf, cmpb


def read_diff(path):
    data = np.load(path)
    return data['arr_0']


def isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def loadObjModels(gn_root, objIds=None):
    '''
    **Function:**
    - load object 3D models of the given obj ids
    **Input:**
    - objIDs: int or list of int of the object ids
    **Output:**
    - a list of open3d.geometry.PointCloud of the models
    '''

    objIds = objIds if isArrayLike(objIds) else [objIds]
    models = []
    for i in objIds:
        plyfile = os.path.join(gn_root, 'models', '%03d' % i, 'nontextured.ply')
        models.append(o3d.io.read_point_cloud(plyfile))
    return models


def create_point_cloud_from_depth_image(depth, k, organized=True):
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    s = 1000.0

    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    cloud = np.stack([points_x, points_y, points_z], axis=-1)  # scene point cloud in shape (720, 1280)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def ssd2pointcloud(cloud, mask, diff, format='img_rear'):
    mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    # Find unit vector of front view pt
    uv = np.sqrt( np.power(cloud[..., 0], 2) + np.power(cloud[..., 1], 2) + np.power(cloud[..., 2], 2))
    uv = cloud[..., :] / np.expand_dims(uv, axis=2)
    uv[np.isnan(uv)] = 0
    fr = uv[..., :] * np.expand_dims(diff, axis=2)  # front view offset: front view unit vector * diff (||FR||)
    cr = (cloud + fr) * mask_obj
    cloudm = cloud * mask_obj
    obj_pt = np.append(cr.reshape(-1, 3), cloudm.reshape(-1, 3), axis=0)

    if format is 'img_rear':  # ori point cloud data type
        return cr
    elif format is 'cloud_rear':
        return cr.reshape(-1,3)

def ssc2pointcloud(mask, cr, format='img_rear'):
    mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    cr = cr * mask_obj

    if format == 'img_rear':
        return cr
    elif format == 'cloud_rear':
        return cr.reshape(-1,3)