import os
import random
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from utils.xmlhandler import xmlReader
from utils.utils import transform_points, parse_posevector
import pptk
from graspnetAPI import GraspNet


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

    # objIds = objIds if isArrayLike(objIds) else [objIds]
    # models = []
    # for i in objIds:
    #     plyfile = os.path.join(gn_root, 'models', '%03d' % i, 'nontextured.ply')
    #     models.append(o3d.io.read_point_cloud(plyfile))
    # return models

gn_root = '/media/dsp520/JTsai/graspnet'

# objIds = [0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
#          18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
#          37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
#          56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
# model_list = loadObjModels(gn_root, objIds)


def save_diff(data, path, frameId):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, 'diff')):
        os.makedirs(os.path.join(path, 'diff'))

    # np.save(os.path.join(path, 'diff', '%04d' % frameId), data)
    np.savez_compressed(os.path.join(path, 'diff', '%04d' % frameId), data)


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


def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int16)
    return mask_seg


def rear_pt_generate(segMask, cmpf, cmpb, pose_list, obj_list, model_list):
    rear_pt_map = np.zeros((720,1280,3), dtype=np.float32)
    h, w = segMask.shape
    for i in range(len(obj_list)):
        mask = np.asarray(segMask == obj_list[i] + 1, np.int16)
        if mask.sum() > 0:
            obj_model = np.asarray(model_list[i].points).T
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.erode(mask, kernel)
            mask = np.expand_dims(mask, axis=2)
            masked_cmpf = cmpf * mask
            masked_cmpb = cmpb * mask
            masked_cmpf = (masked_cmpf[..., :].reshape(-1, 3).T * (
                    obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(h, w,
                                                                                                                  3) * mask
            masked_cmpb = (masked_cmpb[..., :].reshape(-1, 3).T * (
                    obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(h, w,
                                                                                                                  3) * mask
            pose = np.asarray(pose_list[i][:3])
            front_pt = masked_cmpf.reshape(-1, 3).T
            front_pt = np.concatenate((front_pt[:3, :], np.ones((1, front_pt.shape[1]))), axis=0)
            front_pt = pose @ front_pt

            rear_pt = masked_cmpb.reshape(-1, 3).T
            rear_pt = np.concatenate((rear_pt[:3, :], np.ones((1, rear_pt.shape[1]))), axis=0)
            rear_pt = pose @ rear_pt

            rear_pt_map += rear_pt.T.reshape(h, w, 3)
            v = pptk.viewer(rear_pt_map[mask])
            

    return rear_pt

if __name__ == '__main__':
    camera = 'realsense'
    gn = GraspNet(gn_root, camera=camera, split='all')


    for scene in range(101,130):
        for frame in range(256):
            segMask = read_mask_np(os.path.join(gn_root, 'scenes', 'scene_' + str(scene).zfill(4), camera, 'label', '%04d.png' % frame))

            scene_reader = xmlReader(os.path.join(gn_root, 'scenes', 'scene_%04d' % scene, camera, 'annotations', '%04d.xml'% frame))
            posevectors = scene_reader.getposevectorlist()
            obj_list = []
            pose_list = []
            model_list = []
            for posevector in posevectors:
                obj_id, pose = parse_posevector(posevector)
                obj_list.append(obj_id)
                pose_list.append(pose)
            model_list = gn.loadObjModels(obj_list, simplify=True)

            cmpf, cmpb = read_comap_np(os.path.join(gn_root, 'comap', 'scene_%04d' % scene, camera), frame, merge=False)
            rear_pt = rear_pt_generate(segMask, cmpf, cmpb, pose_list, obj_list, model_list)
            # save_diff(diff,os.path.join(gn_root, 'comap', 'scene_%04d' % scene, camera), frame)
            print('save diff in scene: %04d, frame: %04d'%(scene, frame))

    print('ok')
