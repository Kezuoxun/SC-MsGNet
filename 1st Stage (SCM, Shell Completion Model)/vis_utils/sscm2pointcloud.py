import os
import cv2
import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor
# from sklearn.cluster import DBSCAN
# for debug
import matplotlib.pyplot as plt
import pptk

import torch


def sscm2pointcloud(rgb, depth, k, segMask, cmp, obj_list, model_list, format='img_rear'):
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
        if mask.sum > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.erode(mask, kernel)
            mask = np.expand_dims(mask, axis=2)
            masked_scene_pt = scene_points * mask
            masked_cmpf = cmp[..., :3] * mask
            masked_cmpb = cmp[..., 3:] * mask
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
        else:
            continue
        if final_pt is None:
            final_pt = obj_pt
            back_image = cr
        else:
            final_pt = np.append(final_pt, obj_pt, axis=0)
            back_image = back_image + cr

    back_pt = back_image.reshape(-1, 3)
    points = scene_points[filter]
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


def sscm2pointcloud_v2(depth, k, segMask, cmp, obj_list, model_list, format='img_rear'):
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    s = 1000.0

    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    filter = (points_z > 0)
    mask_f = np.expand_dims(filter.astype(np.int32), axis=2)
    scene_points = np.stack([points_x, points_y, points_z], axis=-1)  # scene point cloud in shape (720, 1280)
    final_pt = None

    obj_filter = np.asarray(segMask > 0)
    background_pt = scene_points[~obj_filter]
    for i in range(2):
        background_pt = background_pt[~np.all(background_pt == 0, axis=1)]

        # find plane surface
        scaler = StandardScaler()
        x = background_pt[:, :2]
        y = background_pt[:, 2]
        scaled_X = scaler.fit_transform(x)
        ransac_BG = RANSACRegressor(base_estimator=None, min_samples=300)
        ransac_BG.fit(scaled_X, y)
        inlier_mask = ransac_BG.inlier_mask_
        background_pt = background_pt * np.expand_dims(inlier_mask, axis=1)

    w = ransac_BG.estimator_.coef_
    b = ransac_BG.estimator_.intercept_
    normal = np.concatenate([w, [-1]])
    camera_to_plane = -normal / np.linalg.norm(normal)
    dot_product = np.dot(camera_to_plane, normal)

    for i in range(len(obj_list)):
        mask = np.asarray(segMask == obj_list[i] + 1, np.int16)
        if mask.sum() > 0:
            obj_model = np.asarray(model_list[i].points).T
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.erode(mask, kernel)
            mask = np.expand_dims(mask, axis=2)
            masked_scene_pt = scene_points * mask
            masked_cmpf = cmp[..., :3] * mask
            masked_cmpb = cmp[..., 3:] * mask
            masked_cmpf = (masked_cmpf[..., :].reshape(-1, 3).T * (obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(720, 1280, 3) * mask
            masked_cmpb = (masked_cmpb[..., :].reshape(-1, 3).T * (obj_model.max(1)[:, None] - obj_model.min(1)[:, None]) + obj_model.min(1)[:, None]).T.reshape(720, 1280, 3) * mask

            # Find distance between front and back view (fr: front to rear, cr: camera to rear, uv: unit vector)
            fr_d = masked_cmpb - masked_cmpf
            fr_d = np.sqrt(np.power(fr_d[..., 0], 2) + np.power(fr_d[..., 1], 2) + np.power(fr_d[..., 2], 2)) - 0.005
            # Find unit vector of front view pt
            uv = np.sqrt(np.power(masked_scene_pt[..., 0], 2) + np.power(masked_scene_pt[..., 1], 2) + np.power(masked_scene_pt[..., 2], 2))
            uv = masked_scene_pt[..., :] / np.expand_dims(uv, axis=2)
            uv[np.isnan(uv)] = 0
            fr = uv[..., :] * np.expand_dims(fr_d, axis=2)
            cr = (masked_scene_pt + fr) * mask_f
            obj_pt = np.append(cr.reshape(-1, 3), masked_scene_pt.reshape(-1, 3), axis=0)

            # RANSAC
            # mask_non_zero_cr = (cr.reshape(-1,3) != [0, 0, 0]).all(axis=1)
            # cr_ignore_zero = cr.reshape(-1,3)[~np.all(cr.reshape(-1,3) == 0, axis=1)]
            # # mask_non_zero_cf = (masked_scene_pt.reshape(-1, 3) != [0, 0, 0]).all(axis=1)
            # # cf_ignore_zero = masked_scene_pt.reshape(-1, 3)[~np.all(masked_scene_pt.reshape(-1, 3) == 0, axis=1)]
            # # pt = np.append(cr_ignore_zero, cf_ignore_zero, axis=0)
            # if len(cr_ignore_zero) < 200:
            #     continue
            # x = cr_ignore_zero[:, :2]
            # y = cr_ignore_zero[:, 2]
            # scaled_X = scaler.fit_transform(x)
            # ransac = RANSACRegressor(base_estimator=None, min_samples=50)
            # ransac.fit(scaled_X, y)
            # inlier_mask = ransac.inlier_mask_
            # pt_inlier = cr_ignore_zero*np.expand_dims(inlier_mask, axis=1)
            # cr_inlier = pt_inlier[:len(cr_ignore_zero)]
            # cr = np.zeros((len(mask_non_zero_cr), 3))
            # cr[mask_non_zero_cr.astype(bool)] = cr_inlier
            # cr = cr.reshape(mask.shape[0], mask.shape[1], 3)
            #########
            # obj_pt = np.append(cr.reshape(-1, 3), masked_scene_pt.reshape(-1, 3), axis=0)


            # filter out the outlier behind the plane in back pt
            distance = cr.dot(normal) + b
            # if dot_product < 0:
            #     # Normal vector points away from camera
            #     cr = cr * np.expand_dims(distance >= -0.005, axis=2)
            # else:
            #     # Normal vector points towards camera
            #     cr = cr * np.expand_dims(distance <= 0.005, axis=2)

            obj_pt = np.append(cr.reshape(-1, 3), masked_scene_pt.reshape(-1, 3), axis=0)
        else:
            continue
        if final_pt is None:
            final_pt = obj_pt
            back_image = cr
        else:
            final_pt = np.append(final_pt, obj_pt, axis=0)
            back_image = back_image + cr

    back_pt = back_image.reshape(-1, 3)
    points = scene_points[filter]
    #
    # obj_filter = np.asarray(segMask > 0)
    # background_pt = scene_points[~obj_filter]

    # filter out the noise in background
    # db = DBSCAN(eps=0.005, min_samples=100).fit(background_pt)
    # labels = db.labels_
    # core_mask = np.zeros_like(labels, dtype=bool)
    # core_mask[labels != -1] = True
    # background_pt = background_pt[core_mask]

    # find plane surface and filter out the outlier behind the plane in back pt
    # scaler = StandardScaler()
    # x = background_pt[:, :2]
    # y = background_pt[:, 2]
    # scaled_X = scaler.fit_transform(x)
    # ransac = RANSACRegressor(base_estimator=None, min_samples=50, residual_threshold=0.1)
    # ransac.fit(scaled_X, y)
    # w = ransac.estimator_.coef_
    # b = ransac.estimator_.intercept_
    # normal = np.concatenate([w, [-1]])
    # distance = back_pt.dot(normal) + b
    # camera_to_plane = -normal / np.linalg.norm(normal)
    # dot_product = np.dot(camera_to_plane, normal)
    # if dot_product < 0:
    #     # Normal vector points away from camera
    #     back_pt = back_pt[distance >= 0]
    # else:
    #     # Normal vector points towards camera
    #     back_pt = back_pt[distance <= 0]

    points = np.append(points, back_pt, axis=0)

    if format == 'open3d':
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud
    elif format == 'numpy':
        return points
    elif format == "img_rear":
        return back_image
    else:
        raise ValueError('Format must be either "open3d" or "numpy".')


def ssd2pointcloud(cloud, mask, diff, format='img_rear'):
    mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    # Find unit vector of front view pt
    uv = np.sqrt(
        np.power(cloud[..., 0], 2) + np.power(cloud[..., 1], 2) + np.power(cloud[..., 2], 2))
    uv = cloud[..., :] / np.expand_dims(uv, axis=2)
    uv[np.isnan(uv)] = 0
    fr = uv[..., :] * np.expand_dims(diff, axis=2)
    cr = (cloud + fr) * mask_obj
    cloudm = cloud * mask_obj
    obj_pt = np.append(cr.reshape(-1, 3), cloud.reshape(-1, 3), axis=0)

    if format is 'img_rear':
        return cr
    elif format is 'cloud_rear':
        return cr.reshape(-1,3)


def ssc2pointcloud(mask, cloud, format='img_rear'):

    mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    cr = cloud* mask_obj

    # mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    # cr = cr * mask_obj

    if format is 'img_rear':
        return cr
    elif format is 'cloud_rear':
        return cr.reshape(-1,3)


def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = (arr - min_val) / (max_val - min_val)
    return normalized


def ssd2UVmap(cloud, mask, diff, format='img_rear'):
    mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    diff = diff / 2
    depth = np.sqrt(
        np.power(cloud[..., 0], 2) + np.power(cloud[..., 1], 2) + np.power(cloud[..., 2], 2))
    depth = depth + diff
    w, h = depth.shape
    normalized_w = normalize(np.arange(w))
    normalized_h = normalize(np.arange(h))
    uv = np.zeros((w, h, 3))
    uv[:, :, 0] = np.tile(normalized_w, (h, 1)).T
    uv[:, :, 1] = np.tile(normalized_h, (w, 1))
    uv[:, :, 2] = depth
    uv = uv*mask_obj

    if format is 'img_rear':
        return uv


def ssc_prediction_v2(model, rgb, depth):

    object_list=[-1, 0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
                18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
                37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
                56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
    #Let object list fit mask
    object_list=(np.asarray(object_list)[:] + 1).tolist()
    mapping={}
    for x in range(len(object_list)):
        mapping[x] = object_list[x]
        rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)/1.

    input_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()
    input_tensor = torch.unsqueeze(input_tensor, 0).cuda()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    probs = torch.softmax(output_tensor[:, :len(object_list), ...], dim=1)
    class_masks = (probs > 0.6).float()  # threshold = 0.5
    segMask_pre = torch.argmax(class_masks, dim=1)
    segMask_pre = torch.squeeze(segMask_pre).cpu().numpy()
    for k in mapping.keys():
        segMask_pre[segMask_pre == k] = mapping[k]

    map_pre = torch.squeeze(output_tensor[:, len(object_list):, ...]).permute(1, 2, 0).cpu().numpy()
    map_pre = map_pre * np.expand_dims(np.asarray(segMask_pre > 0), axis=2)


    return segMask_pre, map_pre