import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
# 40 obj
object_list=[-1, 0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
             18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
             37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
             56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
#Let object list fit mask
object_list=(np.asarray(object_list)[:] + 1).tolist()
mapping={}
for x in range(len(object_list)):
  mapping[x] = object_list[x]


def resizer_up(mask, cmp, pred_depth=False, pred_cloud=False):
    original_width, original_height = mask.shape
    mask = cv2.resize(mask, (int(original_height * 2), int(original_width * 2)), interpolation=cv2.INTER_NEAREST)
    if pred_depth or pred_cloud:
        cmp = cv2.resize(cmp, (int(original_height * 2), int(original_width * 2)), interpolation=cv2.INTER_NEAREST)
    else:
        cmpf = cmp[..., :3]
        cmpb = cmp[..., 3:]
        cmpf = cv2.resize(cmpf, (int(original_height * 2), int(original_width * 2)), interpolation=cv2.INTER_CUBIC)
        cmpb = cv2.resize(cmpb, (int(original_height * 2), int(original_width * 2)), interpolation=cv2.INTER_CUBIC)
        cmp = np.append(cmpf, cmpb, axis=2)
    return mask, cmp


def sscm_prediction(checkpoints, model, rgb, depth, pred_depth=False, pred_cloud=False):
    checkpoint = torch.load(checkpoints)
    model = model.cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])

    rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)/1.
    original_width, original_height, _ = rgb.shape
    rgbd = cv2.resize(rgbd, (int(original_height / 2), int(original_width / 2)), interpolation=cv2.INTER_AREA)

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

    if pred_depth:
        map_pre = torch.squeeze(output_tensor[:, len(object_list):, ...]).cpu().numpy()
        map_pre = map_pre * np.asarray(segMask_pre > 0)
    else:
        map_pre = torch.squeeze(output_tensor[:, len(object_list):, ...]).permute(1, 2, 0).cpu().numpy()
        map_pre = map_pre * np.expand_dims(np.asarray(segMask_pre > 0), axis=2)

    segMask_pre, map_pre = resizer_up(segMask_pre, map_pre, pred_depth=pred_depth, pred_cloud=pred_cloud)

    return segMask_pre, map_pre


def sscm_prediction_v2(model, rgb, depth, pred_depth=False, pred_cloud=False):
    rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)/1.
    original_width, original_height, _ = rgb.shape
    rgbd = cv2.resize(rgbd, (int(original_height / 2), int(original_width / 2)), interpolation=cv2.INTER_AREA)

    input_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()
    input_tensor = torch.unsqueeze(input_tensor, 0).cuda()

    with torch.no_grad():
        output_tensor = model(input_tensor)  # SSD Res2Net

    probs = torch.softmax(output_tensor[:, :len(object_list), ...], dim=1)
    class_masks = (probs > 0.6).float()  # threshold = 0.5
    segMask_pre = torch.argmax(class_masks, dim=1)
    segMask_pre = torch.squeeze(segMask_pre).cpu().numpy()
    for k in mapping.keys():
        segMask_pre[segMask_pre == k] = mapping[k]

    if pred_depth is True:
        map_pre = torch.squeeze(output_tensor[:, len(object_list):, ...]).cpu().numpy()
        map_pre = map_pre * np.asarray(segMask_pre > 0)  # FROM
    else:
        map_pre = torch.squeeze(output_tensor[:, len(object_list):, ...]).permute(1, 2, 0).cpu().numpy()
        map_pre = map_pre * np.expand_dims(np.asarray(segMask_pre > 0), axis=2)

    segMask_pre, map_pre = resizer_up(segMask_pre, map_pre, pred_depth=pred_depth, pred_cloud=pred_cloud)

    return segMask_pre, map_pre


def ssc_prediction_v2(model, rgb, depth):
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


def ssv_prediction_v2(model, rgb, depth, cloud):
    rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)/1.

    input_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()
    input_tensor = torch.unsqueeze(input_tensor, 0).cuda()

    FR_CLOUD = torch.from_numpy(cloud).permute(2, 0, 1).float()
    FR_CLOUD = torch.unsqueeze(FR_CLOUD, 0).cuda()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    probs = torch.softmax(output_tensor[:, :len(object_list), ...], dim=1)
    class_masks = (probs > 0.6).float()  # threshold = 0.5
    segMask_pre = torch.argmax(class_masks, dim=1)
    segMask_pre = torch.squeeze(segMask_pre).cpu().numpy()
    for k in mapping.keys():
        segMask_pre[segMask_pre == k] = mapping[k]

    map_pre = output_tensor[:, len(object_list):, ...]
    map_pre = map_pre + FR_CLOUD
    map_pre = torch.squeeze(map_pre).permute(1, 2, 0).cpu().numpy()
    map_pre = map_pre * np.expand_dims(np.asarray(segMask_pre > 0), axis=2)

    return segMask_pre, map_pre
