import cv2
import numpy as np
import math


def crop_or_padding_to_fixed_size_instance(img, mask, comap, th, tw, overlap_ratio=0.8):
    h, w, _ = img.shape
    hs, ws = np.nonzero(mask)

    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax = np.min(ws), np.max(ws)
    fh, fw = hmax - hmin, wmax - wmin
    hpad, wpad = th >= h, tw >= w

    hrmax = int(min(hmin + overlap_ratio * fh, h - th))  # h must > target_height else hrmax<0
    hrmin = int(max(hmin + overlap_ratio * fh - th, 0))
    wrmax = int(min(wmin + overlap_ratio * fw, w - tw))  # w must > target_width else wrmax<0
    wrmin = int(max(wmin + overlap_ratio * fw - tw, 0))

    hbeg = 0 if hpad else np.random.randint(hrmin, hrmax)
    hend = hbeg + th
    wbeg = 0 if wpad else np.random.randint(wrmin, wrmax)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]
    comap = comap[hbeg:hend, wbeg:wend]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, img.shape[-1]], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)
        # new_mask = np.zeros([mask.shape[0], th, tw], dtype=mask.dtype)
        new_comap = np.zeros([th, tw, comap.shape[-1]], dtype=comap.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask
        new_comap[hbeg:hbeg + nh, wbeg:wbeg + nw] = comap

        img, mask, comap = new_img, new_mask, new_comap

    return img, mask, comap


def crop_resize_instance_v1(img, mask, comap, cloud, diff, imheight, imwidth,
                            overlap_ratio=0.8, ratio_min=0.8, ratio_max=1.2):
    '''
    crop a region with [imheight*resize_ratio,imwidth*resize_ratio]
    which at least overlap with foreground bbox with overlap
    :param img:
    :param mask:
    :param imheight:
    :param imwidth:
    :param overlap_ratio:
    :param ratio_min:
    :param ratio_max:
    :return:
    '''
    resize_ratio = np.random.uniform(ratio_min, ratio_max)
    target_height = int(imheight * resize_ratio)
    target_width = int(imwidth * resize_ratio)

    img, mask, comap = crop_or_padding_to_fixed_size_instance(
        img, mask, comap, target_height, target_width, overlap_ratio)

    img = cv2.resize(img, (imwidth, imheight), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (imwidth, imheight), interpolation=cv2.INTER_NEAREST)
    comap = cv2.resize(comap, (imwidth, imheight), interpolation=cv2.INTER_LINEAR)
    cloud = cv2.resize(cloud, (imwidth, imheight), interpolation=cv2.INTER_NEAREST)
    diff = cv2.resize(diff, (imwidth, imheight), interpolation=cv2.INTER_LINEAR)

    return img, mask, comap, cloud, diff


def resizer(rgbd, mask, comap, cloud, diff):
    original_width, original_height = mask.shape
    if not isinstance(rgbd, np.ndarray):
        raise ValueError("rgbd 必须是 NumPy 数组")

    # 检查 rgbd 的维度是否正确
    if len(rgbd.shape) != 3:
        raise ValueError("rgbd 的形状必须为 (height, width, channels)")

    # 检查目标大小是否合法
    target_height, target_width = int(original_height/2), int(original_width/2)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("目标大小必须是正整数")
    # rgbd = cv2.resize(rgbd, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_AREA)
    # 调整图像大小
    try:
        rgbd_resized = cv2.resize(rgbd, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    except cv2.error as e:
        print("OpenCV 错误:", e)
        # 尝试使用其他插值方法
        rgbd_resized = cv2.resize(rgbd, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        print("尝试使用其他插值方法后的结果:", rgbd_resized)
    # mask = cv2.resize(mask, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    # cmpf = comap[...,:3]
    # cmpb = comap[...,3:]
    # cmpf = cv2.resize(cmpf, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_AREA)
    # cmpb = cv2.resize(cmpb, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_AREA)
    # comap = np.append(cmpf, cmpb, axis=2)
    # cloud = cv2.resize(cloud, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    # diff = cv2.resize(diff, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_AREA)

    return rgbd, mask, comap, cloud, diff

def resizer_input(rgbd, mask):
    original_width, original_height = mask.shape
    rgbd = cv2.resize(rgbd, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (int(original_height/2), int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    return rgbd, mask

def rotate_instance(img, mask, comap, cloud, diff, rot_ang_min=-30, rot_ang_max=30):
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    hs, ws = np.nonzero(mask)
    # hs, ws = np.nonzero(mask.max(0))
    R = cv2.getRotationMatrix2D((np.mean(ws), np.mean(hs)), degree, 1)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    comap = cv2.warpAffine(comap, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cloud = cv2.warpAffine(cloud, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    diff = cv2.warpAffine(diff, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return img, mask, comap, cloud, diff

def rgb_normalize():
    return 0