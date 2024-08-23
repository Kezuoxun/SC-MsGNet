from torch import nn
# import torchvision
import torch.nn.functional as F
import torch
import numpy as np


class COMAP_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, target, use_mask=True):
        mask = target[:, 0, ...]
        if use_mask:
            mask = torch.gt(mask, torch.tensor(0)) *1.   # 際上是在創建一個二值掩碼，只保留原 mask 中大於 0 的部分。
        else:
            mask = torch.ones_like(mask)

        mask = torch.unsqueeze(mask, 1)
        cmp = target[:, 1:, ...]

        loss = smooth_l1_loss(inputs, cmp, mask)
        # loss = ruber_loss(inputs, cmp, mask)

        return loss


class SS_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target, reduction=True):

        # Focal loss
        alpha = torch.ones(input.shape[1], device=input.device)
        gamma = 2
        ce_loss = F.cross_entropy(input, target.long(), weight=alpha, reduction='none')
        # ce_loss = F.cross_entropy(input, target, weight=alpha, reduction='none')
        in_loss = ((1-torch.exp(-ce_loss)) ** gamma * ce_loss)

        if reduction:
            in_loss = torch.mean(in_loss)

        return in_loss
    

def rear_pt_eval(cloud1, cloud2):
    """
    Calculate the pixel-wise Chordal Distance between two point cloud images.
    
    Args:
        cloud1 (np.ndarray): First point cloud image of shape (h, w, 3).
        cloud2 (np.ndarray): Second point cloud image of shape (h, w, 3).
        
    Returns:
        np.ndarray: Pixel-wise Chordal Distance for each pixel pair.
    """
    squared_distances = np.sum((cloud1 - cloud2)**2, axis=2)
    pixel_cd = np.sqrt(squared_distances)
    return np.mean(pixel_cd)


def smooth_l1_loss(vertex_pred, vertex_targets, mask, sigma=10, normalize=True, reduction=True):
    '''
    :param reduction:
    :param vertex_pred:     [b,k,h,w]
    :param vertex_targets:  [b,k,h,w]
    :param mask:  [b,1,h,w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    '''
    # sigma = 1.0
    # sigma = 5.0
    b, ver_dim, _, _ = vertex_pred.shape
    # sigma_2 = sigma ** 2
    #  MAE
    abs_diff = abs(mask * (vertex_pred - vertex_targets))

    smoothL1_sign = (abs_diff < 1. / sigma).detach().float()
    in_loss = abs_diff ** 2 * (sigma / 2.) * smoothL1_sign + (abs_diff - (0.5 / sigma)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (2. * torch.sum(mask.view(b, -1), 1) + 1e-9)

    if reduction:
        in_loss = torch.mean(in_loss)

    return in_loss


def mae_loss(vertex_pred, vertex_targets, mask, scale, normalize=True, reduction=True):
    """
    :param err_pred:
    :param reduction:
    :param vertex_pred:     [b,vn*2,h,w]
    :param vertex_targets:  [b,vn*2,h,w]
    :param mask:  [b,1,h,w]
    :param normalize:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    in_loss = abs(mask * (vertex_pred - vertex_targets)) / scale

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * torch.sum(mask.view(b, -1), 1) + 1e-6)

    if reduction:
        in_loss = torch.mean(in_loss)

    return in_loss


def ruber_loss(vertex_pred, vertex_targets, mask, percent=25, normalize=True, reduction=True):
    """
    :param percent:
    :param vertex_pred:     [b,vn*2,h,w]
    :param vertex_targets:  [b,vn*2,h,w]
    :param mask:  [b,1,h,w]
    :param normalize:
    :param reduction:
    :return:
    """
    # error_weights = (err_pred[:, None, :, :] < 0.5).detach()
    b, ver_dim, _, _ = vertex_pred.shape
    abs_diff = abs(mask * (vertex_pred - vertex_targets))
    c = abs_diff.view(b, ver_dim, -1).max(-1)[0][:, :, None, None] * percent / 100.
    ruber_sign = (abs_diff <= c).detach().float()
    in_loss = abs_diff * ruber_sign + (torch.sqrt((2. * c * abs_diff - c ** 2) * (1. - ruber_sign) + 1e-12) - 1e-6)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (1. * torch.sum(mask.view(b, -1), 1) + 1e-6)

    if reduction:
        in_loss = torch.mean(in_loss)

    return in_loss


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()


