from torchvision import transforms
from dataset import SSCM_dataloader
from utils import metrics
from core.res2net_v2 import SSD_net, SSC_net, SSD_net_AFF
from tqdm import tqdm
from PIL import Image
import scipy.io as scio
import numpy as np
import torch
import argparse
import os
import cv2
import random


'''
If  pred vector =False  dimen  (test self EX3 SSD) & rear point cloud  (test self  EX1 SSC)
If   pred_vec = True   pred vector =3  dimen   fr+rear pc      (test self EX2 SSV )
'''
pred_vec = False

object_list=[-1, 0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
             18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
             37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
             56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
#Let object list fit mask
object_list=(np.asarray(object_list)[:] + 1).tolist()
mapping={}
for x in range(len(object_list)):
  mapping[x] = object_list[x]

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

    if  format == 'img_rear' :
        return cr
    elif  format == 'cloud_rear' :
        return cr.reshape(-1,3)
    
def ssc2pointcloud(mask, cr, format='img_rear'):
    mask_obj = np.expand_dims(np.asarray(mask>0), axis=2)
    cr = cr * mask_obj

    if format == 'img_rear':
        return cr
    elif format == 'cloud_rear':
        return cr.reshape(-1,3)
    
class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def read_diff(path):
    data = np.load(path)
    return data['arr_0']

def resizer_up(mask, cmp):
    original_width, original_height = mask.shape
    mask = cv2.resize(mask, (int(original_height * 2), int(original_width * 2)), interpolation=cv2.INTER_NEAREST)
    cmp = cv2.resize(cmp, (int(original_height * 2), int(original_width * 2)), interpolation=cv2.INTER_NEAREST)
    return mask, cmp


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("-checkpoints", type=str, required=True, help="checkpoint weight file")
    '''評估點雲生成訓練的好壞, 在seen做驗證測試'''
    parser.add_argument("--checkpoints", type=str, default="checkpoints/SSD_Zuo_SF/SSD_Zuo_SF_checkpoint_0037.pt", help="checkpoint weight file")
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--gn", default="/media/ntfs/graspnet", type=str, help="gn_root")
    parser.add_argument("--camera", default="realsense", type=str, help="camera(kinect/realsense)")
    parser.add_argument("--name", default="testing", type=str, help="testing dump name")
    parser.add_argument("--result", default="results", type=str, help="dump dir")
    parser.add_argument("--PLUS", default=True, type=bool)    # True: SSD ( front-rear pc)   False: SSC   (rear pc & front+rear pc)
    args = parser.parse_args()

    if args.PLUS:
        model = SSD_net().eval()
        # model = SSD_net_AFF().eval()
        print('SSD')
    else:
        model = SSC_net().eval()

    checkpoint = torch.load(args.checkpoints)
    model = model.cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    rgbpath = []
    depthpath = []
    labelpath = []
    scenename = []
    frameid = []
    diff_path = []
    metapath = []
    sceneIds = list(range(100, 130))  # Valid seen scene 
    sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in sceneIds]
    for x in tqdm(sceneIds, desc='Loading data path and collision labels...'):
        for img_num in range(256):
            rgbpath.append(os.path.join(args.gn, 'scenes', x, args.camera, 'rgb', str(img_num).zfill(4) + '.png'))
            depthpath.append(os.path.join(args.gn, 'scenes', x, args.camera, 'depth', str(img_num).zfill(4) + '.png'))
            labelpath.append(os.path.join(args.gn, 'scenes', x, args.camera, 'label', str(img_num).zfill(4) + '.png'))
            diff_path.append(os.path.join(args.gn, 'comap', x, args.camera, 'diff', str(img_num).zfill(4) + '.npz'))
            metapath.append(os.path.join(args.gn, 'scenes', x, args.camera, 'meta', str(img_num).zfill(4) + '.mat'))
            scenename.append(x.strip())
            frameid.append(img_num)

    print("Data Length: ", len(rgbpath))
    performances = []
    for index in range(len(rgbpath)):
        rgb = np.array(Image.open(rgbpath[index]))
        depth = np.array(Image.open(depthpath[index]))
        seg = np.array(Image.open(labelpath[index]))
        meta = scio.loadmat(metapath[index])
        diff = read_diff(diff_path[index])

        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        FR_CLOUD = torch.from_numpy(cloud).permute(2, 0, 1).float()
        FR_CLOUD = torch.unsqueeze(FR_CLOUD, 0).to(device)

        rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)/1.
        if args.PLUS:
            original_width, original_height, _ = rgb.shape
            rgbd = cv2.resize(rgbd, (int(original_height / 2), int(original_width / 2)), interpolation=cv2.INTER_AREA)
        input_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()
        input_tensor = torch.unsqueeze(input_tensor, 0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        probs = torch.softmax(output_tensor[:, :len(object_list), ...], dim=1)
        class_masks = (probs > 0.6).float()  # threshold = 0.5
        segMask_pre = torch.argmax(class_masks, dim=1)
        segMask_pre = torch.squeeze(segMask_pre).cpu().numpy()
        for k in mapping.keys():
            segMask_pre[segMask_pre == k] = mapping[k]

        if args.PLUS:
            map_pre = torch.squeeze(output_tensor[:, len(object_list):, ...]).cpu().numpy()
            map_pre = map_pre * np.asarray(segMask_pre > 0)
            segMask_pre, map_pre = resizer_up(segMask_pre, map_pre)
            rear_pt_pre = ssd2pointcloud(cloud, segMask_pre, map_pre)
        else:
            map_pre = output_tensor[:, len(object_list):, ...]
            if pred_vec:
                map_pre = map_pre + FR_CLOUD
            map_pre = torch.squeeze(map_pre).permute(1, 2, 0).cpu().numpy()
            map_pre = map_pre * np.expand_dims(np.asarray(segMask_pre > 0), axis=2)
            rear_pt_pre = ssc2pointcloud(segMask_pre, map_pre)

        rear_pt_GT = ssd2pointcloud(cloud, seg, diff)
        per = metrics.rear_pt_eval(rear_pt_pre, rear_pt_GT)
        performances.append(per)
        print('\rMean Accuracy for image: %04d = %f' % (index, per), end='', flush=True)

    PERFORMANCE = sum(performances) / len(performances)
    print("\nPERFORMANCE: ", PERFORMANCE)


    





    