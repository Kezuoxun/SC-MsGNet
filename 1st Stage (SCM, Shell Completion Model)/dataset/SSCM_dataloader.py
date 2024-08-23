import sys
sys.path.append('/home/dsp520/grasp/6DCM_Grasp/6DCM/')

import os
import glob
import random
import torch
import torchvision
import numpy as np
from dataset.augment import *
from dataset.reader import *
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset, Sampler

# For debug
from PIL import Image, ImageFile
import time
import matplotlib.pyplot as plt
from utils.hparams import HParam


object_list=[-1, 0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
             18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
             37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
             56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
#Let object list fit mask 41 obj
object_list=(np.asarray(object_list)[:] + 1).tolist()
mapping={}
for x in range(len(object_list)):
  mapping[x] = object_list[x]


#Sementic segmentation mask and 6DCM dataset
class SSCM_dataset(Dataset):
    def __init__(self, gn_root, camera, split='train', rgb_only=False, pred_depth=False, pred_cloud=False):
        # 定義初始化參數
        self.gn_root = gn_root
        self.camera = camera
        self.split = split
        self.rgb_only = rgb_only
        self.pred_depth = pred_depth
        self.pred_cloud = pred_cloud
      # 讀取資料集路徑
        self.rgbList=[]
        self.depthList=[]
        self.comapfList=[]
        self.comapbList = []
        self.diffList=[]
        self.segMaskList=[]
        self.kList=[]

        # self.key_list = list(mapping.keys())
        # self.val_list = list(mapping.values())
        if pred_depth is False and pred_cloud is False:    #  only have obj id
            self.objIds = [0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
                           18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
                           37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
                           56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
            # self.model_list = loadObjModels(self.gn_root, self.objIds)

        data_set = range(0, 100)
        # data_set = range(1)
        if self.split == 'valid':
            data_set = range(100, 130, 3)
        if self.split == 'test':
            data_set = range(100, 130)
        
        for sceneId in data_set:
            try:

                self.rgbList += sorted(glob.glob(os.path.join(self.gn_root, 'scenes', 'scene_'+str(sceneId).zfill(4), self.camera, 'rgb', '*.png')))
                self.depthList += sorted(glob.glob(os.path.join(self.gn_root, "scenes", "scene_"+str(sceneId).zfill(4), self.camera, 'depth', '*.png')))
                self.comapfList += sorted(glob.glob(os.path.join(self.gn_root, 'comap', 'scene_'+str(sceneId).zfill(4), self.camera, 'cmpf', '*.png')))
                self.comapbList += sorted(glob.glob(os.path.join(self.gn_root, 'comap', 'scene_' + str(sceneId).zfill(4), self.camera, 'cmpb', '*.png')))
                self.diffList += sorted(glob.glob(os.path.join(self.gn_root, 'comap', 'scene_' + str(sceneId).zfill(4), self.camera, 'diff','*.npz')))
                self.segMaskList += sorted(glob.glob(os.path.join(self.gn_root, 'scenes', 'scene_'+str(sceneId).zfill(4), self.camera, 'label', '*.png')))

                for frameId in range(0, 256):
                    self.kList.append(os.path.join(self.gn_root, 'scenes', 'scene_' + str(sceneId).zfill(4), self.camera, 'camK.npy'))
            except:
                continue

        # del self.rgbList[1::2]
        # del self.depthList[1::2]
        # del self.comapList[1::2]
        # del self.segMaskList[1::2]


    def create_rgbd(self, rgb, depth):
        if self.rgb_only:
            return rgb
        else:
            rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)
            return rgbd

    def create_groundtruth(self, comap, segMask, diff, cloud):
        # if self.pred_depth:
        #     map = np.expand_dims(diff, axis=2)
        # elif self.pred_cloud:
        #     map = cloud
        # else:
        #     map = comap
        map = cloud

        for i, cls in enumerate(mapping):
            segMask = np.where(segMask == mapping[i], i, segMask)

        gt = np.append(np.expand_dims(segMask, axis=2), map, axis=2)
        return gt

    def one_hot_encoding(self, segMask):
        h, w = segMask.shape
        one_hot_encode = np.zeros((h, w, len(object_list)))
        for i, cls in enumerate(mapping):
            one_hot_encode[:, :, i] = np.asarray(segMask == mapping[i])
        return one_hot_encode

    def one_hot_decoding(self, one_hot):
        segMask = np.zeros(one_hot.shape[:2])
        single_layer = np.argmax(one_hot, axis=-1)
        for k in mapping.keys():
            segMask[single_layer == k] = mapping[k]
        segMask = np.asarray(segMask, dtype='int')
        return segMask

    def get_onject_list(self):
        return object_list
    
    #  讀取每筆原始資料
    def __getraw__(self, index):
        rgb = read_rgb_np(self.rgbList[index])
        depth = read_depth_np(self.depthList[index])
        # comap = read_pickle(self.comapList[index])
        comap = read_comap_np(self.comapfList[index], self.comapbList[index])
        segMask = read_mask_np(self.segMaskList[index])
        diff = read_diff(self.diffList[index])
        k = np.load(self.kList[index])
        return rgb, depth, comap, segMask, diff, k

    def __getitem__(self, index_tuple):
        if self.split == 'train':
            index, height, width = index_tuple
        else:
            index = index_tuple
        # index = index_tuple
        # 讀取每次迭代的資料集中第 idx  資料
        rgb, depth, comap, segMask, diff, k = self.__getraw__(index)
        rgbd = self.create_rgbd(rgb, depth)
        #  Create scene_cloud (as  GT of front point cloud) , use for predict rear point cloud
        scene_cloud = create_point_cloud_from_depth_image(depth, k)  # front pc  label
        # GT of rear cloud
        '''NOTE chose pred Depth or PCD'''
        if self.pred_depth:
            rear_cloud = ssd2pointcloud(scene_cloud, segMask, diff)  # rear pc  label  diff
        elif   self.pred_cloud:
            rear_cloud = ssc2pointcloud(scene_cloud, segMask, diff)  # rear pc  label   pc
        #  data augment
        if self.split == 'train' and self.pred_cloud ==False :
            rgbd, segMask, comap, rear_cloud, diff = resizer(rgbd, segMask, comap, rear_cloud, diff)
            rgbd, segMask, comap, rear_cloud, diff = rotate_instance(rgbd, segMask, comap, rear_cloud, diff)
            rgbd, segMask, comap, rear_cloud, diff = crop_resize_instance_v1(rgbd, segMask, comap, rear_cloud, diff, height, width)
        gt = self.create_groundtruth(comap, segMask, diff, rear_cloud)  # data list

        # 進行前處理 (torchvision.Transform 等)
        input_tensor = torch.from_numpy(rgbd).permute(2,0,1).float()
        # print('input tensor: ', input_tensor.shape[1]/2)
        # input_tensor=torchvision.transforms.functional.resize(input_tensor, (360, 640))
        # print('input tensor: ', input_tensor.shape)
        gt_tensor = torch.from_numpy(gt).permute(2,0,1).float()
        fr_cloud_tensor = torch.from_numpy(scene_cloud).permute(2,0,1).float()
        # gt_tensor = torchvision.transforms.functional.resize(gt_tensor, (int(gt_tensor.shape[1]/2), int(gt_tensor.shape[2]/2)))
        # print('gt tensor: ', gt_tensor.shape)
        data = {"rgbd":input_tensor, "gt":gt_tensor, "FR_CLOUD":fr_cloud_tensor}
        return data

    def __len__(self):
        # 計算資料集總共數量
        return len(self.rgbList)


# Used in training 
class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, cfg):

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of torch.util.data.Sampler, but got sampler={}".format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin = cfg.hmin  # 256
        self.hmax = cfg.hmax  # 480
        self.wmin = cfg.wmin  # 256
        self.wmax = cfg.wmax  # 640
        self.size_int = cfg.size_int  # 8
        self.hint = (self.hmax - self.hmin) // self.size_int + 1
        self.wint = (self.wmax - self.wmin) // self.size_int + 1

    def generate_height_width(self):
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            # h, w = self.generate_height_width()
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            h, w = self.generate_height_width()
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size 


# Just for debug and testing
if __name__ == '__main__':
    t0 = time.time()
    gn_root = "/media/dsp520/JTsai/graspnet"
    config = '/home/dsp520/grasp/6DCM_Grasp/6DCM/configs/default.yaml'
    camera = 'realsense'

    hp = HParam(config)
    dataset = SSCM_dataset(gn_root, camera, split="test", pred_depth=True)
    val_dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    t1 = time.time()
    print('start')

    for idx, data in enumerate(val_dataloader):
        # t1 = time.time()
        input=data['rgbd']
        label=data['gt']  # (b, 4, h, w)
        # print(idx)
    t2 = time.time()
    print(t2 - t1)

