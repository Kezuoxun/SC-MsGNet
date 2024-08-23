import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning, RuntimeWarning))
import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse

from torch.utils.data import DataLoader, RandomSampler
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

from models.graspnet import GraspNet, GraspNet_Self_Attention_Fuse, GraspNet_Self_Attention_Fuse_UV, pred_decode
from models.loss import get_loss, get_loss_stage
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels
from util.collision_detector import ModelFreeCollisionDetector

dataset_root = '/media/ntfs/graspnet'
camera='realsense'
voxel_size=0.005
num_point=15000
batch_size=4

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

VALID_DATASET = GraspNetDataset(dataset_root, split="valid_seen", camera=camera, num_points=num_point,
                                   voxel_size=voxel_size, remove_outlier=True, augment=False, load_label=False)
print('VALID_DATASET length: ', len(VALID_DATASET))


VALID_DATALOADER = DataLoader(VALID_DATASET, batch_size=batch_size, shuffle=False,
                              num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

test_dataset = VALID_DATASET
scene_list = test_dataset.scene_list()
test_dataloader = VALID_DATALOADER
seed_feat_dim=512
net = GraspNet_Self_Attention_Fuse(seed_feat_dim=seed_feat_dim, num_point=num_point, is_training=True)

# Init the model
net.eval()
net.is_training = False

batch_interval = 100
net.eval()
t = time.localtime()
t = time.strftime("%m/%d, %H:%M:%S", t)

checkpoint_path='/home/dsp/6DCM_Grasp/graspness_v2/logs/log_ssd_AFF_test_envs/minkuresunet_epoch18.tar'
resume=False
CHECKPOINT_PATH = checkpoint_path if checkpoint_path is not None and resume else None

import torch
import torch.optim as optim
learning_rate=0.001
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    EPOCH_CNT = start_epoch
    print("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
print(' ---- VALID ---- epoch: %03d  ---- Time: %s' % (EPOCH_CNT, t))
tic = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
for batch_idx, batch_data in enumerate(test_dataloader):
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)

    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
    voxel_size_cd=0.01
    # Dump results for evaluation
    for i in range(batch_size):
        data_idx = batch_idx * batch_size + i
        preds = grasp_preds[i].detach().cpu().numpy()

        gg = GraspGroup(preds)
        # collision detection
        collision_thresh=-0.01
        if collision_thresh > 0:
            cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        # save grasps
        save_dir = os.path.join(dump_dir, scene_list[data_idx], camera)
        save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        gg.save_npy(save_path)

    if (batch_idx + 1) % batch_interval == 0:
        toc = time.time()
        print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
        tic = time.time()



    dump_dir='logs/dump'
    ge = GraspNetEval(root=dataset_root, camera=camera, split='test')
    res, ap = ge.eval_seen(dump_folder=dump_dir, proc=4)
    t = time.localtime()
    t = time.strftime("%m/%d, %H:%M:%S", t)
    print(' ---- VALID ---- epoch: %03d ---- AP: %06f ---- Time: %s' % (EPOCH_CNT, ap, t))