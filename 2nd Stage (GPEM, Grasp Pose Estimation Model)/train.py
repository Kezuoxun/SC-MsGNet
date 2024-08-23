import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning, RuntimeWarning))
import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

'''Set path of 6DCM net weight'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'util'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnetAPI'))
sys.path.append('/home/dsp/6DCM_Grasp/zuo/MSCG') 

from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels
from dataset.graspnet_dataset_ori import Graspnet_dataset_ori, minkowski_collate_fn, load_grasp_labels
from dataset.graspnet_wonoise_dataset import GraspNetDataset_mix, minkowski_collate_fn, load_grasp_labels

from models.graspnet import GraspNet,  pred_decode
from models.graspnet import GraspNet_MSCG_context_seed_global_high, GraspNet_MSCG_context_high_Gated
from models.loss import get_loss
from util.collision_detector import ModelFreeCollisionDetector
from util.model_size import get_model_parameters, get_model_size


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/dsp/JTsai1/graspnet')  # media/ntfs
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--log_dir', default='logs/log')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim, default 512')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
parser.add_argument('--collision_thresh', type=float, default=-0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default="logs/dump")  # grasp pose
parser.add_argument('--model', default='AFF')   # cat=concat    Choose  Grasp pose Net
parser.add_argument('--NcM' , default=False,  help="whether use NcM")  #  help="whether use NcM"
parser.add_argument('--Ori' , default=False,  help="whether use Ori")  #  help="whether use NcM"

cfgs = parser.parse_args()
cfgs.dump_dir = os.path.join(cfgs.log_dir, "dump_valid")
print(cfgs.num_point)
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

'''NOTE GSNet  選擇1 stage SCM 網路 as input'''
grasp_labels = load_grasp_labels(cfgs.dataset_root)
print('Ori T/F:', cfgs.Ori)

if cfgs.NcM:
    TRAIN_DATASET = GraspNetDataset_mix(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True)
elif cfgs.Ori:
    TRAIN_DATASET = Graspnet_dataset_ori(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True)
else:   
    TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True)

# VALID_DATASET = Graspnet_dataset_ori(cfgs.dataset_root, split="valid_seen", camera=cfgs.camera, num_points=cfgs.num_point,
#                                     voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)

VALID_DATASET = GraspNetDataset(cfgs.dataset_root, split="valid_seen", camera=cfgs.camera, num_points=cfgs.num_point,
                                    voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)

print('train dataset length: ', len(TRAIN_DATASET))
print('valid dataset length: ', len(VALID_DATASET))

#  TODO: do experiment results
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                            num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

VALID_DATALOADER = DataLoader(VALID_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                            num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

print('train dataloader length: ', len(TRAIN_DATALOADER))
print('valid dataloader length: ', len(VALID_DATALOADER))

'''Bollin version choose Net '''
print(cfgs.model)
if cfgs.model == "GraspNet_MSCG_context_seed_global_high":
    net = GraspNet_MSCG_context_seed_global_high( seed_feat_dim=cfgs.seed_feat_dim, num_point=cfgs.num_point,  num_view=300, cylinder_radius=0.05, hmin=-0.02, hmax=0.06,  is_training=True) 
elif cfgs.model == "GraspNet_MSCG_context_high_Gated":
    net = GraspNet_MSCG_context_high_Gated( seed_feat_dim=cfgs.seed_feat_dim, num_point=cfgs.num_point,  num_view=300, cylinder_radius=0.05, hmin=-0.02, hmax=0.06,  is_training=True) 
else:
    net = GraspNet( seed_feat_dim=cfgs.seed_feat_dim, num_point=cfgs.num_point,  num_view=300, cylinder_radius=0.08, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],  is_training=True) 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
get_model_size(net)
get_model_parameters(net)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    EPOCH_CNT = start_epoch
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    net.is_training=True
    batch_interval = 200
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        # Forward pass
        end_points = net(batch_data_label)

        # loss, end_points = get_loss_stage(end_points, EPOCH_CNT)
        '''TODO 换方法时要改 loss'''
        loss, end_points = get_loss(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            t = time.localtime()
            t = time.strftime("%m/%d, %H:%M:%S", t)
            log_string(' ----epoch: %03d  ---- batch: %03d ---- Time: %s' % (EPOCH_CNT, batch_idx + 1, t))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / (batch_idx+1)))

        del batch_data_label
        del loss
        del end_points


def valid_one_epoch():
    stat_dict = {}  # collect statistics
    net.eval()
    batch_interval = 200
    for batch_idx, batch_data_label in enumerate(VALID_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        end_points = net(batch_data_label)
        # loss, end_points = get_loss_stage(end_points, EPOCH_CNT)
        loss, end_points = get_loss(end_points)

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            t = time.localtime()
            t = time.strftime("%m/%d, %H:%M:%S", t)
            log_string(' ---- VALID ---- epoch: %03d  ---- batch: %03d ---- Time: %s' % (EPOCH_CNT, batch_idx + 1, t))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('VALID: mean %s: %f' % (key, stat_dict[key] / (batch_idx+1)))
                # stat_dict[key] = 0

        del loss
        del end_points


def valid_v2():
    test_dataset = VALID_DATASET
    scene_list = test_dataset.scene_list()
    test_dataloader = VALID_DATALOADER

    # Init the model
    net.eval()
    net.is_training = False

    batch_interval = 100
    net.eval()
    t = time.localtime()
    t = time.strftime("%m/%d, %H:%M:%S", t)
    log_string(' ---- VALID ---- epoch: %03d  ---- Time: %s' % (EPOCH_CNT, t))
    tic = time.time()

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

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection
            if cfgs.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()

    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    res, ap = ge.eval_seen(dump_folder=cfgs.dump_dir, proc=4)
    t = time.localtime()
    t = time.strftime("%m/%d, %H:%M:%S", t)
    log_string(' ---- VALID ---- epoch: %03d ---- AP: %06f ---- Time: %s' % (EPOCH_CNT, ap, t))



def train(start_epoch):
    global EPOCH_CNT
    # valid_v2()
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        # valid_one_epoch()
        train_one_epoch()

        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(), 'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))
        t = time.localtime()
        t = time.strftime("%m/%d, %H:%M:%S", t)
        log_string('---- Save checkpoint successfully!!! ---- Time: %s' % (t))
        
        if epoch > 14:  
            valid_v2()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)
    train(start_epoch)
