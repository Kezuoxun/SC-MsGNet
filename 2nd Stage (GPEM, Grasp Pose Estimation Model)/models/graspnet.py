""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14 import MinkUNet14D
from models.backbone_resunet14 import Pointnet2Backbone, PointTransformerBackbone_light  # for new test backbone votenet ...
# from models.voting_module import VotingModule
from models.modules import ApproachNet, GraspableNet, CloudCrop, CloudCrop_context,  MS_CAM, SWADNet, AFF, CloudCrop_ABL
from models.modules import iAFF, GFF_AFF, FeatureFusionModule, FeatureFusionModule_LocalAtt, MS_CAM_LA
from util.loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from util.label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation, three_interpolate, three_nn
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from models.attention import Local_attention


class GraspNet(nn.Module):
    '''Ori_GraspNet'''
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, num_point=15000, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW
        self.num_point = num_point

        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)  
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim*2)  # graspable detection
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim*2, is_training=self.is_training)
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim*2)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        if self.is_training:
            self.rotation.is_training = True
        else:
            self.rotation.is_training = False
        seed_xyz = end_points['point_clouds'][:, :self.num_point, :]  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size
        # point-wise features   MinkUNet
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num*2, -1).transpose(1, 2)  # 256

        seed_features = torch.cat((seed_features[..., :self.num_point], seed_features[..., self.num_point:]), 1)  # 512
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.swad(group_features, end_points)

        return end_points



class GraspNet_MSCG_context_seed_global_high(nn.Module):
    '''new test '''
    def __init__(self, seed_feat_dim=512, num_point=15000, num_view=300,cylinder_radius=0.05, hmin= -0.02, hmax= 0.06,  is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH  # 4
        self.num_angle = NUM_ANGLE  # 12
        self.M_points = M_POINT
        self.num_view = NUM_VIEW   # 300
        self.num_point = num_point

        '''stage 1'''
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.fuse_model = AFF(channels=self.seed_feature_dim)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
      
        '''stage 2'''
       
        '''top: new_version'''
        self.crop1 = CloudCrop_context(16, self.seed_feature_dim, cylinder_radius * 0.5, hmin, hmax) # small
        self.crop2 = CloudCrop_context(32, self.seed_feature_dim, cylinder_radius * 0.75, hmin, hmax)
        self.crop3 = CloudCrop_context(32, self.seed_feature_dim, cylinder_radius * 1.0 , hmin, hmax)
        self.crop4 = CloudCrop_context(64, self.seed_feature_dim, cylinder_radius * 1.5, hmin, hmax)  # larger
 
        self.fuse_multi_scale = nn.Conv1d(512 * 4, 512, 1)

        self.gate_fusion = nn.Sequential(nn.Conv1d(512, 512, 1), nn.Sigmoid())  # MLP
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        if self.is_training:
            self.rotation.is_training = True
        else:
            self.rotation.is_training = False
        '''gsnet version'''
        '''------------------------------ stage 1 ------------------------------'''

        '''MinkUNet14D'''          # point-wise features
        seed_xyz = end_points['point_clouds'][:, :self.num_point, :]  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size
        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        # seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num*2, -1).transpose(1, 2)
        
        # print('seed_features',seed_features.shape)  # 4 512 30000
        '''fuse rerar+front points'''
        seed_features = self.fuse_model(seed_features[..., :self.num_point], seed_features[..., self.num_point:])
        # print('seed_features',seed_features.shape)  # 4 512 15000
        '''graspness & objectness'''
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        # print('seed_features_flipped',seed_features_flipped.shape)  #  4 15000 512
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.

        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        # print('seed_features_graspable ', seed_features_graspable .shape)   #  [4, 512, 1024]

        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        '''------------------------------ stage 2 ------------------------------'''
        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']


        vp_features1 = self.crop1(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        vp_features2 = self.crop2(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        vp_features3 = self.crop3(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        vp_features4 = self.crop4(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        B,  num_seed, num_depth = vp_features1.size()
        # print('B,  num_seed, num_depth', B,  num_seed, num_depth)  # 4 512 1024
        # print('vp_features1.size()', vp_features1.size())  # 4 512 1024

        vp_features_concat = torch.cat([vp_features1, vp_features2, vp_features3, vp_features4], dim=1)
        # print('vp_features_concat.size()', vp_features_concat.size())  # 4 512 1024

        vp_features_concat = self.fuse_multi_scale(vp_features_concat)
        # print('vp_features_concat_fuse.size()', vp_features_concat.size())  # 4 512 1024
# # # # # # # # # # # # # # # # # 
        seed_features_gate = self.gate_fusion(seed_features_graspable.contiguous()) * seed_features_graspable.contiguous()        
        # print('seed_features_gate.size()', seed_features_gate.size())  # 4 512 1024
        
        group_features = vp_features_concat + seed_features_gate   # group features
        # try  MS-CAM
        # like gsnet swad
        end_points = self.swad(group_features, end_points)

        return end_points


class GraspNet_MSCG_context_high_Gated(nn.Module):
    '''new test '''

    def __init__(self, seed_feat_dim=512, num_point=15000, num_view=300,cylinder_radius=0.05, hmin=-0.02, hmax= 0.06,  is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH  # 4
        self.num_angle = NUM_ANGLE  # 12
        self.M_points = M_POINT
        self.num_view = NUM_VIEW   # 300
        self.num_point = num_point

        '''stage 1'''
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.fuse_model = AFF(channels=self.seed_feature_dim)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
      
        '''stage 2'''

        '''top: new_version'''
        
        self.crop1 = CloudCrop_context(16, self.seed_feature_dim, cylinder_radius * 0.5, hmin, hmax) # small
        self.crop2 = CloudCrop_context(32, self.seed_feature_dim, cylinder_radius * 0.75, hmin, hmax)
        self.crop3 = CloudCrop_context(32, self.seed_feature_dim, cylinder_radius * 1.0 , hmin, hmax)
        self.crop4 = CloudCrop_context(64, self.seed_feature_dim, cylinder_radius * 1.5, hmin, hmax)  # larger

        self.fuse_multi_scale = nn.Conv1d(512 * 4, 512, 1)
        self.local_gate_fusion = nn.Sequential(nn.Conv1d(512, 512, 1), nn.Sigmoid())  # MLP
        self.locat_att = Local_attention(channels=self.seed_feature_dim, r=4)
        self.global_gate_fusion = nn.Sequential(nn.Conv1d(512, 512, 1), nn.Sigmoid())  # MLP
        
        self.MS_CAM = MS_CAM(channels=self.seed_feature_dim)
        # self.fuse_crop_model = GFF_AFF(channels=self.seed_feature_dim)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        if self.is_training:
            self.rotation.is_training = True
        else:
            self.rotation.is_training = False
        '''gsnet version'''
        '''------------------------------ stage 1 ------------------------------'''

        '''MinkUNet14D'''          # point-wise features
        seed_xyz = end_points['point_clouds'][:, :self.num_point, :]  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size
        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num*2, -1).transpose(1, 2)
        
        # print('seed_features',seed_features.shape)  # 4 512 30000
        '''fuse rerar+front points'''
        seed_features = self.fuse_model(seed_features[..., :self.num_point], seed_features[..., self.num_point:])
        # print('seed_features',seed_features.shape)  # 4 512 15000
        '''graspness & objectness'''
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        # print('seed_features_flipped',seed_features_flipped.shape)  #  4 15000 512
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.

        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_feat)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        # print('seed_features_graspable ', seed_features_graspable .shape)   #  [4, 512, 1024]

        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        '''------------------------------ stage 2 ------------------------------'''
        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']


        vp_features1 = self.crop1(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        vp_features2 = self.crop2(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        vp_features3 = self.crop3(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        vp_features4 = self.crop4(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        B,  num_c, num_seed = vp_features1.size()
        # print('B,  num_c, num_seed', B,  num_c, num_seed)  # 4 512 1024
        '''GFF'''
        g_f1 = self.local_gate_fusion(vp_features1)
        g_f2 = self.local_gate_fusion(vp_features2)
        g_f3 = self.local_gate_fusion(vp_features3)
        g_f4 = self.local_gate_fusion(vp_features4)
        # print('g_f1.size()', g_f1.size())  # 4 512 1024


        vp_features1 = (1 + g_f1)*vp_features1 + (1 - g_f1)*(g_f2*vp_features2 + g_f3*vp_features3 + g_f4*vp_features4)
        vp_features2 = (1 + g_f2)*vp_features2 + (1 - g_f2)*(g_f1*vp_features1 + g_f3*vp_features3 + g_f4*vp_features4)
        vp_features3 = (1 + g_f3)*vp_features3 + (1 - g_f3)*(g_f1*vp_features1 + g_f2*vp_features2 + g_f4*vp_features4)
        vp_features4 = (1 + g_f4)*vp_features4 + (1 - g_f4)*(g_f1*vp_features1 + g_f2*vp_features2 + g_f3*vp_features3)
        '''GFF'''
        # 後續對 gated fusion 之後的局部特徵做進一步的 2D 卷積和 ReLU 操作,也是很常見的做法。這可以進一步提取和整合融合特徵中的信息。
        # print('vp_features1.size()', vp_features1.size())  # 4 512 1024

        vp_features_concat = torch.cat([ vp_features1, vp_features2, vp_features3, vp_features4], dim=1)
        # print('vp_features_concat.size()', vp_features_concat.size())  # 4 512 1024

        vp_features_concat = self.fuse_multi_scale(vp_features_concat)
        # print('vp_features_concat_fuse.size()', vp_features_concat.size())  # 4 512 1024

        vp_features_concat = self.locat_att(vp_features_concat)
        # print('locat_att.size()', vp_features_concat.size())  # 4 512 1024
        
        seed_features_gate = self.global_gate_fusion(seed_features_graspable.contiguous()) * seed_features_graspable.contiguous()        
        # print('seed_features_gate.size()', seed_features_gate.size())  # 4 512 1024
        
        ''' GFF local-global '''

        group_features = vp_features_concat + seed_features_gate   # group features
        group_features = self.MS_CAM(group_features)
        # print('group_features.size()', group_features.size())  # 4 512 1024

        # like gsnet swad
        end_points = self.swad(group_features, end_points)

        return end_points 


def pred_decode(end_points):
    # final of swadnet
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    ## load predictions
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()  # 48, N
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        # discrete
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        ## convert to rotation matrix
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds
