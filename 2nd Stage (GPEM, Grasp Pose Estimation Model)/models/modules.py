import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup, CylinderQueryAndGroup_att
from util.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from models.attention import AttentionModule, Point_Transformer, Local_attention
# from models.FPT import FastPointTransformer
# from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)  feature=FPS graspable points
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=   0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        # mlps = [3+self.in_dim, 64, 128, 256]   # use xyz, so plus 3  graspnet
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3  gsnet

        '''ori  gsnet'''
        self.groupers = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                        use_xyz=True, normalize_xyz=True)
        '''ori  graspnet'''
        # self.groupers = []
        # for hmax in hmax_list:
        #     self.groupers.append(CylinderQueryAndGroup(
        #         cylinder_radius, hmin, hmax, nsample, use_xyz=True, normalize_xyz=True
        #     ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        '''ori  gsnet'''
        grouped_features = self.groupers(seed_xyz_graspable, seed_xyz_graspable, vp_rot,  seed_features_graspable)  # B*3 + feat_dim*M*K
        vp_features = self.mlps(grouped_features )  # (batch_size, mlps[-1], num_seed*num_depth, nsample)  extract features
        vp_features = F.max_pool2d( vp_features, kernel_size=[1, vp_features.size(3)]  )  # (batch_size, mlps[-1], num_seed*num_depth, 1)
        new_features = vp_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        '''ori  graspnet'''
        # B, num_seed, _, _ = vp_rot.size()
        # grouped_features = []
        # for grouper in self.groupers:
        #     grouped_features.append(grouper(
        #         seed_xyz_graspable, seed_xyz_graspable, vp_rot, seed_features_graspable
        #     ))  # (batch_size, feature_dim, num_seed, nsample)
        # grouped_features = torch.stack(grouped_features,
        #                                dim=3)  # (batch_size, feature_dim, num_seed, num_depth, nsample)
        # grouped_features = grouped_features.view(B, -1, num_seed * num_depth,
        #                                          self.nsample)  # (batch_size, feature_dim, num_seed*num_depth, nsample)

        # vp_features = self.mlps(grouped_features )  # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        # vp_features = F.max_pool2d( vp_features, kernel_size=[1, vp_features.size(3)] )  # (batch_size, mlps[-1], num_seed*num_depth, 1)
        # new_features = vp_features.view(B, -1, num_seed, num_depth)

        return new_features
    
class CloudCrop_context(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=   0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        '''--------- CONTEXT LEARNING ---------'''
        self.pt = Point_Transformer(512)
        # self.fpt = FastPointTransformer(512, 512)
        # self.context = point_context_network(nsample)
        # self.context = PointAttentionNetwork(256)
        # self.attention_module = AttentionModule(dim=self.in_dim* 2, n_head=1, msa_dropout=0.05)
        '''--------- CONTEXT LEARNING ---------'''

        self.groupers = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                                                                            use_xyz=True, normalize_xyz=True)

        # mlps = [3+self.in_dim, 64, 128, 256]   # use xyz, so plus 3  graspnet-1-B
        # mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3  gsnet
        mlps = [3 + self.in_dim, 512,512]   # use xyz, so plus 3  new_zuo
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        # seed_features_graspable  [4, 512, 1024]
        '''--------- CONTEXT LEARNING ---------'''
        seed_xyz_graspable = seed_xyz_graspable.transpose(2, 1).contiguous()   #  [4, 3 , 1024]
        seed_features_graspable = self.pt(seed_xyz_graspable, seed_features_graspable, 10)
        # seed_features_graspable = self.pt(seed_xyz_graspable, seed_features_graspable, 16)
        # seed_features_graspable = self.fpt(seed_xyz_graspable, seed_features_graspable)
        seed_xyz_graspable = seed_xyz_graspable.transpose(1, 2).contiguous()     #  [4, 1024 , 3 ]
        # print('vp_rot.size()', vp_rot.size())  # 4 512 1024
        # print('seed_xyz_graspable.size()', seed_xyz_graspable.size())  # 4 512 1024
        # print('seed_features_graspable.size()', seed_features_graspable.size())  # 4 512 1024

        '''--------- CONTEXT LEARNING ---------'''
        # Sampled point encoder
        grouped_features = self.groupers(seed_xyz_graspable, seed_xyz_graspable, vp_rot, seed_features_graspable)  # B*3 + feat_dim*M*K  [4, 515, 1024, 16]
        # print('grouped_features.size()', grouped_features.size())  # 4 512 1024
        '''sampled pe'''
        vp_features = self.mlps(grouped_features)  # (batch_size, mlps[-1], num_seed*num_depth, nsample)  extract features  [4, 256, 1024, 16]
        # print('vp_features 1.size()', vp_features.size())  # 4 512 1024
        vp_features = F.max_pool2d( vp_features, kernel_size=[1, vp_features.size(3)]  )  # (batch_size, mlps[-1], num_seed*num_depth, 1) [4, 256, 1024, 1]
        # print('vp_features 2.size()', vp_features.size())  # 4 512 1024
        new_features = vp_features.squeeze(-1)   # (batch_size, mlps[-1], M)

        return new_features

class CloudCrop_ABL(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=   0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius


        self.groupers = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                                                                            use_xyz=True, normalize_xyz=True)

        mlps = [3 + self.in_dim, 512, 512]   # use xyz, so plus 3
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
  
    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):

        grouped_features = self.groupers(seed_xyz_graspable, seed_xyz_graspable, vp_rot, seed_features_graspable)  # B*3 + feat_dim*M*K  [4, 515, 1024, 16]
        vp_features = self.mlps(grouped_features)  # extract features  [4, 256, 1024, 16]

        vp_features = F.max_pool2d( vp_features, kernel_size=[1, vp_features.size(3)]  )  # (batch_size, mlps[-1], num_seed*num_depth, 1) [4, 256, 1024, 1]
        new_features = vp_features.squeeze(-1)   # (batch_size, mlps[-1], M)

        return new_features
    
#  no do it
class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        '''WRONG'''
        # self.conv1 = nn.Conv1d(512, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        # self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)
        '''CORRECT GOOD PERFORMANCE'''
        self.conv1 = nn.Conv1d(512, 512, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(512, 2*num_angle*num_depth, 1)   # M × (A × D × 2)

    def forward(self, vp_features, end_points):
        B, _, num_seed= vp_features.size() # num_seed = grasp point num
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points

class ConcatSelfAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(ConcatSelfAttentionFusion, self).__init__()

        self.concat_dim = 2 * input_dim  # Concatenated feature dimension

        # self.self_attention = SelfAttention(self.concat_dim)
        self.self_attention = nn.MultiheadAttention(self.concat_dim, num_heads=2, batch_first=True)

        # self.fc = nn.Conv1d(self.concat_dim, input_dim, 1)
        # self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=True)  # Enable affine transformation

        self.linear = nn.Linear(self.concat_dim, input_dim)

    def forward(self, features_A, features_B):
        B, _, N = features_A.size()

        # Concatenate features
        concat_features = torch.cat((features_A, features_B), dim=1)

        # Reshape for self-attention
        concat_features = concat_features.permute(0, 2, 1)  # (B, N, D)

        # Apply self-attention
        attention_output, _ = self.self_attention(concat_features, concat_features, concat_features)

        # Linear transformation
        # fused_features = self.fc(attention_output)
        fused_features = self.linear(attention_output)

        # Reshape back to original shape
        fused_features = fused_features.permute(0, 2, 1)  # (B, input_dim, N)

        return fused_features


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class SelfAttention_v1(nn.Module):
    def __init__(self, in_features, num_heads):
        super(SelfAttention_v1, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.query = nn.Linear(in_features, self.head_dim * num_heads)
        self.key = nn.Linear(in_features, self.head_dim * num_heads)
        self.value = nn.Linear(in_features, self.head_dim * num_heads)
        self.output_projection = nn.Linear(self.head_dim * num_heads, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, _, N,_ = x.size()
        Q = self.query(x).view(B, self.num_heads, self.head_dim, N)
        K = self.key(x).view(B, self.num_heads, self.head_dim, N)
        V = self.value(x).view(B, self.num_heads, self.head_dim, N)

        Q = Q.transpose(2, 3)  # B x num_heads x N x head_dim
        K = K.transpose(2, 3)  # B x num_heads x N x head_dim
        V = V.transpose(2, 3)  # B x num_heads x N x head_dim

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # B x num_heads x N x N
        attention_scores = attention_scores / (self.head_dim ** 0.5)  # Scale by head_dim

        attention_probs = self.softmax(attention_scores)  # B x num_heads x N x N

        attended_values = torch.matmul(attention_probs, V)  # B x num_heads x N x head_dim
        attended_values = attended_values.transpose(2, 3).contiguous()  # B x num_heads x head_dim x N
        attended_values = attended_values.view(B, -1, N)  # B x (num_heads * head_dim) x N

        output = self.output_projection(attended_values)  # B x in_features x N

        return output


class AFF(nn.Module):
    '''
    多特征融合 AFF   https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
    '''

    def __init__(self, channels=512, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)


        self.local_att = nn.Sequential(
            # context aggregator
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # point-wise conv
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # point-wise conv
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual  # element-wise summation

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
    
class GFF_AFF(nn.Module):
    '''
    多特征融合 AFF   https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
    '''

    def __init__(self, channels=512, r=4):
        super(GFF_AFF, self).__init__()
        inter_channels = int(channels // r)
        self.gate_x =  nn.Sequential(nn.Conv1d(512, 512, 1), nn.Sigmoid())
        self.gate_residual =  nn.Sequential(nn.Conv1d(512, 512, 1), nn.Sigmoid())

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # '''new gate fusion'''
        g_f1 = self.gate_x(x)
        g_f2 = self.gate_residual(residual)
        x_f1 = (1 + g_f1)*x + (1 - g_f1)*(g_f2*residual)
        res_f2 = (1 + g_f2)*residual + (1 - g_f2)*(g_f1*x)
        xa = x_f1+res_f2
        xl = self.local_att(xa)

        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x_f1 * wei + 2 * res_f2 * (1 - wei)
        return xo

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, xa):
        # xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        fused_feat = self.sigmoid(xlg)

        # xo = 2 * x * wei + 2 * residual * (1 - wei)
        return fused_feat

class MS_CAM_LA(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM_LA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att_ori = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.local_att = Local_attention(channels, r)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, xa):
        # xa = x + residual
        
        xl = self.local_att_ori(xa)  
        # xl = self.local_att(xa)  # new
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        # xo = 2 * x * wei + 2 * residual * (1 - wei)
        return wei

class MS_CAM_new(nn.Module):
    '''
    new AFF idea from 冠廷学长 modify MS-CAM 架构
    多特征融合 AFF   https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
    '''

    def __init__(self, channels=512, r=4):
        super(MS_CAM_new, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            # context aggregator
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # point-wise conv
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        '''try other idea from gpt'''
        # # global_att
        self.avg_pool =  nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # from senior idea

        self.global_att = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),  # from senior idea
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # point-wise conv
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, group_features):

        xl = self.local_att(group_features)

        # feat_avg_pool = self.avg_pool(group_features)
        # feat_max_pool = self.max_pool(group_features)  # from senior idea
        # pooled_feat = torch.cat((feat_avg_pool, feat_max_pool), dim=1)
        # xg = self.global_att(pooled_feat)   # 這邊要記得刪除在global_att 中的 AdaptiveMaxPool1d 那行

        xg = self.global_att(group_features)
        xlg = xl + xg

        fused_feat  = self.sigmoid(xlg)

        return fused_feat 

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=512, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class FeatureFusionModule(nn.Module):
    '''使用1D卷积   from claude
        FeatureFusionModule使用1D卷积来对特征进行融合,1D卷积计算效率较高,同时能够很好地捕获特征之间的局部相关性。与使用全连接层相比,1D卷积可以保留一定的局部结构信息。

        融合两个特征源
        该模块的输入是两个特征张量,可以分别来自不同的特征提取模块或不同的区域。通过将这两个特征在通道维度进行拼接,然后使用两个1D卷积层进行融合,可以灵活地融合不同来源的特征信息。

        保持特征维度不变
        为了保证输入和输出的特征维度一致,该模块中第一个1D卷积层的输出通道数为原特征通道数的一半,然后第二个卷积层将通道数恢复到原始大小。这样就能够在融合时保持特征维度不变,便于模块的串联和拓展。

        使用BN和ReLU
        在卷积层之间使用了BN(BatchNormalization,批归一化)和ReLU激活函数,有助于加速收敛并提供一定的非线性表达能力。

        简单且高效
        相较于一些复杂的注意力机制或Transformer模型,FeatureFusionModule结构非常简单,同时由于只使用了少量的1D卷积,参数量和计算量都较小,能够高效地进行特征融合。

        总的来说,FeatureFusionModule的特点和优势在于结构简单、高效灵活,可以方便地集成到各种网络架构中,以完成不同特征源的融合任务。同时相较于一些更复杂的融合模块,它的设计理念更加直接,实现和调试也更加方便。
        '''
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat1, feat2):
        # feat = torch.cat([feat1, feat2], dim=1)
        feat = feat1+feat2
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat)))
        return feat


class FeatureFusionModule_LocalAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule_LocalAtt, self).__init__()
        
        self.local_att = nn.Sequential(
            nn.Conv1d(in_channels, int(out_channels//4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(int(out_channels//4)),   #         # r is the channel reduction ratio = 4
            nn.ReLU(inplace=True),
            nn.Conv1d(int(out_channels//4), out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels)
        )
        
        self.fusion_module = FeatureFusionModule(in_channels, out_channels)
        
        self.weight_local = nn.Parameter(torch.ones(1))
        self.weight_fusion = nn.Parameter(torch.ones(1))
        
    def forward(self, feat1, feat2):
        # concat_feat = torch.cat([feat1, feat2], dim=1)  # 在通道维度拼接
        concat_feat = feat1+feat2
        
        local_feat = self.local_att(concat_feat)

        fusion_feat = self.fusion_module(feat1, feat2)
        
        combined_feat = self.weight_local * local_feat + self.weight_fusion * fusion_feat
        
        return combined_feat
