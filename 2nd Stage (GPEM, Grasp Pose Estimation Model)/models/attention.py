import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(queries, keys, values, mask=None, dropout=None):
        """
            queries: B x H x S x headdim
            keys: B x H x L x headdim
            values: B x H x L x headdim
            mask: B x 1 x S x L
        """
        head_dim = queries.size(-1)
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(head_dim)  # B x H x S x L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values  # B x H x S x head_dim

    def forward(self, query, key, value, mask=None):
        """  (bs, max_len, word_feat_dim)
            query: B x S x D
            key: B x L x D
            value: B x L x D
            mask: B x S x L
        """
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # B x 1 x S x L, 1 for heads
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]  # (bs, nhead, max_len, head_dim) for word feat
        result = self.attention(queries, keys, values, mask, self.dropout)  # (bs, nhead, max_len, head_dim)
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)  # (bs, max_len, dim)

        return self.linears[-1](result)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head, msa_dropout):
        super().__init__()
        self.dim = dim
        self.msa = MultiHeadAttn(dim, n_head, dropout=msa_dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, q, k, v, mask):
        msa = self.msa(q, k, v, mask)
        x = self.norm1(v + msa)

        return x

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)  #  矩陣相乘
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def transformer_neighbors(x, feature, k=20, idx=None):
    '''
    負責處理點雲特徵並計算相關的鄰域信息，從而使模型能夠學習到點之間的關係和上下文信息。
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)  # 1024
    # print('num_points', num_points)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    '''position encoding'''
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size*num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    position_vector = (x - neighbor_x).permute(0, 3, 1, 2).contiguous() # B,3,N,k
    '''position encoding'''

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size*num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims) 
    neighbor_feat = neighbor_feat.permute(0, 3, 1, 2).contiguous() # B,C,N,k
  
    return position_vector, neighbor_feat

class Point_Transformer(nn.Module):
    def __init__(self, input_features_dim):
        super(Point_Transformer, self).__init__()
        #  position encoding function MLP 
        self.conv_theta1 = nn.Conv2d(3, input_features_dim, 1)
        self.conv_theta2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_theta = nn.BatchNorm2d(input_features_dim)
        #  scaler attention  linear transformation
        self.conv_phi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_psi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_alpha = nn.Conv2d(input_features_dim, input_features_dim, 1)
        # mapping fusction
        self.conv_gamma1 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_gamma2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_gamma = nn.BatchNorm2d(input_features_dim)

    def forward(self, xyz, features, k):
        '''position encoding'''
        position_vector, x_j = transformer_neighbors(xyz, features, k=k)
        #  position Encoding function  將位置向量轉換為特徵向量
        delta = F.relu(self.bn_conv_theta(self.conv_theta2(self.conv_theta1(position_vector)))) # B,C,N,k
        '''position encoding'''
        # corrections for x_i
        x_i = torch.unsqueeze(features, dim=-1).repeat(1, 1, 1, k) # B,C,N,k
        #  鄰域點特徵 x_i 和 x_j 的線性變換  feature transformation
        linear_x_i = self.conv_phi(x_i) # B,C,N,k
        linear_x_j = self.conv_psi(x_j) # B,C,N,k
        relation_x = linear_x_i - linear_x_j + delta # B,C,N,k   feature transformation add position
        # mapping fusction  r
        relation_x = F.relu(self.bn_conv_gamma(self.conv_gamma2(self.conv_gamma1(relation_x)))) # B,C,N,k
        # attention generation
        weights = F.softmax(relation_x, dim=-1) # B,C,N,k

        #  self.conv_alpha(x_j)  將 x_j 中的特徵向量進行線性轉換 + position 特征加权
        features_transform = self.conv_alpha(x_j) + delta # B,C,N,k    
        '''aggregation'''
        f_out = weights * features_transform # B,C,N,k
        f_out = torch.sum(f_out, dim=-1) # B,C,N

        return f_out
    


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    # https://github.com/kaiyuyue/cgnl-network.pytorch/blob/master/model/resnet.py
    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                   'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        
        t = t.view(b, 1, c * h * w)   #  theta
        p = p.view(b, 1, c * h * w)  #   phi
        g = g.view(b, c * h * w, 1)  #  gama

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x
    
class PointAttentionNetwork(nn.Module):
    def __init__(self,C, ratio = 8):
        super(PointAttentionNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(C//ratio)
        self.bn2 = nn.BatchNorm1d(C//ratio)
        self.bn3 = nn.BatchNorm1d(C)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn2,
                                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
                                self.bn3,
                                nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b,c,n,_ = x.shape
        print('x.shape', x.shape)

        a = self.conv1(x).permute(0,2,1) # b, n, c/ratio

        b = self.conv2(x) # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b)) # b,n,n

        d = self.conv3(x) # b,c,n
        out = x + torch.bmm(d, s.permute(0, 2, 1))

        return out
    
class point_context_network(nn.Module):
    # https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/modules.py
    def __init__(self, num_points = 4096,out_size = 1024):
        super(point_context_network, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(256, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        return x
    
class LAEConvOperation(nn.Module):
    '''in point attention network paper generated from GPT'''
    def __init__(self, K, in_channels, out_channels):
        super(LAEConvOperation, self).__init__()
        self.K = K
        self.conv_transform = nn.Conv1d(in_channels, out_channels, 1)
        self.graph_attention = nn.Linear(out_channels, out_channels)

    def forward(self, h, pi):
        # h ; point clouds (local)
        # Step 1: Search K neighbor points pj of pi in the point cloud h
        distances, indices = torch.topk(torch.norm(h - pi.unsqueeze(2), dim=1), k=self.K, largest=False)
        neighbor_points = h[:, indices, :]  # Shape: (batch_size, K, in_channels)
        
        # Step 2: pj - pi: move points pj to local coordinate system of pi
        neighbor_points_relative = neighbor_points - pi.unsqueeze(2)

        # Step 3: W(pj - pi): transform the input points into higher-level features
        transformed_points = self.conv_transform(neighbor_points_relative)  # Shape: (batch_size, out_channels, K)

        # Step 4: Compute normalized attention edge coefficients with softmax
        attention_weights = F.softmax(torch.matmul(transformed_points.transpose(1, 2), transformed_points), dim=-1)

        # Step 5: Use graph attention aggregator to obtain updated feature at pi
        updated_feature = torch.matmul(attention_weights, transformed_points.transpose(1, 2))  # Shape: (batch_size, K, out_channels)
        updated_feature = self.graph_attention(updated_feature.mean(dim=1))  # Shape: (batch_size, out_channels)

        # Step 6: Feature transformation operation
        output = F.relu(updated_feature)

        return output


class Local_attention(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // r, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // r, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #  x.shape   b*n, c, k
        x_q = self.q_conv(x).permute(0, 2, 1)  # b*n, k, c
        x_k = self.k_conv(x)  # b*n, c, k
        x_v = self.v_conv(x)  # b*n, c, k
        # energy = x_q @ x_k  # b*n, k, k
        energy = torch.bmm(x_q , x_k)  # b*n, k, k
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdims=True))
        x_r = x_v @ attention  # b*n, c, k
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        # Transpose to (B, N, 512)
        # x = x.permute(0, 2, 1)  # (B, 512, N) -> (B, N, 512)
        return x

class GlobalAttention(nn.Module):
    '''https://github.com/shengfly/BrainAgingNet'''
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(hidden_size, 1024)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        print('x.size', x.size())
        return x.permute(0, 2, 1, 3)
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        print('context_layer', context_layer.size())
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        print('new_context_layer_shape', new_context_layer_shape)
        context_layer = context_layer.view(*new_context_layer_shape)
        print('context_layer', context_layer.size())

        attention_output = self.out(context_layer)
        print('attention_output', attention_output.size())
        attention_output = self.proj_dropout(attention_output)
        print('attention_output', attention_output.size())
        # 在forward函数的最后添加新的线性变换
        attention_output = self.output_proj(attention_output)
        return attention_output

# https://github.com/yanx27/PointASNL/blob/master/models/pointasnl_sem_seg.py
def SampleWeights(new_point, grouped_xyz, mlps, is_training, bn_decay, weight_decay, scope, bn=True, scaled=True):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
        grouped_xyz: (batch_size, npoint, nsample, 3)
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, 1)
    """
    with torch.no_grad():
        batch_size, npoint, nsample, channel = new_point.shape()
        bottleneck_channel = max(32,channel//2)
        normalized_xyz = grouped_xyz - torch.tile(torch.unsqueeze(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
        new_point =torch.cat([normalized_xyz, new_point], axis=-1) # (batch_size, npoint, nsample, channel+3)

        transformed_feature = nn.Conv2d(new_point, bottleneck_channel * 2, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=bn, is_training=is_training,
                                             scope='conv_kv_ds', bn_decay=bn_decay, weight_decay=weight_decay,
                                             activation_fn=None)
        transformed_new_point = nn.Conv2d(new_point, bottleneck_channel, [1, 1],
                                               padding='VALID', stride=[1, 1],
                                               bn=bn, is_training=is_training,
                                               scope='conv_query_ds', bn_decay=bn_decay, weight_decay=weight_decay,
                                               activation_fn=None)

        transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
        feature = transformed_feature[:, :, :, bottleneck_channel:]

        weights = torch.matmul(transformed_new_point, transformed_feature1, transpose_b=True)  # (batch_size, npoint, nsample, nsample)
        if scaled:
            weights = weights / torch.sqrt(torch.tensor(bottleneck_channel, torch.float32))
        weights = nn.Softmax(weights, dim=-1)
        channel = bottleneck_channel

        new_group_features = torch.matmul(weights, feature)
        new_group_features = torch.reshape(new_group_features, (batch_size, npoint, nsample, channel))
        for i, c in enumerate(mlps):
            activation = nn.ReLU() if i < len(mlps) - 1 else None
            new_group_features = nn.Conv2d(new_group_features, c, [1, 1],
                                               padding='VALID', stride=[1, 1],
                                               bn=bn, is_training=is_training,
                                               scope='mlp2_%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay,
                                               activation_fn=activation)
        new_group_weights = nn.Softmax(new_group_features, dim=2)  # (batch_size, npoint,nsample, mlp[-1)
        return new_group_weights
# It is expected to be used to change the position of Point Transformer (it can also be considered to be used in other places to enhance point cloud features)
def AdaptiveSampling(group_xyz, group_feature, num_neighbor, is_training, bn_decay, weight_decay, scope, bn):
    with torch.no_grad():
        nsample, num_channel = group_feature.shape[-2:]
        if num_neighbor == 0:
            new_xyz = group_xyz[:, :, 0, :]
            new_feature = group_feature[:, :, 0, :]
            return new_xyz, new_feature

        shift_group_xyz = group_xyz[:, :, :num_neighbor, :]
        shift_group_points = group_feature[:, :, :num_neighbor, :]
        sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [32, 1 + num_channel], is_training, bn_decay, weight_decay, scope, bn)
        new_weight_xyz = torch.tile(torch.unsqueeze(sample_weight[:,:,:, 0], dim=-1), (1, 1, 1, 3))
        new_weight_feture = sample_weight[:,:,:, 1:]
        new_xyz = torch.sum(torch.mul(shift_group_xyz, new_weight_xyz), dim=2)
        new_feature = torch.sum(torch.mul(shift_group_points, new_weight_feture), dim=2)

    return new_xyz, new_feature


class PointAttentionNet(nn.Module):
    def __init__(self, args):
        super(PointAttentionNet, self).__init__()
        self.args = args
        args.att_heads = 2
        self.attn1 = nn.MultiheadAttention(64, args.att_heads)
        self.attn2 = nn.MultiheadAttention(64, args.att_heads)
        self.attn3 = nn.MultiheadAttention(128, args.att_heads)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, args.number_classes)

    def perform_att(self, att, x):
        residual = x
        # print(x.shape)

        x_T = x.transpose(1, 2)
        x_att, _ = att(x_T,x_T,x_T)
        x = x_att.transpose(1,2)
        # x, _ = att(x,x,x)

        return x + residual

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.perform_att(self.attn1, self.conv2(x))))
        x = F.relu(self.bn3(self.perform_att(self.attn2, self.conv3(x))))
        x = F.relu(self.bn4(self.perform_att(self.attn3, self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    
class Local_attention(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // r, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // r, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # spatial attention
        # 旨在學習不同空間位置的重要性權重。特徵位置的重要性
        #  x.shape   b*n, c, 
        # q 表示目前位置需要注意的內容
        x_q = self.q_conv(x).permute(0, 2, 1)  # b*n, k, c  對輸入特徵 x 做一些線性變換,使得 q、k、v 有不同的特徵表示
        x_k = self.k_conv(x)  # b*n, c, k  k 表示其他位置可以提供 q 什麼樣的訊息
        x_v = self.v_conv(x)  # b*n, c, k  v 表示其他位置的實際特徵值
        # energy = x_q @ x_k  # b*n, k, k 
        energy = torch.bmm(x_q , x_k)  # b*n, k, k
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdims=True))
        x_r = x_v @ attention  # b*n, c, k
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        # Transpose to (B, N, 512)
        # x = x.permute(0, 2, 1)  # (B, 512, N) -> (B, N, 512)
        return x