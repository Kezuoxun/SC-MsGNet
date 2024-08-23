import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from core.modules import ResidualConv, Upsample, Upsample_interpolate
from torchvision import transforms


__all__ = ['Res2Net', 'res2net50']

model_urls = {
    # 'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000, output_stride=16, rgb_only=False):
        self.current_stride = 4
        self.output_stride = output_stride
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        if rgb_only:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(4, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Sequential(nn.Conv2d(self.inplanes, 512, 3, 1, 1, bias=False),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.current_stride == self.output_stride:
                stride = 1
            else:
                self.current_stride = self.current_stride * stride
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale)) 
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)
        x = self.maxpool(x2s)

        x4s = self.layer1(x)
        x8s = self.layer2(x4s)
        x16s = self.layer3(x8s)
        x = self.layer4(x16s)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x2s, x4s, x8s, x16s, x


def res2net50(rgb_only=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if rgb_only:
        model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, rgb_only=True)
    else:
        model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)
    return model


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
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


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

########################################################################

# Used for predict 6DCM (6-channel)
class SSCM_net(nn.Module): 
    def __init__(self, fcdim=512, s8dim=128, s4dim=64, s2dim=32, t0_dim=32, t1_dim=32, raw_dim2=16, class_num=41):
        super(SSCM_net, self).__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        self.conv16s = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True)
        )

        self.conv8s = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(128 + 64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )

        self.conv1s = nn.Sequential(
            nn.Conv2d(4 + 32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, class_num+6, 1, 1)
        )

    def forward(self, input):
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)

        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])

        fm1 = self.conv8s(torch.cat([fm1, x8s], 1))
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])

        fm1 = self.conv4s(torch.cat([fm1, x4s], 1))
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv1s(torch.cat([fm1, input], 1))

        # out1 = self.tail1(x)
        # out2 = self.tail2(x)
        # out = torch.cat((out1, out2), 1)

        return fm1


# Used for predict offset between rear-side cloud and front-side cloud (1-channel)
class SSD_net(nn.Module):
    def __init__(self, fcdim=512, s8dim=128, s4dim=64, s2dim=32, t0_dim=32, t1_dim=32, raw_dim2=16, class_num=41):
        super(SSD_net, self).__init__() 

        self.encoder = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        self.conv16s = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True)
        )

        self.conv8s = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(128 + 64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )

        self.conv1s = nn.Sequential(
            nn.Conv2d(4 + 32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, class_num+1, 1, 1)    # You can change predicted channel num at this line 
        )

    def forward(self, input):  # auto-encoder
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)

        # decoder
        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])

        fm1 = self.conv8s(torch.cat([fm1, x8s], 1))
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])

        fm1 = self.conv4s(torch.cat([fm1, x4s], 1))
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv1s(torch.cat([fm1, input], 1))

        return fm1


# Same as SSD, but use AFF as Skip connnection
class SSD_net_AFF(nn.Module):
    def __init__(self, fcdim=512, s8dim=128, s4dim=64, s2dim=32, t0_dim=32, t1_dim=32, raw_dim2=16, class_num=41):
        super(SSD_net_AFF, self).__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        self.conv16s = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True)
        )

        self.fusion512 = AFF(512)
        self.conv8s = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )

        # x4s->64
        self.fusion256 = AFF(256)
        self.conv4s = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(128 + 64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )

        self.conv1s = nn.Sequential(
            nn.Conv2d(4 + 32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, class_num+1, 1, 1)
        )

    def forward(self, input):
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)

        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])

        fm1 = self.fusion512(fm1, x8s)
        fm1 = self.conv8s(fm1)
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])

        fm1 = self.fusion256(fm1, x4s)
        fm1 = self.conv4s(fm1)
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv1s(torch.cat([fm1, input], 1))

        return fm1


# Used for predict point cloud and vector (both 3-channels)
class SSC_net(nn.Module):
    def __init__(self, fcdim=512, s8dim=128, s4dim=64, s2dim=32, t0_dim=32, t1_dim=32, raw_dim2=16, class_num=41):
        super(SSC_net, self).__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        self.conv16s = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True)
        )

        self.conv8s = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(128 + 64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )

        self.conv1s = nn.Sequential(
            nn.Conv2d(4 + 32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, class_num + 3, 1, 1)
        )

    def forward(self, input):
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)

        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])

        fm1 = self.conv8s(torch.cat([fm1, x8s], 1))
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])

        fm1 = self.conv4s(torch.cat([fm1, x4s], 1))
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv1s(torch.cat([fm1, input], 1))

        # out1 = self.tail1(x)
        # out2 = self.tail2(x)
        # out = torch.cat((out1, out2), 1)

        return fm1