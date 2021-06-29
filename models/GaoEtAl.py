# -*- coding: utf-8 -*-
"""
@Date:   2020/4/22 23:23
@Author: Pangpd
@FileName: GaoEtAl.py
@IDE: PyCharm
@Description:
    Convolutional neural network for spectral–spatial classification
    of hyperspectral images
"""

import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F

def conv_BR(in_planes, out_planes, kernel_size):
    "convolution with padding and BN,Relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class GaoEtAl(nn.Module):

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
    #         init.kaiming_uniform_(m.weight)
    #         m.bias.data.fill_(0)

    def __init__(self, input_channels, n_classes):
        super(GaoEtAl, self).__init__()
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.final_conv7 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(n_classes)

        self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # self.features_size = self._get_final_flattened_size()
        # self.apply(self.weight_init)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = self.conv1(x)  # 3x3
        x = F.relu(self.bn1(x))
        x = self.conv2(x)  # 1x1  后层输入的开始
        x = F.relu(self.bn2(x))

        x_dens1 = self.conv3(x)  # 1x1
        x_dens1 = F.relu(self.bn3(x_dens1))

        x_dens1 = self.conv4(x_dens1)  # 1x1
        x_dens1 = F.relu(self.bn4(x_dens1))
        x_dens1 = torch.cat((x, x_dens1), 1)  # 第一个dens块的输出

        x_dens2 = self.conv5(x_dens1)  # 1x1
        x_dens2 = F.relu(self.bn5(x_dens2))
        x_dens2 = self.conv6(x_dens2)  # 1x1
        x_dens2 = F.relu(self.bn6(x_dens2))
        x_dens2 = torch.cat((x, x_dens1, x_dens2), 1)  # 第二个dens块的输出

        x = self.final_conv7(x_dens2)
        x = F.relu(self.bn7(x))

        x = self.avgpool(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x
