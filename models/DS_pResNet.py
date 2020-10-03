# -*- coding: utf-8 -*-
"""
@Date:   2020/4/8 20:20
@Author: Pangpd
@FileName: DS_pResNet.py
@IDE: PyCharm
@Description: pResnet
"""
import math

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)  # 向上取整
            #self.conv1 = SeparableConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            #self.conv1 = SeparableConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 可分离卷积
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 可分离卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, padding=1, bias=False)  # same可分离卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)  # 标准卷积
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                            featuremap_size[1]).cuda())
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class pResNet(nn.Module):


    def __init__(self, nums_ResUnit, alpha, num_classes, n_bands, spatial_size, inplanes):
        '''
        :param nums_ResUnit: 残差单元数
        :param alpha: alpha因子
        :param num_classes: 分类数
        :param n_bands: 输入波段数
        :param spatial_size: 邻域大小
        :param inplanes: 初始维度
        '''

        super(pResNet, self).__init__()

        block = BasicBlock
        self.inplanes = inplanes
        self.addrate = alpha / (nums_ResUnit * 1.0)
        self.input_featuremap_dim = self.inplanes
        self.featuremap_dim = self.input_featuremap_dim

        # first 1x1 conv
        self.conv1 = nn.Conv2d(n_bands, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        # Res moudle
        self.layer1 = self.pyramidal_make_layer(block, nums_ResUnit)  # n 表示残差单元数量

        # End
        self.final_featuremap_dim = self.input_featuremap_dim
        # self.bn_final1 = nn.BatchNorm2d(self.final_featuremap_dim)
        # self.relu_final = nn.ReLU(inplace=True)

        # final 1x1 conv
        self.conv_final = nn.Conv2d(self.final_featuremap_dim, num_classes, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(num_classes)
        self.relu_final = nn.ReLU(inplace=True)

        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.apply(self.weight_init)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, nums_ResUnit, stride=1):
        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride))

        for i in range(1, nums_ResUnit):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)), int(round(temp_featuremap_dim)), 2))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x
