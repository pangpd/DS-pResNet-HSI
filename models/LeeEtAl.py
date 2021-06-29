# -*- coding: utf-8 -*-
"""
@Date:   2020/4/23 13:15
@Author: Pangpd
@FileName: LeeEtAl.py
@IDE: PyCharm
@Description:
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
"""
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


class LeeEtAl(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.normal_(m.weight, mean=0, std=0.01)
            init.constant_(m.bias, 1)

    def __init__(self, in_channels, n_classes, width):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_5x5 = nn.Conv2d(in_channels, width, kernel_size=5, padding=0, stride=1)
        # nn.init.normal_(self.conv_5x5.weight, mean=0, std=0.01)
        # nn.init.constant_(self.conv_5x5.bias, 1)

        self.conv_3x3 = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, stride=1)
        self.pool_1 = nn.MaxPool2d((3, 3), stride=3)
        self.conv_1x1 = nn.Conv2d(in_channels, width, kernel_size=1, padding=2, stride=1)
        self.pool_2 = nn.MaxPool2d((5, 5), stride=5)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(width * 3, width, (1, 1))
        self.conv2 = nn.Conv2d(width, width, (1, 1))
        self.conv3 = nn.Conv2d(width, width, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(width, width, (1, 1))
        self.conv5 = nn.Conv2d(width, width, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(width, width, (1, 1))
        self.conv7 = nn.Conv2d(width, width, (1, 1))

        self.conv8 = nn.Conv2d(width, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(width * 3)
        self.lrn2 = nn.LocalResponseNorm(width)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)
        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_5x5 = self.conv_5x5(x)
        x_3x3 = self.pool_1(self.conv_3x3(x))
        x_1x1 = self.pool_2(self.conv_1x1(x))
        # cat -> ReLU+LRN
        x = torch.cat((x_5x5, x_3x3, x_1x1), 1)
        x = F.relu(self.lrn1(x))

        # 1x1 conv -> ReLU+LRN
        x = self.conv1(x)
        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        # 1x1Conv -> ReLU -> 1x1 -> SUM -> ReLU
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        # 1x1Conv -> ReLU -> 1x1 -> SUM -> ReLU
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        # Fianl three 1x1 Conv
        # 1x1Conv -> ReLU -> Dropout -> 1x1Conv -> ReLU -> Dropout
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        return x
