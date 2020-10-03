# -*- coding: utf-8 -*-
"""
@Date:   2020/9/14 11:04
@Author: Pangpd
@FileName: test_joint.py
@IDE: PyCharm
@Description: 
"""
import os
import sys

import numpy as np

from utils.data_preprocess import loadData, createImageCubes
from utils.disjoint_sample_split_data import load_disjoint_data, load_disjoint_hyper
from utils.show_maps import show_label

np.set_printoptions(linewidth=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

dataset = 'PU'
path = r'D:\\UseTools\OneDrive\codes\New-Research\\figures'
save_path = os.path.join(path, "disjoint_" + dataset)  # 保存路径

data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'data')
train_loader, test_loader, val_loader, num_classes, bands = load_disjoint_hyper(data_path, dataset, spatial_size=7)
