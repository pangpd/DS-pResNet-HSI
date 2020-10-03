# -*- coding: utf-8 -*-
"""
@Date:   2020/9/14 9:41
@Author: Pangpd
@FileName: read_tif.py
@IDE: PyCharm
@Description: 
"""
import copy
import os

import scipy.io as sio
import skimage.io
import numpy as np
import sys

from utils.show_maps import show_label

np.set_printoptions(linewidth=800)
np.set_printoptions(threshold=sys.maxsize)
# imgpath = r'D:\UseTools\OneDrive\codes\New-Research\data\indianpines_ts.tif'
# imggt = skimage.io.imread(imgpath)
# sio.savemat(r"D:\UseTools\OneDrive\codes\New-Research\data\indianpines_ts.mat", {'imggt': imggt})

dataset = 'IP'
root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = os.path.join(root_path, 'data')
path = r'D:\\UseTools\OneDrive\codes\New-Research\\figures'
save_path = os.path.join(path, "disjoint_" + dataset)  # 保存路径


ts = sio.loadmat(os.path.join(data_path, 'indianpines_ts.mat'))['imggt']

changes = [0, 2, 3, 5, 6, 8, 10, 11, 12, 14, 1, 4, 7, 9, 13, 15, 16]
y_train = copy.deepcopy(ts)

for i, val in enumerate(changes): y_train[ts == i] = val
show_label(y_train, y_train, 16, save_path)