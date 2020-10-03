# -*- coding: utf-8 -*-
"""
@Date:   2020/3/8 16:15
@Author: Pangpd
@FileName: testSplitdata.py
@IDE: PyCharm
@Description: 
"""
import random
import sys
from math import ceil

import numpy as np

# -*- coding: utf-8 -*-
import scipy
import torch
from sklearn.model_selection import train_test_split
import utils.data_preprocess
from utils import data_preprocess

"""
@Date:   2020/2/21 9:48
@Author: Pangpd
@FileName: test.py
@IDE: PyCharm
@Description: 用来测试编写的函数
"""
import numpy as np
import os

np.set_printoptions(linewidth=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels, percent, splitdset="custom", rand_state=1000):
    splitdset = "sklearn"
    if splitdset == "sklearn":
        return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        pixels_number = np.unique(labels, return_counts=1)[1]
        train_set_size = [int(np.ceil(a * percent)) for a in pixels_number]
        tr_size = int(sum(train_set_size))
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size] + list(pixels.shape)[1:])
        sizete = np.array([te_size] + list(pixels.shape)[1:])
        train_x = np.empty((sizetr));
        train_y = np.empty((tr_size), dtype=int);
        test_x = np.empty((sizete));
        test_y = np.empty((te_size), dtype=int)
        trcont = 0;
        tecont = 0;
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
            for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                if cont < train_set_size[cl]:
                    train_x[trcont, :, :, :] = a
                    train_y[trcont] = b
                    trcont += 1
                else:
                    test_x[tecont, :, :, :] = a
                    test_y[tecont] = b
                    tecont += 1
        train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
        test_x, test_y = random_unison(test_x, test_y, rstate=rand_state)
        return train_x, test_x, train_y, test_y


# def split_data_percent(pixels, labels, train_samples, val_samples, rand_state=None):
#     train_set_size = []  # 存储每类地物训练样本数
#     for cl in np.unique(labels):
#         pixels_cl = len(pixels[labels == cl])  # 第i类地物样本总数
#         train_pixels_cl = min(ceil(pixels_cl * 0.3), train_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
#         train_set_size.append(train_pixels_cl)  # 存储每类地物的训练样本数
#
#     val_set_size = [ceil(i * val_samples) for i in train_set_size]  # 存储每类地物的验证样本数
#
#     pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数
#
#     tr_size = int(sum(train_set_size))
#     val_size = int(sum(val_set_size))
#     te_size = int(sum(pixels_number)) - tr_size - val_size
#     sizetr = np.array([tr_size] + list(pixels.shape)[1:])
#     sizeval = np.array([val_size] + list(pixels.shape)[1:])
#     sizete = np.array([te_size] + list(pixels.shape)[1:])
#
#     X_train = np.empty((sizetr))
#     y_train = np.empty((tr_size), dtype=int)
#     X_val = np.empty((sizeval))
#     y_val = np.empty((val_size), dtype=int)
#     X_test = np.empty((sizete))
#     y_test = np.empty((te_size), dtype=int)
#     trcont = 0;
#     valcont = 0;
#     tecont = 0;
#
#     for cl in np.unique(labels):
#         pixels_cl = pixels[labels == cl]
#         labels_cl = labels[labels == cl]
#         pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
#         for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
#             if cont < train_set_size[cl]:
#                 X_train[trcont, :, :, :] = a
#                 y_train[trcont] = b
#                 trcont += 1
#             elif cont < train_set_size[cl] + val_set_size[cl]:
#                 X_val[valcont, :, :, :] = a
#                 y_val[valcont] = b
#                 valcont += 1
#             else:
#                 X_test[tecont, :, :, :] = a
#                 y_test[tecont] = b
#                 tecont += 1
#
#     X_train, y_train = random_unison(X_train, y_train, rstate=rand_state)
#     # X_test, y_test = random_unison(X_test, y_test, rstate=rand_state)
#     X_val, y_val = random_unison(X_val, y_val, rstate=rand_state)
#     return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'data')
    data, labels, num_class, label_names = data_preprocess.loadData(data_path, 'KSC')
    pixels, labels = data_preprocess.createImageCubes(data, labels, windowSize=5, removeZeroLabels=True)
    seed = 1331
    train_samples = 50
    val_samples = 0.5
    class_no = 14
    # for i in range(1):
    #     rand = seed + i

    X_train, y_train, X_val, y_val, X_test, y_test = utils.data_preprocess.split_data_percent(pixels, labels,
                                                                                              train_samples,
                                                                                              val_samples,
                                                                                              rand_state=1)
    # for i in range(class_no):
    #     print('--------------', i, '----------------')
    #     print("训练集样本数:", len(y_train[y_train == i]))
    #     print('验证集样本数:', len(y_val[y_val == i]))
    #     print('测试集样本数:', len(y_test[y_test == i]))

    # print('------------------------------')
    # print("训练集样本数:", len(y_train))
    # print('验证集样本数:', len(y_val))
    # print('测试集样本数:', len(y_test))
    # print('-------------------------')
    # print(y_train)
    # print(y_val)
    # print(y_test)
