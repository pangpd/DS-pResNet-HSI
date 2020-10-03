# -*- coding: utf-8 -*-
"""
@Date:   2020/6/30 10:31
@Author: Pangpd
@FileName: testCreateImageCubes.py
@IDE: PyCharm
@Description: 
"""
import os

from utils import data_preprocess
import numpy as np


# padding
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 创建像素立方体
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    num_labels = np.count_nonzero(y[:, :])  # 标签样本总数
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchIndex = 0
    if removeZeroLabels == True:
        patchesData = np.zeros((num_labels, windowSize, windowSize, X.shape[2]), dtype='float32')
        patchesLabels = np.zeros(num_labels)

        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                if y[r - margin, c - margin] > 0:
                    patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                    import matplotlib.pyplot as plt
                    plt.imshow(patch[:, :, 1])
                    plt.show()
                    patchesData[patchIndex, :, :, :] = patch
                    patchesLabels[patchIndex] = y[r - margin, c - margin]
                    patchIndex = patchIndex + 1

    if removeZeroLabels == False:  # 表示使用全部像素
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype="float32")
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchIndex = patchIndex + 1
    patchesLabels -= 1

    return patchesData, patchesLabels.astype("int")


if __name__ == '__main__':
    data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'data')
    data, labels, num_class, label_names = data_preprocess.loadData(data_path, 'IP')
    pixels, labels = createImageCubes(data, labels, windowSize=9, removeZeroLabels=True)
