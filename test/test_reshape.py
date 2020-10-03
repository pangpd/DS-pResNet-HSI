# -*- coding: utf-8 -*-
"""
@Date:   2020/6/3 21:20
@Author: Pangpd
@FileName: test_reshape.py
@IDE: PyCharm
@Description: 
"""
from math import ceil

import numpy as np

# a = np.zeros((1, 3, 1, 5, 5))
# a = np.arange(0, 90, 1)
# a = a.reshape(2, 5, 1, 3, 3)
# print(a)
# b = np.squeeze(a)
# print('-----------------')
# print(b)
# b = np.reshape(a, (a.shape[0], a.shape[2], a.shape[1], a.shape[3], a.shape[4]))
# print('--------------------------')


a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a[2:])
#b = [ceil(i * 0.5) for i in a]
# l = len(a)
# l_1 = ceil(l * 0.3)
# l_2 = ceil(l - l_1)
#
# print(a[:l_1])
# print(a[l_1:])
# print(a[:-3])
