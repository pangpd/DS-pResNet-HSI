# -*- coding: utf-8 -*-
"""
@Date:   2020/7/11 9:33
@Author: Pangpd
@FileName: test_my_model.py
@IDE: PyCharm
@Description: 
"""
import torch

checkpoint = torch.load(r"D:\UseTools\OneDrive\codes\New-Research\logs\9x9_64_32_200\Experiment_1\best_model.pth.tar")
best_acc = checkpoint['best_acc']
start_epoch = checkpoint['epoch']

print(checkpoint)