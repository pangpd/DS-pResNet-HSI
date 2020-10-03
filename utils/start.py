# -*- coding: utf-8 -*-
"""
@Date:   2020/4/27 0:06
@Author: Pangpd
@FileName: start.py
@IDE: PyCharm
@Description: 
"""
import os
import sys
import time

import numpy as np
from utils import evaluate
import torch
import torch.nn.parallel


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))


def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            losses[batch_idx] = criterion(outputs, targets).item()
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(test_loader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda: inputs = inputs.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))  # 1-149:0.1，150-200:0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
