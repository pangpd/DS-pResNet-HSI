import datetime
import os
import sys

import torch
import logging
from models.GaoEtAl import GaoEtAl
from models.HeEtAl import HeEtAl
from models.HuEtAl import HuEtAl
from models.paoLetti import pResNet as paoLetti
from models.LeeEtAl import LeeEtAl as LeeEtAl
from models.SSRN import SSRN


def get_params(dataset, model_name):
    use_cuda = torch.cuda.is_available()
    # if use_cuda: torch.backends.cudnn.benchmark = True
    params = {}
    if model_name == 'gao':
        params['learn_rate'] = 0.01
        params['batch_size'] = 64
        params['optimizer:'] = 'Adam'
        if dataset == 'IP':
            params['g'] = 32
            params['components'] = 25
            params['spatial_size'] = 11
            model = GaoEtAl(params['components'], 16)
        elif dataset == 'PU':
            params['g'] = 32
            params['components'] = 20
            params['spatial_size'] = 13
            model = GaoEtAl(params['components'], 9)
        elif dataset == 'SA':
            params['g'] = 32
            params['components'] = 25
            params['spatial_size'] = 7
            model = GaoEtAl(params['components'], 16)
        elif dataset == 'KSC':  # 论文中没有使用该数据集,暂时按照这个参数去设置
            params['g'] = 32
            params['components'] = 25
            params['spatial_size'] = 9
            model = GaoEtAl(params['components'], 13)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), params['learn_rate'])
        return params, model, optimizer

    elif model_name == 'pao':
        params['depth'] = 32
        params['alpha'] = 50
        params['inplanes'] = 16
        params['components'] = None
        params['learn_rate'] = 0.1
        params['batch_size'] = 100
        params['momentum'] = 0.9
        params['weight_decay'] = 1e-4
        params['optimizer:'] = 'SGD'
        if dataset == 'IP':
            params['spatial_size'] = 11
            model = paoLetti(32, 48, 16, 200, params['spatial_size'], 16, bottleneck=True)
        elif dataset == 'PU':
            params['spatial_size'] = 13
            model = paoLetti(32, 48, 9, 103, params['spatial_size'], 16, bottleneck=True)
        elif dataset == 'SA':
            model = paoLetti(32, 48, 16, 224, params['spatial_size'], 16, bottleneck=True)
        elif dataset == 'KSC':
            params['spatial_size'] = 9
            model = paoLetti(32, 48, 13, 176, params['spatial_size'], 16, bottleneck=True)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learn_rate'], momentum=params['momentum'],
                                    weight_decay=params['weight_decay'])

        return params, model, optimizer

    elif model_name == 'hu':
        params['spatial_size'] = 1
        params['components'] = None
        params['learn_rate'] = 0.3  # 论文中没有给出，自己根据实验推断
        params['batch_size'] = 100
        params['components'] = None
        params['optimizer:'] = 'SGD'
        if dataset == 'IP':
            model = HuEtAl(200, 16)
        if dataset == 'PU':
            model = HuEtAl(103, 9)
        if dataset == 'SA':
            model = HuEtAl(224, 16)
        if dataset == 'KSC':
            model = HuEtAl(176, 13)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), params['learn_rate'])
        return params, model, optimizer

    elif model_name == 'he':
        # params['spatial_size'] = 7 #论文中给出的参数
        params['components'] = None
        params['learn_rate'] = 0.01
        params['batch_size'] = 40
        params['momentum'] = 0.9
        params['weight_decay'] = 0.01
        params['optimizer'] = 'Adagrad'
        if dataset == 'IP':
            params['spatial_size'] = 11
            model = HeEtAl(200, 16, params['spatial_size'])
        elif dataset == 'PU':
            params['spatial_size'] = 13
            model = HeEtAl(103, 9, params['spatial_size'])
        elif dataset == 'SA':
            model = HeEtAl(224, 16, params['spatial_size'])
        elif dataset == 'KSC':
            params['spatial_size'] = 9
            model = HeEtAl(176, 13, params['spatial_size'])
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=params['learn_rate'],
                                        weight_decay=params['weight_decay'])

        return params, model, optimizer
    elif model_name == 'lee':
        # params['spatial_size'] = 5 #论文中邻域大小
        params['components'] = None
        params['learn_rate'] = 0.001
        params['batch_size'] = 10
        params['components'] = None
        params['optimizer:'] = 'SGD'
        params['momentum'] = 0.9
        params['weight_decay'] = 1e-4
        if dataset == 'IP':
            params['spatial_size'] = 11
            model = LeeEtAl(200, 16, 128)  # 不同于其他模型，模型的输入要加一个宽度参数
        if dataset == 'PU':
            params['spatial_size'] = 13
            model = LeeEtAl(103, 9, 128)
        if dataset == 'SA':
            model = LeeEtAl(224, 16, 192)
        if dataset == 'KSC':
            model = LeeEtAl(176, 13, 192)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learn_rate'], momentum=params['momentum'],
                                    weight_decay=params['weight_decay'])
        return params, model, optimizer

    elif model_name == 'SSRN':
        # params['spatial_size'] = 9 #论文中邻域大小
        params['components'] = None
        params['learn_rate'] = 0.01
        params['components'] = None
        params['optimizer:'] = 'SGD'
        params['momentum'] = 0.9
        params['weight_decay'] = 1e-4
        if dataset == 'IP':
            params['spatial_size'] = 11
            params['batch_size'] = 64
            model = SSRN(200, 16)
        if dataset == 'PU':
            params['spatial_size'] = 13
            params['batch_size'] = 32
            model = SSRN(103, 9)
        if dataset == 'SA':  # 暂时不用这个数据集
            model = SSRN(224, 16)
        if dataset == 'KSC':
            params['batch_size'] = 128
            params['spatial_size'] = 9
            model = SSRN(176, 13)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learn_rate'], momentum=params['momentum'],
                                    weight_decay=params['weight_decay'])

        return params, model, optimizer

    else:
        print("NO MODEL")
        exit()