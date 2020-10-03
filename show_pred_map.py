# -*- coding: utf-8 -*-
"""
@Date:   2020/3/11 12:27
@Author: Pangpd
@FileName: show_pred_map.py
@IDE: PyCharm
@Description:用来测试显示gt图和预测结果图
"""
import os
import sys
import torch
import numpy as np
from utils.get_model_param import get_params
from utils.show_maps import *
from utils.data_preprocess import loadData, createImageCubes
from utils.hyper_pytorch import HyperData
from models.DS_pResNet import pResNet as DS_pResNet
from utils.start import predict

np.set_printoptions(linewidth=400000)
np.set_printoptions(threshold=sys.maxsize)


def start(dataset, best_model_path, model, params):
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    dataset_path = os.path.join(root_path, 'data')  # 数据集路径
    save_path = os.path.join(root_path, 'final_pre_map')  # 保存结果图的路径
    data, labels, nums, label_names = loadData(dataset_path, dataset)
    n_bands = data.shape[-1]

    # removeZeroLabels为False使用的是全部数据，为True时使用的是非0标签的数据
    patchesData, patchesLabels = createImageCubes(data, labels, windowSize=params['spatial_size'],
                                                  removeZeroLabels=False)
    test_hyper = HyperData((np.transpose(patchesData, (0, 3, 1, 2)).astype("float32"), patchesLabels), None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=params['batch_size'], shuffle=False, **kwargs)
    if params['model_name'] == 'my':
        model = DS_pResNet(params['nums_ResUnit'], params['alpha'], nums, n_bands, params['spatial_size'],
                           params['inplanes'])

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    checkpoint = torch.load(best_model_path + "/best_model.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    predict_values = np.argmax(predict(test_loader, model, use_cuda), axis=1)  # 预测结果(地物种类是从0开始标的)
    predict_values = np.array(list(map(lambda x: x + 1, predict_values)))
    show_pred(predict_values, labels, nums, dataset, save_path, removeZeroLabels=True)


if __name__ == '__main__':
    dataset = 'IP'
    model_name = 'my'
    best_model_path = 'E:\Pangpd\My-Research\logs\我的模型试验记录\Indian pines\IP_11x11_初38_全程0.01\Experiment_9'  # my_ip

    # best_model_path = 'E:\Pangpd\My-Research\logs\我的模型试验记录\PU\PU_13x13_初42_全程0.01_残2\Experiment_8'  # my_pu

    # best_model_path = 'E:\Pangpd\My-Research\logs\我的模型试验记录\KSC\KSC_9x9_初32_全程0.01\Experiment_1' #my_ksc

    if model_name == 'my':
        params = {}
        if dataset == 'IP':
            params['batch_size'] = 64
            params['spatial_size'] = 11
            params['inplanes'] = 38
            params['nums_ResUnit'] = 3

        if dataset == 'PU':
            params['batch_size'] = 128
            params['spatial_size'] = 13
            params['inplanes'] = 42
            params['nums_ResUnit'] = 2

        if dataset == 'KSC':
            params['batch_size'] = 32
            params['spatial_size'] = 9
            params['inplanes'] = 32
            params['nums_ResUnit'] = 3
        params['model_name'] = 'my'
        params['alpha'] = 48
        model = None
    else:
        params, model, optimizer = get_params(dataset, model_name)
        params['model_name'] = model_name
    start(dataset, best_model_path, model, params)
