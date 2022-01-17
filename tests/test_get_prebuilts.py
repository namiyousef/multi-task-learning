import unittest
from models.utils import get_prebuilt_model
from itertools import product, combinations, permutations
from data.data import OxpetDataset, fast_loader
import os
import torch

class TestGetPrebuilts(unittest.TestCase):

    def test_get_models(self):
        encoder = 'resnet34'
        decoders = ['seg', 'bb', 'class']
        losses = {
            'seg': 'SegDiceLoss',
            'bb':'L1Loss',
            'class':'CrossEntropyLoss'
        }
        configurations = []
        for r in range(1, len(decoders)+1):
            for config in permutations(decoders, r):
                configurations.append('+'.join(config))

        for config in configurations:
            print(config)
            get_prebuilt_model(encoder=encoder, decoders=config, losses='SegDiceLoss+L1Loss+CrossEntropyLoss')

    def test_random_weights(self):
        encoder = 'resnet34'
        tasks = ['seg', 'bb', 'class']
        decoders = '+'.join(tasks)
        losses = 'SegDiceLoss+0.0032*L1Loss+CrossEntropyLoss'
        model, loss = get_prebuilt_model(encoder, decoders, losses, weights='dynamic::1::1')

        data_dir = '../datasets/data_new/'
        splits = ['train', 'test', 'val']
        datasets = [OxpetDataset(os.path.join(data_dir, split), tasks) for split in splits]
        batch_size = [32, 32, 32]
        trainloader, valloader, testloader = [fast_loader(dataset, batch_size) for dataset, batch_size in
                                              zip(datasets, batch_size)]




    def test_dynamic_weights(self):

        from criterion.loss_functions import DynamicCombinedLoss

        loss_dict = {
            'bb': torch.nn.L1Loss(),
            'reg': torch.nn.MSELoss()
        }
        loss = DynamicCombinedLoss(loss_dict, sf={'bb':0.0032}, temperature=1.0, frequency=5)

        inputs = {
            'bb':torch.rand((4,4)),
            'reg':torch.rand((4,4))
        }
        for key in inputs:
            inputs[key].requires_grad = True
        outputs = {
            'bb':torch.rand((4,4)),
            'reg':torch.rand((4,4))
        }
        for key in outputs:
            outputs[key].requires_grad = True
        for i in range(10):
            print(i)
            loss_ = loss(inputs, outputs)


    def test_simple_weights(self):
        pass