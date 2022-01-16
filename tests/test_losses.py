import unittest
from models.utils import get_prebuilt_model
from itertools import product, combinations, permutations
from data.data import OxpetDataset, fast_loader
import os
import torch

class TestLosses(unittest.TestCase):

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


    def test_novel_weights(self):
        from criterion.loss_functions import NormalisedDynamicCombinedLoss

        loss_dict = {
            'bb': torch.nn.L1Loss(),
            'reg': torch.nn.MSELoss()
        }
        loss = NormalisedDynamicCombinedLoss(loss_dict, sf={'bb': 0.0032}, temperature=1.0, frequency=5)

        inputs = {
            'bb': torch.rand((4, 4)),
            'reg': torch.rand((4, 4))
        }
        for key in inputs:
            inputs[key].requires_grad = True
        outputs = {
            'bb': torch.rand((4, 4)),
            'reg': torch.rand((4, 4))
        }
        for key in outputs:
            outputs[key].requires_grad = True
        for i in range(10):
            loss_ = loss(inputs, outputs)