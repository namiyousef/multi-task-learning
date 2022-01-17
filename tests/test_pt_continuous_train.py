import unittest
import random
from data.data import get_dataloader, get_dataset
import os
import torch

def _get_loaders():
    os.chdir('..')

    task_config = {
        "Model": 'mtl model',
        "Tasks": {
            # "Class":2,
            "Segmen": 1,
            "BB": 4
        },
        "Loss Lambda": {
            # "Class":1,
            "Segmen": 1,
            "BB": 1}

    }
    train_dataset = get_dataset(task_config, "train")
    batch_size = 32
    train_dataloader = get_dataloader(train_dataset, batch_size)

    return train_dataloader

class TestPTContinuousTrain(unittest.TestCase):

    """def test_train_break(self):

        for i, batch in enumerate(train_dataloader):
            true_batch = batch


        torch.manual_seed(0) # manually change the seed, e.g. replicate new run"""

    def test_recreate_loader_rng(self):
        """checks that dataloader shuffles the data in the exact same way regardless of the epoch number
        """
        train_loader = _get_loaders()
        batch = [0] * 2
        for epoch in range(2):
            a = torch.get_rng_state()

            for i, data in enumerate(train_loader):
                if i == 0:
                    batch[epoch] = data
                    break
            break

        torch.set_rng_state(a)
        for i, data in enumerate(train_loader):
            if i == 0:
                test_batch = data
                break

        for d1, d2 in zip(batch[0].values(), test_batch.values()):
            assert torch.equal(d1, d2)