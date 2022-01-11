from data.data import OxfordPetDataset, OxpetDataset, OxpetDatasetIterable
from torch.utils.data import DataLoader, IterableDataset, BatchSampler, SequentialSampler, RandomSampler, SubsetRandomSampler, Sampler
import unittest
import torch
import time
import h5py
import math

def _load_data(index, dir):
    with  h5py.File(dir, 'r') as file:
        key = list(file.keys())[0]
        elems = file[key][index]
        return elems

class TestLoader(unittest.TestCase):

    """def test_loaders_equal(self):
        split = 'train'
        loader1 = OxfordPetDataset(config={'Tasks':{
            'Class':0, 'Segmen':0, 'BB':0
        }},split=split, mini_batch_size=0)

        dir_path = '../datasets/data_new/train/'
        loader2 = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        assert len(loader2) == len(loader1)
        batch_size = 32
        dloader1 = DataLoader(loader1, batch_size, shuffle=False)
        dloader2 = DataLoader(loader2, batch_size, shuffle=False)
        for (inputs1, targets1), batch2 in zip(dloader2, dloader1):
            assert torch.equal(inputs1, batch2['image'])
            break


    def test_old_loader_perf(self):
        split = 'train'
        loader = OxfordPetDataset(config={'Tasks': {
            'Class': 0, 'Segmen': 0, 'BB': 0
        }}, split=split, mini_batch_size=0)
        batch_size = 32
        loader = DataLoader(loader, batch_size, shuffle=False)
        s = time.time()
        for batch in loader:
            break
        print(f'avg. time: {(time.time() - s)}')"""


    """def test_new_loader_perf(self):
        dir_path = '../datasets/data_new/train/'
        loader = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        batch_size = 1
        loader = DataLoader(loader, batch_size, shuffle=False)
        s = time.time()
        for batch in loader:
            print(batch.shape)
        print(f'avg. time: {(time.time() - s)}')"""



    def test_iterable_data_loader(self):
        class RandomBatchSampler(Sampler):
            def __init__(self, dataset, batch_size):
                self.batch_size = batch_size
                self.dataset_length = len(dataset)
                self.n_batches = self.dataset_length / self.batch_size

                self.batch_ids = torch.randperm(int(self.n_batches))

            def __len__(self):
                return self.batch_size

            def __iter__(self):
                for id in self.batch_ids:
                    for index in range(id*self.batch_size, (id+1)*self.batch_size):
                        yield index
                if int(self.n_batches) < self.n_batches:
                    for index in range(int(self.n_batches)*self.batch_size, self.dataset_length):
                        yield index

        dir_path = '../datasets/data_new/train/'
        loader = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        a = DataLoader(loader, batch_size=None, sampler=RandomBatchSampler(loader, batch_size=32))
        a = DataLoader(loader, batch_size=None, sampler=BatchSampler(SequentialSampler(loader), batch_size=32, drop_last=False))
        a = DataLoader(loader, batch_size=None, sampler=BatchSampler(RandomBatchSampler(loader, batch_size=32), batch_size=32, drop_last=False))

        split = 'train'
        data_old = OxfordPetDataset(config={'Tasks': {
            'Class': 0, 'Segmen': 0, 'BB': 0
        }}, split=split, mini_batch_size=0)
        old_loader = DataLoader(loader, batch_size=32, shuffle=True)
        #for (inputs, targets), batch in zip(a, old_loader):
        #    assert torch.equal(inputs, batch['image'])
        s = time.time()
        for (inputs, targets) in a:
            pass
            #print(time.time() - s)

        """s = time.time()
        for data in old_loader:
            print(time.time() - s)
            break"""
        """for i, (inputs, targets) in enumerate(a):
            if i == 0:
                print(inputs)
                raise Exception()
                assert _load_data(0, dir_path+'images.h5')[:32] == inputs
            if i == 69:
                assert _load_data(69, dir_path+'images.h5')[:-32] == inputs"""


if __name__ == '__main__':
    unittest.main()

    """path = '../datasets/data_new/train/images.h5'
    a = _load_data(0, path)
    print(a)
    a = _load_h5_file(path)
    print(a)"""