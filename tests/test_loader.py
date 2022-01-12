from data.data import OxfordPetDataset, OxpetDataset, RandomBatchSampler
from torch.utils.data import DataLoader, IterableDataset, BatchSampler, SequentialSampler, RandomSampler, SubsetRandomSampler, Sampler
import unittest
import torch
import time
import h5py

def _load_data(index, dir):
    with  h5py.File(dir, 'r') as file:
        key = list(file.keys())[0]
        elems = file[key][index]
        return elems

class TestLoader(unittest.TestCase):

    def test_datasets_equal(self):
        """Using the same loader, tests that both datasets are equal
        """
        # prepare first dataset
        split = 'train'
        dset1 = OxfordPetDataset(config={'Tasks':{
            'Class':0, 'Segmen':0, 'BB':0
        }},split=split, mini_batch_size=0)

        # prepare second dataset
        dir_path = '../datasets/data_new/train/'
        dset2 = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        assert len(dset2) == len(dset1)

        batch_size = 32
        dloader1 = DataLoader(dset1, batch_size, shuffle=False)  # set shuffle to false to ensure same loading
        dloader2 = DataLoader(dset2, batch_size, shuffle=False)
        for (inputs1, targets1), batch_dict in zip(dloader2, dloader1):
            assert torch.equal(inputs1, batch_dict['image'])

    def test_dataloader_transform(self):
        # empty unless transform added
        pass


    def test_new_dataset_speed(self):
        """tests that new dataset is faster
        """

        # prepare first dataset
        split = 'train'
        dset1 = OxfordPetDataset(config={'Tasks': {
            'Class': 0, 'Segmen': 0, 'BB': 0
        }}, split=split, mini_batch_size=0)

        # prepare second dataset
        dir_path = '../datasets/data_new/train/'
        dset2 = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        assert len(dset2) == len(dset1)

        batch_size = 32
        limit = 10
        dloader1 = DataLoader(dset1, batch_size, shuffle=False)  # set shuffle to false to ensure same loading
        dloader2 = DataLoader(dset2, batch_size, shuffle=False)
        s = time.time()
        for i, (inputs1, targets1) in enumerate(dloader2, 1):
            if i == limit:
                break
        dloader2_time = (time.time() - s) / limit

        s = time.time()
        for i, batch in enumerate(dloader1, 1):
            if i == limit:
                break
        dloader1_time = (time.time() - s) / limit

        print(f'Old dloader time: {dloader1_time}')
        print(f'New dloader time: {dloader2_time}')

        assert dloader2_time > dloader1_time

    def test_batch_sampling(self):
        """tests if batch sampling with no shuffle gives the same results as normal dataloading
        """
        dir_path = '../datasets/data_new/train/'
        dataset = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        batchloader = DataLoader(dataset, batch_size=None, sampler=BatchSampler(SequentialSampler(dataset), batch_size=32, drop_last=False))
        normalloader = DataLoader(dataset, batch_size=32, drop_last=False)
        for (inputs1, targets1), (inputs2, targets2) in zip(batchloader, normalloader):
            assert torch.equal(inputs1, inputs2)
            for target1, target2 in zip(targets1.values(), targets2.values()):
                assert torch.equal(target1, target2)

    def test_random_batch_sampling_out_shuffle(self):
        """tests if random batch shuffling works as expected
        """
        dir_path = '../datasets/data_new/train/'
        dataset = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        torch.manual_seed(0)

        batchloader = DataLoader(dataset, batch_size=None,
                                 sampler=BatchSampler(RandomBatchSampler(dataset, batch_size=32, in_batch_shuffle=False), batch_size=32, drop_last=False))
        ids = iter(torch.randperm(len(dataset)//32))
        for i, (inputs, targets) in enumerate(batchloader):
            if i < len(ids):
                id = ids[i]
            else:
                id = 69
            index = list(range(id*32, (id+1)*32))
            inputs_file = _load_data(index, dir_path+'images.h5')
            assert torch.equal(inputs_file, inputs)

    def test_random_batch_sampling_in_shuffle_true(self):
        """tests if random batch shuffling works as expected with internal shuffles as well
        """
        dir_path = '../datasets/data_new/train/'
        dataset = OxpetDataset(dir_path, ['class', 'seg', 'bb'])
        torch.manual_seed(0)

        batchloader = DataLoader(dataset, batch_size=None,
                                 sampler=BatchSampler(
                                     RandomBatchSampler(dataset, batch_size=32, in_batch_shuffle=True), batch_size=32,
                                     drop_last=False))
        ids = iter(torch.randperm(len(dataset) // 32))
        for i, (inputs, targets) in enumerate(batchloader):
            if i < len(ids):
                id = ids[i]
            else:
                id = 69
            index = list(range(id * 32, (id + 1) * 32))
            inputs_file = _load_data(index, dir_path + 'images.h5')[torch.randperm(len(index))]
            assert torch.equal(inputs_file, inputs)


    def test_iterable_data_loader(self):
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