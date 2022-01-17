from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
import torch
import os
import h5py

class OxfordPetDataset(Dataset):

    def __init__(self, config, split):
        root = os.getcwd()
        root = os.path.join(root, 'datasets/data_new')
        root = os.path.join(root, split)
        self.split = split
        img_path = r'images.h5'
        mask_path = r'masks.h5'
        bbox_path = r'bboxes.h5'
        bin_path = r'binary.h5'

        self.image_dir = os.path.join(root, img_path)
        self.seg_dir = os.path.join(root, mask_path)
        self.bbox_dir = os.path.join(root, bbox_path)
        self.bin_dir = os.path.join(root, bin_path)

        self.seg_task = "Segmen" in config["Tasks"].keys()
        self.bb_task = "BB" in config["Tasks"].keys()
        self.bin_task = "Class" in config["Tasks"].keys()


    def __getitem__(self, index):
        sample = {}

        _img = self._load_data(index, self.image_dir)

        sample['image'] = torch.from_numpy(_img).float()

        if self.seg_task:
            _seg = self._load_data(index, self.seg_dir)
            sample['Segmen'] = torch.from_numpy(_seg).float()

        if self.bb_task:
            _bb = self._load_data(index, self.bbox_dir)
            sample['BB'] = torch.from_numpy(_bb).float()

        if self.bin_task:
            _bin = self._load_data(index, self.bin_dir)
            sample['Class'] = torch.from_numpy(_bin).float()

        return sample

    def __len__(self):

        if self.split == "train":
            return 2210
        if self.split == "val":
            return 738
        if self.split == "test":
            return 738

    # I believe this is loading the data each time.. can you measure memory usag ?
    def _load_data(self, index, dir):
        with  h5py.File(dir, 'r') as file:
            key = list(file.keys())[0]
            elems = file[key][index]
            return elems


#### UTIL FUNCTS #########

def get_dataset(config, split):
    dataset = OxfordPetDataset(config, split, 32)
    return dataset


def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader

def get_fast_dataloader(dataset, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(
            RandomBatchSampler(dataset=dataset, batch_size=batch_size),
            batch_size=batch_size,
            drop_last=False
        )
    )
    return dataloader

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
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)