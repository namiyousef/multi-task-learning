import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import h5py
import os

class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    """
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

def fast_loader(dataset, batch_size=32, drop_last=False):
    """Implements fast loading by taking advantage of .h5 dataset

    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )

class OxpetDataset(Dataset):
    task_to_file = {
        'class': 'binary.h5',
        'seg': 'masks.h5',
        'bb': 'bboxes.h5'
    }
    # TODO think about adding custom tasks with the same data
    # TODO think about replacing tasks, about arbitrary combinations also?
    def __init__(self, dir_path, tasks, transform=None, target_transforms=None, shuffle=False, max_size=None):

        super(OxpetDataset, self).__init__()

        self.dir_path = dir_path
        self.inputs = self._load_h5_file_with_data('images.h5')
        self.targets = {
            task: self._load_h5_file_with_data(self.task_to_file[task]) for task in tasks if task in self.task_to_file
        }
        self.transform = transform
        self.shuffle = shuffle # TODO implement shuffle after data loaded
        self.max_size = max_size
        # TODO add default transforms, make sure target transforms only add to it, e.g. using transform compose?

    def __getitem__(self, index):
        inputs = self.inputs['data'][index]
        if self.transform:
            inputs = self.transform(inputs) # TODO need to test transform
        else:
            inputs = torch.from_numpy(inputs).float()

        targets = {
            task: torch.from_numpy(self.targets[task]['data'][index]).float() for task in self.targets
        }

        # manual TensorFlow to PyTorch shape conversion for dense tensors # TODO parametrize
        inputs = inputs.permute([0, 3, 2, 1]) # TODO need to think about generalising this with losses. Don't like this right now
        if 'seg' in targets:
            targets['seg'] = targets['seg'].permute([0,3, 2, 1])
        if 'class' in targets:
            targets['class'] = torch.reshape(targets["class"],(-1,)).type(torch.LongTensor)
        return (inputs, targets)

    def __len__(self):
        return self.max_size if self.max_size else self.inputs['data'].shape[0]

    def _load_h5_file_with_data(self, file_name):
        path = os.path.join(self.dir_path, file_name)
        file = h5py.File(path)
        key = list(file.keys())[0]
        data = file[key]
        return dict(file=file, data=data)

    def clear_files(self):
        pass




   

