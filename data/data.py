import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torchvision.transforms import Compose
import os
import h5py

class OxpetDataset(Dataset):
    """Dataset to load data from the Oxford pet dataset .h5 files

    :param dir_path: path to directory containing data (e.g. train, test or val)
    :type dir_path: str
    :param tasks: list of tasks to load data for, must be in 'class', 'seg' or 'bb'
    :type tasks: list
    :param transform: transformation to apply to the images
    :type transform: list
    :param target_transforms: transforms to add to each target, of the form {task:[transforms]}
    :type target_transforms: dict
    :param shuffle: parameter to enable shuffling within batch after loading
    :type shuffle: bool
    :param max_size: maximum size of data to draw from (useful for debugging purposes)
    :type max_size: int
    """
    task_to_file = {
        'class': 'binary.h5',
        'seg': 'masks.h5',
        'bb': 'bboxes.h5'
    }
    def __init__(self, dir_path, tasks, transform=[], target_transforms={}, shuffle=True, max_size=None):

        super(OxpetDataset, self).__init__()

        self.dir_path = dir_path
        self.inputs = self._load_h5_file_with_data('images.h5')
        self.targets = {
            task: self._load_h5_file_with_data(self.task_to_file[task]) for task in tasks if task in self.task_to_file
        }

        self.transform = Compose([self._from_numpy, self._permute_tf_to_torch]+transform)
        self.target_transforms = self._prepare_default_target_transforms(target_transforms)

        self.shuffle = shuffle
        self.max_size = max_size

    def __getitem__(self, index):

        inputs = self.inputs['data'][index]

        if self.shuffle:
            inputs = inputs[torch.randperm(len(index))]

        inputs = self.transform(inputs)

        targets = {
            task: transform(self.targets[task]['data'][index]) for task, transform in self.target_transforms.items()
        }
        return (inputs, targets)

    def __len__(self):
        return self.max_size if self.max_size else self.inputs['data'].shape[0]

    def _load_h5_file_with_data(self, file_name):
        path = os.path.join(self.dir_path, file_name)
        file = h5py.File(path)
        key = list(file.keys())[0]
        data = file[key]
        return dict(file=file, data=data)

    def _permute_tf_to_torch(self, tensor):
        """Function to load PIL images in correct format required by PyTorch
        This extends the capabiliy of torchvision.transforms.ToTensor to 4D arrays
        """
        return tensor.permute([0, 3, 2, 1])

    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _prepare_class_task(self, tensor):
        """Prepares classification data to correct shape required by task
        """
        return torch.reshape(tensor, (-1,)).type(torch.LongTensor)

    def _prepare_default_target_transforms(self, target_transforms):
        """
        Prepares the default target transformations for the tasks
        """
        task_transorm_dict = {}
        for task in self.targets:
            task_transform = [self._from_numpy]
            if task == 'seg':
                task_transform += [self._permute_tf_to_torch]
            elif task == 'class':
                task_transform += [self._prepare_class_task]

            if task in target_transforms:
                task_transform += target_transforms[task]
            task_transorm_dict[task] = Compose(task_transform)
        return task_transorm_dict


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'

    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
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


def normal_loader(dataset, batch_size=32, drop_last=False, shuffle=True):
    """Implements a normal loading scheme
    This scheme indexes the dataset one index at a time. It is slow because the .h5 causes a bottleneck that
    scales linearly with the number of calls made to it. However, this allows strong shuffling to be used.

    :param dataset: dataset
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch_size
    :type batch_size: int
    :param drop_last: bool to determine if last batch dropped if not full size
    :type drop_last: bool
    :returns: batched dataset
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.

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

