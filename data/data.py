import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import h5py
import os
import time
# TODO combine RandomBatchSampler and SequentialSampler using in_batch and out_batch shuffle params!
class BatchSequentialSampler(Sampler):
    """Generalised sampling class to enable
    """
    def __init__(self, dataset, batch_size, shuffle='both'):
        assert shuffle in ['both', 'batch', 'in_batch']
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset_length = dataset
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.range(0, int(self.n_batches))
        if shuffle == 'batch' or shuffle == 'both':
            self.batch_ids = self.batch_ids[torch.randperm(int(self.n_batches))]

    def __init__(self):
        # TODO self.batch_size or self.dataset_length??
        pass

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


class OxpetDataset(Dataset):
    task_to_file = {
        'class': 'binary.h5',
        'seg': 'masks.h5',
        'bb': 'bboxes.h5'
    }
    # TODO think about adding custom tasks with the same data
    # TODO think about replacing tasks, about arbitrary combinations also?
    def __init__(self, dir_path, tasks, transform=None, target_transforms=None):

        super(OxpetDataset, self).__init__()

        self.dir_path = dir_path
        self.inputs = self._load_h5_file_with_data('images.h5')
        self.targets = {
            task: self._load_h5_file_with_data(self.task_to_file[task]) for task in tasks if task in self.task_to_file
        }
        self.transform = transform
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
        return self.inputs['data'].shape[0]

    def _load_h5_file_with_data(self, file_name):
        path = os.path.join(self.dir_path, file_name)
        file = h5py.File(path)
        key = list(file.keys())[0]
        data = file[key]
        return dict(file=file, data=data)

    def clear_files(self):
        pass
class OxfordPetDataset(Dataset):

    def __init__(self,config,split, mini_batch_size, transform=None):
        # TODO minibatchsize not used?
        root = "datasets/data_new/"
        root = root + split
        self.split = split
        img_path = r'images.h5'
        mask_path = r'masks.h5'
        bbox_path = r'bboxes.h5'
        bin_path = r'binary.h5'

        self.image_dir = os.path.join(root, img_path)
        self.seg_dir = os.path.join(root, mask_path)
        self.bbox_dir = os.path.join(root, bbox_path)
        self.bin_dir = os.path.join(root, bin_path)

        self.seg_task= "Segmen" in config["Tasks"].keys()
        self.bb_task= "BB" in config["Tasks"].keys()
        self.bin_task= "Class" in config["Tasks"].keys()

        self.transform = transform
    

    def __getitem__(self,index):
        sample = {}

        _img = self._load_data(index,self.image_dir)

        # TODO how to make this dynamic, and also allowing for multiple transforms to occur?
        sample['image'] = torch.from_numpy(_img).float()

        if self.seg_task:
            _seg = self._load_data(index,self.seg_dir)
            sample['Segmen'] = torch.from_numpy(_seg).float()

        if self.bb_task:
            _bb = self._load_data(index,self.bbox_dir)
            sample['BB'] = torch.from_numpy(_bb).float()

        if self.bin_task:
            _bin = self._load_data(index,self.bin_dir)
            sample['Class'] = torch.from_numpy(_bin).float()


        if self.transform is not None:
            # TODO need to make sure it only transforms the images, not the outputs, but in a dynamic manner
            pass

        return sample  

    def __len__(self):

        if self.split == "train":
            return 2210
        if self.split == "val":
            return 738
        if self.split == "test":
            return 738

    # I believe this is loading the data each time.. can you measure memory usag ?
    def _load_data(self,index,dir):
        with  h5py.File(dir , 'r') as file:
            key = list(file.keys())[0]
            elems = file[key][ index]
            return  elems



    
#### UTIL FUNCTS #########

def get_dataset(config,split):

    dataset = OxfordPetDataset(config,split,32)
    return dataset
    #return 1

def get_dataloader(dataset, batch_size):

    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader
   

