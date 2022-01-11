#from torch._C import double
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.utils.data as data
import h5py
import os
import numpy as np
import time
import random
def _load_h5_file(path):
    with h5py.File(path, 'r') as f:
        key = f.keys()

class OxpetDatasetIterable(IterableDataset):
    task_to_file = {
        'class': 'binary.h5',
        'seg': 'masks.h5',
        'bb': 'bboxes.h5'
    }
    # TODO think about adding custom tasks with the same data
    # TODO think about replacing tasks, about arbitrary combinations also?
    def __init__(self, dir_path, tasks, transform=None, target_transforms=None):

        super(OxpetDatasetIterable, self).__init__() # TODO needed?

        self.dir_path = dir_path
        s = time.time()
        self.inputs = self._load_h5_file_with_data('images.h5')
        print(time.time())
        self.targets = {
            task: self._load_h5_file_with_data(self.task_to_file[task]) for task in tasks if task in self.task_to_file
        }

        self.transform = transform


    def __iter__(self):
        s = time.time()
        inputs = self.inputs['data']
        """print(time.time() - s)
        if self.transform:
            inputs = self.transform(inputs) # TODO need to test transform
        else:
            s = time.time()
            inputs = torch.from_numpy(inputs).float()
            print(inputs.shape)
            print(time.time() - s)

        s = time.time()
        targets = {
            task: torch.from_numpy(self.targets[task]['data'][index]).float() for task in self.targets
        }
        print(time.time() -s)"""
        return iter(inputs)


    def __len__(self):
        return self.inputs['data'].shape[0]

    def _load_h5_file_with_data(self, file_name):
        path = os.path.join(self.dir_path, file_name)
        file = h5py.File(path)
        key = list(file.keys())[0]
        data = file[key]
        return dict(file=file, data=data)
class OxpetDataset(Dataset):
    task_to_file = {
        'class': 'binary.h5',
        'seg': 'masks.h5',
        'bb': 'bboxes.h5'
    }
    # TODO think about adding custom tasks with the same data
    # TODO think about replacing tasks, about arbitrary combinations also?
    def __init__(self, dir_path, tasks, transform=None, target_transforms=None):

        super(OxpetDataset, self).__init__() # TODO needed?

        self.dir_path = dir_path
        self.inputs = self._load_h5_file_with_data('images.h5')
        self.targets = {
            task: self._load_h5_file_with_data(self.task_to_file[task]) for task in tasks if task in self.task_to_file
        }

        self.transform = transform


    def __getitem__(self, index):
        #print(index)
        s = time.time()
        inputs = self.inputs['data'][index]
        print(time.time() - s, index[0], index[-1], len(index))
        if len(index) != 32:
            raise Exception()
        if self.transform:
            inputs = self.transform(inputs) # TODO need to test transform
        else:
            inputs = torch.from_numpy(inputs).float()

        targets = {
            task: torch.from_numpy(self.targets[task]['data'][index]).float() for task in self.targets
        }
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
        root = "../datasets/data_new/"
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
   

