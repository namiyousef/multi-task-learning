#from torch._C import double
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import h5py
import os

class OxfordPetDataset(Dataset):

    def __init__(self,config,split):
    
        root = "/home/cwatts/COMP0090/Coursework2/data/datasets-oxpet/"
        root = root + split
        
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


    def __getitem__(self,index):
        sample = {}

        _img = self._load_data(index,self.image_dir)
        #sample['image'] = _img
        sample['image'] = torch.from_numpy(_img).float()

        if self.seg_task:
            _seg = self._load_data(index,self.seg_dir)
            sample['Segmen'] = torch.from_numpy(_seg).float()
            #sample['seg'] = _seg

        if self.bb_task:
            _bb = self._load_data(index,self.bbox_dir)
            sample['BB'] = torch.from_numpy(_bb).float()
            #sample['bb'] = _bb 

        if self.bin_task:
            _bin = self._load_data(index,self.bin_dir)
            sample['Class'] = torch.from_numpy(_bin).float()
            # sample['bin'] = _bin 

        return sample  

    def __len__(self):
        return 32

    def _load_data(self,index,dir):
        with  h5py.File(dir , 'r') as file:
            key = list(file.keys())[0]
            elems = file[key][ index]
            return  elems
    
#### UTIL FUNCTS #########

def get_dataset(config,split):

    dataset = OxfordPetDataset(config,split)
    return dataset

def get_dataloader(dataset, batch_size):

    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader