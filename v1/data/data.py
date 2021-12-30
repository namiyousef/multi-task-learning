from torch.utils.data import Dataset, DataLoader
import h5py
import os

class OxfordPetDataset(Dataset):

    def __init__(self,config,split):

        root = "./data"
        root = root + split
        
        img_path = r'images.h5'
        mask_path = r'masks.h5'
        bbox_path = r'bboxes.h5'
        bin_path = r'binary.h5'

        self.image_dir = os.path.join(root, img_path)
        self.seg_dir = os.path.join(root, mask_path)
        self.bbox_dir = os.path.join(root, bbox_path)
        self.bin_dir = os.path.join(root, bin_path)

        # these return True if the strings of the respective task name is in the dictionary config, which is a dictionary of the tasks
        self.seg_task= "SEGSEM" in config["Tasks"].keys()
        self.bb_task= "BB" in config["Tasks"].keys()
        self.bin_task= "Class" in config["Tasks"].keys()


    def __getitem__(self,index):
        # Instead of returning a tensor we build a dictionary with the data corresponding to the tasks we have 
        sample = {}
        # images are always loaded as well as the segmentation masks
        _img = self._load_data(index,self.image_dir)
        sample['image'] = _img
        _seg = self._load_data(index,self.seg_dir)
        sample['seg'] = _seg
        # Selectively adding the relevant tasks
        if self.bb_task:
            _bb = self._load_data(index,self.bbox_dir)
            sample['bb'] = _bb 
        if self.bin_task:
            _bin = self._load_data(index,self.bin_dir)
            sample['bin'] = _bin 

        return sample  

    # def __len__(self):
    #     return len()

    def _load_data(index,dir):
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