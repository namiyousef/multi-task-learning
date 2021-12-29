import torch.nn.functional as F
import h5py
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

######################################################################################################

# This is the MTL architecture that I came up with initially, but we decided to go with Connor's because it's more convenient for analysis. Uploading this for future reference

######################################################################################################


# number of samples to load from h5 file number of sample: [train- 2210, val - 738, test - 738]
# num_samples = 5
total_samples_train, total_samples_val, total_samples_test = 2210, 738, 738
# generate random indices which are to be sampled from the dataset
inds_to_sample_train, inds_to_sample_val, inds_to_sample_test = np.sort(np.random.choice(total_samples_train,size=total_samples_train,replace=False)), np.sort(np.random.choice(total_samples_val,size=total_samples_val,replace=False)), np.sort(np.random.choice(total_samples_test,size=total_samples_test,replace=False))
# define a function to load only samples at the specified indices
def load_data_from_h5 ( path , inds_to_sample ):
    with h5py . File ( path , 'r') as file :
        key = list ( file . keys () ) [0]
        elems = file [ key ][ inds_to_sample ]
    return elems
# load data at the randomly generated indices
masks_train, masks_val, masks_test  = torch.from_numpy(load_data_from_h5( './train/masks.h5', inds_to_sample_train)),torch.from_numpy(load_data_from_h5( './val/masks.h5', inds_to_sample_val)),torch.from_numpy(load_data_from_h5( './test/masks.h5', inds_to_sample_test)) #(256, 256, 256, 1) --- this is 256 samples of masks of images which are 256 x 256 with 1 value per entry. Each value is an animal can take value 0 or 1 
masks_train, masks_val, masks_test = (masks_train.resize_((masks_train.shape[0],1,256,256))).squeeze(1).float(), (masks_val.resize_((masks_val.shape[0],1,256,256))).squeeze(1).float(), (masks_test.resize_((masks_test.shape[0],1,256,256))).squeeze(1).float()
imgs_train, imgs_val, imgs_test = torch.from_numpy(load_data_from_h5( './train/images.h5' , inds_to_sample_train)),torch.from_numpy(load_data_from_h5( './val/images.h5' , inds_to_sample_val)),torch.from_numpy(load_data_from_h5( './test/images.h5' , inds_to_sample_test))   # (256, 256, 256, 3) --- this is 256 samples of images which are 256 x 256 with 3 values per entry (RGB)
imgs_train, imgs_val, imgs_test =  imgs_train.resize_((imgs_train.shape[0],3,256,256)), imgs_val.resize_((imgs_val.shape[0],3,256,256)), imgs_train.resize_((imgs_test.shape[0],3,256,256))
bboxes_train, bboxes_val, bboxes_test = load_data_from_h5( './train/bboxes.h5' , inds_to_sample_train), load_data_from_h5( './val/bboxes.h5' , inds_to_sample_val),  load_data_from_h5( './test/bboxes.h5' , inds_to_sample_test)# (256, 4) --- this is 256 samples of ???
binarys_train, binarys_val, binarys_test = load_data_from_h5( './train/binary.h5' , inds_to_sample_train), load_data_from_h5( './val/binary.h5' , inds_to_sample_val), load_data_from_h5( './test/binary.h5' , inds_to_sample_test) # (256, 1) --- either 0 or 1 whether a cat or a dog
# Normalizing 
imgs_train, imgs_val, imgs_test = imgs_train/255.0, imgs_val/255.0, imgs_test/255.0

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, chs=(3,32,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return x, ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class MTLDataset(Dataset):
  def __init__(self, images, masks, binary_outputs, transform=None):
    self.input_images, self.target_masks, self.binary_outputs = images, masks, binary_outputs
    self.transform = transform
    #self.binary_true = binary_true
  def __len__(self):
    return len(self.input_images)
  def __getitem__(self, idx):
    image = self.input_images[idx]
    mask = self.target_masks[idx]
    binary_outputs = self.binary_outputs[idx]
    if self.transform:
      image = self.transform(image)
   # if binary_true==1:
    return [image, mask, binary_outputs]
   # else:
    #    return [image, mask]

class MTLLoss(nn.Module):
    def __init__(self, weight=None, size_average = True):
        super(MTLLoss, self).__init__()
    def forward(self, inputsBC, inputsSeg, targetsBC, targetsSeg, lmbda = 0.8, smooth=1):
        # First we're going to define the dice loss for the main task
        inputsSeg = torch.nn.functional.sigmoid(inputsSeg)       
        # flatten label and prediction tensors
        inputsSeg = inputsSeg.view(-1)
        targetsSeg = targetsSeg.view(-1)
        intersection = (inputsSeg * targetsSeg).sum()                            
        dice = (2.*intersection + smooth)/(inputsSeg.sum() + targetsSeg.sum() + smooth)  
        DL = 1 - dice
        # Now we're going to build the loss for the binary classification task
        inputsBC = inputsBC.view(-1)
        targetsBC = targetsBC.view(-1)
        bce = - (torch.log(inputsBC+0.1)*targetsBC + (1-targetsBC)* torch.log(1-inputsBC)).sum()/(len(inputsBC))
        # Now we're going to return the total loss of these tasks
        total_loss = lmbda*DL + (1-lmbda)*bce
        print(f"DL: {DL}")
        print(f"bce {bce} ")
        return total_loss

class MTLNetwork(nn.Module):
    def __init__(self, num_tasks = 2, enc_chs=(3,32,64), dec_chs=(64, 32), num_class=1, retain_dim=True, out_sz=(256,256)):
        super().__init__()
        self.encoder     = Encoder(enc_chs) # shared weights
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
    def forward(self, x):
        x, enc_ftrs = self.encoder(x) 
        # Segmentation output
        x_seg = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        x_seg = out      = self.head(x_seg)
        if self.retain_dim:
            x_seg = torch.nn.functional.interpolate(x_seg, (256,256))
        # Binary output    
        x_bc = F.relu(nn.Conv2d(64, 16, 3)(x))
        x_bc = nn.MaxPool2d(2)(x_bc)
        x_bc = torch.flatten(x_bc,1)
        x_bc = F.relu(nn.Linear(16*29*29, 64)(x_bc))
        x_bc = nn.Linear(64, 1)(x_bc)
        # 2 outputs: 1 for the segmentation and 1 for binary classification
        return x_seg, x_bc

if __name__ == "__main__":

    toySetImg, toySetMask, toySetBinary = imgs_train[:500], masks_train[:500], binarys_train[:500]
    trainset = MTLDataset(toySetImg, toySetMask, toySetBinary)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    net = MTLNetwork().float()
    criterion = MTLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(5):  # loop over the dataset multiple times (in this case we loop only 2 times)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels_seg, labels_binary = data
            optimizer.zero_grad()
            outputs_seg, outputs_bc = net(inputs.float()) 
    
            loss = criterion(outputs_bc,  outputs_seg.squeeze(1),  labels_binary, labels_seg.type(torch.FloatTensor)) # Computing the the loss 
            
            loss.backward() # Computes the gradient 
            optimizer.step() # Performs a single optimization step (parameter update).
            
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    print('Training done.')
