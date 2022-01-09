
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, out, label):
        return self.loss(out,label)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, out, label):
        return self.loss(out,label)        

class BCEWithLogitsLoss(nn.Module):

    # maybe better loss
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, out, label):
        return self.loss(out,label)

class DiceLoss(nn.Module):

    ### FROM online REWORD 
    
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, out, label):
        return self.loss(out,label)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss( reduction="none")

    def forward(self, out, label):
        return self.loss(out,label).sum(1).sum()/10000




