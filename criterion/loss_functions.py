
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

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, out, label):
        return self.loss(out,label)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, out, label):
        return self.loss(out,label,reduction="none").sum(1)      




