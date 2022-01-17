import torch.nn as nn

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

    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, out, label):
        return self.loss(out,label)

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth =1

    def forward(self, outputs, targets):
        
        outputs = outputs.flatten()
        targets = targets.flatten()
        intersection = (outputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(outputs.sum() + targets.sum() + self.smooth)
        
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
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, out, label):
        return self.loss(out,label).sum(1).sum()/10000