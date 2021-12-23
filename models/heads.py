import torch.nn as nn
import torch
import torch.nn.functional as F

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__(
            #not sure about size here
            nn.AdaptiveAvgPool2d(output_size=1),
            #AdaptiveConcatPool2d,
            nn.Flatten(full=False),
            nn.BatchNorm1d(num_classes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features=num_classes, out_features=512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )

class SegmentationHead(nn.Sequential):
    ## FROM ONLINE 
    ## LOOK INTO /REWRITE
    def __init__(self, backbone_channels, num_outputs):
        super(SegmentationHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
    
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x    



## possibly include


class AdaptiveConcatPool2d(nn.Module):
    #from internet
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)