import torch.nn as nn
import torch
import torch.nn.functional as F

class ClassificationHead(nn.Sequential):

    def __init__(self, backbone_channels=512, num_classes=1):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(backbone_channels, num_classes)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = nn.Sigmoid()(x)

            return x

class SegmentationHead(nn.Sequential):

    def __init__(self, backbone_channels, num_outputs):
        super(SegmentationHead, self).__init__()
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels,out_channels=last_inp_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=last_inp_channels,out_channels= num_outputs,kernel_size= 1,stride = 1,padding = 0))
    
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x    
