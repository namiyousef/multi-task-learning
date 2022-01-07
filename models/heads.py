from os import X_OK
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms as trans
#from models.model import  ConvLayer

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        #self.pool = nn.AvgPool2d(8)
        self.pool = nn.MaxPool2d(8)
        self.soft = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, inputs,skips):
        x = self.pool(inputs)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.soft(x)
        return x.double()

class BBHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(BBHead, self).__init__()
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, inputs,skips):
        x = self.pool(inputs)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self,filters):
        super(SegmentationHead, self).__init__()

        ### NEEDS REWORDING

        #filters=[32, 32, 64, 128]

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.decode_conv1 = ConvLayer(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.decode_conv2 = ConvLayer(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.decode_conv3 = ConvLayer(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, skips):
        
        x = self.upsample_1(x)
        x = self.decode_conv1(torch.cat([x, skips[2]], dim=1))
        x = self.upsample_2(x)
        x = self.decode_conv2(torch.cat([x, skips[1]], dim=1))
        x = self.upsample_3(x)
        x = self.decode_conv3(torch.cat([x, skips[0]], dim=1))
        output = self.output_layer(x)
        return output  

class ConvLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ConvLayer, self).__init__()

        ### TAKEN FROM INTERNET NEEDS REWORDING
       
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        ### TAKEN FROM INTERNET NEEDS REWORDING

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
