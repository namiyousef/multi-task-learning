import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms as trans
#from models.model import  ConvLayer

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, inputs,skips):
        x = self.pool(inputs)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # poss argmax but doesnt work with cross entropy loss
        #torch.argmax(x, dim=1).float()
        return x.double()

class BBHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(BBHead, self).__init__()
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, inputs,skips):
        x = self.pool(inputs)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SegmentationHeadOLD(nn.Sequential):

    def __init__(self, init_ch=512, num_levels=5, out_ch=1):
        super(SegmentationHeadOLD, self).__init__()
        self.decoder = [self._segmen_block(2**(-i)*init_ch, type='up') for i in range(num_levels)] 
        self.out_layer = self._conv2d_layer(16,out_ch, is_output=True)

    def forward(self, inputs,skips):
        #x = self.first_layer(inputs)
        x = inputs
        x = self.decoder[0](inputs)
        for up , skip in zip(self.decoder[1:3], reversed(skips)):
            x = self._resize_to(x,skip) + skip
            x = up(x)
        for up in self.decoder[3:]:
            x = up(x)
        return self.out_layer(x)
    
    def _resize_to(self,x,y):
        #from tutorial
        if x.shape[1:4]==y.shape[1:4]:
            return x
        resize = trans.Compose([
            trans.Resize(list(x.shape[1:4]))])
        return resize(y)

    def _conv2d_layer(self, in_ch,out_ch, is_output=False):

        _use_bias = True if is_output else False
        _activation = 'sigmoid' if is_output else 'relu'

        def _call(_input):
            # PADDING SAME ISSUE
            # should be padding='same'
            x= nn.Conv2d(in_channels = in_ch,
                out_channels = out_ch,
                kernel_size = 3,
                stride=1, padding=1,
                bias=_use_bias, padding_mode='zeros')(_input)
            if _activation == 'sigmoid':
                _act = nn.Sigmoid()
            else: 
                _act = nn.ReLU()
            output = _act(x)

            # not sure on this bit
            y = output if is_output else nn.BatchNorm2d(out_ch)(output)

            return y
        
        return _call

    def _segmen_block(self, ch, type, bn=True):

        def _call(_input):
            
            # resnet layer
            
            x = self._conv2d_layer(int(ch),int(ch))(_input)
            x = self._conv2d_layer(int(ch),int(ch))(x)
            x += _input

            # sampling layer
            
            if type == "up":
                # same_pad = (2*(output-1) - input - 3)*(1 / 2)
                # not sure how to replication padding = "same"
                # in pytorch
                out_chan,in_chan =int(ch/2),int(ch)
                x = nn.ConvTranspose2d(in_channels = in_chan,out_channels =out_chan,
                    kernel_size = 3, stride=2,
                    padding=1, output_padding=1,
                    groups=1, bias=False,
                    dilation=1, padding_mode='zeros')(x)
                _act = nn.ReLU()
                x = _act(x)
                y = nn.BatchNorm2d(out_chan)(x) if bn else x
            else:  #none
                y = x
            
            return y

        return _call


class SegmentationHead(nn.Module):
    def __init__(self):
        super(SegmentationHead, self).__init__()

        ### NEEDS REWORDING

        filters=[32, 32, 64, 128]

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