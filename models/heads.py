import torch.nn as nn
import torch
import torch.nn.functional as F

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
            # not sure about size here
            # this could be better
            # bit of a black box
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, inputs):
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
            # not sure about size here
            # this could be better
            # bit of a black box
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, inputs):
        x = self.pool(inputs)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, init_ch=512, num_levels=5, out_ch=1):
        super(SegmentationHead, self).__init__()
        self.decoder = [self._segmen_block(2**(-i)*init_ch, type='up') for i in range(num_levels)] 
        self.out_layer = self._conv2d_layer(16,out_ch, is_output=True)

    def forward(self, inputs):
        #x = self.first_layer(inputs)
        x = inputs
        
        for up in self.decoder:
            x = up(x)
        return self.out_layer(x)

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

    

