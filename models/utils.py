import torch
from models.resnet import resnet18, resnet34
#from models.heads import BBHeadNEW, ClassificationHead, SegmentationHead, BBHead
from models.heads import ClassificationHead, SegmentationHead, BBHead
from models.bodys import ResUBody, ResUBodyNEW
import torch.nn as nn
from torchsummaryX import summary


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ConvLayer, self).__init__()

       
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




def get_body(filters):

    #shared_net = ResUBody(filters)
    #arch =summary(shared_net,torch.rand((1,3,256,256)))
    shared_net = resnet34(False)
    #arch =summary(shared_net,torch.rand((1,3,256,256)))
    shared_net_chan = filters[3]
    return shared_net ,shared_net_chan
    
def get_heads(config,tasks,encoder_chan,filters):

    return torch.nn.ModuleDict({task: get_head(config, encoder_chan, task,filters) for task in tasks})

def get_head(config, encoder_chan, task,filters): 

    if task == "Class":
        return ClassificationHead(encoder_chan,config['Tasks'][task])

    if task == "Segmen":
        return SegmentationHead(filters)

    if task == "BB":
        return BBHead(encoder_chan,config['Tasks'][task])

