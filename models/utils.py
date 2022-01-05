import torch
from models.resnet import resnet18
from models.heads import ClassificationHead, SegmentationHead, BBHead
from models.bodys import ResUBody
import torch.nn as nn

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




def get_body():

    #PRE_TRAINED = False
    RESU_FILTS = [32, 32, 64, 128]
    #shared_net = resnet18(PRE_TRAINED)
    shared_net = ResUBody(RESU_FILTS)
    shared_net_chan = RESU_FILTS[3]
    return shared_net ,shared_net_chan
    
def get_heads(config,tasks,encoder_chan):

    return torch.nn.ModuleDict({task: get_head(config, encoder_chan, task) for task in tasks})

def get_head(config, encoder_chan, task): 

    if task == "Class":
        return ClassificationHead(encoder_chan,config['Tasks'][task])

    if task == "Segmen":
        #return SegmentationHead(encoder_chan,num_levels=5,out_ch= config['Tasks'][task])
        return SegmentationHead()

    if task == "BB":
        return BBHead(encoder_chan,config['Tasks'][task])
