import torch
from models.resnet import resnet18
from models.heads import ClassificationHead, SegmentationHead

def get_body():

    PRE_TRAINED = False
    shared_net = resnet18(PRE_TRAINED)
    shared_net_chan = 512
    return shared_net ,shared_net_chan
    

def get_heads(config,tasks,encoder_chan):

    return torch.nn.ModuleDict({task: get_head(config, encoder_chan, task) for task in tasks})

def get_head(config, encoder_chan, task): 

    if task == "Class":
        return ClassificationHead(encoder_chan,config['Tasks'][task])

    if task == "Segmen":
        return SegmentationHead(encoder_chan,num_levels=5,out_ch= config['Tasks'][task])

    if task == "BB":
        return ClassificationHead(encoder_chan,config['Tasks'][task])
