import torch
from models.resnet import resnet18
from models.heads import ClassificationHead, SegmentationHead

def get_body():

    shared_net = resnet18()
    shared_net_chan = 512
    return shared_net ,shared_net_chan
    

def get_heads(config,tasks,encoder_chan):

    return torch.nn.ModuleDict({task: get_head(config, encoder_chan, task) for task in tasks})

def get_head(config, encoder_chan, task): 

    if task == "Class":
        return ClassificationHead(config['Tasks'][task],encoder_chan)

    if task == "Seg":
        return SegmentationHead(config['Tasks'][task],encoder_chan)

    if task == "BB":
        return ClassificationHead(config['Tasks'][task],encoder_chan)
