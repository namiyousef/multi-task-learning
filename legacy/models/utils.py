import torch
from legacy.models.resnet import resnet34
from legacy.models.heads import ClassificationHead, SegmentationHead, BBHead

def get_body(filters):
    shared_net = resnet34(False)
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