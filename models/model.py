import torch
from torch import nn
from models.utils import get_body, get_heads, get_head
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, config: dict, filters):
        super(Model, self).__init__()
        self.model = config["Model"]
        self.tasks = config["Tasks"].keys()
        self.encoder, self.encoder_chan = get_body(filters)
        self.decoders = get_heads(config, self.tasks, self.encoder_chan,filters)

    def forward(self, x):
        output, skips = self.encoder(x)
        return {task:self.decoders[task](output, skips) for task in self.tasks}


class HardMTLModel(nn.Module):
    """Base MTL model. Builds MTL from a single encoder and can have multiple decoders
    :param encoder:
    :type encoder:
    :param decoder:
    :type decoder:
    :param weights:
    :type weights:
    """
    def __init__(self, encoder, decoders):
        super(HardMTLModel, self).__init__()
        self.encoder = encoder
        self.decoders = torch.nn.ModuleDict({task: decoder for task, decoder in decoders.items()})

    def forward(self, x):
        output, skips = self.encoder(x) # TODO the encoder MAY not accept skips... need to make this robust
        return {
            task: decoder(output, skips) if decoder.has_skips else decoder(output) for task, decoder in self.decoders.items()
        }


def resnet34_class():
    pass

def resnet34_seg():
    pass

def resnet34_seg_class():
    pass

def resnet34_seg_class_bb():
    pass

def resnet34_seg_bb():
    pass
