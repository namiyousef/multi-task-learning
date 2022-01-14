import torch
from torch import nn
from models.heads import ClassificationHead, SegmentationHead, BBHead
from models.utils import get_body, get_heads
from models.resnet import resnet34

class Model(nn.Module):
    
    def __init__(self, config: dict, filters):
        super(Model, self).__init__()
        self.model = config["Model"]
        self.tasks = config["Tasks"].keys()
        self.encoder, self.encoder_chan = get_body(filters)
        self.decoders = get_heads(config, self.tasks, self.encoder_chan,filters)

    def forward(self, x):
        print(type(self.encoder(x)))
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
        outputs = self.encoder(x) # TODO the encoder MAY not accept skips... need to make this robust
        # TODO need to add good documentation explaining how to use this
     
        return {
            task: decoder(*outputs) for task, decoder in self.decoders.items()
        }


def resnet34_class(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'class': ClassificationHead(in_channels=decoder_out_channels, num_classes=2),
    }
    return HardMTLModel(encoder, decoders)

def resnet34_seg_class(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'class': ClassificationHead(in_channels=decoder_out_channels, num_classes=2),
        'seg': SegmentationHead(filters=filters)
    }
    return HardMTLModel(encoder, decoders)

def resnet34_seg_class_bb(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'class': ClassificationHead(in_channels=decoder_out_channels, num_classes=2),
        'seg': SegmentationHead(filters=filters),
        'bb':BBHead(in_channels=decoder_out_channels, num_classes=4)
    }
    return HardMTLModel(encoder, decoders)

def resnet34_seg_bb(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'seg': SegmentationHead(filters=filters),
        'bb':BBHead(in_channels=decoder_out_channels, num_classes=4)
    }
    return HardMTLModel(encoder, decoders)

def resnet34_seg(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'seg': SegmentationHead(filters=filters),
    }
    return HardMTLModel(encoder, decoders)

def resnet34_bb(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'bb': BBHead(in_channels=decoder_out_channels, num_classes=4)
    }
    return HardMTLModel(encoder, decoders)

def resnet34_class_bb(pretrained=False):
    encoder = resnet34(pretrained)
    filters = [64, 128, 256, 512]
    decoder_out_channels = filters[-1]
    decoders = {
        'class': ClassificationHead(in_channels=decoder_out_channels, num_classes=2),
        'bb':BBHead(in_channels=decoder_out_channels, num_classes=4)
    }
    return HardMTLModel(encoder, decoders)
