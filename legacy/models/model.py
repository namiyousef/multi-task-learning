import torch
from torch import nn
from legacy.models.heads import ClassificationHead, SegmentationHead, BBHead
from legacy.models.resnet import resnet34

class HardMTLModel(nn.Module):
    """Base MTL model. Builds MTL from a single encoder and can have multiple decoders
    :param encoder: encoder to use. Can have skip layers
    :type encoder: torch.nn.Module
    :param decoder: decoders to use, in the form {task: decoder}
    :type decoder: dict
    """
    def __init__(self, encoder, decoders):
        super(HardMTLModel, self).__init__()
        self.encoder = encoder
        self.decoders = torch.nn.ModuleDict({task: decoder for task, decoder in decoders.items()})

    def forward(self, x):
        outputs = self.encoder(x)
        return {
            task: decoder(*outputs) for task, decoder in self.decoders.items()
        }

# default models. These are models that are proven to work.
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
