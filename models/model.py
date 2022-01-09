import torch
import torch.nn as nn

from models.utils import get_body, get_heads
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, config: dict, filters):
        super(Model, self).__init__()
        self.model = config["Model"]
        self.tasks = config["Tasks"].keys()
        self.encoder, self.encoder_chan = get_body(filters)
        self.decoders = get_heads(config,self.tasks,self.encoder_chan,filters)
         

    def forward(self, x):
        output, skips = self.encoder(x)
        return {task:self.decoders[task](output, skips) for task in self.tasks }

