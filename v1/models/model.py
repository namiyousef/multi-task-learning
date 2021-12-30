import torch
import torch.nn as nn

from models.utils import get_body, get_heads
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, config: dict):
        super(Model, self).__init__()
        self.model = config["Model"]
        self.tasks = config["Tasks"].keys()
        #possibly include option in dictionary to cust body
        self.encoder, self.encoder_chan = get_body()
        self.decoders = get_heads(config,self.tasks,self.encoder_chan)
         
    def forward(self, x):
        output = self.encoder(x)
        return {task:self.decoders[task](output) for task in self.tasks }
        
