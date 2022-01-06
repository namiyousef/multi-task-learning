import torch
import torch.nn as nn
import numpy as np

def get_loss(task):
    

    if task == 'Segmen':
        from criterion.loss_functions import DiceLoss
        return DiceLoss()

    elif task == 'Class':
        from criterion.loss_functions import CrossEntropyLoss
        return CrossEntropyLoss()

    elif task == 'BB':
        from criterion.loss_functions import L1Loss
        return L1Loss()
    
    elif task == 'RNL':
        from criterion.loss_functions import DiceLoss
        return DiceLoss()

    return

class Criterion(nn.Module):
    def __init__(self, config, typee):
        super(Criterion, self).__init__()
        self.type = typee
        self.tasks = config["Tasks"].keys()
        lmbda = 0.8
        self.loss_fncts = torch.nn.ModuleDict({task: get_loss(task) for  task in self.tasks}) 
        if len(config["Tasks"].keys())==1:
            lmbda_list = [1]
            self.task_lambda = [(lmbda_list[i], list(self.tasks)[i]) for i in range(1)]
        if len(config["Tasks"].keys())==2:
            #lmbda_list = [lmbda, (1-lmbda)]
            lmbda_list = [0.9, 0.1]
            self.task_lambda = [(lmbda_list[i], list(self.tasks)[i]) for i in range(2)]
        if len(config["Tasks"].keys())==3:
            lmbda_list = [0.9,0.0999, 0.0001]
            self.task_lambda = [(lmbda_list[i], list(self.tasks)[i]) for i in range(3)]

    def forward(self, prediction, truth):
        if self.type == "train":
            loss_dict = {task: self.loss_fncts[task](prediction[task], truth[task]) for task in self.tasks}
            loss_dict['total'] = torch.sum(torch.stack([loss_dict[pair[1]]*pair[0] for pair in self.task_lambda]))
            return loss_dict
        if self.type == "test":
            loss_dict = {task: self.loss_fncts[task](prediction[task], truth[task]) for task in self.tasks}
            loss_dict['total'] = torch.sum(torch.stack([loss_dict[pair[1]]*pair[0] for pair in self.task_lambda])).detach().item()
            return loss_dict