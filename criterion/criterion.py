import torch
import torch.nn as nn



def get_loss(task):
    
    if task == 'Class':
        from criterion.loss_functions import CrossEntropyLoss
        return CrossEntropyLoss()

    elif task == 'Segmen':
        from criterion.loss_functions import BCEWithLogitsLoss
        return BCEWithLogitsLoss()

    elif task == 'BB':
        from criterion.loss_functions import L1Loss
        return L1Loss()

    return

class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.tasks = config["Tasks"].keys()
        self.loss_fncts = torch.nn.ModuleDict({task: get_loss(task) for task in self.tasks})

    
    def forward(self, prediction, truth):
        loss_dict = {task: self.loss_fncts[task](prediction[task], truth[task]) for task in self.tasks}
        loss_dict['total'] = torch.sum(torch.stack([loss_dict[task] for task in self.tasks]))
        return loss_dict
