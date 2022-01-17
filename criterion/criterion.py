import torch
import torch.nn as nn



def get_loss(task):
    
    if task == 'Class':
        from criterion.loss_functions import CrossEntropyLoss
        return CrossEntropyLoss()

    elif task == 'Segmen':
        #from criterion.loss_functions import BCEWithLogitsLoss
        #return BCEWithLogitsLoss()
        from criterion.loss_functions import DiceLoss
        return DiceLoss()

    elif task == 'BB':
        from criterion.loss_functions import L1Loss
        return L1Loss()

    return

class Criterion(nn.Module):
    def __init__(self, config, weights=None):
        super(Criterion, self).__init__()
        self.tasks = config["Tasks"].keys()
        self.lambdas = config["Loss Lambda"]
        self.loss_fncts = torch.nn.ModuleDict({task: get_loss(task) for task in self.tasks})
    
    def forward(self, prediction, truth):
        # TODO not convinced. Are we sure that when we do loss.backward(), that is bases it on the total sum?
        loss_dict = {task: self.loss_fncts[task](prediction[task], truth[task]) for task in self.tasks}
        loss_dict['total'] = torch.sum(torch.stack([loss_dict[task]*self.lambdas[task] for task in self.tasks]))
        return loss_dict

# TODO add scaling
# TODO add weighting option
