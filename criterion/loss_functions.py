
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F

class CombinedLoss(
    torch.nn.Module):  # TODO can you modify the combined loss to make it follow the strat of simple combined loss?
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.is_combined_loss = True
        self.loss_values = None

    def _update_history(self, loss_dict):
        for task, loss in self.loss_values.items():
            if task != self.__class__.__name__:
                loss_val = loss.item()
                loss_val = [loss_val] if isinstance(loss_dict[task], list) else loss_val
                loss_dict[task] += loss_val
        return loss_dict

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, out, label):
        return self.loss(out,label)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, out, label):
        return self.loss(out,label)        

class BCEWithLogitsLoss(nn.Module):

    # maybe better loss
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, out, label):
        return self.loss(out,label)

class DiceLoss(nn.Module):

    ### FROM online REWORD 
    
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, out, label):
        return self.loss(out,label)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, out, label):
        return self.loss(out,label).sum(1).sum()/10000



# TODO DiceLoss for BB, class, inputs configs, def forward(outputs, targets)
# outputs -> tensors, targets -> tensors
# TODO newer iterations need to make the subclassing better to avoid name clashes!
class SimpleCombinedLoss(CombinedLoss):
    def __init__(self, loss_dict, weights=None):
        super(SimpleCombinedLoss, self).__init__()  # you must inherit superclass
        self.loss_dict = loss_dict  # you must define this as such

        # feel free to add any configurations for the loss of your choice
        # in this case, we've done for a simple weighted loss.
        if weights is None:
            self.weights = torch.ones(size=(len(self.loss_dict),))
        else:
            self.weights = weights

    def forward(self, outputs, targets):
        # while the contents of the dictionary may vary, you MUST set self.losses to a dictionary that contains your losses
        self.loss_values = {
            task: weight * loss(outputs[task], targets[task]) for (task, loss), weight in
            zip(self.loss_dict.items(), self.weights)
        }
        return sum(self.loss_values.values())


class RandomCombinedLoss(CombinedLoss):
    # TODO currently no extra support for scaling weights manually
    """Adds random weights to the loss before summation

    :param loss_dict: dictionary with keys and values, {task_name: callable_loss}
    :type loss_dict: dict
    :param prior: partially initialised prob function, callable without any extra params, or str input
    :type prior: functools.partial OR str
    :param frequency: number of mini-batches before update. This should be math.ceil(data_length / batch_size) for epoch
    :type frequency: int
    """

    def __init__(self, loss_dict, prior, frequency):
        super(RandomCombinedLoss, self).__init__()
        self.loss_dict = loss_dict
        self.prior = getattr(self, prior) if isinstance(prior, str) else prior
        self.frequency = frequency
        self.mini_batch_counter = 0
        self.weights = None

    def forward(self, outputs, targets):
        if self.mini_batch_counter % self.frequency == 0:
            self._update_weights()
        k0 = list(outputs.keys())[0]
        if outputs[k0].requires_grad:
            self.mini_batch_counter += 1
        self.loss_values = {
            task: weight * loss(
                outputs[task], targets[task]
            ) for (task, loss), weight in zip(self.loss_dict.items(), self.weights)
        }
        return sum(self.loss_values.values())

    def _update_weights(self):
        self.weights = self.prior()

    def normal(self):
        return torch.softmax(torch.randn(size=(len(self.loss_dict),)), dim=-1)

    def uniform(self):
        return torch.softmax(torch.rand(size=(len(self.loss_dict),)), dim=-1)

    def bernoulli(self):
        return torch.randint(0, 2, size=(len(self.loss_dict),))

    def constrained_bernoulli(self):
        probas = torch.randint(0, 2, size=(len(self.loss_dict),))
        return probas / torch.sum(probas, dtype=torch.float)