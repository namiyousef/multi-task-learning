import torch
import torch.nn as nn

# TODO try to improve this using setters and getters. For now this is OK, but makes the definition of custom losses complex
class CombinedLoss(torch.nn.Module):
    """Superclass for defining all combined losses. Any custom combined loss must subclass from this. If the custom
    losses have their own weights, then they will be multiplied by the scaling factors before the total loss is computed
    for performing backward in the training loop

    :param loss_dict: the losses used for each task {task_name: pytorch_loss}
    :type loss_dict: dict
    :param sf: any scaling associated with the losses
    :type sf: dict
    """
    def __init__(self, loss_dict, eval_test=False, sf=None):
        super(CombinedLoss, self).__init__()
        self.loss_dict = loss_dict
        if sf is None:
            self.sf = torch.ones(size=(len(loss_dict),))
        else:
            self.sf = [sf[task] if task in sf else 1.0 for task in self.loss_dict]

        self.eval_test = eval_test
        self.loss_values = None

    def _update_history(self, loss_dict):
        for task, loss in self.loss_values.items():
            if task != self.__class__.__name__:
                loss_val = loss.item()
                loss_val = [loss_val] if isinstance(loss_dict[task], list) else loss_val
                loss_dict[task] += loss_val
        return loss_dict

    def caclulate_loss(self, outputs, targets, weights):
        if self.eval_test:
            self.loss_values = {
                task: weight * sf_ * loss(outputs[task], targets[task]) for (task, loss), weight, sf_ in
                zip(self.loss_dict.items(), weights, self.sf)
            }
        else:
            self.loss_values = {
                task: (weight if outputs[task].requires_grad else 1.0) * sf_ * loss(outputs[task], targets[task]) \
                for (task, loss), weight, sf_ in zip(self.loss_dict.items(), weights, self.sf)
            }
        return self.loss_values



class SimpleCombinedLoss(CombinedLoss):
    def __init__(self, loss_dict, weights=None, **kwargs):
        super(SimpleCombinedLoss, self).__init__(loss_dict, **kwargs)  # inherit superclass
        if weights is None:
            self.weights = torch.ones(size=(len(self.loss_dict),))
        else:
            self.weights = weights

    def forward(self, outputs, targets):
        losses = self.caclulate_loss(outputs, targets, self.weights)
        return sum(losses)


class RandomCombinedLoss(CombinedLoss):
    """Adds random weights to the loss before summation

    :param loss_dict: dictionary with keys and values, {task_name: callable_loss}
    :type loss_dict: dict
    :param prior: partially initialised prob function, callable without any extra params, or str input
    :type prior: functools.partial OR str
    :param frequency: number of mini-batches before update. This should be math.ceil(data_length / batch_size) for epoch
    :type frequency: int
    """

    def __init__(self, loss_dict, prior, frequency, **kwargs):
        super(RandomCombinedLoss, self).__init__(loss_dict, **kwargs)
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

        losses = self.caclulate_loss(outputs, targets, self.weights)
        return sum(losses.values())

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
        while torch.all(probas == 0):
            probas = torch.randint(0, 2, size=(len(self.loss_dict),))

        return probas / torch.sum(probas, dtype=torch.float)


# TODO there are some type casting problems here.. make sure all the inputs are tensors for future
class DynamicCombinedLoss(CombinedLoss):
    def __init__(self, loss_dict, temperature, frequency, **kwargs):
        super(DynamicCombinedLoss, self).__init__(loss_dict, **kwargs)
        self.frequency = frequency
        self.temperature = temperature
        self.mini_batch_counter = 0
        self.epoch = 0
        self.weights = {task: torch.ones(len(loss_dict), dtype=torch.float) for task in self.loss_dict}

    def forward(self, outputs, targets):
        weights = {
            task: torch.exp(
                self.weights[task][-1].item() / (self.weights[task][-2].item() * self.temperature)
            ) for task in self.loss_dict
        }
        weights_sum = sum(weights.values())
        weights = {task: weight / weights_sum for task, weight in weights.items()}

        loss = self.caclulate_loss(outputs, targets, torch.ones(size=(len(self.loss_dict),)))

        k0 = list(outputs.keys())[0]
        if outputs[k0].requires_grad:
            self.mini_batch_counter += 1

        if self.mini_batch_counter % self.frequency == 0:
            self._update_weights(loss)

        losses = self.caclulate_loss(outputs, targets, weights)
        return sum(losses.values())

    def _update_weights(self, loss):
        if self.epoch > 1:
            for task in self.loss_dict:
                self.weights[task][self.epoch % 2] = loss[task]

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





if __name__ == '__main__':
    a = torch.rand(size=(5,4))
    b = torch.rand(size=(5,4))
    import torch
    loss1 = torch.nn.L1Loss(reduction='none')
    loss2 = torch.nn.L1Loss()
    print(loss1(a, b).sum(1).sum()/20)
    print(loss2(a, b))

    loss = RandomCombinedLoss(loss_dict = {}, prior='uniform', frequency=1)
    print(isinstance(loss, CombinedLoss))
