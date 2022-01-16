import torch
import torch.nn as nn
import math

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
        return sum(losses.values())


class RandomCombinedLoss(CombinedLoss):
    """Adds random weights to the loss before summation as appears in https://arxiv.org/abs/2111.10603

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
    """Dynamic loss, as appears in https://arxiv.org/pdf/1803.10704.pdf
    """
    def __init__(self, loss_dict, temperature, frequency, **kwargs):
        super(DynamicCombinedLoss, self).__init__(loss_dict, **kwargs)
        assert frequency != 1
        self.frequency = frequency  # this is the reset frequency, e.g. epoch
        self.temperature = temperature
        self.mini_batch_counter = 0
        self.prev_losses = {task: torch.ones(2, dtype=torch.float) for task in self.loss_dict}

    def forward(self, outputs, targets):


        k0 = list(outputs.keys())[0]
        if outputs[k0].requires_grad:
            self.mini_batch_counter += 1

        weights = {
            task: math.exp(
                self.prev_losses[task][-1].item() / (self.prev_losses[task][-2].item() * self.temperature)
            ) for task in self.loss_dict
        }
        Z = sum(weights.values()) / len(self.loss_dict)
        weights = {task: weight / Z for task, weight in weights.items()}
        losses = self.caclulate_loss(outputs, targets, weights.values())
        total_loss = sum(losses.values())
        self._update_weights(losses)


        if self.mini_batch_counter // self.frequency:
            self.mini_batch_counter = 0  # reset the counter, that means a whole new epoch has started
            self.prev_losses = self._get_default_losses() # reset prev losses


        return total_loss

    def _update_weights(self, losses):
        for task in self.loss_dict:
            tmp_list = self.prev_losses[task]
            tmp_list = torch.flip(tmp_list, dims=[0])
            tmp_list[1] = losses[task]
            self.prev_losses[task] = tmp_list

    def _get_default_losses(self):
        return {task: torch.ones(2, dtype=torch.float) for task in self.loss_dict}

class NormalisedDynamicCombinedLoss(CombinedLoss):

    def __init__(self, loss_dict, temperature, frequency, **kwargs):
        super(NormalisedDynamicCombinedLoss, self).__init__(loss_dict, **kwargs)
        assert frequency != 1
        self.frequency = frequency
        self.temperature = temperature
        self.mini_batch_counter = 0
        self.epoch = 0
        self.weights = {task: torch.ones(2, dtype=torch.float) for task in self.loss_dict}
        self.weight_totals = torch.ones(2, dtype=torch.float)

    def forward(self, outputs, targets):
        k0 = list(outputs.keys())[0]
        if outputs[k0].requires_grad:
            self.mini_batch_counter += 1

        weights = {
            task: math.exp(
                (self.weights[task][-1].item() - self.weights[task][-2].item()) * \
                (self.weight_totals[-1].item() - self.weight_totals[-2].item()) * \
                (self.weights[task][-1].item() / (self.weights[task][-2].item() * self.temperature))
            ) for task in self.loss_dict
        }

        Z = sum(weights.values()) / len(self.loss_dict)
        weights = {task: weight / Z for task, weight in weights.items()}

        losses = self.caclulate_loss(outputs, targets, weights.values())
        total_loss = sum(losses.values())

        if self.mini_batch_counter % self.frequency == 0:
            self.epoch = self.mini_batch_counter // self.frequency
            self._update_weights(losses, total_loss)

        return total_loss

    def _update_weights(self, losses, total_loss):
        if self.epoch > 1:
            for task in self.loss_dict:
                self.weights[task][self.epoch % 2] = losses[task]
                self.weights[task][self.epoch % 2] = total_loss


class SegDiceLoss(nn.Module):
    """Segmentation Dice Loss as appears here https://arxiv.org/abs/2006.14822
    """
    def __init__(self, smooth=1):
        super(SegDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        
        outputs = outputs.flatten()
        targets = targets.flatten()
        intersection = (outputs * targets).sum()
        dice_score = (2.*intersection + self.smooth)/(outputs.sum() + targets.sum() + self.smooth)

        return 1 - dice_score


# TODO try to improve this using setters and getters. For now this is OK, but makes the definition of custom losses complex
