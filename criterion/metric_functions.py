import torch

# TODO Qingyu and Anny
# recall, precision, f1
# Design classes that subclass from torch.nn.Module. Make sure the init initialises the super class
# make sure any definitions you need to make to customize the function are in __init__. This for example can be
# any smoothing paramters that you have to add
# make a forward method that takes in 2 tensors, outputs and targets. The return of that method should be the score of
# the metric that you are calculating. A simple example is shown below
# make sure to think of edge cases as well.

class MultiAccuracy(torch.nn.Module):
    """Calculates accuracy for multiclass inputs (batchsize, feature length) by determining the most likely class
    using argmax -> (batchsize,) and then comparing with targets which are also (batchsize,)
    """
    def __init__(self):
        super(MultiAccuracy, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1)
        return torch.sum(outputs == targets) / len(targets)

class PixelAccuracy(torch.nn.Module):
    def __init__(self):
        super(PixelAccuracy, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.where(outputs >= 0.5, 1.0, 0.0)
        outputs = outputs.flatten()
        targets = targets.flatten()
        return torch.sum(outputs == targets) / len(targets)


class Jaccard(torch.nn.Module):

    def __init__(self):
        super(Jaccard, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.where(outputs >= 0.5, 1.0, 0.0)
        outputs = outputs.flatten()
        targets = targets.flatten()
        intersection = torch.sum(outputs == targets)
        union = len(targets) + len(outputs) - intersection
        return intersection / union

class Precision(torch.nn.Module):

    def __init__(self):
        super(Precision, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.where(outputs >= 0.5, 1.0, 0.0)
        outputs = outputs.flatten()
        targets = targets.flatten()

        # filter only positive predictions
        ids_pos = targets == 1.0

        pos_targets = targets[ids_pos]
        pos_outputs = outputs[ids_pos]
        neg_targets = targets[~ids_pos]
        neg_outputs = outputs[~ids_pos]
        tp = torch.sum(pos_outputs == pos_targets)
        fp = torch.sum(neg_targets != neg_outputs)
        return tp/(tp+fp)

class Recall(torch.nn.Module):
    def __init__(self):
        super(Recall, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.where(outputs >= 0.5, 1.0, 0.0)
        outputs = outputs.flatten()
        targets = targets.flatten()

        # filter only positive predictions
        ids_pos = targets == 1.0

        pos_targets = targets[ids_pos]
        pos_outputs = outputs[ids_pos]

        tp = torch.sum(pos_outputs == pos_targets)
        fn = torch.sum(pos_targets != pos_outputs)

        return tp / (tp + fn)

class F1Score(torch.nn.Module):
    def __init__(self):
        super(F1Score, self).__init__()
        self.precision = Precision()
        self.recall = Recall()

    def forward(self, outputs, targets):

        recall = self.recall(outputs, targets)
        precision = self.precision(outputs, targets)

        return 2 * (precision * recall) / (precision + recall)
