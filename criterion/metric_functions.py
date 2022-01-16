import torch

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
    """Determines pixel accuracy for a binary classificaion problem that that has pixel values between [0, 1]
    """
    def __init__(self):
        super(PixelAccuracy, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.where(outputs >= 0.5, 1.0, 0.0)
        outputs = outputs.flatten()
        targets = targets.flatten()
        return torch.sum(outputs == targets) / len(targets)


class Jaccard(torch.nn.Module):
    """Implements the Jaccard score for a binary classification problem with pixel values between [0, 1]
    """
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
    """Implements the Precision score for a binary classification problem with pixel values between [0, 1]
    """
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
    """Implements the Recall score for a binary classification problem with pixel values between [0, 1]
    """
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

class FScore(torch.nn.Module):
    """Implements the F1Score score for a binary classification problem with pixel values between [0, 1]
    """
    def __init__(self, beta=1):
        super(FScore, self).__init__()
        self.precision = Precision()
        self.recall = Recall()
        self.beta_squared = beta**2

    def forward(self, outputs, targets):

        recall = self.recall(outputs, targets)
        precision = self.precision(outputs, targets)

        return (1+self.beta_squared) * (precision * recall) / (self.beta_squared*precision + recall)
