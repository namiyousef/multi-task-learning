import torch

def accuracy(y_true, y_pred):
    y_class = torch.argmax(y_pred, dim=1)
    return (y_true == y_class).sum()/len(y_true)
