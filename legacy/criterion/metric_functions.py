import torch

def accuracy(y_true, y_pred):
    y_class = torch.argmax(y_pred, dim=1)
    return (y_true == y_class).sum()/len(y_true)

# TODO Qingyu and Anny
# recall, precision, f1
# Design classes that subclass from torch.nn.Module. Make sure the init initialises the super class
# make sure any definitions you need to make to customize the function are in __init__. This for example can be
# any smoothing paramters that you have to add
# make a forward method that takes in 2 tensors, outputs and targets. The return of that method should be the score of
# the metric that you are calculating. A simple example is shown below
# make sure to think of edge cases as well.

class Accuracy(torch.nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1)
        # remember, the edge case here is that this would not work if you have a 2D matrix!
        return torch.sum(outputs == targets) / len(targets)


