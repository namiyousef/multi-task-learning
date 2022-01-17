import torch

def accuracy(y_true, y_pred):
    y_class = torch.argmax(y_pred, dim=1)
    return (y_true == y_class).sum()/len(y_true)

class Accuracy(torch.nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1)
        # remember, the edge case here is that this would not work if you have a 2D matrix!
        return torch.sum(outputs == targets) / len(targets)


