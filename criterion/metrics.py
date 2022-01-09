def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum()/len(y_true)
