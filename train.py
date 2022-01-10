import torch
import numpy as np
#from data.data import OxfordPetDataset, _get_data
from data.data import OxfordPetDataset
from utils import _prepare_data , _update_loss_dict, _print_epoch_results
from criterion.metrics import accuracy


def model_train(config, model, criterion, optimizer, train_dataloader, val_dataloader=None, metrics=None):
    # TODO validation and mini_batch_not used, need to fix
     
    model.train()

    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    # TODO metrics needs to be dynamically created... it should at least have the losses?
    metrics_epoch_dict = {
        task: [0] * len(metric_names) for task, metric_names in metrics.items()
    }
    train_accuracy = 0
    for i, mini_batch in enumerate(train_dataloader):
        
        mini_batch = _prepare_data(mini_batch,config)
        inputs = mini_batch["image"]

        task_targets = {task:mini_batch[task]for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,task_targets)
        loss['total'].backward()
        optimizer.step()
        if 'Class' in outputs:
            train_accuracy += accuracy(mini_batch['Class'], outputs['Class'])
        loss_epoch_dict = _update_loss_dict(loss_epoch_dict,loss, config)

        for task, output in outputs.items():
            pass

    _print_epoch_results(loss_epoch_dict , config)
    print(f'Training accuracy: {accuracy/(i+1):.3g}')

    return model
        


