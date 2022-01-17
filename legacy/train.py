import torch
from legacy.data.data import OxfordPetDataset
from legacy.utils import _prepare_data , _update_loss_dict, _print_epoch_results
from legacy.criterion.metric_functions import accuracy
import time
import math
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    device = torch.device('cuda:0')
    print('CUDA device detected. Running on GPU.')
else:
    device = torch.device('cpu')
    print('CUDA device not detected. Running on CPU instead.')

def model_train(config, model, criterion, optimizer, train_dataloader, val_dataloader=None, metrics=None, prior=None, apply_prior='batch'):
    # TODO validation and mini_batch_not used, need to fix
    from functools import partial
    model.to(device)
    model.train()
    if isinstance(prior, float):
        class DLW:
            def __init__(self, size, temperature, config):
                self.tasks = config.keys()
                self.weights = {task: torch.ones(size, dtype=torch.float) for task in self.tasks}
                self.temperature = temperature

            def __call__(self, i, loss):
                weights = {
                    task: math.exp(
                        self.weights[task][-1].item()/(self.weights[task][-2].item() * self.temperature)
                    ) for task in self.tasks
                }
                if i > 1:
                    for task in self.tasks:
                        self.weights[task][i % 2] = loss[task]
                return [weight/ (sum(weights.values()) / len(self.tasks)) for weight in weights.values()]

        prior = DLW(len(config['Tasks']), prior, config['Tasks'])

    elif isinstance(prior, partial):
        weights = prior()
    else:
        weights = torch.ones(len(config['Tasks']), dtype=torch.float)

    loss_epoch_dict = {"Seg":[],"Class":[],"BB":[]}
    # TODO metrics needs to be dynamically created... it should at least have the losses?
    #metrics_epoch_dict = {
    #    task: [0] * len(metric_names) for task, metric_names in metrics.items()
    #}
    train_accuracy = 0
    start_load = time.time()
    for i, mini_batch in enumerate(train_dataloader):
        start_train = time.time()
        print(f'Batch Loaded, time taken: {start_train - start_load:.3g}')
        mini_batch = _prepare_data(mini_batch,config)
        inputs = mini_batch["image"].to(device)

        task_targets = {task:mini_batch[task].to(device) for task in config["Tasks"].keys()}
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,task_targets)

        if apply_prior == 'batch':
            if isinstance(prior, DLW):
                weights = prior(i, loss)
                print(weights)
            else:
                weights = prior()
        total_loss = {task: weight * loss_ * scaling_factor for (task, loss_), weight, scaling_factor in zip(loss.items(), weights, criterion.lambdas.values()) if task != 'total'}

        sum(total_loss.values()).backward()
        optimizer.step()
        if 'Class' in outputs:
            train_accuracy += accuracy(mini_batch['Class'].to(device), outputs['Class'])
        loss_epoch_dict = _update_loss_dict(loss_epoch_dict,loss, config)
        end_train = time.time()
        print(f'Minibatch {i+1} complete. Time taken: load({start_train - start_load:.3g}),'
              f'train({end_train - start_train:.3g}), total({end_train - start_load:.3g})')
        start_load = time.time()
        #break

    _print_epoch_results(loss_epoch_dict , config)
    print(f'Training accuracy: {train_accuracy/(i+1):.3g}')


    return model