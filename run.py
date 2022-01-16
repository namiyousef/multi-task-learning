"""
First version
Things to do for future improvements:
# TODO take track_history into __init__ and create a class variable self.track_metrics. use it to replace measure metrics
# TODO add model saving and loading
# TODO make update epoch functions explicit
# TODO move all tests regarding MTL outside of RunTorchModel framework and into MTL loading functions
"""
# public imports
import time
import torch

# private imports
from utils import get_device
from criterion.loss_functions import CombinedLoss

class RunTorchModel:
    """Class for easy running of PyTorch models, similar to that of Keras API

    :param model: initialised PyTorch model (subclasses from torch.nn.Module)
    :type model: class
    :param optimizer: initialised optimizer from PyTorch
    :type optimizer: <class 'torch.optim.{optim_name}.{optim_cls_name}'>
    :param loss: initialised PyTorch loss
    :type loss: class
    :param metrics: PyTorch metrics (either objects or string names)
    :type metrics: list
    """
    def __init__(self, model, optimizer, loss, metrics=None):
        self.model = model
        self.optimizer = optimizer

        self.device = get_device()

        self.loss = loss

        self.is_mtl = isinstance(self.loss, CombinedLoss)

        self.history = {'loss': {'train': self._create_init_loss_history()}}

        if metrics:
            self.metrics = metrics


        if self.is_mtl:
            self._assert_dicts_compatible(model.decoders.keys(), loss.loss_dict.keys())


    def train(self, trainloader, epochs=1, valloader=None, verbose=0, track_history=False):

        measure_metrics = track_history and hasattr(self, 'metrics')

        if measure_metrics:
            self.history['metric'] = {'train': self._create_init_metric_history()}
        if valloader:
            self.history['loss']['val'] = self._create_init_loss_history()
            if measure_metrics:
                self.history['metric']['val'] = self._create_init_metric_history()

        history_keys = self.history.keys()
        create_init_histories = {key: getattr(self, f'_create_init_{key}_history') for key in history_keys}
        update_epoch_histories = {key: getattr(self, f'_update_{key}_epoch_history') for key in history_keys}

        print('Training model...')
        self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()
            start_epoch_message = f'EPOCH {epoch+1} STARTED'
            print(start_epoch_message)
            print(f'{"-" * len(start_epoch_message)}')
            start_epoch = time.time()
            # TODO add model saving here
            start_load = time.time()

            epoch_train_history = {key: create_init_history(domain='epoch') for key, create_init_history in create_init_histories.items()}

            for i, (inputs, targets) in enumerate(trainloader):
                start_train = time.time()
                inputs = self._move(inputs)
                targets = self._move(targets)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                for key in update_epoch_histories:
                    tmp_dict = epoch_train_history
                    if key == 'loss':
                        # TODO don't like implicit defs
                        update_func = update_epoch_histories[key]
                        tmp_dict = update_func(tmp_dict, loss)
                        epoch_train_history = tmp_dict

                    else:
                        update_func = update_epoch_histories[key]
                        tmp_dict = tmp_dict[key] # TODO need to make this consistent
                        tmp_dict = update_func(tmp_dict, outputs, targets)
                        epoch_train_history[key] = tmp_dict

                loss.backward()
                self.optimizer.step()

                end_train = time.time()

                if verbose > 1:
                    print(
                        f'Batch {i+1} complete. Time taken: load({start_train - start_load:.3g}), '
                        f'train({end_train - start_train:.3g}), total({end_train - start_load:.3g}). '
                    )
                if verbose > 2:
                    for key in epoch_train_history:
                        print(f'{", ".join([f"{loss_name}({loss_val/(i+1):.3g})" for loss_name, loss_val in epoch_train_history[key].items()])}')
                start_load = time.time()
            epoch_train_history = {key: {loss_name: loss_val/(i+1) for loss_name, loss_val in losses.items()} for key, losses in epoch_train_history.items()}
            self._update_history(epoch_train_history, 'train')
            if valloader:
                epoch_val_history = self.test(valloader, measure_val_metrics=measure_metrics)
                self._update_history(epoch_val_history, 'val')

            end_epoch = time.time()

            print_message = f'Epoch {epoch+1}/{epochs} complete. Time taken: {end_epoch - start_epoch:.3g}. ' \
                            f'Loss: {self._get_loss_print_msg(self.history["loss"])}'

            if verbose:
                print(f'{"-"*len(print_message)}')
                print(print_message)
                print(f'{"-"*len(print_message)}')

    def test(self, testloader, measure_val_metrics=True):
        self.model.eval()
        measure_metrics = measure_val_metrics if not measure_val_metrics else hasattr(self, 'metrics')
        epoch_test_history = {'loss': self._create_init_loss_history('epoch')}
        if measure_metrics:
            epoch_test_history['metric'] = self._create_init_metric_history('epoch')

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(testloader):
                inputs = self._move(inputs)
                targets = self._move(targets)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                epoch_test_history = self._update_loss_epoch_history(epoch_test_history, loss)  # update losses of epoch
                if measure_metrics:
                    epoch_test_history['metric'] = self._update_metric_epoch_history(epoch_test_history['metric'], outputs, targets)  # update metrics of epoch

        epoch_test_history = {key: {loss_name: loss_val / (i + 1) for loss_name, loss_val in losses.items()} for key, losses in epoch_test_history.items()}
        return epoch_test_history

    def save_model(self):
        raise NotImplementedError('Saving models has not yet been implemented.')

    def get_history(self):
        return self.history

    def _get_init_history_value(self, domain):
        if domain == 'history':
            return []
        if domain == 'epoch':
            return 0

    def _create_init_loss_history(self, domain='history'):
        name = self._get_cls_name(self.loss)
        value = self._get_init_history_value(domain)
        loss_history = {name: value}
        if self.is_mtl:
            for key in self.loss.loss_dict:
                loss_history[key] = value

        return loss_history

    def _create_init_metric_history(self, domain='history'):
        value = self._get_init_history_value(domain)
        if self.is_mtl:

            return {self._get_cls_name(metric): value for  key, task_metrics in self.metrics.items()  for metric in task_metrics}
        else:
            return {self._get_cls_name(metric): value for metric in self.metrics}

    def _update_history(self, epoch_history_dict, split):
        for key in self.history:
            if key == 'loss':
                for name, val in epoch_history_dict[key].items():
                    tmp_list = self.history[key][split][name].copy()
                    tmp_list.append(val)
                    self.history[key][split][name] = tmp_list
            else:
                if self.is_mtl:
                    for task in self.metrics:
                        task_metrics = self.metrics[task]
                        for metric in task_metrics:
                            name = self._get_cls_name(metric)
                            tmp_list = self.history[key][split][name].copy()
                            tmp_list.append(epoch_history_dict[key][name])
                            self.history[key][split][name] = tmp_list
                else:
                    for metric in self.metrics:
                        name = self._get_cls_name(metric)
                        self.history[key][split][name].append(epoch_history_dict[key][name])

    def _update_loss_epoch_history(self, history_dict, loss):
        name = self._get_cls_name(self.loss)
        history_dict['loss'][name] += loss.item()
        if self.is_mtl:
            history_dict['loss'] = self.loss._update_history(history_dict['loss'])
        return history_dict

    def _update_metric_epoch_history(self, history_dict, outputs, targets):
        if self.is_mtl:
            for task in self.metrics:
                task_metrics = self.metrics[task]
                for metric in task_metrics:
                    name = self._get_cls_name(metric)
                    history_dict[name] += metric(outputs[task], targets[task]).item()
        else:
            for metric in self.metrics:
                name = self._get_cls_name(metric)
                history_dict[name] += metric(outputs, targets).item()
        return history_dict

    def _move(self, data):
        if torch.is_tensor(data):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {task: tensor.to(self.device) for task, tensor in data.items()}
        elif isinstance(data, list):
            raise NotImplementedError('Currenlty no support for tensors stored in lists.')
        else:
            raise TypeError('Invalid data type.')

    def _get_cls_name(self, cls):
        return cls.__class__.__name__

    def _get_loss_print_msg(self, loss_dict):
        return ', '.join(
            [
                f'{split}[{", ".join([f"{loss_name}({sum(loss_val)/len(loss_val):.3g})" for loss_name, loss_val in loss_dict[split].items()])}]' for split in loss_dict.keys()
            ]
        )

    def _assert_dicts_compatible(self, *iterables):
        iterables = [sorted(iterable) for iterable in iterables]
        assert all(iterables)