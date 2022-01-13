import torch
from utils import _prepare_data, _update_loss_dict, _update_performance_dict, _print_epoch_results, _get_device
from models.model import Model
from models import model
from criterion.criterion import Criterion
from criterion.metric_functions import accuracy
from criterion import metric_functions
from data.data import get_dataloader, get_dataset, RandomBatchSampler
from train import model_train
import time
from data.data import OxpetDataset
from torch.utils.data import DataLoader, BatchSampler
import math

configuration = {
        'save_params':'s',
        'encoder': {
            'name':'resnet34',
            'params':[64, 128, 256, 512],
        },
        'decoders': {
            # TODO must call them class, seg or bb... no other support provided...
            'class':{'n_output_features':2, 'loss':'bce'},
            'seg':{'n_output_features': 1, 'loss':'dice'},
            'bb':{'n_output_features':'', 'loss':'l1'}
        },
        'weights': '',
    }



# TODO don't like this, would like this to be in one place only. Currently also done in train.py!
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    device = torch.device('cuda:0')
    print('CUDA device detected. Running on GPU.')
else:
    device = torch.device('cpu')
    print('CUDA device not detected. Running on CPU instead.')


class SimpleCombinedLoss(torch.nn.Module):
    def __init__(self, loss_dict, weights=None):
        self.loss_dict = loss_dict
        if weights is None:
            self.weights = {task: 1 for task in self.loss_dict}
        else:
            self.weights = weights

    def forward(self, outputs, targets):
        losses = {
            task: weight * loss(outputs, targets) for (task, loss), weight in zip(self.loss_dict.items(), self.weights)
        }

        return sum(losses.values())

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.is_combined_loss = True
        self.loss_values = None
    def _update_history(self, loss_dict):
        for task, loss in self.loss_values.items():
            if task != self.__class__.__name__:
                loss_val = loss.item()
                loss_val = [loss_val] if isinstance(loss_dict[task], list) else loss_val
                loss_dict[task] += loss_val
        return loss_dict


# TODO currently no support for scaling weights

class RandomCombinedLoss(CombinedLoss):
    """Adds random weights to the loss before summation

    :param loss_dict: dictionary with keys and values, {task_name: callable_loss}
    :type loss_dict: dict
    :param prior: partially initialised prob function, callable without any extra params, or str input
    :type prior: functools.partial OR str
    :param frequency: number of mini-batches before update. This should be math.ceil(data_length / batch_size) for epoch
    :type frequency: int
    """
    def __init__(self, loss_dict, prior, frequency):
        super(RandomCombinedLoss, self).__init__()
        self.loss_dict = loss_dict
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
        self.loss_values = {
            task: weight * loss(
                outputs[task], targets[task]
            ) for (task, loss), weight in zip(self.loss_dict.items(), self.weights)
        }
        return sum(self.loss_values.values())

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
        return probas / torch.sum(probas, dtype=torch.float)

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

        self.device = _get_device()

        self.loss = loss

        self.history = {'loss': {'train': self._create_init_loss_history()}}

        if metrics:
            self.metrics = metrics


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

                if verbose == 2:
                    print(
                        f'Batch {i+1} complete. Time taken: load({start_train - start_load:.3g}), '
                        f'train({end_train - start_train:.3g}), total({end_train - start_load:.3g}). '
                    )
                start_load = time.time()
            epoch_train_history = {key: {loss_name: loss_val/(i+1) for loss_name, loss_val in losses.items()} for key, losses in epoch_train_history.items()}
            self._update_history(epoch_train_history, 'train')
            if valloader:
                epoch_val_history = self.test(valloader, measure_val_metrics=measure_metrics)
                self._update_history(epoch_val_history, 'val')

            end_epoch = time.time()

            print_message = f'Epoch {epoch+1}/{epochs} complete. Time taken: {end_epoch - start_epoch:.3g}. ' \
                            f'Loss: {self._get_loss_print_msg()}'

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
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                epoch_test_history = self._update_loss_epoch_history(epoch_test_history, loss)  # update losses of epoch
                if measure_metrics:
                    epoch_test_history['metric'] = self._update_metric_epoch_history(epoch_test_history['metric'], outputs, targets)  # update metrics of epoch
        epoch_test_history = {key: loss_val / (i + 1) for key, loss_val in epoch_test_history.items()}
        return epoch_test_history

    def save_model(self):
        pass

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
        if hasattr(self.loss, 'is_combined_loss'):
            for key in self.loss.loss_dict:
                loss_history[key] = value

        return loss_history

    def _create_init_metric_history(self, domain='history'):
        value = self._get_init_history_value(domain)
        if hasattr(self.loss, 'is_combined_loss'):

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
                if hasattr(self.loss, 'is_combined_loss'):
                    for task in self.metrics:
                        task_metrics = self.metrics[task]
                        for metric in task_metrics:
                            name = self._get_cls_name(metric)
                            self.history[key][split][name].append(epoch_history_dict[key][name])
                else:
                    for metric in self.metrics:
                        name = self._get_cls_name(metric)
                        self.history[key][split][name].append(epoch_history_dict[key][name])

    def _update_loss_epoch_history(self, history_dict, loss):
        name = self._get_cls_name(self.loss)
        history_dict['loss'][name] += loss.item()
        if hasattr(self.loss, 'is_combined_loss'):
            history_dict['loss'] = self.loss._update_history(history_dict['loss'])
        return history_dict

    def _update_metric_epoch_history(self, history_dict, outputs, targets):
        if hasattr(self.loss, 'is_combined_loss'):
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

    def _get_loss_print_msg(self):
        loss_dict = self.history['loss']
        return ', '.join(
            [
                f'{split}[{", ".join([f"{loss_name}({sum(loss_val)/len(loss_val):.3g})" for loss_name, loss_val in loss_dict[split].items()])}]' for split in loss_dict.keys()
            ]
        )



def main(config, epochs=1, batch_size=32,
         metrics=None, losses=None, validation_data=True): # TODO later change to false!
    """
    :param config:
    # either dict or string. If string will use configuration that exists. If dict will build the model
    :param epochs:
    :param batch_size:
    :param metrics:
    :return:
    """
    if isinstance(config, str):
        try:
            net = getattr(model, config)()
        except:
            print(f'Model with name {config} is not pre-built. Creating one based on input instead...')
            pass
    else:
        pass
        #net = _build_model(config) # TODO needs changing.

    model_config = config['model']
    task_config = config['mtl']
    net = Model(task_config, model_config)

    # TODO need
    criterion = Criterion(task_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-04)

    train_dataset = get_dataset(task_config, "train")
    test_dataset = get_dataset(task_config, "test")

    train_dataloader = get_dataloader(train_dataset, batch_size)
    test_dataloader = get_dataloader(test_dataset, batch_size)

    if validation_data:
        val_dataset = get_dataset(task_config, "val")
        val_dataloader = get_dataloader(val_dataset, batch_size)

    # TODO can even add losses here! Just need to think of a smart way to do it with an if statement
    #callable_metrics = {
    #    task : [getattr(metric_functions, metric) for metric in metric_names] for task, metric_names in metrics.items()
    #}
    # train loop

    print("Train loop started...")

    for i, epoch in enumerate(range(epochs)):
        print(f"Epoch {i+1}") # TODO beautify this with verbose later
        model_eval = model_train(
            config=task_config, model=net, criterion=criterion, optimizer=optimizer, train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, #metrics=callable_metrics
        )

    print("Test loop started...")
    model_eval.eval()
    perfomance_dict = {"Seg": [], "Class": [], "BB": []}
    test_accuracy = 0
    with torch.no_grad():
        for i, mini_batch in enumerate(test_dataloader):

            mini_batch = _prepare_data(mini_batch, task_config)
            inputs = mini_batch["image"].to(device)

            task_targets = {task: mini_batch[task].to(device) for task in task_config["Tasks"].keys()}
            test_output = model_eval(inputs)
            if 'Class' in task_targets:
                test_accuracy += accuracy(task_targets['Class'].to(device), test_output['Class'])
            loss = criterion(test_output, task_targets)

            perfomance_dict = _update_performance_dict(perfomance_dict, loss, test_output,mini_batch,task_config)

    _print_epoch_results(perfomance_dict, config['mtl'])
    print(f'Test Accuracy: {test_accuracy/(i+1)}')

if __name__ == '__main__':

    run = 'NEW'
    if run == 'NEW':
        from models.model import resnet34_class, resnet34_seg_class
        from functools import partial
        from criterion.loss_functions import DiceLoss
        from criterion.metric_functions import Accuracy

        # model configuration
        model = resnet34_seg_class(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)
        loss = RandomCombinedLoss(loss_dict={
            'class': torch.nn.CrossEntropyLoss(), 'seg': DiceLoss()
        }, prior='normal', frequency=70)
        # data
        root_path = 'datasets/data_new/{}'
        train_data = OxpetDataset(root_path.format('train'), tasks=['class', 'seg'])
        val_data = OxpetDataset(root_path.format('val'), tasks=['class', 'seg'])
        train_bs = 32
        train_loader = DataLoader(
            train_data, batch_size=None,
            sampler=BatchSampler(RandomBatchSampler(train_data, train_bs), batch_size=train_bs, drop_last=False)
        )


        run_instance = RunTorchModel(model=model, optimizer=optimizer, loss=loss, metrics={'class':[Accuracy()]})
        run_instance.train(train_loader, epochs=2, verbose=2, track_history=True)
        print(run_instance.test(train_loader))
        print(run_instance.get_history())

        configuration = {
            'save_params': 's',
            'encoder': {
                'name': 'resnet34',
                'params': [64, 128, 256, 512],
            },
            'decoders': {
                # TODO must call them class, seg or bb... no other support provided...
                'class': {'name':'ClassificationHead',
                          'params': {'n_output_features':2},
                          'loss': 'bce'},
                'seg': {'name':'SegmentationHead', 'params':{'filters':[64, 128, 256, 512]}, 'loss': 'dice'},
                'bb': {'name':'BBHead', 'params':{'n_output_features': 4}, 'loss': 'l1'}
            },
            'weights': '',
        }
        # create model
        # create loss
        # create optimizer
        # get loaders
        run_instance = RunTorchModel()
        run_instance.train()

    elif run == 'OLD':
        config = {
            'model': [64, 128, 256, 512],
            'mtl': {
                "Model": 'mtl model', # TODO why do we need this?
                "Tasks":{
                    "Class":2,
                    #"Segmen":1,
                    #"BB":4
                },
                "Loss Lambda":{
                    "Class":1,
                    #"Segmen":1,
                    #"BB":1
                    }

            }
        }

        main(config=config, epochs=1, batch_size=32)



# TODO for later, DO NOT USE
def _build_multi_criterion():
    pass


# DO NOT USE YET
# code for building models from string names. Currently have no need for this, but can be a useful way to quickly build models
def _get_model(model_name):
    model_components = model_name.split('_')
    decoder = model_components[0]
    tasks = model_components[1:]
    filters = 0 # TODO get decoder default fitlers
    # get default model weights
    pass
def _create_config_from_str(model_name):
    model_components = model_name.split('_')
    decoder_name = model_components
    task_names = model_components[1:]
    pass



    import re
    def _convert_class_to_func(cls_name):
        cls_comps = re.findall('[A-Z][^A-Z]*', cls_name)
        func_name = '_'.join([comp.lower() for comp in cls_comps])
        return func_name

    def _convert_func_to_class():
        pass

    def build_mtl_from_config(config):
        encoder_name, encoder_params = configuration['encoder'].values()
        # TODO to generalise this will have to change, for now we are just working with strings
        encoder = getattr(bodys, encoder_name)(encoder_params)  # TODO maybe kwargs this for robustness
        decoders = configuration['decoders']
        decoder_dict = {
            task: getattr(heads, decoder_info['name']) for task, decoder_info in decoders.items()
        }
        # TODO honestly, I don't like this at all. I want default models
        # TODO add this
        model = HardMTLModel(encoder, decoder_dict)
        return model(encoder, decoders), loss
