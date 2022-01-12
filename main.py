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
    # TODO solve loss problem by subclassing from 'combined loss' class
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.is_combined_loss = True
        self.losses = None
    def update_history(self, loss_history):
        pass


class RandomCombinedLoss(CombinedLoss):
    """Adds random weights to the loss before summation

    :param loss_dict: dictionary with keys and values, {task_name: callable_loss}
    :type loss_dict: dict
    :param prior: partially initialised prob function, callable without any extra params.
    :type prior: functools.partial
    :param frequency: number of mini-batches before update. This should be math.ceil(data_length / batch_size)
    :type frequency: int
    """
    def __init__(self, loss_dict, prior, frequency=1):
        super(RandomCombinedLoss, self).__init__()
        self.loss_dict = loss_dict
        self.prior = prior
        self.frequency = frequency
        self.mini_batch_counter = 0
        self.weights = None

    def forward(self, outputs, targets):
        if self.mini_batch_counter % self.frequency == 0:
            self._update_weights()
        self.mini_batch_counter += 1
        print(f'Epoch {self.mini_batch_counter // self.frequency}, minibatch {self.mini_batch_counter}: weights: {self.weights}')
        losses = {
            task: weight * loss(
                outputs[task], targets[task]
            ) for (task, loss), weight in zip(self.loss_dict.items(), self.weights)
        }
        return sum(losses.values())

    def _update_weights(self):
        self.weights = self.prior()

class RunTorchModel:
    """
    loss : needs to apply to be self contained. E.g. with the data that you provide, it should be able to perform backwards!
    """
    def __init__(self, model, optimizer, loss, metrics={}):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss # TODO need to think about how weights would work with this, in conjunction with history, e.g. getting each loss individually and the weights applied
        self.device = _get_device()
        self.loss_history = {self.loss.__class__.__name__: []} # TODO how to update loss_history for combined and single losses
        self.metrics = metrics
        if hasattr(self.loss, 'is_combined_loss'):
            self.metrics_history = {metric.__class__.__name__: [] for task_metrics in metrics for metric in task_metrics}
        else:
            self.metrics_history = {metric.__class__.__name__: [] for metric in metrics} # TODO also this is wrong, need to deal with multitask or single task case, maybe base on loss subclass?

    def train(self, trainloader, epochs=1, valloader=None, verbose=0, track_history=False):

        print('Training model...')
        self.model.to(self.device)
        self.model.train()

        for epoch in range(epochs):
            start_epoch_message = f'EPOCH {epoch+1} STARTED'
            print(start_epoch_message)
            print(f'{"-" * len(start_epoch_message)}')
            start_epoch = time.time()
            # TODO add model saving here
            start_load = time.time()
            for i, (inputs, targets) in enumerate(trainloader):
                start_train = time.time()
                inputs = self._move(inputs)
                targets = self._move(targets)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                # make this into a function, because we will need to use it in the testing loop as well, for both val and test
                # make sure the only inputs you need are outputs, inputs. Make default storing of loss_history
                if track_history:
                    self.loss_history[self.loss.__class__.__name__].append(loss.item())
                    if hasattr(self.loss, 'is_combined_loss'):
                        self.loss.update_history(self.loss_history)
                        for task in self.metrics:
                            metrics = self.metrics[task]
                            for metric in metrics:
                                self.metrics_history[metric.__class__.__name__].append(metric(outputs, targets))
                    else:
                        for metric in self.metrics:
                            self.metrics_history[metric.__class__.__name__].append(metric(outputs, targets))
                            # TODO the indices print from dataset, need to remove that
                loss.backward()
                self.optimizer.step()
                # metrics here
                # TODO need to have a print loss function for each loss? How do you print individual losses while
                # maintaining generalisability?
                end_train = time.time()
                if verbose == 2:
                    print(f'Batch {i+1} complete. Time taken: load({start_train - start_load}), '
                          f'train({end_train - start_train}), total({end_train - start_load}). Loss: {"ignore"}')
                start_load = time.time()
            # TODO add valloader

            if valloader:
                # TODO needs to create entry for the validation metrics, perhaps with val_metricname, or completely different entry?
                # TODO need to add validation histories and metrics as well!
                pass

            end_epoch = time.time()
            print_message = f'Epoch {epoch+1}/{epochs} complete. Time taken: {end_epoch - start_epoch:.3g}. ' \
                            f'Loss: {"ignore"}'

            if verbose:
                print(f'{"-"*len(print_message)}')
                print(print_message)
                print(f'{"-"*len(print_message)}')

    def test(self, testloader, verbose=0, track_history=False):
        with torch.no_grad():
            pass
        pass

    def save_model(self):
        pass

    def get_history(self):
        for name, history in self.loss_history:
            pass
        metrics = {}
        return self.loss_history, self.metrics

    def _move(self, data):
        if torch.is_tensor(data):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {task: tensor.to(self.device) for task, tensor in data.items()}
        elif isinstance(data, list):
            raise NotImplementedError('Currenlty no support for tensors stored in lists.')
        else:
            raise TypeError('Invalid data type.')





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
        from models.model import resnet34_class
        from functools import partial

        # model configuration
        model = resnet34_class(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)
        # TODO in order to make the frequency more robust, e.g. if you wanted to add 'epoch' or 'batch' then you would have to use the update history function as an internal counter for epochs...!
        loss = RandomCombinedLoss(loss_dict={'class': torch.nn.CrossEntropyLoss()}, prior=partial(torch.rand, size=(1,)), frequency=5)
        # data
        root_path = 'datasets/data_new/{}'
        train_data = OxpetDataset(root_path.format('train'), tasks=['class'])
        train_bs = 512
        train_loader = DataLoader(
            train_data, batch_size=None,
            sampler=BatchSampler(RandomBatchSampler(train_data, train_bs), batch_size=train_bs, drop_last=False)
        )

        run_instance = RunTorchModel(model=model, optimizer=optimizer, loss=loss)
        run_instance.train(train_loader, epochs=2, verbose=1)

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
        run_instance.test()

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
