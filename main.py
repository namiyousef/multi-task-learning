import torch
from utils import _prepare_data, _update_loss_dict, _update_performance_dict, _print_epoch_results, get_device
from models.model import Model
from models import model
from criterion.criterion import Criterion
from criterion.metric_functions import accuracy
from criterion import metric_functions
from data.data import get_dataloader, get_dataset, RandomBatchSampler
from train import model_train
from data.data import OxpetDataset
from torch.utils.data import DataLoader, BatchSampler
from run import RunTorchModel

configuration = {
    'save_params': 's',
    'encoder': {
        'name': 'resnet34',
        'params': [64, 128, 256, 512],
    },
    'decoders': {
        # TODO must call them class, seg or bb... no other support provided...
        'class': {'n_output_features': 2, 'loss': 'bce'},
        'seg': {'n_output_features': 1, 'loss': 'dice'},
        'bb': {'n_output_features': '', 'loss': 'l1'}
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


class CombinedLoss(
    torch.nn.Module):  # TODO can you modify the combined loss to make it follow the strat of simple combined loss?
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


# TODO newer iterations need to make the subclassing better to avoid name clashes!
class SimpleCombinedLoss(CombinedLoss):
    def __init__(self, loss_dict, weights=None):
        super(SimpleCombinedLoss, self).__init__()  # you must inherit superclass
        self.loss_dict = loss_dict  # you must define this as such

        # feel free to add any configurations for the loss of your choice
        # in this case, we've done for a simple weighted loss.
        if weights is None:
            self.weights = torch.ones(size=(len(self.loss_dict),))
        else:
            self.weights = weights

    def forward(self, outputs, targets):
        # while the contents of the dictionary may vary, you MUST set self.losses to a dictionary that contains your losses
        self.loss_values = {
            task: weight * loss(outputs[task], targets[task]) for (task, loss), weight in
            zip(self.loss_dict.items(), self.weights)
        }
        return sum(self.loss_values.values())


class RandomCombinedLoss(CombinedLoss):
    # TODO currently no extra support for scaling weights manually
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


def main(config, epochs=1, batch_size=32,
         metrics=None, losses=None, validation_data=True):  # TODO later change to false!
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
        # net = _build_model(config) # TODO needs changing.

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
    # callable_metrics = {
    #    task : [getattr(metric_functions, metric) for metric in metric_names] for task, metric_names in metrics.items()
    # }
    # train loop

    print("Train loop started...")

    for i, epoch in enumerate(range(epochs)):
        print(f"Epoch {i + 1}")  # TODO beautify this with verbose later
        model_eval = model_train(
            config=task_config, model=net, criterion=criterion, optimizer=optimizer, train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,  # metrics=callable_metrics
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

            perfomance_dict = _update_performance_dict(perfomance_dict, loss, test_output, mini_batch, task_config)

    _print_epoch_results(perfomance_dict, config['mtl'])
    print(f'Test Accuracy: {test_accuracy / (i + 1)}')


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
        train_data = OxpetDataset(root_path.format('train'), tasks=['class', 'seg'], max_size=1)
        val_data = OxpetDataset(root_path.format('val'), tasks=['class', 'seg'])
        train_bs = 32
        train_loader = DataLoader(
            train_data, batch_size=None,
            sampler=BatchSampler(RandomBatchSampler(train_data, train_bs), batch_size=train_bs, drop_last=False)
        )

        run_instance = RunTorchModel(model=model, optimizer=optimizer, loss=loss, metrics={'class': [Accuracy()]})
        run_instance.train(train_loader, epochs=2, verbose=2, track_history=True)
        print(run_instance.test(train_loader))
        print(run_instance.get_history())

        """configuration = {
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
        run_instance.train()"""

    elif run == 'OLD':
        config = {
            'model': [64, 128, 256, 512],
            'mtl': {
                "Model": 'mtl model',  # TODO why do we need this?
                "Tasks": {
                    "Class": 2,
                    # "Segmen":1,
                    # "BB":4
                },
                "Loss Lambda": {
                    "Class": 1,
                    # "Segmen":1,
                    # "BB":1
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
    filters = 0  # TODO get decoder default fitlers
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
