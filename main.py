import torch
from utils import _prepare_data, _update_performance_dict, _print_epoch_results
from legacy.models.model import Model
from legacy.models import model
from legacy.criterion.criterion import Criterion
from legacy.data.data import get_dataloader, get_dataset
from legacy.train import model_train
from data.data import OxpetDataset
from torch.utils.data import DataLoader, BatchSampler
from run import RunTorchModel

def main():
    pass

def _split_equation(string):
    string = "".join(string.lower().split())
    comps = string.split('+')
    return comps

def get_prebuilt_model(encoder, decoders, losses, weights):

    decoders = _split_equation(decoders)
    losses = _split_equation(losses)
    scaling_factors = {task: float(loss.split('*')[0]) if '*' in loss else 1.0 for task, loss in zip(decoders, losses)}
    losses = {task: loss.split('*')[-1] for task, loss in zip(decoders, losses)}

    if isinstance(weights, list):
        assert len(decoders) == len(weights)

    elif isinstance(weights, str):
        if '::' in weights:
        else:
            pass

    decoders = sorted(decoders)
    model = getattr(bodys, f'{encoder}_{"_".join(decoders)}')



    # get losses, combine them



    decoders = sorted(scaling_factors.values())



    model_name = f'{encoder}_{decoders}'
    pass

if __name__ == '__main__':
    get_prebuilt_model('resnet34', 'class,bb,seg', 0, 0, 0)
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

