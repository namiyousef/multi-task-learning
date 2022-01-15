import torch
from data.data import OxpetDataset, fast_loader
from torch.utils.data import DataLoader, BatchSampler
from run import RunTorchModel
from criterion.loss_functions import RandomCombinedLoss
import os

from models.utils import get_prebuilt_model
config = {
    'data_dir': 'datasets/data_new/',
    'encoder': 'resnet34',
    'decoders': 'class+seg+bb',
    'losses': 'CrossEntropyLoss+DiceLoss+0.0032*L1Loss',
    'epochs':20,
    'batch_size':32,
}
def main(data_dir, encoder, decoders, losses, batch_size):

    # prepare data
    tasks = decoders.split('+')
    splits = ['train', 'test', 'val']
    datasets = [OxpetDataset(os.path.join(data_dir, split), tasks) for split in splits]

    if isinstance(batch_size, list):
        assert len(batch_size) == 3
        batch_sizes = batch_size
    else:
        batch_sizes = [batch_size] * 3

    trainloader, valloader, testloader = [fast_loader(dataset, batch_size) for dataset, batch_size in zip(datasets, batch_sizes)]





if __name__ == '__main__':
    from models.model import resnet34_class, resnet34_seg_class, resnet34_seg_class_bb
    from criterion.loss_functions import DiceLoss, DynamicCombinedLoss, SimpleCombinedLoss
    from criterion.metric_functions import MultiAccuracy, PixelAccuracy, Recall, Precision, F1Score, Jaccard
    from torch.nn import L1Loss
    # model configuration
    model = resnet34_seg_class_bb(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)
    train_bs = 8

    loss_dict = {
        'class': torch.nn.CrossEntropyLoss(), 'seg': DiceLoss(), 'bb': L1Loss()
    }
    loss = DynamicCombinedLoss(loss_dict, frequency=1, temperature=1, sf={'bb':train_bs/10000})
    loss = SimpleCombinedLoss(loss_dict, sf={'bb':train_bs/10000})
    # data
    root_path = 'datasets/data_new/{}'
    train_data = OxpetDataset(root_path.format('train'), tasks=['class', 'seg', 'bb'])
    val_data = OxpetDataset(root_path.format('val'), tasks=['class', 'seg', 'bb'])
    train_loader = fast_loader(train_data, train_bs)

    run_instance = RunTorchModel(model=model, optimizer=optimizer, loss=loss, metrics={
        'class': [MultiAccuracy()],
        'seg':[PixelAccuracy(), Precision(), Recall(), F1Score(), Jaccard()]})
    run_instance.train(train_loader, epochs=2, verbose=3, track_history=True)
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

