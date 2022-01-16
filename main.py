import torch
from data.data import OxpetDataset, fast_loader
from run import RunTorchModel
import os
from utils import _split_equation
from criterion import metric_functions
from models.utils import get_prebuilt_model
from warnings import warn

def main(data_dir, encoder, decoders, losses, metrics, train_params={'epochs':2, 'verbose':2}, weighting_strategy={}, batch_size=32):

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
    print('Dataloaders created')
    model, loss = get_prebuilt_model(encoder, decoders, losses, **weighting_strategy)
    print('Model and loss created')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)
    for task, metric_equation in metrics.items():
        metric_names = _split_equation(metric_equation, lower=False)
        task_metrics = []
        for metric_name in metric_names:
            try:
                task_metrics.append(getattr(metric_functions, metric_name)())
            except:
                warn(f'The metric {metric_name} has not yet been implemented... continuing without', stacklevel=2)
        metrics[task] = task_metrics


    run_instance = RunTorchModel(
        model=model, optimizer=optimizer, loss=loss, metrics=metrics
    )
    print('Run instance created')

    run_instance.train(trainloader, valloader=valloader, track_history=True, **train_params)

    print('Test performance:')
    print(run_instance.test(testloader))

    return run_instance.get_history()

if __name__ == '__main__':
    config = {
        'data_dir': 'datasets/data_new/',
        'encoder': 'resnet34',
        'decoders': 'class+seg+bb',
        'losses': 'CrossEntropyLoss+SegDiceLoss+0.0032*L1Loss',
        #'weighting_strategy': {},
        'batch_size': 32,
        'metrics': {
            'class': 'Accuracy+MultiAccuracy',
            'seg': 'PixelAccuracy+Precision+Recall+FScore+Jaccard'
        },
        'train_params': {
            'epochs': 20,
            'verbose': 3,
        },
    }
    main(**config)
