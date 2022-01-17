import torch
from utils import _prepare_data, _update_performance_dict, _print_epoch_results
from models.model import Model
from criterion.criterion import Criterion
from criterion.metric_functions import accuracy
from data.data import get_dataloader, get_dataset
from train import model_train
from google.colab import drive
drive.mount('/content/gdrive')
drive_base_path = '/content/gdrive/MyDrive/'

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    device = torch.device('cuda:0')
    print('CUDA device detected. Running on GPU.')
else:
    device = torch.device('cpu')
    print('CUDA device not detected. Running on CPU instead.')

def continue_train(config, batch_size, epochs, model_name, last_saved_model_epoch):
    path = f'/content/gdrive/MyDrive/{model_name}{last_saved_model_epoch}.pt'
    print(f'Continuing train from epoch {last_saved_model_epoch+1}')

    model_config = config['model']
    task_config = config['mtl']

    net = Model(task_config, model_config)
    net.load_state_dict(torch.load(path))
    print(f'Successfully loaded model with name: {model_name}{last_saved_model_epoch}')


    criterion = Criterion(task_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-04)

    train_dataset = get_dataset(task_config, "train")
    test_dataset = get_dataset(task_config, "test")

    train_dataloader = get_dataloader(train_dataset, batch_size)
    test_dataloader = get_dataloader(test_dataset, batch_size)

    for i, epoch in enumerate(range(last_saved_model_epoch, epochs)):
        print(f"Epoch {epoch + 1}")  # TODO beautify this with verbose later
        model_eval = model_train(
            config=task_config, model=net, criterion=criterion, optimizer=optimizer, train_dataloader=train_dataloader,
        )
        model_save_name = f'{task_config["Model"]}{epoch + 1}.pt'
        path = drive_base_path + model_save_name
        torch.save(model_eval.state_dict(), path)
        print(f'Wrote model {model_save_name} to personal drive.')


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