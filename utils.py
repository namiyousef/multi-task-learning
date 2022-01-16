import torch

def get_device():
    CUDA_AVAILABLE = torch.cuda.is_available()

    if CUDA_AVAILABLE:
        device = torch.device('cuda:0')
        print('CUDA device detected. Running on GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA device not detected. Running on CPU instead.')
    return device

def _split_equation(string, lower=True):
    if lower:
      string = string.lower()
    string = "".join(string.split())
    comps = string.split('+')
    return comps