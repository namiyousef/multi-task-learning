import torch

def get_device():
    """Function to get cuda device if avaiable
    """
    CUDA_AVAILABLE = torch.cuda.is_available()

    if CUDA_AVAILABLE:
        device = torch.device('cuda:0')
        print('CUDA device detected. Running on GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA device not detected. Running on CPU instead.')
    return device

def _split_equation(string, lower=True):
    """Function to split string equation of form v1+v2+v3...
    """
    if lower:
      string = string.lower()
    string = "".join(string.split())
    comps = string.split('+')
    return comps