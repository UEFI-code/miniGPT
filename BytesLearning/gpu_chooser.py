import torch

def choose_gpu():
    # check macOS M1
    if torch.backends.mps.is_available():
        return torch.device('mps')
    # check CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')