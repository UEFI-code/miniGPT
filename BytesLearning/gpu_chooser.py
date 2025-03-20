import os
import torch

def choose_gpu():
    # check macOS M1
    if os.name == "Darwin":
        return torch.device('mps')
    # check CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')