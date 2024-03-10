#!/usr/bin/python3

import torch

def get_device_name():
    device = "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    # elif torch.backends.cuda.is_available() and torch.backends.cuda.is_built():
    elif torch.backends.cuda.is_built():
        device = "cuda"
    
    return device