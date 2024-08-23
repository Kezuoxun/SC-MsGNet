import sys
import numpy as np
import torch

def get_model_parameters(net):
    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total number of parameters: {total_params}')

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Print Model Size: {size_all_mb:.3f} MB")