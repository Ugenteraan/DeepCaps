'''
Helper functions.
'''

import os
import torch


def check_path(path):
    '''
    Checks if a given path exists. If not, create the directory.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def get_device():
    '''
    Checks if GPU is available to be used. If not, CPU is used.
    '''
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def onehot_encode(tensor, num_classes, device):
    '''
    Encodes the given tensor into one-hot vectors.
    '''
    return torch.eye(num_classes).to(device).index_select(dim=0, index=tensor.to(device))

