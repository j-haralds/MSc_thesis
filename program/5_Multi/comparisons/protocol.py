import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np


def get_NN_params():
    epochs = 150
    lr = 1e-3
    threshold_rmse = 0.01
    return epochs, lr, threshold_rmse


def RMSE(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2) + 1e-8)