import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
import sys
sys.path.append('..')
import GP

import pandas as pd
from torchinfo import summary
import numpy as np


def exp_log(model,history, name):
    '''Log experiment results to a CSV file and save model summary to a text file.
    Args:
        model: The trained PyTorch model.
        history: A dictionary containing training history, including 'd_loss', 'p_loss', and 'time per ep'.
        name: A string name for the experiment (used in filenames).
        d_loss_type: A string describing the type of data loss used.
        p_loss: Optional; a string describing the type of physics loss used (if any).
    '''

    N_ep = len(history['loss'])
    train_time = np.sum(history['time per ep'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    df_existing = pd.read_csv("comparisons/comp_exps.csv")
    rmse_noise = pd.read_csv(f'comparisons/comp_models/noise_robustness_{name}_train.csv')['mean_rmse'].values[-1]
    rmse = pd.read_csv(f'comparisons/comp_models/noise_robustness_{name}_train.csv')['mean_rmse'].values[0]
    
    # mean of the 5 last training RMSE values as an estimate of the final training RMSE
    trained_rmse = np.mean(history['loss'][-5:])
    data = np.array([name,n_params,N_ep,train_time,trained_rmse,rmse,rmse_noise ]).reshape(1,7)

    df_new = pd.DataFrame(data, columns=df_existing.columns.values)
    df_combined = pd.concat([df_existing,df_new], ignore_index=True)

        #df_combined.to_excel("exp_log/Euler_exps.xlsx", index=False)
    df_combined.to_csv("comparisons/comp_exps.csv", index=False)

    with open(f"comparisons/comp_models/{name}.txt", "w") as file:
        file.write(f'{summary(model)}')

def get_previous_exps():
    df = pd.read_csv('comparisons/comp_exps.csv')
    return df


def get_NN_params():
    epochs = 150
    lr = 1e-3
    threshold_rmse = 0.01
    return epochs, lr, threshold_rmse


def RMSE(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2) + 1e-8)


def gen_noise(i,u,noise_lvl = 0.):
    C_to_I = 4.72930472709413 / 1.9
    u_par_to_u = 2.587185069984447 / 18.0
    I_max = 5.0 * C_to_I
    u_max = 30. * u_par_to_u

    # Assumed measurement error for I: 5% of max current
    I_noise_std = noise_lvl * I_max
    u_noise_std = noise_lvl * u_max
    i_noise = torch.normal(0, I_noise_std, size=i.shape)
    u_noise = torch.normal(0, u_noise_std, size=u.shape)
    return i_noise, u_noise



