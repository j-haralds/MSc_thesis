import pandas as pd
from torchinfo import summary
import numpy as np


def exp_log(model,history, name, N_trajs, mse, mae):
    '''Log experiment results to a CSV file and save model summary to a text file.
    Args:
        model: The trained PyTorch model.
        history: A dictionary containing training history, including 'd_loss', 'p_loss', and 'time per ep'.
        name: A string name for the experiment (used in filenames).
        d_loss_type: A string describing the type of data loss used.
        p_loss: Optional; a string describing the type of physics loss used (if any).
    '''
    mean_mse_V = np.mean(mse[:,0])
    mean_mse_F = np.mean(mse[:,1])
    mean_mae_V = np.mean(mae[:,0])
    mean_mae_F = np.mean(mae[:,1]) 
    N_ep = len(history['loss'])
    train_time = np.sum(history['time per ep'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    df_existing = pd.read_csv("exp_log/multi_exps.csv")

    data = np.array([name,n_params,N_ep,train_time,N_trajs, mean_mse_V,mean_mse_F, mean_mae_V, mean_mae_F]).reshape(1,9)

    df_new = pd.DataFrame(data, columns=df_existing.columns.values)
    df_combined = pd.concat([df_existing,df_new], ignore_index=True)

        #df_combined.to_excel("exp_log/Euler_exps.xlsx", index=False)
    df_combined.to_csv("exp_log/multi_exps.csv", index=False)

    with open(f"exp_log/{name}.txt", "w") as file:
        file.write(f'{summary(model)}')

def get_previous_exps():
    df = pd.read_csv('exp_log/multi_exps.csv')
    return df