import pandas as pd
from torchinfo import summary
import numpy as np


def exp_log(model,history,current, name, d_loss_type, p_loss = None):
    '''Log experiment results to a CSV file and save model summary to a text file.
    Args:
        model: The trained PyTorch model.
        history: A dictionary containing training history, including 'd_loss', 'p_loss', and 'time per ep'.
        name: A string name for the experiment (used in filenames).
        d_loss_type: A string describing the type of data loss used.
        p_loss: Optional; a string describing the type of physics loss used (if any).
    '''
    
    N_ep = len(history['d_loss'])
    d_loss = np.mean(history['d_loss'][-3:])
    train_time = np.sum(history['time per ep'])
    if p_loss is not None:
        p_loss = np.mean(history['p_loss'][-3:])
    else: 
        p_loss = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    df_existing = pd.read_csv("exp_log/PENN_exps.csv")

    data = np.array([name,current,n_params,N_ep,train_time,d_loss,d_loss_type, p_loss]).reshape(1,8)

    df_new = pd.DataFrame(data, columns=df_existing.columns.values)
    df_combined = pd.concat([df_existing,df_new], ignore_index=True)

        #df_combined.to_excel("exp_log/Euler_exps.xlsx", index=False)
    df_combined.to_csv("exp_log/PENN_exps.csv", index=False)

    with open(f"exp_log/{name}.txt", "w") as file:
        file.write(f'{summary(model)}')

def get_previous_exps():
    df = pd.read_csv('exp_log/PENN_exps.csv')
    return df