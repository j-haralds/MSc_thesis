import pandas as pd
from torchinfo import summary
import numpy as np


def exp_log(model,history, name, d_loss_type, p_loss = None):
    N_ep = len(history)
    d_loss = np.mean(history['d_loss'][-3:])
    train_time = np.sum(history['time per ep'])
    if p_loss is not None:
        p_loss = np.mean(history['p_loss'][-3:])
    else: 
        p_loss = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    df_existing = pd.read_csv("exp_log/Euler_exps.csv")

    data = np.array([name,n_params,N_ep,train_time,d_loss,d_loss_type, p_loss]).reshape(1,7)

    df_new = pd.DataFrame(data, columns=df_existing.columns.values)
    df_combined = pd.concat([df_existing,df_new], ignore_index=True)

        #df_combined.to_excel("exp_log/Euler_exps.xlsx", index=False)
    df_combined.to_csv("exp_log/Euler_exps.csv", index=False)

    with open(f"exp_log/{name}.txt", "w") as file:
        file.write(f'{summary(model)}')

def get_previous_exps():
    df = pd.read_csv('exp_log/Euler_exps.csv')
    return df