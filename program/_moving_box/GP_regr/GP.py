import sklearn.gaussian_process as gp
import pandas as pd
import numpy as np
import torch

def get_data(file_name):
    df = pd.read_csv(
        f"../GP_regr/{file_name}.txt",
        sep=r"\s+",
        comment="%",
        header=None
    )

    df.columns = ['u_par','C','t','E_cell (V)','I_cell (A)','Rx_cell (N)','u_cell (m)','E_ocv_cell (V)','soc_cell (1)']
    df.columns = ['u_par', 'C', 't', 'V', 'I', 'F', 'u', 'Ue', 'soc']

    # Sort by time within each batch
    return df

def GP_process():
    sig = 1
    l = 0.1
    alpha = [l, sig]
    data = get_data('GP_run')

    x_gp = data['soc'].to_numpy().copy()
    y_gp = data['Ue'].to_numpy().copy()

    x_gp[x_gp<0] = 0 
    kernel_GP = gp.kernels.RBF(length_scale=alpha[0]) * gp.kernels.ConstantKernel(constant_value=alpha[1])
    gp_model = gp.GaussianProcessRegressor(kernel=kernel_GP,optimizer=None,normalize_y=False)
    gp_model.fit(x_gp.reshape(-1,1), y_gp.reshape(-1,1))
    return gp_model

def soc_to_Ue(soc, gp_model, return_torch = False):
    
    soc = np.asarray(soc)
    soc[soc<0] = 0; soc[soc>1] = 1
    if return_torch:
        return torch.from_numpy(gp_model.predict(np.asarray(soc).reshape(-1,1))).float()
    else:
        return gp_model.predict(soc.reshape(-1,1))