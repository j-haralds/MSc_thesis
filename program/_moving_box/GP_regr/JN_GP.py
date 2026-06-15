import sklearn.gaussian_process as gp
import pandas as pd
import numpy as np
import torch

def get_data(file_name):
    df = pd.read_csv(
        f"../Multi_data/other/{file_name}.txt",
        sep=r'\s+',
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
    # `.values` returns a read-only view in modern pandas/NumPy — copy so we
    # can safely clip negative SOCs in-place.
    x_gp = data['soc'].values.copy()
    y_gp = data['Ue'].values.copy()
    x_gp[x_gp < 0] = 0
    kernel_GP = gp.kernels.RBF(length_scale=alpha[0]) * gp.kernels.ConstantKernel(constant_value=alpha[1])
    gp_model = gp.GaussianProcessRegressor(kernel=kernel_GP, optimizer=None, normalize_y=False)
    gp_model.fit(x_gp.reshape(-1, 1), y_gp.reshape(-1, 1))
    return gp_model



def soc_to_Ue(soc, gp_model, return_torch=False):
    soc = np.asarray(soc)
    if return_torch:
        return torch.from_numpy(gp_model.predict(np.asarray(soc).reshape(-1, 1))).float()
    else:
        return gp_model.predict(soc.reshape(-1, 1))
