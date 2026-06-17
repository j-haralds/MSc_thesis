"""
Ue_GP.py — fast Ue(SOC) lookup for the battery NODE model.

Wraps sklearn's Gaussian process with a tabulated cache: the GP is evaluated
once on a dense SOC grid, and every subsequent lookup is an np.interp against
that grid (~150x faster than calling GP.predict directly, ~1 uV error).

----------
get_gp()                          -> raw sklearn GP model (slow path, for compat)
warm_cache(n_grid, soc_lo, soc_hi) -> pre-build the lookup grid (auto-fires on first use)
soc_to_Ue(soc, return_torch=False) -> main entry point; numpy or float32 torch
"""

import os
import numpy as np
import torch
import pandas as pd
import sklearn.gaussian_process as gp


_GP_MODEL = None
_GRID     = None
_VALUES   = None

# Anchor data lookups to this file's directory so the module keeps working
# after the project is moved. The notebook CWD is no longer relevant.
_HERE = os.path.dirname(os.path.abspath(__file__))


def get_data(file_name):
    path = os.path.join(_HERE, f"{file_name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Ue_GP.get_data: cannot find GP data file at {os.path.abspath(path)}. "
        )

    df = pd.read_csv(
        path,
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
    data = get_data('Ue_run_comsol')
    # `.values` returns a read-only view in modern pandas/NumPy, copy so we
    # can safely clip negative SOCs in-place.
    x_gp = data['soc'].values.copy()
    y_gp = data['Ue'].values.copy()
    x_gp[x_gp < 0] = 0
    kernel_GP = gp.kernels.RBF(length_scale=alpha[0]) * gp.kernels.ConstantKernel(constant_value=alpha[1])
    gp_model = gp.GaussianProcessRegressor(kernel=kernel_GP, optimizer=None, normalize_y=False)
    gp_model.fit(x_gp.reshape(-1, 1), y_gp.reshape(-1, 1))
    return gp_model


def get_gp():
    """Return the sklearn GP for Ue(SOC)."""
    global _GP_MODEL
    if _GP_MODEL is None:
        print("Loading GP for Comsol Ue(SOC) ...")
        _GP_MODEL = GP_process()
    return _GP_MODEL


def warm_cache(n_grid=2000, soc_lo=0.0, soc_hi=1.05, verbose=True):
    """Pre-tabulate Ue(SOC) on a dense grid. Call once explicitly if you want
    to control timing/resolution; otherwise auto-fires on first soc_to_Ue.

    The grid and values are published to the module globals only after the GP
    evaluation succeeds, so a failure here (e.g. a missing data file) cannot
    leave a half-built cache that breaks later np.interp lookups."""
    global _GRID, _VALUES
    if verbose:
        print(f"Pre-tabulating Ue(SOC) on {n_grid}-point grid over [{soc_lo}, {soc_hi}] ...")
    grid   = np.linspace(soc_lo, soc_hi, n_grid)
    values = np.asarray(get_gp().predict(grid.reshape(-1, 1))).reshape(-1)
    _GRID, _VALUES = grid, values


def soc_to_Ue(soc, return_torch=False):
    """Look up Ue(SOC). Accepts scalar / list / numpy array / torch tensor.
    Returns numpy of the same shape by default; float32 torch if requested.
    SOC values < 0 are clipped to 0 to match GP training; values above the
    grid upper bound are clamped (np.interp default)."""
    if _GRID is None or _VALUES is None:
        warm_cache()
    if isinstance(soc, torch.Tensor):
        soc_np = soc.detach().cpu().numpy()
    else:
        soc_np = np.asarray(soc)
    soc_flat = np.clip(np.asarray(soc_np, dtype=float).reshape(-1), 0.0, None)
    Ue_flat  = np.interp(soc_flat, _GRID, _VALUES)
    Ue       = Ue_flat.reshape(soc_np.shape)
    if return_torch:
        return torch.tensor(Ue, dtype=torch.float32)
    return Ue