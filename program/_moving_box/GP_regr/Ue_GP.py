"""
Ue_GP.py — fast Ue(SOC) lookup for the battery NODE model.

Wraps JN_GP's Gaussian process with a tabulated cache: the GP is evaluated
once on a dense SOC grid, and every subsequent lookup is an np.interp against
that grid (~150× faster than calling GP.predict directly, ~1 µV error).

Public API
----------
get_gp()                          → raw sklearn GP model (slow path, for compat)
warm_cache(n_grid, soc_lo, soc_hi) → pre-build the lookup grid (auto-fires on first use)
soc_to_Ue(soc, return_torch=False) → main entry point; numpy or float32 torch
"""

import numpy as np
import torch
import JN_GP


_GP_MODEL = None
_GRID     = None
_VALUES   = None


def get_gp():
    """Return the sklearn GP for Ue(SOC)."""
    global _GP_MODEL
    if _GP_MODEL is None:
        print("Loading GP for Ue(SOC) ...")
        _GP_MODEL = JN_GP.GP_process()
    return _GP_MODEL


def warm_cache(n_grid=2000, soc_lo=0.0, soc_hi=1.05, verbose=True):
    """Pre-tabulate Ue(SOC) on a dense grid. Call once explicitly if you want
    to control timing/resolution; otherwise auto-fires on first soc_to_Ue."""
    global _GRID, _VALUES
    if verbose:
        print(f"Pre-tabulating Ue(SOC) on {n_grid}-point grid over [{soc_lo}, {soc_hi}] ...")
    grid     = np.linspace(soc_lo, soc_hi, n_grid)
    _GRID    = grid
    _VALUES  = get_gp().predict(grid.reshape(-1, 1)).reshape(-1)


def soc_to_Ue(soc, return_torch=False):
    """Look up Ue(SOC). Accepts scalar / list / numpy array / torch tensor.
    Returns numpy of the same shape by default; float32 torch if requested.
    SOC values < 0 are clipped to 0 to match GP training; values above the
    grid upper bound are clamped (np.interp default)."""
    if _GRID is None:
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
