
#  Same physics as the original, but with batched training:
#  all trajectories (padded to equal length) are integrated
#  simultaneously, giving ~5-10× speedup on CPU.
#
#  Physics:
#      SOC(t)  = SOC0 − I·t/Q0                     (analytical)
#      U1(0)   = 0
#      U1(n+1) = U1(n) + I/C1 − U1(n)/(R1(n)·C1)  (Euler, dt=1s)
#      V(n)    = Ue(SOC(n)) − I·R0 − U1(n)
#
#  Learned:  R1Net(SOC, I, u) → R1 > 0   (small feedforward NN)
#            C1                            (one scalar)
#  Known:    Ue(SOC) from data,  R0(u, I) fitted function

import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
import time as _time

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# FILE_PATH = os.getcwd()
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()


DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, '2_merged_data.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')

Q0          = 17921.57581
TRAIN_SPLIT = 0.8
N_HIDDEN    = 32
EPOCHS      = 2
LR          = 1e-3
BATCH_SIZE  = 1        # Trajectories per batch


# ══════════════════════════════════════════════════════════
#  R1 NETWORK
# ══════════════════════════════════════════════════════════════

class R1Net(nn.Module):
    """(SOC, I, u) → R1 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        # Works for any shape — just needs matching last dims
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.01 + 1e-5
    
class R1NetConstrained(nn.Module):
    """(SOC, I, u) → R1 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, config, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.R1_min = config.get('R1_min')
        self.R1_max = config.get('R1_max')
        print(f'R1 constrained to [{self.R1_min}, {self.R1_max}] Ohm')

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
        return self.R1_min + s * (self.R1_max - self.R1_min)
    
# ══════════════════════════════════════════════════════════
#  C1 NETWORK
# ══════════════════════════════════════════════════════════

class C1Net(nn.Module):
    """(SOC, I, u) → C1 > 0  [F].  One hidden layer, softplus output."""
    def __init__(self, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        # Works for any shape — just needs matching last dims
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 2000

class C1NetConstrained(nn.Module):
    """(SOC, I, u) → C1 > 0  [F].  One hidden layer, softplus output, linear constraint."""
    def __init__(self, config, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.C1_min = config.get('C1_min')
        self.C1_max = config.get('C1_max')
        print(f'C1 constrained to [{self.C1_min}, {self.C1_max}] F')

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
        return self.C1_min + s * (self.C1_max - self.C1_min)

# ══════════════════════════════════════════════════════════
#  R0 NETWORK
# ══════════════════════════════════════════════════════════════

class R0Net(nn.Module):
    """(SOC, I, u) → R0 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        # Works for any shape — just needs matching last dims
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.01 + 1e-5
    
class R0NetConstrained(nn.Module):
    """(SOC, I, u) → R0 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, config, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.R0_min = config.get('R0_min')
        self.R0_max = config.get('R0_max')
        print(f'R0 constrained to [{self.R0_min}, {self.R0_max}] Ohm')

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
        return self.R0_min + s * (self.R0_max - self.R0_min)

def R0_func(u, I):
    return u * (-0.0001887521) - 7.049519e-5 * I + 0.008446693

class R0NetNoSOC(nn.Module):
    """(I, u) → R0 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, config, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.R0_min = config.get('R0_min')
        self.R0_max = config.get('R0_max')
        if config.get('R0_constrained', 'false') == 'true':
            print(f'R0 constrained to [{self.R0_min}, {self.R0_max}] Ohm')
        else:
            print('R0 unconstrained')

    def forward(self, I_norm, u):
        x = torch.stack([I_norm, u], dim=-1)   # (..., 2)
        if self.config.get('R0_constrained', 'false') == 'true':
            s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.R0_min + s * (self.R0_max - self.R0_min)
        else:
            return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.01 + 1e-5
    
# ══════════════════════════════════════════════════════════
#  k NETWORK (static)
# ══════════════════════════════════════════════════════════════

class kNet(nn.Module):
    """(SOC, I, u) → k > 0  [GN/mm].  Algebraic — no integration.

    Predicts the instantaneous stiffness k(SOC, I, u).  The reaction force
    follows directly as F_r = -k * u.

    The reference k0 (initial elastic baseline from the data, e.g. -F(0)/u(0))
    is stored as `self.k` so the swelling decomposition F_s = (k0 - k)*u and
    plotting code can refer to it.  k0 is NOT used as an initial condition —
    there is no longer any state to initialise.
    """
    def __init__(self, n_hidden=32, k=53.0, k_scale=None):
        super().__init__()
        self.k = float(k)                           # reference k0 from data
        # If k_scale not given, default to k0 itself so softplus output ~ O(1)
        # gives k values around k0.
        self.k_scale = float(k_scale) if k_scale is not None else float(k)
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        return self.net(x).squeeze(-1) * self.k_scale

# ══════════════════════════════════════════════════════════
#  BATCHED ECMM MODEL
# ════════════════════════════════════════════════════════════

class BatteryECMM(nn.Module):
    """
    Batched ECMM:  forward() now accepts tensors of shape (B,) for
    scalar-per-trajectory quantities and returns (B, T_max) tensors.

    The Euler loop steps ALL trajectories simultaneously at each
    timestep — no per-trajectory Python loop.
    """
    def __init__(self, config, Ue_interp, R0_func, Q0, C1_init=1500.0, I_ref=20.0, k=53.0):
        super().__init__()
        self.Ue_interp = Ue_interp
        self.Q0        = Q0
        self.I_ref     = I_ref
        self.k         = k
        self.config    = config
        nh = config.get('n_hidden', 32)

        # ── k network (always; static algebraic stiffness) ──
        k_scale = config.get('k_scale', None)        # default = k0
        self.k_net = kNet(n_hidden=nh, k=k, k_scale=k_scale)

        # ── R1 net — always network, optionally constrained ──
        if config.get('R1_constrained', 'false') == 'true':
            self.r1_net = R1NetConstrained(config, n_hidden=nh, I_ref=I_ref)
        else:
            print('R1 unconstrained')
            self.r1_net = R1Net(n_hidden=nh, I_ref=I_ref)

        # ── C1 net — always network, optionally constrained ──
        if config.get('C1_constrained', 'false') == 'true':
            self.C1_net = C1NetConstrained(config, n_hidden=nh, I_ref=I_ref)
        else:
            print('C1 unconstrained')
            self.C1_net = C1Net(n_hidden=nh, I_ref=I_ref)

        # ── R0 — multiple modes still supported ──
        m = config['R0_mode']
        if m == 'net':
            if config.get('R0_constrained', 'false') == 'true':
                self.R0_net = R0NetConstrained(config, n_hidden=nh, I_ref=I_ref)
            else:
                print('R0 unconstrained')
                self.R0_net = R0Net(n_hidden=nh, I_ref=I_ref)
        elif m == 'func':
            self.R0_func = R0_func
        elif m == 'param':
            self.log_R0 = nn.Parameter(torch.tensor(np.log(config.get('R0_param', 0.01)), dtype=torch.float32))
        elif m == 'net_no_soc':
            self.R0_net = R0NetNoSOC(config, n_hidden=nh, I_ref=I_ref)
        else:
            raise ValueError(f"Unknown R0_mode: {m!r}. Use 'net', 'func', 'param', or 'net_no_soc'.")

    # ── Dispatchers ──
    def _R1(self, soc, I_norm, u):
        return self.r1_net(soc, I_norm, u)

    def _C1(self, soc, I_norm, u):
        return self.C1_net(soc, I_norm, u)

    def _R0(self, soc, I_norm, u, I_batch, u_batch):
        """Returns R0 broadcastable to (B, T)."""
        m = self.config['R0_mode']
        if m == 'net':
            return self.R0_net(soc, I_norm, u)
        if m == 'param':
            return torch.exp(self.log_R0).expand_as(soc)
        if m == 'func':
            B = I_batch.shape[0]
            return torch.tensor(
                [self.R0_func(u_batch[b].item(), I_batch[b].item()) for b in range(B)],
                dtype=torch.float32).unsqueeze(1)
        if m == 'net_no_soc':
            return self.R0_net(I_norm, u)

    def forward(self, I_batch, u_batch, soc0_batch, T_max, V_mode='dynamic'):
        """
        V_mode : 'dynamic' — full Euler U1 integration  (Stage 2 / production)
                 'static'  — algebraic U1 = I·R1        (Stage 1 / Brucker)
                             C1_net is NOT called in this mode — guarantees
                             C1 plays no role during Stage 1 training.
        F is dynamic in both modes (k integrated via Euler).
        """
        B = I_batch.shape[0]
        t_idx = torch.arange(T_max, dtype=torch.float32).unsqueeze(0)
        soc = soc0_batch.unsqueeze(1) - I_batch.unsqueeze(1) / self.Q0 * t_idx
        I_norm = (I_batch / self.I_ref).unsqueeze(1).expand(B, T_max)
        u_exp  = u_batch.unsqueeze(1).expand(B, T_max)

        # Parameters along the trajectory  (B, T_max)
        R1 = self._R1(soc, I_norm, u_exp)
        R0 = self._R0(soc, I_norm, u_exp, I_batch, u_batch)

        # ── F branch (static): k is an algebraic function, no state ──
        k = self.k_net(soc, I_norm, u_exp)              # (B, T_max)

        # ── V branch: static or dynamic U1 ──
        if V_mode == 'static':
            # Steady-state of the RC: U1 = I · R1.  C1 is *not* used.
            U1 = I_batch.unsqueeze(1) * R1
        elif V_mode == 'dynamic':
            C1 = self._C1(soc, I_norm, u_exp)
            U1_steps = [torch.zeros(B)]
            for n in range(T_max - 1):
                C1_n = C1[:, n] if C1.ndim == 2 else C1
                # Semi-implicit Euler — unconditionally stable
                U1_next = (U1_steps[n] + I_batch / C1_n) / (1.0 + 1.0 / (R1[:, n] * C1_n))
                U1_steps.append(U1_next)
            U1 = torch.stack(U1_steps, dim=1)
        else:
            raise ValueError(f"V_mode must be 'static' or 'dynamic', got {V_mode!r}")

        with torch.no_grad():
            Ue = torch.tensor(self.Ue_interp(soc.detach().numpy()), dtype=torch.float32)

        V  = Ue - I_batch.unsqueeze(1) * R0 - U1
        Fr = -k * u_exp
        # Fs = swelling-induced part beyond the elastic baseline (-k0·u),
        # so Fs_pred matches Fs_true_plot = F_data + k0·u
        Fs = (self.k_net.k - k) * u_exp

        return V, Fr, soc, U1, R1, Fs

    # Keep single-trajectory forward for inference / plotting
    def forward_single(self, I_val, u_val, soc0_val, T):
        """Convenience wrapper matching the original forward() signature."""
        I_b    = torch.tensor([I_val], dtype=torch.float32)
        u_b    = torch.tensor([u_val], dtype=torch.float32)
        soc0_b = torch.tensor([soc0_val], dtype=torch.float32)
        V, Fr, soc, U1, R1, Fs = self.forward(I_b, u_b, soc0_b, T)
        return V[0], Fr[0], soc[0], U1[0], R1[0], Fs[0]
    
    def forward_pulse(self, I_seq, u_batch, soc0_batch):
        """
        I_seq      : (B, T) — current per trajectory per timestep
        u_batch    : (B,)
        soc0_batch : (B,)
        """
        B, T = I_seq.shape

        # SOC cumulative integration
        dsoc = -I_seq / self.Q0
        soc = soc0_batch.unsqueeze(1) + torch.cumsum(dsoc, dim=1) - dsoc[:, :1]

        I_norm = I_seq / self.I_ref
        u_exp  = u_batch.unsqueeze(1).expand(B, T)

        R1 = self._R1(soc, I_norm, u_exp)
        C1 = self._C1(soc, I_norm, u_exp)

        m = self.config['R0_mode']
        if m == 'net':
            R0 = self.R0_net(soc, I_norm, u_exp)
        elif m == 'param':
            R0 = torch.exp(self.log_R0).expand_as(soc)
        elif m == 'func':
            R0 = (u_exp * (-0.0001887521) - 7.049519e-5 * I_seq + 0.008446693)
        elif m == 'net_no_soc':
            R0 = self.R0_net(I_norm, u_exp)

        # k is algebraic (static)
        k = self.k_net(soc, I_norm, u_exp)              # (B, T)

        U1_steps = [torch.zeros(B)]
        for n in range(T - 1):
            C1_n = C1[:, n] if C1.ndim == 2 else C1
            # Semi-implicit Euler for U1
            U1_next = (U1_steps[n] + I_seq[:, n] / C1_n) / (1.0 + 1.0 / (R1[:, n] * C1_n))
            U1_steps.append(U1_next)
        U1 = torch.stack(U1_steps, dim=1)

        with torch.no_grad():
            Ue = torch.tensor(self.Ue_interp(soc.detach().numpy()), dtype=torch.float32)

        V  = Ue - I_seq * R0 - U1
        Fr = -k * u_exp
        Fs = (self.k_net.k - k) * u_exp

        return V, Fr, soc, U1, R1, Fs

def get_C1(model, scalar=True, soc_ref=0.5, I_ref_val=10.0, u_ref=-0.06,
           soc=None, I_norm=None, u_exp=None):
    """Return a representative C1 value.

    scalar=True  → a single float at the (soc_ref, I_ref_val, u_ref) reference point
    scalar=False → trajectory-shape numpy array, evaluated at the given (soc, I_norm, u_exp)
    """
    if scalar:
        soc_t  = torch.tensor([soc_ref], dtype=torch.float32)
        I_norm_t = torch.tensor([I_ref_val / model.I_ref], dtype=torch.float32)
        u_t    = torch.tensor([u_ref], dtype=torch.float32)
        with torch.no_grad():
            return model._C1(soc_t, I_norm_t, u_t).mean().item()
    else:
        return model._C1(soc, I_norm, u_exp).detach().numpy()

# ══════════════════════════════════════════════════════════
#  DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════

def prepare_data(data, R0_func):
    trajs = []
    for _, grp in data.sort_values(['trajectory', 't']).groupby('trajectory'):
        grp = grp.reset_index(drop=True)
        I_val, u_val = float(grp['I'].iloc[0]), float(grp['u'].iloc[0])
        C_val  = float(grp['C'].iloc[0])
        u_per = float(grp['u_par'].iloc[0])      
        R0_val = R0_func(u_val, I_val)
        trajs.append(dict(
            I=I_val, u=u_val, C=C_val, u_per=u_per,
            soc0=float(grp['soc'].iloc[0]), T=len(grp),
            V=torch.tensor(grp['V'].values, dtype=torch.float32),
            F=torch.tensor(grp['F'].values, dtype=torch.float32),
            soc=torch.tensor(grp['soc'].values, dtype=torch.float32),
            U1_true=torch.tensor(grp['Ue'].values - I_val * R0_val - grp['V'].values,
                                 dtype=torch.float32),
        ))
    return trajs

def prepare_pulse_data(pulse_raw):
    pulse_trajs = []
    for _, grp in pulse_raw.sort_values(['trajectory', 't']).groupby('trajectory'):
        grp = grp.reset_index(drop=True)
        pulse_trajs.append(dict(
            I_seq = torch.tensor(grp['I'].values,   dtype=torch.float32),  # sequence!
            u     = float(grp['u'].iloc[0]),
            u_per = float(grp['u_par'].iloc[0]),    
            soc0  = float(grp['soc'].iloc[0]),
            T     = len(grp),
            t     = torch.tensor(grp['t'].values,   dtype=torch.float32),
            V     = torch.tensor(grp['V'].values,   dtype=torch.float32),
            F     = torch.tensor(grp['F'].values,   dtype=torch.float32),
            soc   = torch.tensor(grp['soc'].values, dtype=torch.float32)
        ))
    return pulse_trajs


def estimate_C1(trajs):
    ests = []
    for tr in trajs:
        U1 = tr['U1_true'].numpy()
        U1_ss = np.mean(U1[-max(20, len(U1)//20):])
        R1_ss = U1_ss / tr['I'] if tr['I'] > 0 else np.nan
        target = 0.632 * U1_ss
        idx = np.argmax(U1 > target)
        if idx > 0 and R1_ss > 1e-6:
            ests.append(idx / R1_ss)
    return float(np.median(ests)) if ests else 30000.0

# ══════════════════════════════════════════════════════════
#  BATCH COLLATION
# ══════════════════════════════════════════════════════════════

def collate_batch(trajs):
    """
    Pack a list of trajectory dicts into padded tensors + mask.

    Returns
    -------
    I_batch    : (B,)
    u_batch    : (B,)
    soc0_batch : (B,)
    V_batch    : (B, T_max)    padded with 0
    Fr_batch   : (B, T_max)    padded with 0
    mask       : (B, T_max)    True where data exists
    T_max      : int
    """
    B = len(trajs)
    T_max = max(tr['T'] for tr in trajs)

    I_batch    = torch.tensor([tr['I']    for tr in trajs], dtype=torch.float32)
    u_batch    = torch.tensor([tr['u']    for tr in trajs], dtype=torch.float32)
    soc0_batch = torch.tensor([tr['soc0'] for tr in trajs], dtype=torch.float32)

    V_batch  = torch.zeros(B, T_max)
    Fr_batch = torch.zeros(B, T_max)
    mask     = torch.zeros(B, T_max, dtype=torch.bool)

    for i, tr in enumerate(trajs):
        T = tr['T']
        V_batch[i, :T]  = tr['V']
        Fr_batch[i, :T] = tr['F']
        mask[i, :T]     = True

    return I_batch, u_batch, soc0_batch, V_batch, Fr_batch, mask, T_max


def collate_batch_pulse(trajs):
    """
    Pulse-data analogue of collate_batch.  Current is a *sequence* per traj.

    Returns
    -------
    I_seq_batch : (B, T_max)   padded with 0
    u_batch     : (B,)
    soc0_batch  : (B,)
    V_batch     : (B, T_max)   padded with 0
    Fr_batch    : (B, T_max)   padded with 0
    mask        : (B, T_max)   True where data exists
    T_max       : int
    """
    B = len(trajs)
    T_max = max(tr['T'] for tr in trajs)

    I_seq_batch = torch.zeros(B, T_max)
    u_batch     = torch.tensor([tr['u']    for tr in trajs], dtype=torch.float32)
    soc0_batch  = torch.tensor([tr['soc0'] for tr in trajs], dtype=torch.float32)

    V_batch  = torch.zeros(B, T_max)
    Fr_batch = torch.zeros(B, T_max)
    mask     = torch.zeros(B, T_max, dtype=torch.bool)

    for i, tr in enumerate(trajs):
        T = tr['T']
        I_seq_batch[i, :T] = tr['I_seq']
        V_batch[i, :T]     = tr['V']
        Fr_batch[i, :T]    = tr['F']
        mask[i, :T]        = True

    return I_seq_batch, u_batch, soc0_batch, V_batch, Fr_batch, mask, T_max

# ══════════════════════════════════════════════════════════
#  TRAINING FUNCTION  (batched)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS  (batched, with staged option)
# ══════════════════════════════════════════════════════════════

def _empty_history():
    return {'train': [], 'train_V': [], 'train_Fr': [],
            'test': [], 'time': 0.0,
            'train_rmse': [], 'train_rmse_V': [], 'train_rmse_Fr': [],
            'test_rmse': [],
            'stage': []}                       # which stage produced each epoch


def _train_inner(model, train_trajs, test_trajs,
                 optimizer, scheduler, n_epochs, batch_size, print_every,
                 V_mode='dynamic', stage_label='', history=None,
                 pulse=False):
    """
    One training pass with the given optimizer over n_epochs.
    V_mode is forwarded to model() so the same loop fits Stage 1 (static V)
    and Stage 2 (dynamic V).  F is dynamic in both modes.

    pulse=True switches to pulse-trajectory mode: trajectories carry an
    `I_seq` (per-timestep current) and the model is rolled out via
    forward_pulse() instead of the constant-current forward().  V_mode is
    ignored in this case (forward_pulse is always dynamic).
    """
    if history is None:
        history = _empty_history()
    epoch_offset = len(history['train'])
    t0 = _time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_trajs))
        ep_mse = ep_mse_V = ep_mse_Fr = 0.0
        ep_rmse = ep_rmse_V = ep_rmse_Fr = 0.0
        n_batches = 0

        for start in range(0, len(train_trajs), batch_size):
            idxs  = order[start : start + batch_size]
            batch = [train_trajs[i] for i in idxs]

            optimizer.zero_grad()
            if pulse:
                I_seq_b, u_b, soc0_b, V_true, Fr_true, mask, T_max = collate_batch_pulse(batch)
                V_pred, Fr_pred, _, _, _, _ = model.forward_pulse(I_seq_b, u_b, soc0_b)
            else:
                I_b, u_b, soc0_b, V_true, Fr_true, mask, T_max = collate_batch(batch)
                V_pred, Fr_pred, _, _, _, _ = model(I_b, u_b, soc0_b, T_max, V_mode=V_mode)

            sq_err_V  = (V_pred  - V_true ) ** 2
            sq_err_Fr = (Fr_pred - Fr_true) ** 2
            loss_V  = sq_err_V[mask].mean()
            loss_Fr = sq_err_Fr[mask].mean()
            loss = loss_V + loss_Fr

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rmse_V  = torch.sqrt(loss_V).item()
                rmse_Fr = torch.sqrt(loss_Fr).item()

            ep_mse    += loss.item()
            ep_mse_V  += loss_V.item()
            ep_mse_Fr += loss_Fr.item()
            ep_rmse   += rmse_V + rmse_Fr
            ep_rmse_V += rmse_V
            ep_rmse_Fr+= rmse_Fr
            n_batches += 1

        ep_mse /= n_batches; ep_mse_V /= n_batches; ep_mse_Fr /= n_batches
        ep_rmse /= n_batches; ep_rmse_V /= n_batches; ep_rmse_Fr /= n_batches

        history['train'].append(ep_mse)
        history['train_V'].append(ep_mse_V)
        history['train_Fr'].append(ep_mse_Fr)
        history['train_rmse'].append(ep_rmse)
        history['train_rmse_V'].append(ep_rmse_V)
        history['train_rmse_Fr'].append(ep_rmse_Fr)
        history['stage'].append(stage_label)

        # Test eval — uses same V_mode as training stage (pulse always dynamic)
        model.eval()
        with torch.no_grad():
            test_mse = 0.0
            for tr in test_trajs:
                if pulse:
                    I_seq_b = tr['I_seq'].unsqueeze(0)
                    u_b     = torch.tensor([tr['u']],    dtype=torch.float32)
                    soc0_b  = torch.tensor([tr['soc0']], dtype=torch.float32)
                    V_pred, _, _, _, _, _ = model.forward_pulse(I_seq_b, u_b, soc0_b)
                else:
                    I_b    = torch.tensor([tr['I']],    dtype=torch.float32)
                    u_b    = torch.tensor([tr['u']],    dtype=torch.float32)
                    soc0_b = torch.tensor([tr['soc0']], dtype=torch.float32)
                    V_pred, _, _, _, _, _ = model(I_b, u_b, soc0_b, tr['T'], V_mode=V_mode)
                test_mse += torch.mean((V_pred[0] - tr['V']) ** 2).item()
            test_mse /= len(test_trajs)
            test_rmse = float(np.sqrt(test_mse))
        history['test'].append(test_mse)
        history['test_rmse'].append(test_rmse)

        if scheduler is not None:
            scheduler.step(ep_mse)

        if epoch % print_every == 0 or epoch == 1:
            C1 = get_C1(model, scalar=True)
            eta = (_time.time() - t0) / epoch * (n_epochs - epoch) / 60
            tag = f"[{stage_label}] " if stage_label else ""
            print(f"  {tag}{epoch:4d}/{n_epochs} | ETA {eta:.1f}m "
                  f"| RMSE V {ep_rmse_V:.4f} Fr {ep_rmse_Fr:.4f} "
                  f"| test V {test_rmse:.4f} | C1={C1:.0f}F")

    history['time'] = history.get('time', 0.0) + (_time.time() - t0) / 60
    return history


def train_model(model, train_trajs, test_trajs,
                n_epochs=200, lr=1e-3, batch_size=16, print_every=10,
                V_mode='dynamic'):
    """Single-stage training (default behaviour: dynamic V from epoch 1)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)
    return _train_inner(model, train_trajs, test_trajs,
                        optimizer, scheduler, n_epochs, batch_size, print_every,
                        V_mode=V_mode, stage_label='single')


def train_staged(model, train_trajs, test_trajs,
                 n_epochs_static=100, n_epochs_dynamic=200,
                 lr_static=1e-3, lr_dynamic=1e-3,
                 batch_size=16, print_every=10,
                 on_stage1_done=None,
                 # ── pulse-data extension (Stage 2 only) ──
                 pulse_train_trajs=None, pulse_test_trajs=None,
                 # ── optional Stage 2b: unfreeze R1 and keep training ──
                 n_epochs_unfreeze=0, lr_unfreeze=None,
                 on_stage2_done=None):
    """
    Brucker-style two-stage training, with optional pulse-data Stage 2 and an
    optional R1-unfreeze sub-phase (Stage 2b).

    Stage 1  (static V, CC trajectories):
        V = Ue − I·R0 − I·R1     (algebraic; no U1 dynamics)
        F = -k(SOC, I, u) · u    (algebraic — no integration)
        Trains: r1_net, R0_net (if R0_mode is a net), k_net
        Frozen: C1_net

    Stage 2  (dynamic V):
        V uses full Euler U1 integration with dU1/dt = I/C1 − U1/(R1·C1)
        F unchanged (still algebraic)
        Trains: C1_net
        Frozen: r1_net, R0_net (if a net), k_net

        Data: `pulse_train_trajs` if provided (pulse current sequences via
        forward_pulse), else the constant-current `train_trajs`.

    Stage 2b (optional, only if n_epochs_unfreeze > 0):
        Same dynamics + data as Stage 2, but r1_net is unfrozen so R1 gets
        refined under the dynamic objective.  R0_net and k_net stay frozen.
        Uses lr_unfreeze (defaults to lr_dynamic).

    on_stage1_done : optional callable(model, history) invoked between Stage 1
                     and Stage 2 (useful for plotting the post-Stage-1 state).
    on_stage2_done : optional callable(model, history) invoked between Stage 2
                     and Stage 2b.

    History is concatenated across stages; the 'stage' key marks each epoch
    ('S1' / 'S2' / 'S2b').
    """
    history = _empty_history()

    # ── Stage 1 (always CC trajectories, static V) ──
    print(f"\n========== STAGE 1: static V  ({n_epochs_static} epochs) ==========")
    s1_params = [p for name, p in model.named_parameters() if 'C1_net' not in name]
    n_s1 = sum(p.numel() for p in s1_params)
    print(f"  Stage 1 trainable params: {n_s1}  (C1_net frozen)")
    opt1 = torch.optim.Adam(s1_params, lr=lr_static)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=40, factor=0.5)
    _train_inner(model, train_trajs, test_trajs,
                 opt1, sched1, n_epochs_static, batch_size, print_every,
                 V_mode='static', stage_label='S1', history=history)

    history['stage1_epochs'] = n_epochs_static

    # ── Optional callback between stages ──
    if on_stage1_done is not None:
        print("\n---- post-Stage-1 callback ----")
        on_stage1_done(model, history)

    # ── Stage 2 data routing ──
    use_pulse = pulse_train_trajs is not None
    s2_train  = pulse_train_trajs if use_pulse else train_trajs
    if use_pulse:
        if pulse_test_trajs is None:
            print("  Warning: pulse_train_trajs given but pulse_test_trajs is None — "
                  "evaluating Stage 2 on pulse_train_trajs.")
            s2_test = pulse_train_trajs
        else:
            s2_test = pulse_test_trajs
    else:
        s2_test = test_trajs
    data_tag = 'pulse' if use_pulse else 'CC'

    # ── Stage 2 (R1, R0_net, k frozen — train C1) ──
    print(f"\n========== STAGE 2: dynamic V  ({n_epochs_dynamic} epochs, {data_tag} data) ==========")
    freeze_kw = ('r1_net', 'R0_net', 'k_net')   # R0_net only exists if R0_mode is net or net_no_soc
    s2_params = [p for name, p in model.named_parameters()
                 if not any(kw in name for kw in freeze_kw)]
    n_s2 = sum(p.numel() for p in s2_params)
    print(f"  Stage 2 trainable params: {n_s2}  (r1_net{', R0_net' if hasattr(model,'R0_net') else ''}, k_net frozen)")
    opt2 = torch.optim.Adam(s2_params, lr=lr_dynamic)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=40, factor=0.5)
    _train_inner(model, s2_train, s2_test,
                 opt2, sched2, n_epochs_dynamic, batch_size, print_every,
                 V_mode='dynamic', stage_label='S2', history=history,
                 pulse=use_pulse)

    history['stage2_epochs'] = n_epochs_dynamic

    # ── Optional callback between Stage 2 and Stage 2b ──
    if on_stage2_done is not None:
        print("\n---- post-Stage-2 callback ----")
        on_stage2_done(model, history)

    # ── Stage 2b (optional): unfreeze R1, keep R0_net + k_net frozen ──
    if n_epochs_unfreeze and n_epochs_unfreeze > 0:
        lr_b = lr_unfreeze if lr_unfreeze is not None else lr_dynamic
        print(f"\n========== STAGE 2b: dynamic V, R1 UNFROZEN  "
              f"({n_epochs_unfreeze} epochs, {data_tag} data, lr={lr_b}) ==========")
        freeze_kw_b = ('R0_net', 'k_net')   # R1 + C1 trainable
        s2b_params = [p for name, p in model.named_parameters()
                      if not any(kw in name for kw in freeze_kw_b)]
        n_s2b = sum(p.numel() for p in s2b_params)
        print(f"  Stage 2b trainable params: {n_s2b}  (R1 unfrozen; "
              f"{'R0_net, ' if hasattr(model,'R0_net') else ''}k_net still frozen)")
        opt2b = torch.optim.Adam(s2b_params, lr=lr_b)
        sched2b = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2b, patience=40, factor=0.5)
        _train_inner(model, s2_train, s2_test,
                     opt2b, sched2b, n_epochs_unfreeze, batch_size, print_every,
                     V_mode='dynamic', stage_label='S2b', history=history,
                     pulse=use_pulse)
        history['stage2b_epochs'] = n_epochs_unfreeze

    return history

# ══════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS  (standalone numpy predict — no batched forward)
# ══════════════════════════════════════════════════════════════

def gen_noise(i,u,noise_lvl = 0.):
    C_to_I = 4.72930472709413 / 1.9
    u_par_to_u = 2.587185069984447 / 18.0
    I_max = 5.0 * C_to_I
    u_max = 30. * u_par_to_u

    I_noise_std = noise_lvl * I_max
    u_noise_std = noise_lvl * u_max
    i_noise = torch.normal(0, I_noise_std, size=i.shape)
    u_noise = torch.normal(0, u_noise_std, size=u.shape)
    print(i_noise, u_noise)
    return i_noise, u_noise


@torch.no_grad()
def predict_np(model, config, I_val, u_val, soc0, T,
               noise=False, noise_lvl=0.00, V_mode='dynamic'):
    """Single-trajectory rollout for plotting.

    V_mode='static' uses the Stage-1 V equation (U1 = I·R1) and returns C1=None
    so plotting can omit any C1-dependent panels.
    """
    I_b    = torch.tensor([I_val],  dtype=torch.float32)
    u_b    = torch.tensor([u_val],  dtype=torch.float32)
    soc0_b = torch.tensor([soc0],   dtype=torch.float32)

    if noise:
        i_noise, u_noise = gen_noise(I_b, u_b, noise_lvl=noise_lvl)
        I_b += i_noise
        u_b += u_noise

    V, Fr, soc, U1, R1, Fs = model(I_b, u_b, soc0_b, T, V_mode=V_mode)
    V   = V[0].numpy();   Fr = Fr[0].numpy()
    soc = soc[0].numpy(); U1 = U1[0].numpy()
    R1  = R1[0].numpy();  Fs = Fs[0].numpy()

    soc_t  = torch.from_numpy(soc)
    I_norm = torch.full((T,), I_val / model.I_ref)
    u_t    = torch.full((T,), u_val)
    k      = model.k_net(soc_t, I_norm, u_t).numpy()

    if V_mode == 'static':
        C1 = None       # C1 is meaningless in Stage 1 — don't evaluate it
    else:
        C1 = get_C1(model, scalar=False, soc=soc_t, I_norm=I_norm, u_exp=u_t)

    if config['R0_mode'] in ('net', 'net_no_soc', 'param'):
        R0 = model._R0(soc_t, I_norm, u_t, 0, 0).numpy()
    else:
        R0 = None  # plotting path uses R0_func directly when None

    return V, soc, U1, R1, Fs, Fr, k, C1, R0


def plot_predictions(model, config, trajs, time=False, noise=False,
                     noise_lvl=0.00, title='', n_show=3, V_mode='dynamic'):
    """Per-trajectory diagnostic grid.

    V_mode='dynamic' → 8 rows (V, U1, dU1/dt, R, C1, Fr, Fs, k)
    V_mode='static'  → 6 rows (V, U1, R, Fr, Fs, k)
                       — dU1/dt and C1 are omitted because they have no
                         meaning in Stage 1 (U1 is algebraic, C1 unused).
    """
    if V_mode == 'static':
        rows = ['V', 'U1', 'R', 'Fr', 'Fs', 'k']
    elif V_mode == 'dynamic':
        rows = ['V', 'U1', 'dU1', 'R', 'C1', 'Fr', 'Fs', 'k']
    else:
        raise ValueError(f"V_mode must be 'static' or 'dynamic', got {V_mode!r}")

    n = min(n_show, len(trajs))
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(5 * n, 3.3 * n_rows), squeeze=False)
    model.eval()
    k0 = model.k_net.k                      # scalar reference k0 from data

    for j in range(n):
        tr = trajs[j]
        V, soc_np, U1, R1, Fs, Fr, k_pred, C1, R0 = predict_np(
            model, config, tr['I'], tr['u'], tr['soc0'], tr['T'],
            noise=noise, noise_lvl=noise_lvl, V_mode=V_mode)

        # x-axis: SOC (default) or sample index (time=True)
        x = soc_np if not time else np.arange(tr['T'])

        for r, name in enumerate(rows):
            ax = axes[r, j]

            if name == 'V':
                ax.plot(x, tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
                ax.plot(x, V,               '-',  color=COLORS[0], label=r'Predicted $V$', lw=2)
                ax.set_ylabel(r'$V$ [V]'); ax.legend()
                ax.set_title(f'{title}I={tr["I"]:.1f}, u={tr["u"]:.3f}')

            elif name == 'U1':
                ax.plot(x, tr['U1_true'].numpy(), '--', color=COLORS[1], label=r'True $U_1$', lw=2)
                ax.plot(x, U1,                    '-',  color=COLORS[0], label=r'Predicted $U_1$', lw=2)
                ax.set_ylabel(r'$U_1$ [V]'); ax.legend()

            elif name == 'dU1':
                # Only reachable in V_mode='dynamic' – C1 is not None
                dU1_data = np.gradient(tr['U1_true'].numpy(), 1.0)
                dU1_rc   = tr['I'] / C1 - U1 / (R1 * C1)
                ax.plot(x, dU1_data, '--', color=COLORS[1], label=r'True $dU_1/dt$', lw=2, alpha=0.7)
                ax.plot(x, dU1_rc,   '-',  color=COLORS[0], label=r'Predicted $dU_1/dt$', lw=2)
                ax.set_ylabel(r'$dU_1/dt$ [V/s]')

            elif name == 'R':
                if config['R0_mode'] in ('net', 'net_no_soc'):
                    ax.plot(x, R0 * 1000, ls='--', color=COLORS[0], label=r'$R_0$', lw=2)
                elif config['R0_mode'] == 'func':
                    R0_val = R0_func(tr['u'], tr['I'])
                    ax.axhline(R0_val * 1000, ls='--', color=COLORS[0],
                               label=r'$R_0$' + fr' = {R0_val*1000:.1f} m$\Omega$', lw=2)
                elif config['R0_mode'] == 'param':
                    ax.axhline(R0[0] * 1000, ls='--', color=COLORS[0],
                               label=r'$R_0$' + fr' = {R0[0]*1000:.1f} m$\Omega$', lw=2)
                ax.plot(x, R1 * 1000, '-', color=COLORS[0], label=r'$R_1$', lw=2)
                ax.set_ylabel(r'$R$ [m$\Omega$]'); ax.legend()

            elif name == 'C1':
                ax.plot(x, C1, ls='--', color=COLORS[0], label=r'$C_1$', lw=2)
                ax.set_ylabel(r'$C_1$ [F]'); ax.legend()

            elif name == 'Fr':
                ax.plot(x, tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
                ax.plot(x, Fr,              '-',  color=COLORS[0], label=r'Predicted $F_r$', lw=2)
                ax.set_ylabel(r'$F_r$ [GN]'); ax.legend()

            elif name == 'Fs':
                Fs_true = tr['F'].numpy() + k0 * tr['u']
                ax.plot(x, Fs_true, '--', color=COLORS[1], label=r'True $F_s$', lw=2)
                ax.plot(x, Fs,      '-',  color=COLORS[0], label=r'Predicted $F_s$', lw=2)
                ax.set_ylabel(r'$F_s$ [GN]'); ax.legend()

            elif name == 'k':
                # Empirical stiffness directly from the data: k_true = -F/u
                k_true = -tr['F'].numpy() / tr['u']
                ax.plot(x, k_true, '--', color=COLORS[1], label=r'Empirical $k = -F/u$', lw=2, alpha=0.7)
                ax.plot(x, k_pred, '-',  color=COLORS[0], label=r'Predicted $k$', lw=2)
                # ax.axhline(k0, color='0.6', ls=':', lw=1, label=fr'$k_0 = {k0:.1f}$')
                ax.set_ylabel(r'$k$ [GN/mm]'); ax.legend()

    # x-label + axis direction handled per-mode
    for ax in axes.flat:
        if not time:
            ax.set_xlabel('State of Charge')
            ax.invert_xaxis()
        else:
            ax.set_xlabel('Time [s]')

    fig.tight_layout()
    return fig


def plot_loss(history):
    """Plot RMSE curves with Stage dividers when staged training was used."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    epochs = np.arange(1, len(history['train_rmse']) + 1)

    ax.semilogy(epochs, history['train_rmse_V'],  color=COLORS[0], lw=2,
                label=r'Train $V$  (final {:.4f} V)'.format(history['train_rmse_V'][-1]))
    ax.semilogy(epochs, history['train_rmse_Fr'], color=COLORS[1], lw=2,
                label=r'Train $F_r$ (final {:.4f} GN)'.format(history['train_rmse_Fr'][-1]))
    ax.semilogy(epochs, history['test_rmse'],     color=COLORS[2], lw=2, ls='--',
                label=r'Test $V$  (final {:.4f} V)'.format(history['test_rmse'][-1]))

    # Stage dividers (only present if staged training was used)
    n_s1  = history.get('stage1_epochs', 0)
    n_s2  = history.get('stage2_epochs', 0)
    n_s2b = history.get('stage2b_epochs', 0)
    ymin, ymax = ax.get_ylim()

    if n_s1 > 0 and n_s2 > 0:
        b1 = n_s1 + 0.5
        ax.axvline(b1, color='0.4', ls=':', lw=1.2)
        ax.text(0.5 + n_s1 / 2, ymax, 'Stage 1\n(static $V$)',
                ha='center', va='top', fontsize=9, color='0.3')
        if n_s2b > 0:
            b2 = n_s1 + n_s2 + 0.5
            ax.axvline(b2, color='0.4', ls=':', lw=1.2)
            ax.text(b1 + n_s2 / 2, ymax, 'Stage 2\n(dyn., $R_1$ frozen)',
                    ha='center', va='top', fontsize=9, color='0.3')
            ax.text(b2 + n_s2b / 2, ymax, 'Stage 2b\n(dyn., $R_1$ unfrozen)',
                    ha='center', va='top', fontsize=9, color='0.3')
        else:
            ax.text(b1 + n_s2 / 2, ymax, 'Stage 2\n(dynamic $V$)',
                    ha='center', va='top', fontsize=9, color='0.3')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig


@torch.no_grad()
def predict_pulse_np(model, I_seq, u, soc0, T, noise = False, noise_lvl = 0.00):
    # --- run model ---
    I_b    = I_seq.unsqueeze(0) if I_seq.ndim == 1 else I_seq
    u_b    = torch.tensor([u],    dtype=torch.float32)
    soc0_b = torch.tensor([soc0], dtype=torch.float32)
    if noise: 
        i_noise, u_noise = gen_noise(I_b, u_b, noise_lvl = noise_lvl)
        I_b += i_noise
        u_b += u_noise
    V, Fr, soc, U1, R1, Fs = model.forward_pulse(I_b, u_b, soc0_b)
    V   = V[0].numpy();   Fr = Fr[0].numpy()
    soc = soc[0].numpy(); U1 = U1[0].numpy()
    R1  = R1[0].numpy();  Fs = Fs[0].numpy()

    I_np = I_b[0].numpy()
    u_np = np.full(T, u)
    soc_t = torch.from_numpy(soc.astype(np.float32))
    In_t  = torch.from_numpy((I_np / model.I_ref).astype(np.float32))
    u_t   = torch.from_numpy(u_np.astype(np.float32))
    k     = model.k_net(soc_t, In_t, u_t).numpy()
    C1_t  = model._C1(soc_t, In_t, u_t)

    return V, soc, U1, R1, Fs, Fr, k, C1_t

def plot_predictions_pulse(model, pulse_trajs, time=False, noise=False, noise_lvl=0.00, title='', n_show=3, spec=None):

    n = min(n_show, len(pulse_trajs))
    fig, axes = plt.subplots(9, n, figsize=(4.5 * n, 32), squeeze=False)
    model.eval()
    k = model.k_net.k

    if not time and spec == None:
        for j in range(n):
            tr = pulse_trajs[j]
            T  = tr['T']

            V, soc, U1, R1, Fs, Fr, k_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

            I_np = tr['I_seq'].numpy()
            u_np = np.full(T, tr['u'])

            # TODO: Switch to R0 from config
            R0_np = R0_func(u_np, I_np)
            Ue_np = model.Ue_interp(soc)
            U1_true = Ue_np - I_np * R0_np - tr['V'].numpy()

            # Row 0: I profile vs SOC
            axes[0, j].plot(soc, I_np, '-', color=COLORS[0], lw=2)
            axes[0, j].set_ylabel(r'$I$ [A]')
            axes[0, j].set_title(f'{title}pulse traj {j}, u={tr["u"]:.3f}')

            # Row 1: V
            axes[1, j].plot(soc, tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
            axes[1, j].plot(soc, V, '-', color=COLORS[0], label=r'Predicted $V$', lw=2)
            axes[1, j].set_ylabel(r'$V$ [V]'); axes[1, j].legend()

            # Row 2: SOC consistency check — predicted vs dataset SOC, vs time index
            # (keep this one on a sample index since both curves ARE soc)
            axes[2, j].plot(np.arange(T), tr['soc'].numpy(), '--', color=COLORS[1], label='True SOC', lw=2)
            axes[2, j].plot(np.arange(T), soc, '-', color=COLORS[0], label='Predicted SOC', lw=2)
            axes[2, j].set_ylabel('SOC'); axes[2, j].legend()
            axes[2, j].set_xlabel('Time [s]')

            # Row 3: U1
            axes[3, j].plot(soc, U1_true, '--', color=COLORS[1], label=r'True $U_1$', lw=2)
            axes[3, j].plot(soc, U1,      '-',  color=COLORS[0], label=r'Predicted $U_1$', lw=2)
            axes[3, j].set_ylabel(r'$U_1$ [V]'); axes[3, j].legend()

            # Row 4: R1 (+ R0, time-varying via I)
            axes[4, j].plot(soc, R0_np * 1000, '--', color=COLORS[0], label=r'$R_0$', lw=2)
            axes[4, j].plot(soc, R1    * 1000, '-',  color=COLORS[0], label=r'$R_1$', lw=2)
            axes[4, j].set_ylabel(r'$R$ [m$\Omega$]'); axes[4, j].legend()

            # Row 5: C1
            C1_np = C1_t.numpy() if C1_t.ndim else np.full(T, float(C1_t))
            axes[5, j].plot(soc, C1_np, '-', color=COLORS[0], lw=2)
            axes[5, j].set_ylabel(r'$C_1$ [F]')

            # Row 6: Fr
            axes[6, j].plot(soc, tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
            axes[6, j].plot(soc, Fr,              '-',  color=COLORS[0], label=r'Predicted $F_r$', lw=2)
            axes[6, j].set_ylabel(r'$F_r$ [GN]'); axes[6, j].legend()

            # Row 7: Fs
            Fs_true = tr['F'].numpy() + k * tr['u']
            Fs_plot = Fs 
            axes[7, j].plot(soc, Fs_true, '--', color=COLORS[1], label=r'True $F_s$', lw=2)
            axes[7, j].plot(soc, Fs_plot, '-',  color=COLORS[0], label=r'Predicted $F_s$', lw=2)
            axes[7, j].set_ylabel(r'$F_s$ [GN]'); axes[7, j].legend()

            # Row 8: k
            axes[8, j].plot(soc, k_pred, '-', color=COLORS[0], label=r'Predicted $k$', lw=2)
            axes[8, j].set_ylabel(r'$k$ [GN/mm]'); axes[8, j].legend()

        for ax in axes.flat:
            if ax.get_xlabel() != 'Time [s]':
                ax.set_xlabel('State of Charge')
                ax.invert_xaxis()
    elif time and spec == None:
        for j in range(n):
            tr = pulse_trajs[j]
            T  = tr['T']

            V, soc, U1, R1, Fs, Fr, k_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

            I_np = tr['I_seq'].numpy()
            u_np = np.full(T, tr['u'])

            # TODO: Switch to R0 from config
            R0_np = R0_func(u_np, I_np)
            Ue_np = model.Ue_interp(soc)
            U1_true = Ue_np - I_np * R0_np - tr['V'].numpy()

            # Row 0: I profile vs SOC
            axes[0, j].plot(I_np, '-', color=COLORS[0], lw=2)
            axes[0, j].set_ylabel(r'$I$ [A]')
            axes[0, j].set_title(f'{title}pulse traj {j}, u={tr["u"]:.3f}')

            # Row 1: V
            axes[1, j].plot(tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
            axes[1, j].plot(V, '-', color=COLORS[0], label=r'Predicted $V$', lw=2)
            axes[1, j].set_ylabel(r'$V$ [V]'); axes[1, j].legend()

            # Row 2: SOC consistency check — predicted vs dataset SOC, vs time index
            # (keep this one on a sample index since both curves ARE soc)
            axes[2, j].plot(np.arange(T), tr['soc'].numpy(), '--', color=COLORS[1], label='True SOC', lw=2)
            axes[2, j].plot(np.arange(T), soc, '-', color=COLORS[0], label='Predicted SOC', lw=2)
            axes[2, j].set_ylabel('SOC'); axes[2, j].legend()

            # Row 3: U1
            axes[3, j].plot(U1_true, '--', color=COLORS[1], label=r'True $U_1$', lw=2)
            axes[3, j].plot(U1,      '-',  color=COLORS[0], label=r'Predicted $U_1$', lw=2)
            axes[3, j].set_ylabel(r'$U_1$ [V]'); axes[3, j].legend()

            # Row 4: R1 (+ R0, time-varying via I)
            axes[4, j].plot(R0_np * 1000, '--', color=COLORS[0], label=r'$R_0$', lw=2)
            axes[4, j].plot(R1    * 1000, '-',  color=COLORS[0], label=r'$R_1$', lw=2)
            axes[4, j].set_ylabel(r'$R$ [m$\Omega$]'); axes[4, j].legend()

            # Row 5: C1
            C1_np = C1_t.numpy() if C1_t.ndim else np.full(T, float(C1_t))
            axes[5, j].plot(soc, C1_np, '-', color=COLORS[0], lw=2)
            axes[5, j].set_ylabel(r'$C_1$ [F]')

            # Row 6: Fr
            axes[6, j].plot(tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
            axes[6, j].plot(Fr,              '-',  color=COLORS[0], label=r'Predicted $F_r$', lw=2)
            axes[6, j].set_ylabel(r'$F_r$ [GN]'); axes[6, j].legend()

            # Row 7: Fs
            Fs_true = tr['F'].numpy() + k * tr['u']
            Fs_plot = Fs 
            axes[7, j].plot(Fs_true, '--', color=COLORS[1], label=r'True $F_s$', lw=2)
            axes[7, j].plot(Fs_plot, '-',  color=COLORS[0], label=r'Predicted $F_s$', lw=2)
            axes[7, j].set_ylabel(r'$F_s$ [GN]'); axes[7, j].legend()

            # Row 8: k
            axes[8, j].plot(k_pred, '-', color=COLORS[0], label=r'Predicted $k$', lw=2)
            axes[8, j].set_ylabel(r'$k$ [GN/mm]'); axes[8, j].legend()

        for ax in axes.flat:
            ax.set_xlabel('Time [s]')
    elif spec is not None:
        j = 0
        n = spec
        tr = pulse_trajs[n]
        T  = tr['T']

        V, soc, U1, R1, Fs, Fr, k_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

        I_np = tr['I_seq'].numpy()
        u_np = np.full(T, tr['u'])

        # TODO: Switch to R0 from config
        R0_np = R0_func(u_np, I_np)
        Ue_np = model.Ue_interp(soc)
        U1_true = Ue_np - I_np * R0_np - tr['V'].numpy()

        # Row 0: I profile vs SOC
        axes[0, j].plot(I_np, '-', color=COLORS[0], lw=2)
        axes[0, j].set_ylabel(r'$I$ [A]')
        axes[0, j].set_title(f'{title}pulse traj {j}, u={tr["u"]:.3f}')

        # Row 1: V
        axes[1, j].plot(tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
        axes[1, j].plot(V, '-', color=COLORS[0], label=r'Predicted $V$', lw=2)
        axes[1, j].set_ylabel(r'$V$ [V]'); axes[1, j].legend()

        # Row 2: SOC consistency check — predicted vs dataset SOC, vs time index
        # (keep this one on a sample index since both curves ARE soc)
        axes[2, j].plot(np.arange(T), tr['soc'].numpy(), '--', color=COLORS[1], label='True SOC', lw=2)
        axes[2, j].plot(np.arange(T), soc, '-', color=COLORS[0], label='Predicted SOC', lw=2)
        axes[2, j].set_ylabel('SOC'); axes[2, j].legend()

        # Row 3: U1
        axes[3, j].plot(U1_true, '--', color=COLORS[1], label=r'True $U_1$', lw=2)
        axes[3, j].plot(U1,      '-',  color=COLORS[0], label=r'Predicted $U_1$', lw=2)
        axes[3, j].set_ylabel(r'$U_1$ [V]'); axes[3, j].legend()

        # Row 4: R1 (+ R0, time-varying via I)
        axes[4, j].plot(R0_np * 1000, '--', color=COLORS[0], label=r'$R_0$', lw=2)
        axes[4, j].plot(R1    * 1000, '-',  color=COLORS[0], label=r'$R_1$', lw=2)
        axes[4, j].set_ylabel(r'$R$ [m$\Omega$]'); axes[4, j].legend()

        # Row 5: C1
        C1_np = C1_t.numpy() if C1_t.ndim else np.full(T, float(C1_t))
        axes[5, j].plot(soc, C1_np, '-', color=COLORS[0], lw=2)
        axes[5, j].set_ylabel(r'$C_1$ [F]')

        # Row 6: Fr
        axes[6, j].plot(tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
        axes[6, j].plot(Fr,              '-',  color=COLORS[0], label=r'Predicted $F_r$', lw=2)
        axes[6, j].set_ylabel(r'$F_r$ [GN]'); axes[6, j].legend()

        # Row 7: Fs
        Fs_true = tr['F'].numpy() + k * tr['u']
        Fs_plot = Fs 
        axes[7, j].plot(Fs_true, '--', color=COLORS[1], label=r'True $F_s$', lw=2)
        axes[7, j].plot(Fs_plot, '-',  color=COLORS[0], label=r'Predicted $F_s$', lw=2)
        axes[7, j].set_ylabel(r'$F_s$ [GN]'); axes[7, j].legend()

        # Row 8: k
        axes[8, j].plot(k_pred, '-', color=COLORS[0], label=r'Predicted $k$', lw=2)
        axes[8, j].set_ylabel(r'$k$ [GN/mm]'); axes[8, j].legend()

        for ax in axes.flat:
            ax.set_xlabel('Time [s]')


    fig.tight_layout()
    return fig




def plot_noisy_inputs(trajs, noise_lvl = 0.00):
    tr    = trajs[0]
    I_val = tr['I']
    u_val = tr['u']
    N = 1000  # number of samples

    I_b = torch.full((N,), I_val, dtype=torch.float32)
    u_b = torch.full((N,), u_val, dtype=torch.float32)

    i_noise, u_noise = gen_noise(I_b, u_b, noise_lvl=noise_lvl)
    I_plot = I_b + i_noise
    u_plot = u_b + u_noise
    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(I_plot.numpy(), bins=20, color=COLORS[0], alpha=0.7, edgecolor='black')
    ax[0].axvline(I_val, color=COLORS[0], ls='--', label='True I', lw=2)
    ax[0].legend()
    ax[0].set_xlabel('Current [A]')
    ax[0].set_ylabel('Frequency')
    ax[1].hist(u_plot.numpy() * 10, bins=20, color=COLORS[1], alpha=0.7, edgecolor='black')
    ax[1].axvline(u_val * 10, color=COLORS[1], ls='--', label='True u', lw=2)
    ax[1].legend()
    ax[1].set_xlabel('Displacement [µm]')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    return f


def plot_noisy_preds(model, config, trajs, time=False, title='', n_show=10, pulse = False):
    n = min(n_show, len(trajs))

    model.eval()
    k = model.k_net.k
    noise_max = 0.1
    noise_lvls = np.linspace(0, noise_max, n)
    rmse_noise = np.zeros((n, len(noise_lvls))) # (traj, noise_lvl)
    if not time:
        for j in trange(n):
            tr = trajs[j]
            for i, noise_lvl in enumerate(noise_lvls):
                if pulse:
                    V, soc_np, U1, R1, Fs, Fr, ks, C1 = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise = True, noise_lvl = noise_lvl)
                else:
                    V, soc_np, U1, R1, Fs, Fr, ks, C1 = predict_np(model, config, tr['I'], tr['u'], tr['soc0'], tr['T'], noise = True, noise_lvl = noise_lvl)
                rmse_noise[j,i] = np.sqrt(np.mean((V - tr['V'])**2))

    std_noise = rmse_noise.std(axis=0)
    mean_noise = rmse_noise.mean(axis=0)
    min_noise = mean_noise - std_noise
    max_noise = mean_noise + std_noise
    f,ax = plt.subplots(1,2, figsize=(8,4))

    ax[0].plot(np.unique(noise_lvls)*100, mean_noise,color = COLORS[0], label='Mean RMSE', lw = 3, ls = '--')
    
    ax[0].fill_between(np.unique(noise_lvls)*100, min_noise, max_noise, color=COLORS[0], alpha=0.3, label=r'Mean $\pm\sigma$')
    ax[0].set_xlabel(r'Noise level [\%]')
    ax[0].set_ylabel('RMSE for $V$ [V]')
    ax[0].legend()
    ax[1].hist(rmse_noise[:,0], bins=15, color=COLORS[1], alpha=0.5, label=fr'${noise_lvls.min()*100:.0f} \%$ noise' , edgecolor='black', orientation='vertical')
    ax[1].hist(rmse_noise[:,-1], bins=15, color=COLORS[2], alpha=0.5, label=fr'${noise_lvls.max()*100:.0f} \%$ noise' , edgecolor='black',orientation='vertical')
    ax[1].set_yticks([])
    ax[1].legend()
    ax[1].set_xlabel('RMSE for $V$ [V]')
    plt.tight_layout()
    return f




# ══════════════════════════════════════════════════════════
#  EXTRACT ECM PARAMETERS  (unchanged)
# ══════════════════════════════════════════════════════════════

def extract_ecm_params(model, soc_points, I_val, u_val):
    T = len(soc_points)
    soc_t  = torch.tensor(soc_points, dtype=torch.float32)
    I_norm = torch.full((T,), I_val / model.r1_net.I_ref)
    u_t    = torch.full((T,), u_val)
    model.eval()
    with torch.no_grad():
        R1 = model.r1_net(soc_t, I_norm, u_t).numpy()
    C1 = get_C1(model, soc_ref=0.5, I_ref_val=I_val, u_ref=u_val)
    R0 = R0_func(u_val, I_val)
    return dict(soc=np.array(soc_points), R0=np.full(T, R0),
                R1=R1, C1=C1, tau=R1 * C1, U1_ss=R1 * I_val)


# =════════════════════════════════════════════════════════════════
# RMSE CALC FOR PULSES
# =════════════════════════════════════════════════════════════════

def rmse_pulse(model, pulse_trajs, noise=False, noise_lvl=0.00):
    rmse = []
    for j in range(len(pulse_trajs)):
        tr = pulse_trajs[j]

        V, soc, U1, R1, Fs, Fr, ks_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)
        rmse.append(np.sqrt(np.mean((V - tr['V'].numpy())**2)))
    
    return rmse


# =═════════════════════════════════════════════════════════
# Plotter for ECM parameters
# ══════════════════════════════════════════════════════════════

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_param(model, trajs, param='R1', title=''):
    """
    Plot R0, R1, or C1 across SOC for all given trajectories (one line each).

    Parameters
    ----------
    model : BatteryECMM
    trajs : list of trajectory dicts (e.g. test_trajs)
    param : 'R0', 'R1', or 'C1'
    title : prefix for the plot title
    """
    assert param in ('R0', 'R1', 'C1'), "param must be 'R0', 'R1', or 'C1'"

    fig, ax = plt.subplots(figsize=(6, 4))
    model.eval()

    trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
    C_vals = np.array([tr['C'] for tr in trajs_sorted])

    # base = plt.get_cmap("RdBu_r")  # reversed so blue = low C, red = high C
    # blue_only = LinearSegmentedColormap.from_list(
    #     "RdBu_blue",
    #     base(np.linspace(0.0, 0.4, 256))
    # )
    # cmap = blue_only
    # cmap = plt.cm.Blues # magma # RdBu_r    # coolwarm

    base = plt.cm.Blues_r
    Blues_cut = LinearSegmentedColormap.from_list(
        "Blues_custom",
        base(np.linspace(0.0, 0.8, 256))
    )
    cmap = Blues_cut
    # cmap = plt.cm.copper
    norm = Normalize(vmin=C_vals.min(), vmax=C_vals.max())

    with torch.no_grad():
        for tr in trajs_sorted:
            soc    = tr['soc']
            I_val  = float(tr['I'])
            u_val  = float(tr['u'])
            C_val  = float(tr['C'])
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_t    = torch.full_like(soc, u_val)

            if param == 'R1':
                y = model._R1(soc, I_norm, u_t).numpy() * 1e3
                ylabel = r'$R_1$ [m$\Omega$]'

            elif param == 'C1':
                c1 = model._C1(soc, I_norm, u_t)
                y  = c1.numpy() # if c1.ndim else np.full(len(soc), c1.item())
                ylabel = r'$C_1$ [F]'

            else:  # 'R0'
                m = model.config['R0_mode']
                if m == 'net':
                    y = model.R0_net(soc, I_norm, u_t).numpy() * 1e3
                elif m == 'param':
                    y = model._R0(soc, I_norm, u_t, 0, 0).numpy() * 1e3
                elif m == 'func': 
                    y = R0_func(u_t.numpy(), I_norm.numpy()) * 1e3
                elif m in ('net', 'net_no_soc'):
                    y = model._R0(soc, I_norm, u_t, 0, 0).numpy() * 1e3
                ylabel = r'$R_0$ [m$\Omega$]'

            ax.plot(soc.numpy(), y, '-', color=cmap(norm(C_val)), lw=2)

    # ax.axhline(1000, color='gray', ls='--', lw=1)
    # ax.axhline(5.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('State of Charge')
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()
    ax.ticklabel_format(useOffset=False, style='plain')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='C-rate [a.u.]')

    fig.tight_layout()
    return fig


def data_param(model, trajs):
    """
    Plot R0, R1, or C1 across SOC for all given trajectories (one line each).

    Parameters
    ----------
    model : BatteryECMM
    trajs : list of trajectory dicts (e.g. test_trajs)
    param : 'R0', 'R1', or 'C1'
    """

    model.eval()

    trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
    # C_vals = np.array([tr['C'] for tr in trajs_sorted])

    frames = []

    with torch.no_grad():
        for i, tr in enumerate(trajs_sorted):
            soc    = tr['soc']
            I_val  = float(tr['I'])
            u_val  = float(tr['u'])
            u_per_val = float(tr['u_per'])
            C_val  = float(tr['C'])
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_t    = torch.full_like(soc, u_val)


            R1 = model._R1(soc, I_norm, u_t).numpy()       # Ohm
            C1 = model._C1(soc, I_norm, u_t).numpy()       # F
            R0 = model._R0(soc, I_norm, u_t, 0, 0).numpy()    # Ohm

            frames.append(pd.DataFrame({
                'trajectory': i,
                'soc': soc.numpy(),
                'R1': R1,
                'C1': C1,
                'R0': R0,
                'C': C_val,
                'u_per': u_per_val}))
            
    df = pd.concat(frames, ignore_index=True)

    return df

# =═════════════════════════════════════════════════════════
# Plotter for predictions
# ══════════════════════════════════════════════════════════════

def plot_predicts(model, config, trajs, predict='R1', sort='C_rate'):
    """
    Plot R0, R1, or C1 across SOC for all given trajectories (one line each).

    Parameters
    ----------
    model : BatteryECMM
    trajs : list of trajectory dicts (e.g. test_trajs)
    predict : 'V', or 'F'
    sort : 'C_rate' or 'u_par'
    """
    assert predict in ('V', 'F'), "predict must be 'V' or 'F'"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    model.eval()
    k = model.k_net.k

    if sort == 'C_rate':
        trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
        C_vals = np.array([tr['C'] for tr in trajs_sorted])
        norm = Normalize(vmin=C_vals.min(), vmax=C_vals.max())
        bar_name = 'C-rate [a.u.]'

    elif sort == 'u_per':
        trajs_sorted = sorted(trajs, key=lambda tr: tr['u_per'])
        u_per_vals = np.array([tr['u_per'] for tr in trajs_sorted])
        norm = Normalize(vmin=u_per_vals.min(), vmax=u_per_vals.max())
        bar_name = r'$u$ $[\%]$'

    
    base = plt.cm.Blues_r
    Blues_cut = LinearSegmentedColormap.from_list(
        "Blues_custom",
        base(np.linspace(0.0, 0.8, 256))
    )
    cmap_b = Blues_cut
    base = plt.cm.Reds_r
    Reds_cut = LinearSegmentedColormap.from_list(
        "Reds_custom",
        base(np.linspace(0.0, 0.8, 256))
    )
    cmap_r = Reds_cut

    with torch.no_grad():
        for tr in trajs_sorted:
            soc    = tr['soc']
            I_val  = float(tr['I'])
            u_val  = float(tr['u'])
            C_val  = float(tr['C'])
            u_per_val = float(tr['u_per'])
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_t    = torch.full_like(soc, u_val)

            V, soc_np, U1, R1, Fs, Fr, ks, C1, R0 = predict_np(model, config, tr['I'], tr['u'], tr['soc0'], tr['T'])

            if predict == 'V':
                y_true = tr['V'].numpy()
                y_pred = V
                ylabel = r'$V$ [V]'

            elif predict == 'F':
                y_true = tr['F'].numpy()
                y_pred = Fr
                ylabel = r'$F$ [GN]'

            if sort == 'C_rate':
                bar_val = C_val
            elif sort == 'u_per':
                bar_val = u_per_val

            axes[0].plot(soc_np, y_true, '-', label=f'True {ylabel}', color=cmap_r(norm(bar_val)), lw=2)
            axes[1].plot(soc_np, y_pred, '-', label=f'Predicted {ylabel}', color=cmap_b(norm(bar_val)), lw=2)
            axes[2].plot(soc_np, y_pred, '-', label=f'Predicted {ylabel}', color=cmap_b(norm(bar_val)), lw=2)
            axes[2].plot(soc_np, y_true, '--', label=f'True {ylabel}', color=cmap_r(norm(bar_val)), lw=2)


    # axes[0].set_ylabel(ylabel)
    for ax in axes:
        ax.set_xlabel('State of Charge')
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()

    # cheat legend
    from matplotlib.lines import Line2D
    # mid-color of each cmap
    axes[0].legend(handles=[Line2D([0], [0], color='tab:red', lw=2, label='True')])
    axes[1].legend(handles=[Line2D([0], [0], color='tab:blue', lw=2, label='Predicted')])
    axes[2].legend(handles=[Line2D([0], [0], color='tab:red', lw=2, label='True'), Line2D([0], [0], color='tab:blue', lw=2, label='Predicted')])

    fig.tight_layout()
    sm_true = ScalarMappable(cmap=cmap_b, norm=norm)
    fig.colorbar(sm_true, ax=axes, label=bar_name, pad=0.02)
    return fig