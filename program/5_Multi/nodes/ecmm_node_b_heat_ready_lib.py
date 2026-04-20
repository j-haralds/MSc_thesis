
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
from networkx import config
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
import time as _time
from tqdm import trange

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
#  ks NETWORK
# ══════════════════════════════════════════════════════════════

class ksNet(nn.Module):
    """(SOC, I, u) → ks > 0  [GN].  One hidden layer, softplus output."""
    def __init__(self, n_hidden=32, k=53):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        return self.net(x).squeeze(-1)

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
        self.ks_net    = ksNet(n_hidden=nh, k=k)

        # Instantiate according to configurations
        if config['R1_mode'] == 'net':
            # Choose constrained network or not
            if config.get('R1_constrained', 'false') == 'true':     # When key not availably, default to unconstrained
                self.r1_net = R1NetConstrained(config, n_hidden=nh, I_ref=I_ref)
            else:
                print('R1 unconstrained')
                self.r1_net = R1Net(n_hidden=nh, I_ref=I_ref)

        elif config['R1_mode'] == 'param':
            self.R1 = nn.Parameter(torch.tensor((config.get('R1_param', 0.01)), dtype=torch.float32)) 

        # ------
        if config['C1_mode'] == 'net':
            if config.get('C1_constrained', 'false') == 'true':
                self.C1_net = C1NetConstrained(config, n_hidden=nh, I_ref=I_ref)
            else:
                print('C1 unconstrained')
                self.C1_net = C1Net(n_hidden=nh, I_ref=I_ref)

        elif config['C1_mode'] == 'param':
            self.log_C1 = nn.Parameter(torch.tensor(np.log(C1_init), dtype=torch.float32))
        elif config['C1_mode'] == 'const':
            self.log_C1 = torch.tensor(np.log(C1_init), dtype=torch.float32)

        # ------
        if config['R0_mode'] == 'net':
            if config.get('R0_constrained', 'false') == 'true':
                self.R0_net = R0NetConstrained(config, n_hidden=nh, I_ref=I_ref)
            else:
                print('R0 unconstrained')
                self.R0_net = R0Net(n_hidden=nh, I_ref=I_ref)

        elif config['R0_mode'] == 'func':
            self.R0_func = R0_func
        elif config['R0_mode'] == 'param':
            self.log_R0 = nn.Parameter(torch.tensor(np.log(config.get('R0_param', 0.01)), dtype=torch.float32))  
        elif config['R0_mode'] == 'net_no_soc':
            self.R0_net = R0NetNoSOC(config, n_hidden=nh, I_ref=I_ref)

        # Dispatchers
    def _R1(self, soc, I_norm, u):
        m = self.config['R1_mode']
        if m == 'net':
            return self.r1_net(soc, I_norm, u)
        if m == 'param':
            return nn.functional.softplus(self.R1).expand_as(soc)

    def _C1(self, soc, I_norm, u):
        m = self.config['C1_mode']
        if m == 'net': return self.C1_net(soc, I_norm, u)
        if m == 'param': return torch.exp(self.log_C1)
        elif m == 'const': return torch.exp(self.log_C1)

    def _R0(self, soc, I_norm, u, I_batch, u_batch, config=config):
        """Returns R0 broadcastable to (B, T)."""
        m = self.config['R0_mode']
        if m == 'net':
            return self.R0_net(soc, I_norm, u)
        if m == 'param':
            return torch.exp(self.log_R0).expand_as(soc)
        if m == 'func':
            # scalar-per-trajectory → (B, 1), broadcasts over T
            B = I_batch.shape[0]
            return torch.tensor(
                [self.R0_func(u_batch[b].item(), I_batch[b].item()) for b in range(B)],
                dtype=torch.float32).unsqueeze(1)
        if m == 'net_no_soc':
            return self.R0_net(I_norm, u)

    def forward(self, I_batch, u_batch, soc0_batch, T_max):
        B = I_batch.shape[0]
        t_idx = torch.arange(T_max, dtype=torch.float32).unsqueeze(0)
        soc = soc0_batch.unsqueeze(1) - I_batch.unsqueeze(1) / self.Q0 * t_idx
        I_norm = (I_batch / self.I_ref).unsqueeze(1).expand(B, T_max)
        u_exp  = u_batch.unsqueeze(1).expand(B, T_max)

        R1 = self._R1(soc, I_norm, u_exp)           
        C1 = self._C1(soc, I_norm, u_exp)        
        R0 = self._R0(soc, I_norm, u_exp, I_batch, u_batch)

        U1_steps = [torch.zeros(B)]
        Fs_steps = [torch.zeros(B)]

        for n in range(T_max - 1):
            # C1 may be scalar OR (B, T_max) — index only if 2-D
            C1_n = C1[:, n] if C1.ndim == 2 else C1
            # Euler forward 
            dU1 = I_batch / C1_n - U1_steps[n] / (R1[:, n] * C1_n)
            U1_steps.append(U1_steps[n] + dU1)

            # # Semi-implicit Euler forward
            # U1_next = (U1_steps[n] + I_batch / C1_n) / (1.0 + 1.0 / (R1[:, n] * C1_n))
            # U1_steps.append(U1_next)

            ks = self.ks_net(soc[:, n], I_norm[:, n], u_exp[:, n])
            dFs = ks * (-I_batch / self.Q0)
            Fs_steps.append(Fs_steps[n] + dFs)

        U1 = torch.stack(U1_steps, dim=1)
        Fs = torch.stack(Fs_steps, dim=1)

        with torch.no_grad():
            Ue = torch.tensor(self.Ue_interp(soc.detach().numpy()), dtype=torch.float32)

        # Use the dispatcher result, not a second hard-coded R0_func call
        V = Ue - I_batch.unsqueeze(1) * R0 - U1
        Fr = -self.ks_net.k * u_exp + Fs

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

        R1 = self._R1(soc, I_norm, u_exp)           # local, not self.R1
        C1 = self._C1(soc, I_norm, u_exp)           # local, not self.C1
        
        m = self.config['R0_mode']
        if m == 'net':
            R0 = self.R0_net(soc, I_norm, u_exp)                   # (B, T)
        elif m == 'param':
            R0 = nn.functional.softplus(self.R0).expand_as(soc)    # (B, T)
        elif m == 'func':
            R0 = (u_exp * (-0.0001887521) - 7.049519e-5 * I_seq + 0.008446693)
        elif m == 'net_no_soc':
            R0 = self.R0_net(config, I_norm, u_exp)

        U1_steps = [torch.zeros(B)]
        Fs_steps = [torch.zeros(B)]
        for n in range(T - 1):
            # C1 may be scalar OR (B, T_max) — index only if 2-D
            C1_n = C1[:, n] if C1.ndim == 2 else C1
            dU1 = I_seq[:, n] / C1_n - U1_steps[n] / (R1[:, n] * C1_n)
            U1_steps.append(U1_steps[n] + dU1)

            ks = self.ks_net(soc[:, n], I_norm[:, n], u_exp[:, n])
            dFs = ks * (-I_seq[:, n] / self.Q0)
            Fs_steps.append(Fs_steps[n] + dFs)

        U1 = torch.stack(U1_steps, dim=1)
        Fs = torch.stack(Fs_steps, dim=1)

        with torch.no_grad():
            Ue = torch.tensor(self.Ue_interp(soc.detach().numpy()), dtype=torch.float32)

        # Use the dispatcher result, not a second hard-coded R0_func call
        V = Ue - I_seq * R0 - U1
        Fr = -self.ks_net.k * u_exp + Fs

        return V, Fr, soc, U1, R1, Fs

def get_C1(model, scalar=True, soc_ref=0.5, I_ref_val=10.0, u_ref=-0.06, soc=0, I_norm=0, u_exp=0):
    """Return a representative scalar C1 for display purposes."""
    if model.config['C1_mode'] in ('const', 'param'):
        return torch.exp(model.log_C1).item()
    
    if model.config['C1_mode'] in ('net') and scalar:
        soc_t  = torch.tensor([soc_ref], dtype=torch.float32)
        I_norm = torch.tensor([I_ref_val / model.I_ref], dtype=torch.float32)
        u_t    = torch.tensor([u_ref], dtype=torch.float32)
        with torch.no_grad():
            return model._C1(soc_t, I_norm, u_t).mean().item()
        
    elif model.config['C1_mode'] in ('net'):
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

# ══════════════════════════════════════════════════════════
#  TRAINING FUNCTION  (batched)
# ══════════════════════════════════════════════════════════════

def train_model(model, train_trajs, test_trajs,
                n_epochs=200, lr=1e-3, batch_size=16, print_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)

    history = {'train': [], 'train_V': [], 'train_Fr': [], 'test': [], 'time': [],
               'train_rmse': [], 'train_rmse_V': [], 'train_rmse_Fr': [], 'test_rmse': []}
    t0 = _time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_trajs))
        epoch_loss = 0.0
        epoch_loss_V = 0.0
        epoch_loss_Fr = 0.0
        epoch_loss_rmse_V = 0.0
        epoch_loss_rmse_Fr = 0.0
        epoch_loss_rmse = 0.0
        n_batches  = 0

        # ── mini-batch loop ──
        for start in range(0, len(train_trajs), batch_size):
            idxs  = order[start : start + batch_size]
            batch = [train_trajs[i] for i in idxs]

            I_b, u_b, soc0_b, V_true, Fr_true, mask, T_max = collate_batch(batch)

            optimizer.zero_grad()
            V_pred, Fr_pred, soc_pred, U1_pred, R1_pred, Fs_pred = model(I_b, u_b, soc0_b, T_max)

            # Masked MSE — only score real (non-padded) timesteps
            sq_err_V = (V_pred - V_true) ** 2
            loss_V = sq_err_V[mask].mean() 
            sq_err_Fr = (Fr_pred - Fr_true) ** 2
            loss_Fr = sq_err_Fr[mask].mean()
            loss = loss_V + loss_Fr

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Masked RMSE for reporting
            loss_Fr_rmse = torch.sqrt(sq_err_Fr[mask].mean())
            loss_V_rmse = torch.sqrt(sq_err_V[mask].mean())

            epoch_loss += loss.item()
            epoch_loss_V += loss_V.item()
            epoch_loss_Fr += loss_Fr.item()
            epoch_loss_rmse += loss_V_rmse.item() + loss_Fr_rmse.item()
            epoch_loss_rmse_V += loss_V_rmse.item()
            epoch_loss_rmse_Fr += loss_Fr_rmse.item()
            n_batches += 1

        epoch_loss /= n_batches
        epoch_loss_V /= n_batches
        epoch_loss_Fr /= n_batches
        epoch_loss_rmse /= n_batches
        epoch_loss_rmse_V /= n_batches
        epoch_loss_rmse_Fr /= n_batches
        history['train'].append(epoch_loss)
        history['train_V'].append(epoch_loss_V)
        history['train_Fr'].append(epoch_loss_Fr)
        history['train_rmse'].append(epoch_loss_rmse)
        history['train_rmse_V'].append(epoch_loss_rmse_V)
        history['train_rmse_Fr'].append(epoch_loss_rmse_Fr)

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_loss_rmse = 0.0
            for tr in test_trajs:
                I_b    = torch.tensor([tr['I']],    dtype=torch.float32)
                u_b    = torch.tensor([tr['u']],    dtype=torch.float32)
                soc0_b = torch.tensor([tr['soc0']], dtype=torch.float32)
                V_pred, Fr_pred, _, _, _, _ = model(I_b, u_b, soc0_b, tr['T'])
                test_loss += (torch.mean((V_pred[0] - tr['V'])**2)).item()
                test_loss_rmse += (torch.mean((V_pred[0] - tr['V'])**2)).item()
            test_loss /= len(test_trajs)
            test_loss_rmse /= len(test_trajs)
        history['test'].append(test_loss)
        history['test_rmse'].append(test_loss_rmse)
        scheduler.step(epoch_loss)

        if epoch % print_every == 0 or epoch == 1:
            C1 = get_C1(model, scalar=True)
            eta = (_time.time() - t0) / epoch * (n_epochs - epoch) / 60
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  {epoch:4d}/{n_epochs} "
                  f"| ETA {eta:.1f}m "
                  f"| MSE train {epoch_loss} | train_V {epoch_loss_V} V | train_Fr {epoch_loss_Fr} GN "
                  f"| RMSE train {epoch_loss_rmse:.4f} | train_V {epoch_loss_rmse_V:.4f} V | train_Fr {epoch_loss_rmse_Fr:.4f} GN "
                  f"| test {test_loss_rmse:.4f} V | C1={C1:.0f}F "
                  )

    history['time'] = (_time.time() - t0) / 60

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
def predict_np(model, config, I_val, u_val, soc0, T, noise = False, noise_lvl = 0.00):
    # --- run model ---
    I_b    = torch.tensor([I_val],  dtype=torch.float32)
    u_b    = torch.tensor([u_val],  dtype=torch.float32)
    soc0_b = torch.tensor([soc0],   dtype=torch.float32)

    # Add noise to inputs
    if noise:
        i_noise, u_noise = gen_noise(I_b, u_b, noise_lvl = noise_lvl)
        I_b += i_noise
        u_b += u_noise
    V, Fr, soc, U1, R1, Fs = model(I_b, u_b, soc0_b, T)
    V   = V[0].numpy();   Fr = Fr[0].numpy()
    soc = soc[0].numpy(); U1 = U1[0].numpy()
    R1  = R1[0].numpy();  Fs = Fs[0].numpy()

    # ks along the trajectory (not returned by forward)
    soc_t  = torch.from_numpy(soc)
    I_norm = torch.full((T,), I_val / model.I_ref)
    u_t    = torch.full((T,), u_val)
    ks = model.ks_net(soc_t, I_norm, u_t).numpy()

    if config['C1_mode'] in ('net'):
        C1 = get_C1(model, scalar=False, soc=soc_t, I_norm=I_norm, u_exp=u_t)
    else:
        C1 = get_C1(model, scalar=True)


    if config['R0_mode'] in ('net', 'net_no_soc', 'param'):
        R0 = model._R0(soc_t, I_norm, u_t, 0, 0).numpy()
    else:
        R0 = None

    return V, soc, U1, R1, Fs, Fr, ks, C1, R0


def plot_predictions(model, config, trajs, time=False, noise=False, noise_lvl=0.00, title='', n_show=3):
    n = min(n_show, len(trajs))
    fig, axes = plt.subplots(8, n, figsize=(5 * n, 26), squeeze=False)

    model.eval()
    k = model.ks_net.k

    if not time:
        for j in range(n):
            tr = trajs[j]

            V, soc_np, U1, R1, Fs, Fr, ks, C1, R0 = predict_np(model, config, tr['I'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

            # Row 0: V
            axes[0, j].plot(soc_np, tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
            axes[0, j].plot(soc_np, V, '-', color=COLORS[0], label=r'Predicted $V$', lw=2)
            axes[0, j].set_ylabel(r'$V$ [V]'); axes[0, j].legend()
            axes[0, j].set_title(f'{title}I={tr["I"]:.1f}, u={tr["u"]:.3f}')

            # Row 1: U1
            axes[1, j].plot(soc_np, tr['U1_true'].numpy(), '--', color=COLORS[1], label=r'True $U_1$', lw=2)
            axes[1, j].plot(soc_np, U1, '-', color=COLORS[0], label=r'Predicted $U_1$', lw=2)
            axes[1, j].set_ylabel(r'$U_1$ [V]'); axes[1, j].legend()

            # Row 2: dU1/dt
            dU1_data = np.gradient(tr['U1_true'].numpy(), 1.0)
            dU1_rc   = tr['I'] / C1 - U1 / (R1 * C1)
            axes[2, j].plot(soc_np, dU1_data, '--', color=COLORS[1], label=r'True $dU_1/dt$', lw=2, alpha=0.7)
            axes[2, j].plot(soc_np, dU1_rc, '-', color=COLORS[0], label=r'Predicted $dU_1/dt$', lw=2)
            axes[2, j].set_ylabel(r'$dU_1/dt$ [V/s]')

            # Row 3: R1
            if config['R0_mode'] in ('net', 'net_no_soc'):
                axes[3, j].plot(soc_np, R0 * 1000, ls='--', color=COLORS[0], label=r'$R_0$', lw=2)
            elif config['R0_mode'] in ('func'):
                R0_val = R0_func(tr['u'], tr['I'])
                axes[3, j].axhline(R0_val * 1000, ls='--', color=COLORS[0], label=r'$R_0$' + fr' = {R0_val*1000:.1f} m$\Omega$', lw=2)
            elif config['R0_mode'] in ('param'):
                axes[3, j].axhline(R0[0] * 1000, ls='--', color=COLORS[0], label=r'$R_0$' + fr' = {R0[0]*1000:.1f} m$\Omega$', lw=2)
            axes[3, j].plot(soc_np, R1 * 1000, '-', color=COLORS[0], label=r'$R_1$', lw=2)
            axes[3, j].set_ylabel(r'$R$ [m$\Omega$]'); axes[3, j].legend()


            # Row 4: C1
            if config['C1_mode'] in ('net'):
                axes[4, j].plot(soc_np, C1, ls='--', color=COLORS[0], label=r'$C_1$', lw=2)
            else:
                axes[4, j].axhline(C1, ls='--', color=COLORS[0], label=r'$C_1=$' + f'{C1:.0f} F', lw=2)
            axes[4, j].set_ylabel(r'$C_1$ [F]'); axes[4, j].legend()

            # Row 5: Fr (reaction force)
            axes[5, j].plot(soc_np, tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
            axes[5, j].plot(soc_np, Fr, '-', color=COLORS[0], label=r'Predicted $F_r$', lw=2)
            axes[5, j].set_ylabel(r'$F_r$ [GN]'); axes[5, j].legend()

            # Row 6: Fs (swelling force) — true Fs reconstructed from data: Fs_true = F + k*u
            Fs_true = tr['F'].numpy() + k * tr['u']
            axes[6, j].plot(soc_np, Fs_true, '--', color=COLORS[1], label=r'True $F_s$', lw=2)
            axes[6, j].plot(soc_np, Fs, '-', color=COLORS[0], label=r'Predicted $F_s$', lw=2)
            axes[6, j].set_ylabel(r'$F_s$ [GN]'); axes[6, j].legend()

            # Row 7: ks — compare to empirical slope dFs/dSOC from data
            dSOC     = np.gradient(soc_np)
            dFs_true = np.gradient(Fs_true)
            # ks_true = dFs/dSOC
            ks_true = dFs_true / dSOC
            axes[7, j].plot(soc_np, ks_true, '--', color=COLORS[1], label=r'Empirical $k_s = dF_s/d\mathrm{SOC}$', lw=2, alpha=0.7)
            axes[7, j].plot(soc_np, ks, '-', color=COLORS[0], label=r'Predicted $k_s$', lw=2)
            axes[7, j].set_ylabel(r'$k_s$ [GN]'); axes[7, j].legend()

        for ax in axes.flat:
            ax.set_xlabel('State of Charge')
            ax.invert_xaxis()
    else:
        for j in range(n):
            tr = trajs[j]

            V, soc_np, U1, R1, Fs, Fr, ks, C1, R0 = predict_np(model, config, tr['I'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

            # Row 0: V
            axes[0, j].plot(tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
            axes[0, j].plot(V, '-', color=COLORS[0], label=r'Predicted $V$', lw=2)
            axes[0, j].set_ylabel(r'$V$ [V]'); axes[0, j].legend()
            axes[0, j].set_title(f'{title}I={tr["I"]:.1f}, u={tr["u"]:.3f}')

            # Row 1: U1
            axes[1, j].plot(tr['U1_true'].numpy(), '--', color=COLORS[1], label=r'True $U_1$', lw=2)
            axes[1, j].plot(U1, '-', color=COLORS[0], label=r'Predicted $U_1$', lw=2)
            axes[1, j].set_ylabel(r'$U_1$ [V]'); axes[1, j].legend()

            # Row 2: dU1/dt
            dU1_data = np.gradient(tr['U1_true'].numpy(), 1.0)
            dU1_rc   = tr['I'] / C1 - U1 / (R1 * C1)
            axes[2, j].plot(dU1_data, '--', color=COLORS[1], label=r'True $dU_1/dt$', lw=2, alpha=0.7)
            axes[2, j].plot(dU1_rc, '-', color=COLORS[0], label=r'Predicted $dU_1/dt$', lw=2)
            axes[2, j].set_ylabel(r'$dU_1/dt$ [V/s]')

            # Row 3: R1 (+ R0 reference)
            if config['R0_mode'] in ('net'):
                axes[3, j].plot(R0 * 1000, ls='--', color=COLORS[0], label=r'$R_0$', lw=2)
            elif config['R0_mode'] in ('func'):
                R0_val = R0_func(tr['u'], tr['I'])
                axes[3, j].axhline(R0_val * 1000, ls='--', color=COLORS[0], label=r'$R_0$' + fr' = {R0_val*1000:.1f} m$\Omega$', lw=2)
            axes[3, j].plot(R1 * 1000, '-', color=COLORS[0], label=r'$R_1$', lw=2)
            axes[3, j].set_ylabel(r'$R$ [m$\Omega$]'); axes[3, j].legend()

            # Row 4: C1
            if config['C1_mode'] in ('net'):
                axes[4, j].plot(C1, ls='--', color=COLORS[0], label=r'$C_1$', lw=2)
            else:
                axes[4, j].axhline(C1, ls='--', color=COLORS[0], label=r'$C_1=$' + f'{C1:.0f} F', lw=2)
            axes[4, j].set_ylabel(r'$C_1$ [F]'); axes[4, j].legend()

            # Row 5: Fr (reaction force)
            axes[5, j].plot(tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
            axes[5, j].plot(Fr, '-', color=COLORS[0], label=r'Predicted $F_r$', lw=2)
            axes[5, j].set_ylabel(r'$F_r$ [GN]'); axes[5, j].legend()

            # Row 6: Fs (swelling force) — true Fs reconstructed from data: Fs_true = F + k*u
            Fs_true = tr['F'].numpy() + k * tr['u']
            axes[6, j].plot(Fs_true, '--', color=COLORS[1], label=r'True $F_s$', lw=2)
            axes[6, j].plot(Fs, '-', color=COLORS[0], label=r'Predicted $F_s$', lw=2)
            axes[6, j].set_ylabel(r'$F_s$ [GN]'); axes[6, j].legend()

            # Row 7: ks — compare to empirical slope dFs/dSOC from data
            dSOC     = np.gradient(soc_np)
            dFs_true = np.gradient(Fs_true)
            # ks_true = dFs/dSOC
            ks_true = dFs_true / dSOC
            axes[7, j].plot(ks_true, '--', color=COLORS[1], label=r'Empirical $k_s = dF_s/d\mathrm{SOC}$', lw=2, alpha=0.7)
            axes[7, j].plot(ks, '-', color=COLORS[0], label=r'Predicted $k_s$', lw=2)
            axes[7, j].set_ylabel(r'$k_s$ [GN]'); axes[7, j].legend()

        for ax in axes.flat:
            ax.set_xlabel('Time [s]')

    fig.tight_layout()
    return fig


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history['train_rmse'], color=COLORS[0], label='Loss last RMSE: {:.4f} V'.format(history['train_rmse'][-1]))
    ax.semilogy(history['train_rmse_Fr'], color=COLORS[1], ls='--', label='Loss $F_r$ last RMSE: {:.4f} GN'.format(history['train_rmse_Fr'][-1]))
    ax.semilogy(history['train_rmse_V'], color=COLORS[2], ls='--', label='Loss $V$ last RMSE: {:.4f} V'.format(history['train_rmse_V'][-1]))
    # ax.semilogy(history['test_rmse'], color=COLORS[1], label='Test \n Last RMSE: {:.4f} V'.format(history['test_rmse'][-1]))
    ax.set_xlabel('epoch'); ax.set_ylabel('RMSE'); ax.legend()
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
    ks    = model.ks_net(soc_t, In_t, u_t).numpy()
    C1_t  = model._C1(soc_t, In_t, u_t)

    return V, soc, U1, R1, Fs, Fr, ks, C1_t

def plot_predictions_pulse(model, pulse_trajs, time=False, noise=False, noise_lvl=0.00, title='', n_show=3, spec=None):

    n = min(n_show, len(pulse_trajs))
    fig, axes = plt.subplots(9, n, figsize=(5 * n, 28), squeeze=False)
    model.eval()
    k = model.ks_net.k

    if not time and spec == None:
        for j in range(n):
            tr = pulse_trajs[j]
            T  = tr['T']

            V, soc, U1, R1, Fs, Fr, ks_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

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

            # Row 8: ks
            axes[8, j].plot(soc, ks_pred, '-', color=COLORS[0], label=r'Predicted $k_s$', lw=2)
            axes[8, j].set_ylabel(r'$k_s$ [GN]'); axes[8, j].legend()

        for ax in axes.flat:
            if ax.get_xlabel() != 'Time [s]':
                ax.set_xlabel('State of Charge')
                ax.invert_xaxis()
    elif time and spec == None:
        for j in range(n):
            tr = pulse_trajs[j]
            T  = tr['T']

            V, soc, U1, R1, Fs, Fr, ks_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

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

            # Row 8: ks
            axes[8, j].plot(ks_pred, '-', color=COLORS[0], label=r'Predicted $k_s$', lw=2)
            axes[8, j].set_ylabel(r'$k_s$ [GN]'); axes[8, j].legend()

        for ax in axes.flat:
            ax.set_xlabel('Time [s]')
    elif spec is not None:
        j = 0
        n = spec
        tr = pulse_trajs[n]
        T  = tr['T']

        V, soc, U1, R1, Fs, Fr, ks_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'], noise=noise, noise_lvl=noise_lvl)

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

        # Row 8: ks
        axes[8, j].plot(ks_pred, '-', color=COLORS[0], label=r'Predicted $k_s$', lw=2)
        axes[8, j].set_ylabel(r'$k_s$ [GN]'); axes[8, j].legend()

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

# def _predict_np_noise(model, config, I_val, u_val, soc0, T, noise_lvl = 0.00):
#     # --- run model ---
#     I_b    = torch.tensor([I_val],  dtype=torch.float32)
#     u_b    = torch.tensor([u_val],  dtype=torch.float32)
#     # Add noise to inputs
#     i_noise, u_noise = gen_noise(I_b, u_b, noise_lvl = noise_lvl)
#     I_b += i_noise
#     u_b += u_noise

#     soc0_b = torch.tensor([soc0],   dtype=torch.float32)
#     V, Fr, soc, U1, R1, Fs = model(I_b, u_b, soc0_b, T)
#     V   = V[0].numpy();   Fr = Fr[0].numpy()
#     soc = soc[0].numpy(); U1 = U1[0].numpy()
#     R1  = R1[0].numpy();  Fs = Fs[0].numpy()

#     # ks along the trajectory (not returned by forward)
#     soc_t  = torch.from_numpy(soc)
#     I_norm = torch.full((T,), I_val / model.I_ref)
#     u_t    = torch.full((T,), u_val)
#     ks = model.ks_net(soc_t, I_norm, u_t).numpy()

#     if config['C1_mode'] in ('net'):
#         C1 = get_C1(model, scalar=False, soc=soc_t, I_norm=I_norm, u_exp=u_t)
#     else:
#         C1 = get_C1(model, scalar=True)

#     return V, soc, U1, R1, Fs, Fr, ks, C1


def plot_noisy_preds(model, config, trajs, time=False, title='', n_show=10, pulse = False):
    n = min(n_show, len(trajs))

    model.eval()
    k = model.ks_net.k
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
    k = model.ks_net.k

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