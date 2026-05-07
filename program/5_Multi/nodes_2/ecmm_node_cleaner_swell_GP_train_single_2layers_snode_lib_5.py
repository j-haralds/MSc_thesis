import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import time as _time

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
sys.path.append(os.path.join(FILE_PATH, '..'))          # Up one step — for JN_GP / Ue_GP
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()

# Ue(SOC) lookup — see Ue_GP.py for caching / tabulation details.
# Tabulate the GP on a dense SOC grid once at startup, use np.interp for lookups. 
# GP-quality values with interpolant speed.
import Ue_GP


Q0          = 17921.57581   # cell capacity [Coulombs]
LIMON_CELL0 = 14.37325  # cell length [1e-5m]
TRAIN_SPLIT = 0.8
N_HIDDEN    = 32
EPOCHS      = 2
LR          = 1e-3
PAT         = 400   # # Extrmely high pateience to omitt scheduler (epochs with no improvement on test loss before reducing LR)

# ══════════════════════════════════════════════════════════
#  R1 NETWORK
# ══════════════════════════════════════════════════════════════

class R1Net(nn.Module):
    """(SOC, I, u) → R1 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        # scale output to typical R1 range (mOhm·m)
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.01  # if softplus = 1, out = 10 [mOhm * m]

class R1NetConstrained(nn.Module):
    """(SOC, I, u) → R1 > 0  [Ohm].  One hidden layer, sigmoid+linear constraint."""
    def __init__(self, config, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
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
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        # C1 initialized around softplus(0) = ln(1 + e^0) = ln(2) = 0.693.  0.693 × 2000 = 1386 F
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 2000 * 4     # [F]

class C1NetConstrained(nn.Module):
    """(SOC, I, u) → C1 > 0  [F].  One hidden layer, sigmoid+linear constraint."""
    def __init__(self, config, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
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
#  R0
# ══════════════════════════════════════════════════════════════
class R0Net(nn.Module):
    """(SOC, I, u) → R0 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
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
            nn.Linear(n_hidden, n_hidden),
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
    
class R0NetNoSOC(nn.Module):
    """(I, u) → R0 > 0  [Ohm].  One hidden layer, softplus output."""
    def __init__(self, config, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
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
    
def R0_func(u, I):
    return u * (-0.0001887521) - 7.049519e-5 * I + 0.008446693

# ══════════════════════════════════════════════════════════
#  k NETWORK (static)
# ══════════════════════════════════════════════════════════════

class kNet(nn.Module):
    """(u) → k > 0  [GN/1e-5m].  Algebraic — no integration.
    """
    def __init__(self, config, n_hidden=32, k=53.0):
        super().__init__()
        self.k = float(k)                          # reference k0 from data
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.k_min = config.get('k_min')
        self.k_max = config.get('k_max')
        if config.get('k_constrained', 'false') == 'true':
            print(f'k constrained to [{self.k_min}, {self.k_max}] GN/1e-5m')
        else:
            print('k unconstrained')

    def forward(self, soc, I_norm, u_norm):
        # x = torch.stack([soc, I_norm, u_norm], dim=-1)
        x = torch.stack([soc, I_norm, u_norm], dim=-1)
        if self.config.get('k_constrained', 'false') == 'true':
            s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.k_min + s * (self.k_max - self.k_min)
        else:
            return nn.functional.softplus(self.net(x)).squeeze(-1)
    
# ══════════════════════════════════════════════════════════
#  s NETWORK (static)
# ══════════════════════════════════════════════════════

        
class betaNet(nn.Module):
    """(soc, I_norm, u_norm) → β > 0  [1e-5 m].

    Tier 2 kinematic NODE: ds/dt = -β · I/Q₀.   β has units of length [1e-5 m]
    so that β · I/Q₀ has units of [length / time] (with Q₀ in [As], I in [A],
    dt in [s]).  Equivalently: ds/dSOC = β, so β is the swelling-per-unit-SOC
    gain (its integral over SOC gives the static s(SOC) curve).

    Output is softplus-positive: discharge (I > 0) gives ds/dt < 0 (cell shrinks),
    charge (I < 0) gives ds/dt > 0 (cell swells), with β ≥ 0 throughout —
    consistent with monotone graphite expansion on lithiation.
    """
    def __init__(self, config, n_hidden=32):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.beta_min = config.get('beta_min')
        self.beta_max = config.get('beta_max')
        if config.get('beta_constrained', 'false') == 'true':
            print(f'beta constrained to [{self.beta_min}, {self.beta_max}] 1e-5 m')
        else:
            print('beta unconstrained')

    def forward(self, soc, I_norm, u_norm):
        x = torch.stack([soc, I_norm, u_norm], dim=-1)
        if self.config.get('beta_constrained', 'false') == 'true':
            g = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.beta_min + g * (self.beta_max - self.beta_min)
        else:
            return nn.functional.softplus(self.net(x)).squeeze(-1)

# ══════════════════════════════════════════════════════════
#  ECMM MODEL
# ══════════════════════════════════════════════════════
class BatteryECMM(nn.Module):
    """
    Single-trajectory ECMM (B=1 in all paths; the leading dim is preserved
    so the rest of the code can stay shape-agnostic).

    forward() handles BOTH constant-current trajectories and pulse trajectories
    via a single code path.  The current input shape selects the mode:

        I_batch shape (B,)    → constant current per trajectory  (CC)
        I_batch shape (B, T)  → time-varying current             (pulse)

    SOC is integrated by cumulative sum in both cases (analytically equivalent
    to soc0 - I·t/Q0 when I is constant).  V_mode='static' is only meaningful
    for the CC case; it is rejected if a sequence is provided.
    """
    def __init__(self, config, Q0=Q0, I_ref=24.7915, u_ref=-4.2976, k=53.0):
        super().__init__()
        # Ue(SOC) is sourced from the module-level Ue_GP lookup (cached GP) —
        # no longer a constructor argument. This makes checkpoint loading
        # self-contained and matches how `sr_ode` consumes the GP from JN_GP.
        self.Q0        = Q0
        self.I_ref     = I_ref
        self.u_ref     = u_ref
        self.k         = k
        self.config    = config
        nh = config.get('n_hidden', 32)

        # ── k network (always; static algebraic stiffness) ──
        self.k_net = kNet(config, n_hidden=nh, k=k)

        # ── s network (always; static algebraic stiffness) ──
        self.beta_net = betaNet(config, n_hidden=nh)

        # ── R1 net — always network, optionally constrained ──
        if config.get('R1_constrained', 'false') == 'true':
            self.r1_net = R1NetConstrained(config, n_hidden=nh)
        else:
            print('R1 unconstrained')
            self.r1_net = R1Net(n_hidden=nh)

        # ── C1 net — always network, optionally constrained ──
        if config.get('C1_constrained', 'false') == 'true':
            self.C1_net = C1NetConstrained(config, n_hidden=nh)
        else:
            print('C1 unconstrained')
            self.C1_net = C1Net(n_hidden=nh)

        # ── R0 — multiple modes still supported ──
        m = config['R0_mode']
        if m == 'net':
            if config.get('R0_constrained', 'false') == 'true':
                self.R0_net = R0NetConstrained(config, n_hidden=nh)
            else:
                print('R0 unconstrained')
                self.R0_net = R0Net(n_hidden=nh)
        elif m == 'func':
            self.R0_func = R0_func
        elif m == 'param':
            self.log_R0 = nn.Parameter(torch.tensor(np.log(config.get('R0_param', 0.01)), dtype=torch.float32))
        elif m == 'net_no_soc':
            self.R0_net = R0NetNoSOC(config, n_hidden=nh)
        else:
            raise ValueError(f"Unknown R0_mode: {m!r}. Use 'net', 'func', 'param', or 'net_no_soc'.")

    # ── Dispatchers ──
    def _R1(self, soc, I_norm, u):
        return self.r1_net(soc, I_norm, u)

    def _C1(self, soc, I_norm, u):
        return self.C1_net(soc, I_norm, u)

    def _R0(self, soc, I_norm, u_exp, I_seq):
        """Element-wise R0 evaluated on (B, T) tensors. Returns (B, T).

        Note: u_exp is the *normalized* u (u/u_ref) — this matches what the
        networks were trained on. R0_func, however, was fitted to physical
        u in [1e-5 m], so we de-normalize before calling it.
        """
        m = self.config['R0_mode']
        if m == 'func':
            u_raw = u_exp * self.u_ref           # back to [1e-5 m] for the fitted function
            return self.R0_func(u_raw, I_seq)
        elif m == 'net':
            return self.R0_net(soc, I_norm, u_exp)
        elif m == 'param':
            return torch.exp(self.log_R0)
        elif m == 'net_no_soc':
            return self.R0_net(I_norm, u_exp)
        raise ValueError(f"Unsupported R0_mode: {m!r}.")

    def forward(self, I_batch, u_batch, soc0_batch, T=None, V_mode='dynamic'):
        """
        I_batch    : (B,)     constant current per traj         → CC mode
                     (B, T)   per-timestep current sequence     → pulse mode
        u_batch    : (B,)
        soc0_batch : (B,)
        T          : int, required when I_batch is 1D, ignored when 2D
        V_mode     : 'dynamic' — full Euler U1 integration  (Stage 2 / production)
                     'static'  — algebraic U1 = I·R1        (Stage 1 / Brucker);
                                 only valid for CC input.
                                 C1_net is NOT called in this mode — guarantees
                                 C1 plays no role during Stage 1 training.
        F is algebraic in both modes (k is static).
        """
        # Batch remnant: B size is 1 

        # ── Resolve I_seq shape (B, T) and B, T ──
        if I_batch.ndim == 1:
            assert T is not None, "T must be provided when I_batch is 1D (CC mode)"
            B = I_batch.shape[0]
            I_seq = I_batch.unsqueeze(1).expand(B, T)
        elif I_batch.ndim == 2:
            B, T = I_batch.shape
            I_seq = I_batch
        else:
            raise ValueError(f"I_batch must be 1D or 2D, got shape {tuple(I_batch.shape)}")

        # ── SOC integration ──
        # We want soc[:, 0] = soc0, soc[:, n] = soc0 + sum_{k<n} dsoc[k]
        # cumsum gives sum_{k≤n}; subtract dsoc[:, :1] to shift the index.
        # soc[:, 0] = soc0_batch + dsoc[:,0] - dsoc[:,0] = soc0_batch
        dsoc = -I_seq / self.Q0
        soc  = soc0_batch.unsqueeze(1) + torch.cumsum(dsoc, dim=1) - dsoc[:, :1]    # (B, 1) + (B, T) - (B, 1)

        # Normalize to obtain latent inputs roughly in range [0,1]
        I_norm = I_seq / self.I_ref
        u_norm = u_batch / self.u_ref           # both negative for compression → u_norm > 0
        u_norm_exp  = u_norm.unsqueeze(1).expand(B, T)
        u_phys_exp = u_batch.unsqueeze(1).expand(B, T)

        # Parameters along the trajectory  (B, T)
        R1 = self._R1(soc, I_norm, u_norm_exp)
        R0 = self._R0(soc, I_norm, u_norm_exp, I_seq)

        # ── F branch: Tier 2 charge-driven kinematic NODE ──
        #   ds/dt = -β(SOC, I, u) · I/Q₀
        # No s on RHS → fully vectorizable, no Python loop, no stiffness possible.
        # Forward-Euler integration as a cumulative sum, with the same indexing
        # convention as the SOC integration above (s[:, 0] = 0).
        k = self.k_net(soc, I_norm, u_norm_exp)              # (B, T)
        beta = self.beta_net(soc, I_norm, u_norm_exp)        # (B, T)

        ds_seq = beta * I_seq / self.Q0                    # (B, T)
        s = torch.cumsum(ds_seq, dim=1) - ds_seq[:, :1]      # (B, T), s[:, 0] = 0

        Fr = - k * (u_phys_exp - s)            # GN/ 1e-5m * 1e-5m


        # ── V branch: static or dynamic U1 ──
        with torch.no_grad():
            Ue = Ue_GP.soc_to_Ue(soc, return_torch=True)

        if V_mode == 'static':
            # Steady-state of the RC: U1 = I · R1.  C1 is *not* used.
            U1 = I_seq * R1
            V  = Ue - I_seq * R0 - U1
        elif V_mode == 'static_no_R0':
            U1 = I_seq * R1
            V  = Ue - U1
        elif V_mode == 'dynamic':
            C1 = self._C1(soc, I_norm, u_norm_exp)
            U1_steps = [torch.zeros(B)]
            dt = 1.0
            for n in range(T - 1):
                C1_n = C1[:, n] if C1.ndim == 2 else C1
                # Semi-implicit Euler — unconditionally stable
                U1_next = (U1_steps[n] + dt * I_seq[:, n] / C1_n) / (1.0 + dt / (R1[:, n] * C1_n))
                U1_steps.append(U1_next)
            U1 = torch.stack(U1_steps, dim=1)

            V  = Ue - I_seq * R0 - U1
        else:
            raise ValueError(f"V_mode must be 'static', 'static_no_R0' or 'dynamic', got {V_mode!r}")


        return V, Fr, soc, U1, R1, s, beta


def vmode_from_style(style):
    """Map a CONFIG['style'] to the V_mode the model should run in for inference."""
    if style == 'static_no_R0':
        return 'static_no_R0'
    elif style == 'static':
        return 'static'
    elif style in ('dynamic', 'staged'):
        return 'dynamic'
    else:
        raise ValueError(f"Unknown style: {style!r}")


def get_C1(model, scalar=True, soc_ref=0.5, I_ref_val=10.0, u_ref_val=-0.6,
           soc=None, I_norm=None, u_exp=None):
    """Return a representative C1 value.

    scalar=True  → a single float at the (soc_ref, I_ref_val, u_ref_val) reference point
    scalar=False → trajectory-shape numpy array, evaluated at the given (soc, I_norm, u_exp)

    Note: u_ref_val is a *physical* reference u in [1e-5 m]. It is normalized
    by model.u_ref before being passed to the network. In scalar=False mode,
    u_exp is assumed to already be normalized.
    """
    if scalar:
        soc_t    = torch.tensor([soc_ref], dtype=torch.float32)
        I_norm_t = torch.tensor([I_ref_val / model.I_ref], dtype=torch.float32)
        u_norm_t = torch.tensor([u_ref_val / model.u_ref], dtype=torch.float32)
        with torch.no_grad():
            return model._C1(soc_t, I_norm_t, u_norm_t).mean().item()
    else:
        return model._C1(soc, I_norm, u_exp).detach().numpy()

# ══════════════════════════════════════════════════════════
#  DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════

def prepare_data(data):
    trajs = []
    for _, grp in data.groupby('trajectory', sort=False):
        grp = grp.sort_values('t').reset_index(drop=True)
        I_val, u_val = float(grp['I'].iloc[0]), float(grp['u'].iloc[0])     # u = [1e-5m]
        C_val  = float(grp['C'].iloc[0])
        u_per = float(grp['u_par'].iloc[0])
        trajs.append(dict(
            I=I_val, u=u_val, C=C_val, u_per=u_per,
            soc0=float(grp['soc'].iloc[0]), T=len(grp),
            V=torch.tensor(grp['V'].values, dtype=torch.float32),
            F=torch.tensor(grp['F'].values, dtype=torch.float32),
            soc=torch.tensor(grp['soc'].values, dtype=torch.float32),
            eta=torch.tensor(grp['eta'].values, dtype=torch.float32),
        ))
    return trajs

def prepare_pulse_data(pulse_raw):
    pulse_trajs = []
    for _, grp in pulse_raw.groupby('trajectory', sort=False):
        grp = grp.sort_values('t').reset_index(drop=True)
        pulse_trajs.append(dict(
            I_seq = torch.tensor(grp['I'].values,   dtype=torch.float32),  # sequence!
            u     = float(grp['u'].iloc[0]),
            u_per = float(grp['u_par'].iloc[0]),
            soc0  = float(grp['soc'].iloc[0]),
            T     = len(grp),
            t     = torch.tensor(grp['t'].values,   dtype=torch.float32),
            V     = torch.tensor(grp['V'].values,   dtype=torch.float32),
            F     = torch.tensor(grp['F'].values,   dtype=torch.float32),
            soc   = torch.tensor(grp['soc'].values, dtype=torch.float32),
            eta   = torch.tensor(grp['eta'].values, dtype=torch.float32),
        ))
    return pulse_trajs


# ══════════════════════════════════════════════════════════
#  TRAJECTORY → MODEL INPUT  (single traj, B=1)
# ══════════════════════════════════════════════════════════════

def _traj_inputs(tr):
    """Pack a single trajectory dict into the (I, u, soc0) tensors expected by
    BatteryECMM.forward.  Auto-detects CC vs pulse based on whether the dict
    carries 'I_seq' (pulse) or 'I' (CC scalar).

    Returns: I_b, u_b, soc0_b, T  — all with leading dim B=1.
    """
    u_b    = torch.tensor([tr['u']],    dtype=torch.float32)
    soc0_b = torch.tensor([tr['soc0']], dtype=torch.float32)
    if 'I_seq' in tr:
        I_b = tr['I_seq'].unsqueeze(0)              # (1, T) — pulse mode
    else:
        I_b = torch.tensor([tr['I']], dtype=torch.float32)   # (1,) — CC mode
    return I_b, u_b, soc0_b, tr['T']


# ══════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS  (single-trajectory SGD, with staged option)
# ══════════════════════════════════════════════════════════════

def _empty_history():
    return {'train': [], 'train_V': [], 'train_Fr': [],
            'test': [], 'test_V': [], 'test_F': [], 'time': 0.0,
            'train_rmse': [], 'train_rmse_V': [], 'train_rmse_Fr': [],
            'test_rmse_V': [], 'test_rmse_F': [],
            'stage': []}                       # which stage produced each epoch


def _train_inner(model, train_trajs, test_trajs,
                 optimizer, scheduler, n_epochs, print_every,
                 V_mode='dynamic', stage_label='', history=None):
    """
    One training pass with the given optimizer over n_epochs.

    Per-trajectory stochastic GD: each epoch shuffles the trajectory order
    and takes one optimizer step per trajectory.  The trajectory dict's
    shape (CC scalar 'I' or pulse 'I_seq') drives whether forward() runs
    in CC or pulse mode.

    V_mode is forwarded to model() so the same loop fits Stage 1 (static V)
    and Stage 2 (dynamic V).  V_mode is honoured for both CC and pulse
    trajectories — no implicit override — so e.g. V_mode='static_no_R0' on
    pulse data trains the algebraic equation against pulses (which is the
    diagnostic loss curve showing the algebraic V cannot follow transients).
    """
    if history is None:
        history = _empty_history()

    t0 = _time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_trajs))
        ep_mse = ep_mse_V = ep_mse_Fr = 0.0
        ep_rmse = ep_rmse_V = ep_rmse_Fr = 0.0
        n_steps = 0

        ''' Mark 2: batch-like accumulation of gradients over accum_steps trajectories '''
        accum_steps = 1
        alpha_F = 1  # approx last MSE_V / last MSE_Fr
        optimizer.zero_grad()

        for k, i in enumerate(order):
            tr = train_trajs[i]
            I_b, u_b, soc0_b, T = _traj_inputs(tr)

            V_pred, Fr_pred, *_ = model(I_b, u_b, soc0_b, T=T, V_mode=V_mode)

            loss_V  = ((V_pred[0]  - tr['V']) ** 2).mean()
            loss_Fr = ((Fr_pred[0] - tr['F']) ** 2).mean() * alpha_F  # scale up Fr loss to be in similar order as V loss
            loss = (loss_V + loss_Fr) / accum_steps    # normalize loss to account for accumulation

            loss.backward()

            # Batch like accumulation of gradients: step every accum_steps trajectories or at the end of the epoch
            if (k + 1) % accum_steps == 0 or (k + 1) == len(order):
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                rmse_V  = torch.sqrt(loss_V).item()
                rmse_Fr = torch.sqrt(loss_Fr/alpha_F).item()

            ep_mse    += loss.item() * accum_steps  # scale back up to the true loss for logging
            ep_mse_V  += loss_V.item()
            ep_mse_Fr += loss_Fr.item()
            ep_rmse   += rmse_V + rmse_Fr
            ep_rmse_V += rmse_V
            ep_rmse_Fr+= rmse_Fr
            n_steps += 1


        ep_mse /= n_steps; ep_mse_V /= n_steps; ep_mse_Fr /= n_steps
        ep_rmse /= n_steps; ep_rmse_V /= n_steps; ep_rmse_Fr /= n_steps

        history['train'].append(ep_mse)
        history['train_V'].append(ep_mse_V)
        history['train_Fr'].append(ep_mse_Fr)
        history['train_rmse'].append(ep_rmse)
        history['train_rmse_V'].append(ep_rmse_V)
        history['train_rmse_Fr'].append(ep_rmse_Fr)
        history['stage'].append(stage_label)

        # Test eval — uses the same V_mode as the training stage on every traj.
        model.eval()
        with torch.no_grad():
            test_mse = test_mse_V = test_mse_F = 0.0
            for tr in test_trajs:
                I_b, u_b, soc0_b, T = _traj_inputs(tr)
                V_pred, F_pred, *_ = model(I_b, u_b, soc0_b, T=T, V_mode=V_mode)
                test_mse_V += torch.mean((V_pred[0] - tr['V']) ** 2).item()
                test_mse_F += torch.mean((F_pred[0] - tr['F']) ** 2).item()
                test_mse += torch.mean((V_pred[0] - tr['V']) ** 2 + (F_pred[0] - tr['F']) ** 2).item()
            test_mse /= len(test_trajs)
            test_mse_V /= len(test_trajs)
            test_mse_F /= len(test_trajs)
            test_rmse_V = float(np.sqrt(test_mse_V))
            test_rmse_F = float(np.sqrt(test_mse_F))
        history['test'].append(test_mse)
        history['test_V'].append(test_mse_V)
        history['test_F'].append(test_mse_F)
        history['test_rmse_V'].append(test_rmse_V)
        history['test_rmse_F'].append(test_rmse_F)

        if scheduler is not None:
            scheduler.step(ep_mse)

        if epoch % print_every == 0 or epoch == 1:
            C1 = get_C1(model, scalar=True)
            eta = (_time.time() - t0) / epoch * (n_epochs - epoch) / 60
            tag = f"[{stage_label}] " if stage_label else ""
            print(f"  {tag}{epoch:4d}/{n_epochs} | ETA {eta:.1f}m "
                  f"| RMSE V {ep_rmse_V:.4f} Fr {ep_rmse_Fr:.4f} "
                  f"| test V {test_rmse_V:.4f} F {test_rmse_F:.4f} | C1={C1:.0f}F | LR {optimizer.param_groups[0]['lr']:.2e}")

    history['time'] = history.get('time', 0.0) + (_time.time() - t0) / 60
    return history


def train_model(model, train_trajs, test_trajs,
                n_epochs=200, lr=1e-3, print_every=10,
                V_mode='dynamic', freeze=None):
    """Single-stage training (default behaviour: dynamic V from epoch 1)."""
    freeze_kw = freeze if freeze is not None else ()
    params = [p for name, p in model.named_parameters() if not any(kw in name for kw in freeze_kw)]
    n_ = sum(p.numel() for p in params)
    print(f"  Trainable params: {n_}  ({freeze_kw} frozen)")

    optimizer = torch.optim.Adam(params, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PAT, factor=0.5) 
    return _train_inner(model, train_trajs, test_trajs,
                        optimizer, scheduler, n_epochs, print_every,
                        V_mode=V_mode, stage_label='single')


def train_staged(model, train_trajs, test_trajs,
                 n_epochs_static=100, n_epochs_dynamic=200,
                 lr_static=1e-3, lr_dynamic=1e-3,
                 print_every=10,
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
        Trains: r1_net, k_net
        Frozen: C1_net

    Stage 2  (dynamic V):
        V uses full Euler U1 integration with dU1/dt = I/C1 − U1/(R1·C1)
        F unchanged (still algebraic)
        Trains: C1_net, k_net
        Frozen: r1_net

        Data: `pulse_train_trajs` if provided, else the constant-current
        `train_trajs`.  Pulse data automatically uses dynamic V regardless
        of stage flags (static V requires constant I).

    Stage 2b (optional, only if n_epochs_unfreeze > 0):
        Same dynamics + data as Stage 2, but r1_net is unfrozen so R1 gets
        refined under the dynamic objective.  Uses lr_unfreeze (defaults to
        lr_dynamic).

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
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=PAT, factor=0.5)
    _train_inner(model, train_trajs, test_trajs,
                 opt1, sched1, n_epochs_static, print_every,
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

    # ── Stage 2 (R1 frozen — train C1 + k) ──
    print(f"\n========== STAGE 2: dynamic V  ({n_epochs_dynamic} epochs, {data_tag} data) ==========")
    freeze_kw = ('r1_net',)
    s2_params = [p for name, p in model.named_parameters()
                 if not any(kw in name for kw in freeze_kw)]
    n_s2 = sum(p.numel() for p in s2_params)
    print(f"  Stage 2 trainable params: {n_s2}  (r1_net frozen)")
    opt2 = torch.optim.Adam(s2_params, lr=lr_dynamic)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=PAT, factor=0.5)
    _train_inner(model, s2_train, s2_test,
                 opt2, sched2, n_epochs_dynamic, print_every,
                 V_mode='dynamic', stage_label='S2', history=history)

    history['stage2_epochs'] = n_epochs_dynamic

    # ── Optional callback between Stage 2 and Stage 2b ──
    if on_stage2_done is not None:
        print("\n---- post-Stage-2 callback ----")
        on_stage2_done(model, history)

    # ── Stage 2b (optional): unfreeze R1 ──
    if n_epochs_unfreeze and n_epochs_unfreeze > 0:
        lr_b = lr_unfreeze if lr_unfreeze is not None else lr_dynamic
        print(f"\n========== STAGE 2b: dynamic V, R1 UNFROZEN  "
              f"({n_epochs_unfreeze} epochs, {data_tag} data, lr={lr_b}) ==========")
        s2b_params = list(model.parameters())   # everything trainable
        n_s2b = sum(p.numel() for p in s2b_params)
        print(f"  Stage 2b trainable params: {n_s2b}  (R1 unfrozen)")
        opt2b = torch.optim.Adam(s2b_params, lr=lr_b)
        sched2b = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2b, patience=PAT, factor=0.5)
        _train_inner(model, s2_train, s2_test,
                     opt2b, sched2b, n_epochs_unfreeze, print_every,
                     V_mode='dynamic', stage_label='S2b', history=history)
        history['stage2b_epochs'] = n_epochs_unfreeze

    return history



# ══════════════════════════════════════════════════════════
#  PREDICT  (single-trajectory rollout, returns a dict)
# ══════════════════════════════════════════════════════════════



@torch.no_grad()
def predict_np(model, config, traj, V_mode=None):
    """Single-trajectory rollout for plotting.  Auto-detects CC vs pulse from
    whether `traj` carries 'I_seq' (pulse) or 'I' (CC).

    V_mode controls the V equation — 'static' (V = Ue - I·R0 - I·R1),
    'static_no_R0' (V = Ue - I·R1) or 'dynamic' (Euler RC integration with
    V = Ue - I·R0 - U1).  When V_mode is None (default) it is derived from
    config['style'] via vmode_from_style().  V_mode is honoured for both CC
    and pulse trajectories — no implicit override — so a static-trained model
    can be evaluated on pulses to show the algebraic V can't follow them.

    For V_mode in ('static', 'static_no_R0') C1 is returned as None so plots
    can omit any C1-dependent panels.

    Returns a dict with keys: V, soc, U1, R1, Fr, k, C1, R0, I  — all numpy
    arrays of length T (R0 / I are arrays in pulse mode and constant arrays
    in CC mode, so plotting code can treat them uniformly).
    """
    if V_mode is None:
        V_mode = vmode_from_style(config.get('style', 'dynamic'))
    pulse = 'I_seq' in traj
    I_b, u_b, soc0_b, T = _traj_inputs(traj)

    V, Fr, soc, U1, R1, s, beta = model(I_b, u_b, soc0_b, T=T, V_mode=V_mode)
    V    = V[0].numpy();    Fr   = Fr[0].numpy()
    soc  = soc[0].numpy();  U1   = U1[0].numpy()
    R1   = R1[0].numpy();   s    = s[0].numpy()
    beta = beta[0].numpy()

    # Trajectory-shape arrays for parameter evaluation
    I_np   = traj['I_seq'].numpy() if pulse else np.full(T, traj['I'])
    u_np   = np.full(T, traj['u'])
    soc_t  = torch.from_numpy(soc.astype(np.float32))
    I_norm = torch.from_numpy((I_np / model.I_ref).astype(np.float32))
    u_t    = torch.from_numpy(u_np.astype(np.float32))             # raw u [1e-5 m]
    u_norm = u_t / model.u_ref                                     # normalized — what networks see

    k = model.k_net(soc_t, I_norm, u_norm).numpy()
    R0 = model._R0(soc_t, I_norm, u_norm, I_np).numpy()            # (T,) — _R0 expects u_norm

    if V_mode in ('static', 'static_no_R0'):
        C1 = None       # C1 not used when V is algebraic — don't evaluate it
    else:
        C1 = get_C1(model, scalar=False, soc=soc_t, I_norm=I_norm, u_exp=u_norm)

    return dict(V=V, soc=soc, U1=U1, R1=R1, Fr=Fr, k=k, s=s, beta=beta,
                C1=C1, R0=R0, I=I_np)


# ══════════════════════════════════════════════════════════
#  PLOT PREDICTIONS  (one function for both CC and pulse)
# ════════════════════════════════════════════════

def plot_predictions(model, config, trajs, time=False, title='', n_show=3,
                     V_mode=None):
    """Per-trajectory diagnostic grid.  Auto-detects CC vs pulse per trajectory.

    Row layout:
        pulse traj, V_mode='dynamic'        - 9 rows (I, V, soc, eta, R, C1, Fr, k, s)
        pulse traj, V_mode in static modes  - 8 rows (I, V, soc, eta, R, Fr, k, s) C1 omitted
        CC traj,    V_mode='dynamic'        - 7 rows (V, eta, R, C1, Fr, k, s)
        CC traj,    V_mode in static modes  - 6 rows (V, eta, R, Fr, k, s) C1 omitted

    eta is the overpotential eta = V − Uₑ taken straight from the data column —
    model-independent.  Predicted eta is built from the model's V equation:
        static_no_R0     →  eta_pred = -I·R₁
        static / dynamic →  eta_pred = -(I·R₀ + U₁)

    When V_mode is None (default), it is derived from config['style'] via
    vmode_from_style().
    All trajectories in `trajs` should be the same kind (all CC or all pulse).
    """
    if V_mode is None:
        V_mode = vmode_from_style(config.get('style', 'dynamic'))
    n = min(n_show, len(trajs))
    if n == 0:
        raise ValueError("trajs is empty")

    # Determine kind from first trajectory; assume the rest are the same.
    pulse = 'I_seq' in trajs[0]
    if pulse and V_mode in ('static', 'static_no_R0'):
        # Pulse evaluated under an algebraic V — C1 is unused, omit its panel.
        rows = ['I', 'V', 'soc', 'eta', 'R', 'Fr', 'k', 's', 'beta']
    elif pulse:
        rows = ['I', 'V', 'soc', 'eta', 'R', 'C1', 'Fr', 'k', 's', 'beta']
    elif V_mode in ('static', 'static_no_R0'):
        rows = ['V', 'eta', 'R', 'Fr', 'k', 's', 'beta']
    elif V_mode == 'dynamic':
        rows = ['V', 'eta', 'R', 'C1', 'Fr', 'k', 's', 'beta']
    else:
        raise ValueError(
            f"V_mode must be 'static', 'static_no_R0' or 'dynamic', got {V_mode!r}")

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(5 * n, 3.3 * n_rows), squeeze=False)
    model.eval()

    for j in range(n):
        tr = trajs[j + 3]   # Hard code skip the first 3 trajs to show more interesting ones
        T  = tr['T']
        out = predict_np(model, config, tr, V_mode=V_mode)
        V, soc_np, U1 = out['V'], out['soc'], out['U1']
        R1, Fr, k_pred, s_pred = out['R1'], out['Fr'], out['k'], out['s']
        beta_pred = out['beta']
        C1, R0, I_np = out['C1'], out['R0'], out['I']

        # x-axis: time index when time=True OR for pulse data (SOC isn't monotonic
        # under repeated charge/discharge pulses, so SOC-on-x makes no sense).
        use_time_x = time or pulse
        x = np.arange(T) if use_time_x else soc_np

        # Trajectory header — used in the title of the topmost row
        if pulse:
            traj_header = f'{title}pulse traj {j}, u={tr["u"]:.3f}'
        else:
            traj_header = f'{title}I={tr["I"]:.1f}, u={tr["u"]:.3f}'

        for r, name in enumerate(rows):
            ax = axes[r, j]

            if name == 'I':         # pulse-only
                ax.plot(x, I_np, '-', color=COLORS[0], lw=2)
                ax.set_ylabel(r'$I$ [A]')
                ax.set_title(traj_header)

            elif name == 'V':
                ax.plot(x, tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
                ax.plot(x, V,               '-',  color=COLORS[0], label=r'Predicted $V$', lw=2)
                ax.set_ylabel(r'$V$ [V]'); ax.legend()
                if not pulse:        # for CC, V is the topmost row
                    ax.set_title(traj_header)

            elif name == 'soc':     # pulse-only — SOC consistency check
                ax.plot(np.arange(T), tr['soc'].numpy(), '--', color=COLORS[1],
                        label='True SOC', lw=2)
                ax.plot(np.arange(T), soc_np, '-', color=COLORS[0],
                        label='Predicted SOC', lw=2)
                ax.set_ylabel('SOC'); ax.legend()
                ax.set_xlabel('Time [s]')   # always indexed in time

            elif name == 'eta':
                # eta_true = V_data - Ue, taken directly from the data column —
                # model-independent, no R0 assumption, identical across V_modes.
                # eta_pred follows the model's V equation:
                #   static_no_R0:    eta = V - Ue = -U1                = -I·R1
                #   static / dynamic: eta = V - Ue = -(I·R0 + U1)
                eta_true = tr['eta'].numpy()
                if V_mode == 'static_no_R0':
                    eta_pred = U1
                else:
                    eta_pred = (I_np * R0 + U1)
                ax.plot(x, eta_true, '--', color=COLORS[1], label=r'True $\eta$', lw=2)
                ax.plot(x, eta_pred, '-',  color=COLORS[0], label=r'Predicted $\eta$', lw=2)
                ax.set_ylabel(r'$\eta$ [V]'); ax.legend()

            elif name == 'R':
                if config['R0_mode'] in ('func', 'net', 'net_no_soc'):
                    ax.plot(x, R0 * 1000, '--', color=COLORS[0], label=r'$R_0$', lw=2)
                else:   # 'param'
                    R0_val = float(R0[0])
                    ax.axhline(R0_val * 1000, ls='--', color=COLORS[0],
                               label=r'$R_0$' + fr' = {R0_val*1000:.1f} m$\Omega$', lw=2)
                ax.plot(x, R1 * 1000, '-', color=COLORS[0], label=r'$R_1$', lw=2)
                ax.set_ylabel(r'$R$ [m$\Omega$]'); ax.legend()

            elif name == 'C1':
                ax.plot(x, C1, ls='--', color=COLORS[0], label=r'$C_1$', lw=2)
                ax.set_ylabel(r'$C_1$ [F]'); ax.legend()

            elif name == 'Fr':
                ax.plot(x, tr['F'].numpy(), '--', color=COLORS[1], label=r'True $F_r$', lw=2)
                ax.plot(x, Fr,              '-',  color=COLORS[0], label=r'Predicted $F_r$', lw=2)
                ax.set_ylabel(r'$F_r$ [GN]'); ax.legend()

            elif name == 'k':
                # Empirical stiffness directly from the data: k_true = -F/u
                k_true = -tr['F'] / tr['u'] * 1e2 # Convert u from 1e-5m to 1e-5*1e2 = mm
                k_pred = k_pred * 1e2   # convert back from GN/1e-5m to GN/mm for plotting. 1e2 GN / (1e-2*1e-3 m) = 1e2GN/mm
                # ax.plot(x, k_true, '--', color=COLORS[1], label=r'INVALID! True $k = -F/u$', lw=2, alpha=0.7)
                ax.plot(x, k_pred, '-',  color=COLORS[0], label=r'Predicted $k$', lw=2)
                ax.set_ylabel(r'$k$ [GN/mm]'); ax.legend()

            elif name == 's':
                # s integrated from ds/dt = -β·I/Q₀ inside the model.  s[0] = 0
                # by convention (anchored at trajectory's starting SOC).
                ax.plot(x, s_pred / 100, '-',  color=COLORS[0], label=r'Predicted $s$', lw=2)
                ax.set_ylabel(r'$s$ [mm]'); ax.legend()

            elif name == 'beta':
                # β: kinematic gain ds/dSOC.  Path-independence diagnostic — if
                # β(SOC) overlays across C-rates, the data is path-independent.
                ax.plot(x, beta_pred, '-', color=COLORS[0], label=r'Predicted $\beta$', lw=2)
                ax.set_ylabel(r'$\beta$ [$10^{-5}$ m]'); ax.legend()

    # x-label + axis direction handled per-trajectory-kind
    for ax in axes.flat:
        if ax.get_xlabel() == 'Time [s]':
            continue        # 'soc' panel already labelled itself
        if pulse or time:
            ax.set_xlabel('Time [s]')
        else:
            ax.set_xlabel('State of Charge')
            ax.invert_xaxis()

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
    ax.semilogy(epochs, history['test_rmse_V'],     color=COLORS[2], lw=2, ls='--',
                label=r'Test $V$  (final {:.4f} V)'.format(history['test_rmse_V'][-1]))
    ax.semilogy(epochs, history['test_rmse_F'],     color=COLORS[3], lw=2, ls='--',
                label=r'Test $F_r$  (final {:.4f} GN)'.format(history['test_rmse_F'][-1]))

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
    ax.grid(True, which='both', ls=':', color='0.8')
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig


# =════════════════════════════════════════════════════════════════
# RMSE CALC FOR PULSES
# =════════════════════════════════════════════════════════════════

def rmse_pulse(model, pulse_trajs):
    rmse = []
    for tr in pulse_trajs:
        out = predict_np(model, model.config, tr)
        rmse.append(float(np.sqrt(np.mean((out['V'] - tr['V'].numpy())**2))))
    return rmse


# =═════════════════════════════════════════════════════════
# Plotter for ECM parameters
# ══════════════════════════════════════════════════════════════

def plot_param(model, trajs, param='R1'):
    """
    Plot R0, R1, C1 or k across SOC for all given trajectories (one line each).

    Parameters
    ----------
    model : BatteryECMM
    trajs : list of CC trajectory dicts (e.g. test_trajs)
    param : 'R0', 'R1', 'C1', or 'k'
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    model.eval()

    trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
    C_vals = np.array([tr['C'] for tr in trajs_sorted])

    base = plt.cm.Blues_r
    Blues_cut = LinearSegmentedColormap.from_list(
        "Blues_custom", base(np.linspace(0.0, 0.8, 256)))
    cmap = Blues_cut

    base = plt.cm.Reds_r
    Reds_cut = LinearSegmentedColormap.from_list(
        "Reds_custom", base(np.linspace(0.0, 0.8, 256)))
    cmap_r = Reds_cut

    norm = Normalize(vmin=C_vals.min(), vmax=C_vals.max())
    norm_u = Normalize(vmin=0, vmax=30)

    with torch.no_grad():
        for tr in trajs_sorted:
            soc    = tr['soc']
            I_val  = float(tr['I'])
            u_val  = float(tr['u'])
            print(u_val)
            u_per_val = float(tr['u_per'])
            C_val  = float(tr['C'])
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_t    = torch.full_like(soc, u_val)                # raw u [1e-5 m]
            u_norm = torch.full_like(soc, u_val / model.u_ref)  # what the networks see
            I_real = torch.full_like(soc, I_val)                # raw I [A] for _R0's I_seq arg
            xlabel = 'State of Charge'

            if param == 'R1':
                y = model._R1(soc, I_norm, u_norm).numpy() * 1e3
                ylabel = r'$R_1$ [m$\Omega$]'

            elif param == 'C1':
                y = model._C1(soc, I_norm, u_norm).numpy()
                ylabel = r'$C_1$ [F]'

            elif param == 'R0':
                y = model._R0(soc, I_norm, u_norm, I_real).numpy() * 1e3
                ylabel = r'$R_0$ [m$\Omega$]'

            elif param == 'k':
                y = model.k_net(soc, I_norm, u_norm).numpy()
                ylabel = r'$k$ [GN/mm]'
                # k_true = (-tr['F'] / tr['u']).numpy() * 1e2 # Convert u from 1e-5m to 1e-5*1e2 = mm
                y = y * 1e2   # convert back from GN/1e-5m to GN/mm for plotting. 1e2 GN / (1e-2*1e-3 m) = 1e2GN/mm

                #ax.plot(soc.numpy(), k_true, '--', color=cmap_r(norm_u(u_per_val)), label='True $k$', lw=2)
            
            elif param == 'ku':
                y = model.k_net(soc, I_norm, u_norm).numpy()
                ylabel = r'$k$ [GN/mm]'
                y = y * 1e2   # convert back from GN/1e-5m to GN/mm for plotting. 1e2 GN / (1e-2*1e-3 m) = 1e2GN/mm

            
            elif param == 's':
                # Tier 2: s is path-dependent.  Integrate -β·I/Q₀ along this
                # CC trajectory (dt=1) using the same convention as model.forward.
                beta_t = model.beta_net(soc, I_norm, u_norm)
                ds = beta_t * I_real / model.Q0                  # (T,)
                s_t = torch.cumsum(ds, dim=0) - ds[:1]            # s[0] = 0
                y = s_t.numpy() / 100.0                            # 1e-5 m → mm
                ylabel = r'$s$ [mm]'

            elif param == 'beta':
                # Path-independence diagnostic: overlay β(SOC) across C-rates.
                # If they collapse, β depends only on SOC ⇒ data is path-indep.
                y = model.beta_net(soc, I_norm, u_norm).numpy()
                ylabel = r'$\beta$ [$10^{-5}$ m]'

            else:
                raise ValueError(f"param must be 'R0', 'R1', 'C1', or 'k', got {param!r}")

            if not param == 'ku':
                ax.plot(soc.numpy(), y, '-', color=cmap(norm(C_val)), lw=2)
            else:
                ax.plot(u_t.numpy(), y, 'o', color=cmap(norm(C_val)), lw=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()
    ax.ticklabel_format(useOffset=False, style='plain')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='C-rate [a.u.]')
    # if param == 'k':
    #     sm = ScalarMappable(cmap=cmap_r, norm=norm_u)
    #     fig.colorbar(sm, ax=ax, label='u [%]')

    fig.tight_layout()
    return fig


def plot_force(model, trajs):
    """Plot reaction force F = -k(soc, I, u)·u vs u [%], colored by SOC.

    Each trajectory contributes len(soc) points: x = u_per (constant per traj),
    y = -k·u (varies along the traj because k depends on SOC), color = SOC.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    model.eval()

    base = plt.cm.Blues_r
    cmap = LinearSegmentedColormap.from_list(
        "Blues_custom", base(np.linspace(0.0, 0.8, 256)))

    base = plt.cm.Reds_r
    Reds_cut = LinearSegmentedColormap.from_list(
        "Reds_custom", base(np.linspace(0.0, 0.8, 256)))
    cmap_r = Reds_cut

    norm = Normalize(vmin=0, vmax=1)

    with torch.no_grad():
        for tr in trajs:
            soc       = tr['soc']
            T        = tr['T']
            I_val     = float(tr['I'])
            u_val     = float(tr['u'])
            u_per_val = float(tr['u_per'])
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_norm = torch.full_like(soc, u_val / model.u_ref)   # what the networks see
            u_phys = torch.full_like(soc, u_val)                 # raw u [1e-5 m] for force calculation

            k = model.k_net(soc, I_norm, u_norm).numpy()
            # Tier 2: integrate s = ∫(-β·I/Q₀)dt along this CC trajectory
            beta_t = model.beta_net(soc, I_norm, u_norm)
            ds = -beta_t * I_val / model.Q0
            s = (torch.cumsum(ds, dim=0) - ds[:1]).numpy()       # s[0] = 0

            F = - k * (u_phys.numpy() - s)                       # GN
            F_true = tr['F'].numpy()
            x = np.full(len(soc), u_per_val)                # constant per traj

            ax.scatter(x, F, c=soc.numpy(), cmap=cmap, norm=norm, s=6)
            ax.scatter(x, F_true, c=soc.numpy(), cmap=cmap_r, norm=norm, s=2, linewidths=0.1)

    ax.set_xlabel(r'$u$ $[\%]$')
    ax.set_ylabel(r'$F$ [GN]')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='State of Charge')
    fig.tight_layout()
    return fig

def plot_swelling(model, trajs):
    """Plot reaction force F = -k(soc, I, u)·u vs u [%], colored by SOC.

    Each trajectory contributes len(soc) points: x = u_per (constant per traj),
    y = -k·u (varies along the traj because k depends on SOC), color = SOC.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    model.eval()

    base = plt.cm.Blues_r
    cmap = LinearSegmentedColormap.from_list(
        "Blues_custom", base(np.linspace(0.0, 0.8, 256)))

    base = plt.cm.Reds_r
    Reds_cut = LinearSegmentedColormap.from_list(
        "Reds_custom", base(np.linspace(0.0, 0.8, 256)))
    cmap_r = Reds_cut

    trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
    C_vals = np.array([tr['C'] for tr in trajs_sorted])
    norm = Normalize(vmin=C_vals.min(), vmax=C_vals.max())
    norm_u = Normalize(vmin=0, vmax=1)

    with torch.no_grad():
        for tr in trajs:
            soc       = tr['soc']
            I_val     = float(tr['I'])
            C_val     = float(tr['C'])
            u_per_val = float(tr['u_per'])
            T         = tr['T']
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_norm = torch.full_like(soc, u_per_val / model.u_ref)   # what the networks see

            # Tier 2: s is path-dependent — integrate along this CC trajectory.
            I_real = torch.full_like(soc, I_val)
            beta_t = model.beta_net(soc, I_norm, u_norm)
            ds = beta_t * I_real / model.Q0
            s = (torch.cumsum(ds, dim=0) - ds[:1]).numpy()        # s[0] = 0
            x = np.full(len(soc), u_per_val)                # constant per traj

            # ax.scatter(u_per_val, s.max(), c=C_val, cmap=cmap, norm=norm, s=6)
            ax.scatter(x, s, c=C_val*np.ones_like(s), cmap=cmap, norm=norm, s=6)


    ax.set_xlabel(r'$u$ $[\%]$')
    ax.set_ylabel(r'$s$ [mm]')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='C-rate [a.u.]')
    fig.tight_layout()
    return fig


def element_predict(model, c_rate, u_per, soc, element=None, Q0=Q0, L0=LIMON_CELL0):
    '''
    Element value predictor. Accepts any mix of scalars / lists / 1-D arrays /
    tensors for c_rate, u_per, and soc — they are broadcast to a common shape
    before being fed to the networks (so e.g. a scalar c_rate against a list
    of SOCs Just Works).

    Returns (R1, C1, R0, k, beta) numpy arrays of the broadcast shape, or a
    single array if `element` ('R0' / 'R1' / 'C1' / 'k' / 'beta') is given.

    Note on `s` vs `beta`: in Tier 2 the swelling s is path-dependent (it's the
    time-integral of -β·I/Q₀), so there is no static s(SOC, I, u) map to query
    pointwise.  This function returns the kinematic gain β instead — the
    state-function that *does* exist pointwise.  Use predict_np to obtain s
    along an actual trajectory.

    Notes on units (must match training):
        c_rate  → I_real = c_rate · Q0 / 3600   [A]
                  I_norm = I_real / model.I_ref
        u_per   → u      = u_per · L0           [1e-5 m]   (signed; compression < 0)
    '''
    model.eval()
    with torch.no_grad():
        c_rate = torch.atleast_1d(torch.as_tensor(c_rate, dtype=torch.float32))
        u_per  = torch.atleast_1d(torch.as_tensor(u_per,  dtype=torch.float32))
        soc    = torch.atleast_1d(torch.as_tensor(soc,    dtype=torch.float32))

        # Broadcast to a common shape so callers can mix scalars and lists freely.
        # Without this, the inner torch.stack inside the small networks fails
        # with "stack expects each tensor to be equal size".
        c_rate, u_per, soc = torch.broadcast_tensors(c_rate, u_per, soc)
        c_rate = c_rate.contiguous()
        u_per  = u_per.contiguous()
        soc    = soc.contiguous()

        I_real = c_rate * Q0 / 3600.0          # actual current [A]
        I_norm = I_real / model.I_ref          # what the networks were trained on
        u_real = u_per * L0                    # cell displacement [1e-5 m]
        u_norm = u_real / model.u_ref          # what the networks were trained on

        R1 = model._R1(soc, I_norm, u_norm).numpy()              # Ohm
        C1 = model._C1(soc, I_norm, u_norm).numpy()              # F
        R0 = model._R0(soc, I_norm, u_norm, I_real).numpy()      # Ohm   (I_seq = real I, not normalised)
        k  = model.k_net(soc, I_norm, u_norm).numpy()            # GN/1e-5m
        beta = model.beta_net(soc, I_norm, u_norm).numpy()       # [1e-5 m]

    out = {'R1': R1, 'C1': C1, 'R0': R0, 'k': k, 'beta': beta}
    return out[element] if element is not None else (R1, C1, R0, k, beta)


def data_param(model, trajs):
    """
    Return a long-form DataFrame of R0, R1, C1, k, and β across SOC for all
    given trajectories — one row per (trajectory, soc-sample).

    β here is the kinematic gain ds/dSOC, evaluated pointwise from beta_net.
    The integrated swelling s is path-dependent and not returned by this
    function — use predict_np for trajectory-level s values.
    """
    model.eval()
    trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
    frames = []

    with torch.no_grad():
        for i, tr in enumerate(trajs_sorted):
            soc    = tr['soc']
            I_val  = float(tr['I'])
            u_val  = float(tr['u'])
            u_per_val = float(tr['u_per'])
            C_val  = float(tr['C'])
            I_norm = torch.full_like(soc, I_val / model.I_ref)
            u_norm = torch.full_like(soc, u_val / model.u_ref)
            I_real = torch.full_like(soc, I_val)                     # raw I [A] for _R0's I_seq arg

            R1 = model._R1(soc, I_norm, u_norm).numpy()              # Ohm
            C1 = model._C1(soc, I_norm, u_norm).numpy()              # F
            R0 = model._R0(soc, I_norm, u_norm, I_real).numpy()      # Ohm — pass raw I, not I_norm
            k  = model.k_net(soc, I_norm, u_norm).numpy()            # GN/1e-5m
            beta = model.beta_net(soc, I_norm, u_norm).numpy()       # [1e-5 m]

            frames.append(pd.DataFrame({
                'trajectory': i,
                'C': C_val,
                'u_per': u_per_val,
                'I': I_val,
                'u': u_val,
                'soc': soc.numpy(),
                'R1': R1,
                'C1': C1,
                'R0': R0,
                'k': k,
                'beta': beta}))

    return pd.concat(frames, ignore_index=True)


# =═════════════════════════════════════════════════════════
# Plotter for predictions
# ══════════════════════════════════════════════════════════════

def plot_predicts(model, config, trajs, predict='V', sort='C_rate'):
    """
    Plot V or F prediction vs true across SOC for all given (CC) trajectories.

    Parameters
    ----------
    model : BatteryECMM
    trajs : list of CC trajectory dicts (e.g. test_trajs)
    predict : 'V' or 'F'
    sort : 'C_rate' or 'u_per'
    """
    assert predict in ('V', 'F'), "predict must be 'V' or 'F'"

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    model.eval()

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
        "Blues_custom", base(np.linspace(0.0, 0.8, 256)))
    cmap_b = Blues_cut
    base = plt.cm.Reds_r
    Reds_cut = LinearSegmentedColormap.from_list(
        "Reds_custom", base(np.linspace(0.0, 0.8, 256)))
    cmap_r = Reds_cut

    with torch.no_grad():
        for tr in trajs_sorted:
            C_val     = float(tr['C'])
            u_per_val = float(tr['u_per'])

            out = predict_np(model, config, tr)
            soc_np = out['soc']
            if predict == 'V':
                y_true = tr['V'].numpy(); y_pred = out['V']
                ylabel = r'$V$ [V]'
            elif predict == 'F':
                y_true = tr['F'].numpy(); y_pred = out['Fr']
                ylabel = r'$F$ [GN]'

            bar_val = C_val if sort == 'C_rate' else u_per_val

            ax.plot(soc_np, y_true, '--', color=cmap_r(norm(bar_val)), lw=2)
            ax.plot(soc_np, y_pred, '-',  color=cmap_b(norm(bar_val)), lw=2)

    ax.set_xlabel('State of Charge')
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()

    # legend with two cheat handles
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='tab:red',  lw=2, label='True'),
        Line2D([0], [0], color='tab:blue', lw=2, label='Predicted'),
    ])

    fig.tight_layout()
    sm_true = ScalarMappable(cmap=cmap_b, norm=norm)
    fig.colorbar(sm_true, ax=ax, label=bar_name, pad=0.02)
    return fig

def load_nn_model(model_name, I_ref=None):
    """Load a saved BatteryECMM checkpoint.
    Parameters
    ----------
    model_name : str
        Filename inside ./models/.
    I_ref : float or None
        Reference current used to normalise I.  If None, falls back to the
        value saved in the checkpoint (new checkpoints) and finally to the
        BatteryECMM default (24.79).  Pass an explicit value when loading
        older checkpoints that don't carry I_ref.
    u_ref : float or None
        Reference voltage used to normalise u.  If None, falls back to the
        value saved in the checkpoint (new checkpoints) and finally to the
        BatteryECMM default (-0.0862).  Pass an explicit value when loading
        older checkpoints that don't carry u_ref.
    """
    ckpt_file = os.path.join(FILE_PATH, 'models', model_name)
    ckpt      = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    CONFIG = ckpt['config']
    print(f"Loaded checkpoint with config: {CONFIG}")

    I_ref = ckpt.get('I_ref', 24.7915)    # Use persisted I_ref if model saved it, else default
    u_ref = ckpt.get('u_ref', -4.2976)       # Use persisted u_ref if model saved it, else default
    print(f"Using I_ref = {I_ref} and 'u_ref' = {u_ref} for model parameters")

    model = BatteryECMM(CONFIG, I_ref=I_ref, u_ref=u_ref)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, ckpt

def load_checkpoint(ckpt):
    history  = ckpt['history']
    CONFIG   = ckpt['config']
    C1_final = ckpt['C1_final']
    N_HIDDEN = ckpt['N_HIDDEN']
    EPOCHS_STATIC   = ckpt['EPOCHS_STATIC']
    EPOCHS_DYNAMIC  = ckpt['EPOCHS_DYNAMIC']
    EPOCHS_UNFREEZE = ckpt['EPOCHS_UNFREEZE']
    return history, CONFIG, C1_final, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE