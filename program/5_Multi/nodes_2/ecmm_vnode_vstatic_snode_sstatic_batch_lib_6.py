
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
#  Config helpers — read style_V / style_F with backward compat
# ══════════════════════════════════════════════════════════
#
# CONFIG carries two orthogonal style flags:
#
#   style_V  ('static' / 'static_no_R0' / 'dynamic' / 'staged')
#       Controls the V branch — what equation V follows and (for 'staged')
#       which staged-training schedule to run.  Old checkpoints stored this
#       under the key 'style' instead of 'style_V'; readers fall back to
#       'style' if 'style_V' is absent.
#
#   style_F  ('static' / 'dynamic')
#       Controls the F branch — whether s is algebraic (sNet, lib_3 style)
#       or integrated as a NODE (sdotNet, lib_4 style).  Old checkpoints
#       didn't have this key at all; readers default to 'dynamic' which
#       reproduces the previous lib_4 behaviour exactly.

def _style_V(config):
    """Read CONFIG['style_V'], falling back to legacy 'style' for old checkpoints."""
    return config.get('style_V', config.get('style', 'dynamic'))

def _style_F(config):
    """Read CONFIG['style_F']; default 'dynamic' = previous lib_4 behaviour."""
    return config.get('style_F', 'dynamic')


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
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 5000     # [F]

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
        x = torch.stack([soc, I_norm, u_norm], dim=-1)
        if self.config.get('k_constrained', 'false') == 'true':
            s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.k_min + s * (self.k_max - self.k_min)
        else:
            return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.03  # if softplus = 1, out = 30 [MN/mm]
    
# ══════════════════════════════════════════════════════════
#  s NETWORK (static)
# ══════════════════════════════════════════════════════


class sdotNet(nn.Module):
    """(s, soc, I_norm, u_norm) → ds/dt   [1e-5 m / s].

    Used when CONFIG['style_F'] == 'dynamic'.  Output is the *rate* of swelling,
    not the swelling itself, so the constraint pair (sdot_min, sdot_max) bounds
    ds/dt — i.e. the maximum allowable change per integration step (dt = 1 s).

    Constraint keys:
        sdot_constrained / sdot_min / sdot_max     (preferred)
        s_constrained    / s_min    / s_max        (legacy fallback for old
                                                    checkpoints that pre-date
                                                    the split — silently used
                                                    if the sdot_* keys are
                                                    absent.)
    """
    def __init__(self, config, n_hidden=32):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(4, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        # Prefer the new sdot_* keys; fall back to the legacy s_* keys so that
        # old checkpoints (which only had s_min/s_max/s_constrained, used for
        # both modes before the split) keep loading and behaving identically.
        self.sdot_min = config.get('sdot_min', config.get('s_min'))
        self.sdot_max = config.get('sdot_max', config.get('s_max'))
        if config.get('sdot_constrained', config.get('s_constrained', 'false')) == 'true':
            print(f'ds/dt constrained to [{self.sdot_min}, {self.sdot_max}] 1e-5 m / s  (dynamic sdotNet)')
        else:
            print('ds/dt unconstrained  (dynamic sdotNet)')

    def _is_constrained(self):
        return self.config.get('sdot_constrained', self.config.get('s_constrained', 'false')) == 'true'

    def forward(self, s, soc, I_norm, u_norm):
        x = torch.stack([s, soc, I_norm, u_norm], dim=-1)
        if self._is_constrained():
            sd = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.sdot_min + sd * (self.sdot_max - self.sdot_min)
        else:
            return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.0001 # if softplus = 1, out = 1e-4 [1e-5 m / s]

# ══════════════════════════════════════════════════════════
#  s NETWORK (static, algebraic — lib_3 style)
# ══════════════════════════════════════════════════════════
#
# Algebraic counterpart to sdotNet.  Used when CONFIG['style_F'] == 'static':
#
#     style_F='static'   →  s = sNet(soc, I_norm)            (no integration)
#     style_F='dynamic'  →  ds/dt = sdotNet(s, soc, I_norm, u_norm), Euler-rolled
#
# Architecture: 2 hidden layers, matching every other network in this library
# (R1Net / C1Net / R0Net / kNet / sdotNet).  Inputs (soc, I_norm) are the
# same as lib_3's sNet — u is intentionally not an input, since the algebraic
# s captures the SOC-driven swelling and u enters the force only via the
# (u - s) term in F = -k·(u - s).
#
# Constraint keys (now split from the dynamic sdotNet's keys):
#     s_constrained / s_min / s_max     — bound s itself in [1e-5 m]
# The dynamic sdotNet uses sdot_constrained / sdot_min / sdot_max instead
# (with backward-compat fallback to s_*) — see sdotNet for details.
# ══════════════════════════════════════════════════════════

class sNet(nn.Module):
    """(soc, I_norm) → s ≥ 0  [1e-5 m].  Algebraic — no integration.
    """
    def __init__(self, config, n_hidden=32):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.s_min = config.get('s_min')
        self.s_max = config.get('s_max')
        if config.get('s_constrained', 'false') == 'true':
            print(f's constrained to [{self.s_min}, {self.s_max}] 1e-5 m  (static sNet)')
        else:
            print('s unconstrained  (static sNet)')

    def forward(self, soc, I_norm):
        x = torch.stack([soc, I_norm], dim=-1)   # (..., 2)
        if self.config.get('s_constrained', 'false') == 'true':
            s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.s_min + s * (self.s_max - self.s_min)
        else:
            return nn.functional.softplus(self.net(x)).squeeze(-1)

# ══════════════════════════════════════════════════════════
#  ECMM MODEL
# ══════════════════════════════════════════════════════
# @torch.jit.script
# def _u1_integrate(I_seq: torch.Tensor,
#                   R1: torch.Tensor,
#                   C1: torch.Tensor,
#                   dt: float = 1.0) -> torch.Tensor:
#     """Semi-implicit Euler U1 integration, T iterations, B trajectories.

#     Shapes: I_seq, R1, C1 are (B, T).  Returns U1 of shape (B, T) with
#     U1[:, 0] = 0 and U1[:, n+1] computed from U1[:, n] by one
#     semi-implicit Euler step.
#     """
#     B, T = I_seq.shape
#     out = torch.zeros(B, T, dtype=I_seq.dtype, device=I_seq.device)
#     U1_n = torch.zeros(B, dtype=I_seq.dtype, device=I_seq.device)
#     for n in range(T - 1):
#         U1_n = (U1_n + dt * I_seq[:, n] / C1[:, n]) / (1.0 + dt / (R1[:, n] * C1[:, n]))
#         out[:, n + 1] = U1_n
#     return out

# def _u1_integrate(I_seq: torch.Tensor,
#                   R1: torch.Tensor,
#                   C1: torch.Tensor,
#                   dt: float = 1.0) -> torch.Tensor:
#     B, T = I_seq.shape
#     out  = torch.zeros(B, T, dtype=I_seq.dtype, device=I_seq.device)
#     U1_n = torch.zeros(B, dtype=I_seq.dtype, device=I_seq.device)
#     for n in range(T - 1):
#         U1_n = (U1_n + dt * I_seq[:, n] / C1[:, n]) / (1.0 + dt / (R1[:, n] * C1[:, n]))
#         out[:, n + 1] = U1_n
#     return out

# _u1_integrate = torch.compile(_u1_integrate,
#                               mode='reduce-overhead',
#                               fullgraph=True,
#                               dynamic=True)

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

    Two orthogonal style flags in CONFIG control the V and F branches:

      CONFIG['style_V']  ('static' | 'static_no_R0' | 'dynamic' | 'staged')
          V branch — algebraic vs RC-integrated voltage (and the staged-
          training schedule).  Legacy key 'style' is also accepted.

      CONFIG['style_F']  ('static' | 'dynamic')        default: 'dynamic'
          F branch — controls how s is produced:
            'static'   →  s = sNet(soc, I_norm)              (algebraic, lib_3 style)
            'dynamic'  →  ds/dt = sdotNet(s, soc, I_norm, u_norm), Euler-rolled
                          from s(0)=0  (lib_4 style — the previous default).
          Only one of {self.s_net, self.ds_net} is instantiated, chosen at
          __init__ time from CONFIG['style_F'].  k_net is unaffected.
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

        # ── s/ds network — switch on CONFIG['style_F'] ──
        # 'static'   : sNet(soc, I_norm) → s          (algebraic, lib_3 style)
        # 'dynamic'  : sdotNet(s, soc, I_norm, u_norm) → ds/dt, Euler-rolled
        sF = _style_F(config)
        if sF == 'static':
            self.s_net = sNet(config, n_hidden=nh)
        elif sF == 'dynamic':
            self.ds_net = sdotNet(config, n_hidden=nh)
        else:
            raise ValueError(
                f"Unknown style_F: {sF!r}.  Use 'static' (algebraic sNet) "
                f"or 'dynamic' (integrated sdotNet).")

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

    def _s(self, soc, I_norm, u_norm_exp, B, T):
        """Trajectory-shape s (B, T).

        style_F='static'   →  s = sNet(soc, I_norm)            (one-shot algebraic)
        style_F='dynamic'  →  ds/dt = sdotNet(s, soc, I_norm, u_norm), forward
                              Euler integrated from s(0)=0 with dt=1 s.

        Same return shape (B, T) in both modes, so forward() and downstream
        callers stay shape-agnostic with respect to style_F.
        """
        sF = _style_F(self.config)
        if sF == 'static':
            return self.s_net(soc, I_norm)
        elif sF == 'dynamic':
            s_steps = [torch.zeros(B)]                     # (B,) initial step
            dt = 1.0
            for n in range(T - 1):
                ds = self.ds_net(s_steps[n], soc[:, n], I_norm[:, n], u_norm_exp[:, n])  # (B,)
                s_next = s_steps[n] + ds.squeeze(-1) * dt
                s_steps.append(s_next)
            return torch.stack(s_steps, dim=1)             # (B, T)
        raise ValueError(f"Unsupported style_F: {sF!r}.")

    def _s_diag(self, soc, I_norm, u_norm):
        """Single-shot diagnostic s output for per-state plots (plot_param,
        plot_force, plot_swelling, element_predict, data_param).

        style_F='static'   →  returns s = sNet(soc, I_norm)        in [1e-5 m]
        style_F='dynamic'  →  returns ds/dt at s=0 = sdotNet(0, soc, I_norm, u_norm)

        Note the dynamic branch returns a *rate* (ds/dt at the zero-state),
        not the integrated s.  This preserves the prior lib_4 plotting
        convention (which evaluated `model.ds_net(s_ref=zeros, …)` directly).
        For the integrated s along a trajectory, call predict_np() instead.
        """
        sF = _style_F(self.config)
        if sF == 'static':
            return self.s_net(soc, I_norm)
        elif sF == 'dynamic':
            s_ref = torch.zeros_like(soc)
            return self.ds_net(s_ref, soc, I_norm, u_norm)
        raise ValueError(f"Unsupported style_F: {sF!r}.")

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


        # ––––––––––––––––––––––––––––––––––––––––
        # ── SOC integration ──
        # We want soc[:, 0] = soc0, soc[:, n] = soc0 + sum_{k<n} dsoc[k]
        # cumsum gives sum_{k≤n}; subtract dsoc[:, :1] to shift the index.
        # soc[:, 0] = soc0_batch + dsoc[:,0] - dsoc[:,0] = soc0_batch

        dsoc = -I_seq / self.Q0
        soc_fix  = soc0_batch.unsqueeze(1) + torch.cumsum(dsoc, dim=1) - dsoc[:, :1]    # (B, 1) + (B, T) - (B, 1)

        # Normalize to obtain latent inputs roughly in range [0,1]
        I_norm = I_seq / self.I_ref
        u_norm = u_batch / self.u_ref           # both negative for compression → u_norm > 0
        u_norm_exp  = u_norm.unsqueeze(1).expand(B, T)
        u_phys_exp = u_batch.unsqueeze(1).expand(B, T)

        # Parameters along the trajectory  (B, T)
                
        # ── F branch ──
        # k is always algebraic (no time integration).
        # s is algebraic (style_F='static', sNet) or integrated (style_F='dynamic', sdotNet).
        # The _s dispatcher returns a (B, T) tensor in both cases so the
        # F = -k·(u - s) computation below is shape-agnostic.
        k = self.k_net(soc_fix, I_norm, u_norm_exp)              # (B, T)
        s = self._s(soc_fix, I_norm, u_norm_exp, B, T)           # (B, T)

        Fr = - k * (u_phys_exp - s)            # GN/ 1e-5m * 1e-5m


        # ── V branch: static or dynamic U1 ──
        # with torch.no_grad():
        #     Ue = Ue_GP.soc_to_Ue(soc, return_torch=True)

        if V_mode == 'static':
            # Steady-state of the RC: U1 = I · R1.  C1 is *not* used.
            U1 = I_seq * R1
            V  = Ue - I_seq * R0 - U1
        elif V_mode == 'static_no_R0':
            U1 = I_seq * R1
            V  = Ue - U1
        elif V_mode == 'dynamic':
            U1_steps = [torch.zeros(B)]
            soc = [torch.ones(B)]
            dt = 1.0
            for n in range(T - 1):
                soc_next = soc[n] + dsoc[:, n] * dt
                soc.append(soc_next)

                R1 = self._R1(soc[n], I_norm[:, n], u_norm_exp[:, n])
                C1 = self._C1(soc[n], I_norm[:, n], u_norm_exp[:, n])
                # Semi-implicit Euler — unconditionally stable
                U1_next = (U1_steps[n] + dt * I_seq[:, n] / C1) / (1.0 + dt / (R1 * C1))
                U1_steps.append(U1_next)
            U1 = torch.stack(U1_steps, dim=1)
            soc = torch.stack(soc, dim=1)

        # ── V branch: static or dynamic U1 ──
        with torch.no_grad():
            Ue = Ue_GP.soc_to_Ue(soc, return_torch=True)

        R0 = self._R0(soc, I_norm, u_norm_exp, I_seq)
        V  = Ue - I_seq * R0 - U1

        # –––––––––––––––––––––––––––––––––

        # dsoc = -I_seq / self.Q0
        # soc_dummy  = soc0_batch.unsqueeze(1) + torch.cumsum(dsoc, dim=1) - dsoc[:, :1]    # (B, 1) + (B, T) - (B, 1)

        # # Normalize to obtain latent inputs roughly in range [0,1]
        # I_norm = I_seq / self.I_ref
        # u_norm = u_batch / self.u_ref           # both negative for compression → u_norm > 0
        # u_norm_exp  = u_norm.unsqueeze(1).expand(B, T)
        # u_phys_exp = u_batch.unsqueeze(1).expand(B, T)

        # # Parameters along the trajectory  (B, T)
        

        # # ── F branch ──
        # # k is always algebraic (no time integration).
        # # s is algebraic (style_F='static', sNet) or integrated (style_F='dynamic', sdotNet).
        # # The _s dispatcher returns a (B, T) tensor in both cases so the
        # # F = -k·(u - s) computation below is shape-agnostic.
        # k = self.k_net(soc_dummy, I_norm, u_norm_exp)              # (B, T)
        # s = self._s(soc_dummy, I_norm, u_norm_exp, B, T)           # (B, T)

        # Fr = - k * (u_phys_exp - s)            # GN/ 1e-5m * 1e-5m



        # if V_mode == 'static':
        #     # Steady-state of the RC: U1 = I · R1.  C1 is *not* used.
        #     U1 = I_seq * R1
        #     V  = Ue - I_seq * R0 - U1
        # elif V_mode == 'static_no_R0':
        #     U1 = I_seq * R1
        #     V  = Ue - U1
        # elif V_mode == 'dynamic':
  
        #     U1_steps = [torch.zeros(B)]
        #     soc = [torch.ones(B)]
        #     dt = 1.0
        #     for n in range(T - 1):
        #         soc_next = soc[n] + dsoc[:, n] * dt
        #         soc.append(soc_next)

        #         R1 = self._R1(soc[n], I_norm[:, n], u_norm_exp[:, n])
        #         C1 = self._C1(soc[n], I_norm[:, n], u_norm_exp[:, n])
        #         # print(R1.shape, C1.shape, I_seq[:, n].shape, U1_steps[n].shape)
        #         # Semi-implicit Euler — unconditionally stable
        #         U1_next = (U1_steps[n] + dt * I_seq[:, n] / C1) / (1.0 + dt / (R1 * C1))
        #         U1_steps.append(U1_next)
        #     U1 = torch.stack(U1_steps, dim=1)
        #     soc = torch.stack(soc, dim=1)

        # # ── V branch: static or dynamic U1 ──
        # with torch.no_grad():
        #     Ue = Ue_GP.soc_to_Ue(soc, return_torch=True)

        #     R0 = self._R0(soc, I_norm, u_norm_exp, I_seq)
        #     V  = Ue - I_seq * R0 - U1

        # –––––––––––––––––––––––––––––    

        # Faster than semi-implicit by about 20% ––––
        # elif V_mode == 'dynamic':
        #     C1 = self._C1(soc, I_norm, u_norm_exp)
        #     dt = 1.0
        #     tau = R1 * C1                                      # (B, T)
        #     alpha = torch.exp(-dt / tau)                       # decay factor per step
        #     drive = I_seq * R1                                 # steady-state target per step
        #     beta  = (1.0 - alpha) * drive                 # the (1-α)·I·R1 term
        #     U1_steps = [torch.zeros(B)]

        #     alpha_list = alpha.unbind(dim=1)   # tuple of T tensors of shape (B,)
        #     beta_list  = beta.unbind(dim=1)
        #     for n in range(T - 1):
        #         # U1[n+1] = U1[n]·α + I·R1·(1-α)  — EXACT for piecewise-const coefs
        #         # U1_next = U1_steps[n] * alpha[:, n] + beta[:, n]
        #         U1_next = U1_steps[n] * alpha_list[n] + beta_list[n]
        #         U1_steps.append(U1_next)
        #     U1 = torch.stack(U1_steps, dim=1)
        #     V = Ue - I_seq * R0 - U1
        # ––––

        # elif V_mode == 'dynamic':
        #     C1 = self._C1(soc, I_norm, u_norm_exp)
        #     U1 = _u1_integrate(I_seq, R1, C1, dt=1.0)
        #     V  = Ue - I_seq * R0 - U1


        return V, Fr, soc, U1, R1, s


def vmode_from_style(style):
    """Map a CONFIG['style_V'] (legacy 'style') value to the V_mode the model
    should run in for inference.  Pass `_style_V(config)` for the right key."""
    if style == 'static_no_R0':
        return 'static_no_R0'
    elif style == 'static':
        return 'static'
    elif style in ('dynamic', 'staged'):
        return 'dynamic'
    else:
        raise ValueError(f"Unknown style_V: {style!r}")


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
            C  = float(grp['C'].iloc[0]),
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
#  TRAINING — fully batched, masked-MSE for variable-length trajectories
# ══════════════════════════════════════════════════════════════
#
# Single training path: every epoch processes mini-batches of trajectories
# in parallel through BatteryECMM.forward().  Trajectories with different
# T are zero-padded *per batch* to max(T_i in batch) and a binary mask
# flags valid positions.  The loss is computed only over valid positions
# (masked MSE), so padding contributes nothing to gradients.
#
# Per-batch padding (rather than padding the whole dataset to global max-T)
# matters here because DC trajectories vary up to ~40× in length (low-C
# discharges take much longer than high-C).  Random batching + per-batch
# padding amortizes this cost — each batch only pays the local max-T price.
#
# Loss formulation: per-trajectory masked MSE, then mean across the batch.
# This treats each trajectory equally regardless of its length, matching
# the implicit weighting that per-trajectory SGD used.
#
# Padding safety for the recurrences:
#   - SOC integrates dsoc = -I/Q0; padded I=0 → soc stays constant past T_i
#   - U1 (semi-implicit Euler): with I=0 it relaxes towards 0
#   - s  (forward Euler): sdotNet may be non-zero at I=0, so s drifts in
#     padded positions, but those positions are masked out of the loss and
#     are after the valid window in time — they cannot influence any valid
#     prediction.  Wasted compute, not wrong gradients.
#
# Note: the (B,)-shape U1_steps[0]/s_steps[0] zeros inside the model are
# created with B inferred from the batched I, so per-batch B-variation is
# handled automatically.

def _empty_history():
    return {'train': [], 'train_V': [], 'train_Fr': [],
            'test': [], 'test_V': [], 'test_F': [], 'time': 0.0,
            'train_rmse': [], 'train_rmse_V': [], 'train_rmse_Fr': [],
            'test_rmse_V': [], 'test_rmse_F': []}


def _pad_collate(trajs_batch):
    """Pad a list of variable-length trajectories to a common length and stack.

    All CC trajectories ('I' scalar) are converted to per-step I sequences
    holding the constant value at every valid timestep (zero in padded
    positions).  All trajectories are zero-padded at the end to T_max =
    max(T_i in batch).  The mask is 1.0 at valid positions, 0.0 at padded.

    Returns
    -------
    I_b    : (B, T_max)  per-step current
    u_b    : (B,)        constant displacement per traj   [1e-5 m]
    soc0_b : (B,)        initial SOC per traj
    V_tg   : (B, T_max)  target voltage  (zero-padded)
    F_tg   : (B, T_max)  target force    (zero-padded)
    mask   : (B, T_max)  float, 1.0 at valid positions, 0.0 at padded
    T_max  : int         max trajectory length in this batch
    """
    B = len(trajs_batch)
    Ts = [tr['T'] for tr in trajs_batch]
    T_max = max(Ts)

    I_b  = torch.zeros(B, T_max, dtype=torch.float32)
    V_tg = torch.zeros(B, T_max, dtype=torch.float32)
    F_tg = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.float32)

    for i, tr in enumerate(trajs_batch):
        T_i = tr['T']
        if 'I_seq' in tr:
            I_b[i, :T_i] = tr['I_seq']
        else:
            I_b[i, :T_i] = float(tr['I'])
        V_tg[i, :T_i] = tr['V']
        F_tg[i, :T_i] = tr['F']
        mask[i, :T_i] = 1.0

    u_b    = torch.tensor([tr['u']    for tr in trajs_batch], dtype=torch.float32)
    soc0_b = torch.tensor([tr['soc0'] for tr in trajs_batch], dtype=torch.float32)

    return I_b, u_b, soc0_b, V_tg, F_tg, mask, T_max


def _masked_per_traj_mse(pred, target, mask):
    """Per-trajectory MSE then mean across the batch.

    pred, target, mask : (B, T_max).  Returns a 0-D tensor.

    Per-trajectory normalisation (rather than overall (B,T) mean) keeps the
    weight-per-trajectory invariant to length, which is what the original
    per-traj SGD loop did.  Each traj contributes equally to the gradient.
    """
    sq = (pred - target) ** 2 * mask                                # (B, T_max)
    n_valid = mask.sum(dim=1).clamp(min=1.0)                        # (B,)
    per_traj = sq.sum(dim=1) / n_valid                              # (B,)
    return per_traj.mean()


def train_model(model, train_trajs, test_trajs,
                n_epochs=200, lr=1e-3, print_every=10,
                V_mode='dynamic', freeze=None,
                batch_size=16, eval_every=None):
    """Fully-batched training loop with masked MSE for ragged trajectories.

    Every epoch shuffles the train-trajectory indices, splits them into
    mini-batches of `batch_size`, pads each batch to its own max-T, and
    takes one optimizer step per mini-batch.  Trajectories of any kind
    (CC or pulse, any T) can be mixed freely.

    Parameters
    ----------
    model : BatteryECMM
    train_trajs, test_trajs : list of trajectory dicts (CC or pulse).
        CC dicts carry 'I' (scalar); pulse dicts carry 'I_seq' (Tensor of
        shape (T,)).  Both forms are handled transparently.
    n_epochs : int
    lr : float
    print_every : int
        Epochs between progress prints.  The first and last epochs always
        print, regardless of this value.
    V_mode : 'dynamic' | 'static' | 'static_no_R0'
        Forwarded to BatteryECMM.forward().  'static' / 'static_no_R0' use
        the algebraic V equation (no U1 dynamics), 'dynamic' integrates U1
        with semi-implicit Euler.
    freeze : tuple of str or None
        Substring filters; any parameter whose name contains one of these
        is excluded from the optimizer.  E.g. ('R0_net', 'C1_net') for
        static_no_R0 training.
    batch_size : int
        Mini-batch size.  16-32 is reasonable.
    eval_every : int or None
        Epochs between test-set evaluations.  Defaults to `print_every`.
        On non-eval epochs the test_* history fields repeat the previous
        value so plot axes stay aligned to the epoch index.
    """
    if eval_every is None:
        eval_every = print_every

    freeze_kw = freeze if freeze is not None else ()
    params = [p for name, p in model.named_parameters()
              if not any(kw in name for kw in freeze_kw)]
    n_ = sum(p.numel() for p in params)
    print(f"  Trainable params: {n_}  ({freeze_kw} frozen)")

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PAT, factor=0.5)

    history = _empty_history()
    N = len(train_trajs)

    # Pre-pad the test set once (it doesn't change between epochs, and
    # eval_every is ≥1 so this tensor is reused many times).
    Ite, ute, s0te, Vte, Fte, Mte, T_te_max = _pad_collate(test_trajs)
    n_valid_te = Mte.sum().item()

    Ts_train = [tr['T'] for tr in train_trajs]
    print(f"  Batched training: N_train={N}, N_test={len(test_trajs)}, "
          f"batch_size={batch_size}, eval_every={eval_every}")
    print(f"  Train T range: [{min(Ts_train)}, {max(Ts_train)}]   "
          f"Test T_max (pre-padded): {T_te_max}")

    t0 = _time.time()
    alpha_F = 1.0  # voltage / force loss balance — both already in their natural units

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = np.random.permutation(N)

        ep_mse = ep_mse_V = ep_mse_Fr = 0.0
        ep_rmse = ep_rmse_V = ep_rmse_Fr = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = perm[start:start + batch_size]
            batch = [train_trajs[i] for i in batch_idx]

            I_b, u_b, s0_b, V_tg, F_tg, m_b, T_b = _pad_collate(batch)

            V_pred, Fr_pred, _, _, _, _ = model(I_b, u_b, s0_b, T=T_b, V_mode=V_mode)

            loss_V  = _masked_per_traj_mse(V_pred,  V_tg, m_b)
            loss_Fr = _masked_per_traj_mse(Fr_pred, F_tg, m_b) * alpha_F
            loss = loss_V + loss_Fr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                lv = loss_V.item()
                lf = loss_Fr.item()
                rv = float(np.sqrt(lv))
                rf = float(np.sqrt(lf / alpha_F))
            ep_mse    += lv + lf
            ep_mse_V  += lv
            ep_mse_Fr += lf
            ep_rmse   += rv + rf
            ep_rmse_V += rv
            ep_rmse_Fr += rf
            n_batches += 1

        ep_mse /= n_batches; ep_mse_V /= n_batches; ep_mse_Fr /= n_batches
        ep_rmse /= n_batches; ep_rmse_V /= n_batches; ep_rmse_Fr /= n_batches

        history['train'].append(ep_mse)
        history['train_V'].append(ep_mse_V); history['train_Fr'].append(ep_mse_Fr)
        history['train_rmse'].append(ep_rmse)
        history['train_rmse_V'].append(ep_rmse_V); history['train_rmse_Fr'].append(ep_rmse_Fr)

        # Test eval — single batched forward over the pre-padded test set.
        do_eval = (epoch % eval_every == 0) or epoch == 1 or epoch == n_epochs
        # if do_eval:
        #     model.eval()
        #     with torch.no_grad():
        #         V_pred, F_pred, _, _, _, _ = model(Ite, ute, s0te,
        #                                            T=T_te_max, V_mode=V_mode)
        #         # Per-element masked MSE for reporting (matches RMSE per data point).
        #         test_mse_V = (((V_pred - Vte) ** 2) * Mte).sum().item() / n_valid_te
        #         test_mse_F = (((F_pred - Fte) ** 2) * Mte).sum().item() / n_valid_te
        #         test_mse   = test_mse_V + test_mse_F
        #         test_rmse_V = float(np.sqrt(test_mse_V))
        #         test_rmse_F = float(np.sqrt(test_mse_F))
        #     history['test'].append(test_mse)
        #     history['test_V'].append(test_mse_V); history['test_F'].append(test_mse_F)
        #     history['test_rmse_V'].append(test_rmse_V); history['test_rmse_F'].append(test_rmse_F)
        if do_eval:
            model.eval()
            with torch.no_grad():
                V_pred, F_pred, _, _, _, _ = model(Ite, ute, s0te,
                                                T=T_te_max, V_mode=V_mode)
                # Per-trajectory masked MSE — matches train loss formulation
                test_mse_V = _masked_per_traj_mse(V_pred, Vte, Mte).item()
                test_mse_F = _masked_per_traj_mse(F_pred, Fte, Mte).item()
                test_mse   = test_mse_V + test_mse_F
                test_rmse_V = float(np.sqrt(test_mse_V))
                test_rmse_F = float(np.sqrt(test_mse_F))
            history['test'].append(test_mse)
            history['test_V'].append(test_mse_V); history['test_F'].append(test_mse_F)
            history['test_rmse_V'].append(test_rmse_V); history['test_rmse_F'].append(test_rmse_F)
        else:
            for k in ('test', 'test_V', 'test_F', 'test_rmse_V', 'test_rmse_F'):
                history[k].append(history[k][-1] if history[k] else float('nan'))

        if scheduler is not None:
            scheduler.step(ep_mse)

        if epoch % print_every == 0 or epoch == 1 or epoch == n_epochs:
            C1 = get_C1(model, scalar=True)
            eta = (_time.time() - t0) / epoch * (n_epochs - epoch) / 60
            print(f"  {epoch:4d}/{n_epochs} | ETA {eta:.1f}m "
                  f"| RMSE V {ep_rmse_V:.4f} Fr {ep_rmse_Fr:.4f} "
                  f"| test V {history['test_rmse_V'][-1]:.4f} F {history['test_rmse_F'][-1]:.4f} "
                  f"| C1={C1:.0f}F | LR {optimizer.param_groups[0]['lr']:.2e}")

    history['time'] = history.get('time', 0.0) + (_time.time() - t0) / 60
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
    config['style_V'] (legacy 'style') via vmode_from_style().  V_mode is
    honoured for both CC and pulse trajectories — no implicit override — so a
    static-trained model can be evaluated on pulses to show the algebraic V
    can't follow them.

    For V_mode in ('static', 'static_no_R0') C1 is returned as None so plots
    can omit any C1-dependent panels.

    Returns a dict with keys: V, soc, U1, R1, Fr, k, C1, R0, I  — all numpy
    arrays of length T (R0 / I are arrays in pulse mode and constant arrays
    in CC mode, so plotting code can treat them uniformly).
    """
    if V_mode is None:
        V_mode = vmode_from_style(_style_V(config))
    pulse = 'I_seq' in traj
    I_b, u_b, soc0_b, T = _traj_inputs(traj)

    V, Fr, soc, U1, R1, s = model(I_b, u_b, soc0_b, T=T, V_mode=V_mode)
    V   = V[0].numpy();   Fr = Fr[0].numpy()
    soc = soc[0].numpy(); U1 = U1[0].numpy()
    R1  = R1[0].numpy();  s  = s[0].numpy()

    # Trajectory-shape arrays for parameter evaluation
    I_np   = traj['I_seq'].numpy() if pulse else np.full(T, traj['I'])
    u_np   = np.full(T, traj['u'])
    soc_t  = torch.from_numpy(soc.astype(np.float32))
    I_norm = torch.from_numpy((I_np / model.I_ref).astype(np.float32))
    u_t    = torch.from_numpy(u_np.astype(np.float32))             # raw u [1e-5 m]
    u_norm = u_t / model.u_ref                                     # normalized — what networks see

    k = model.k_net(soc_t, I_norm, u_norm).numpy()
    R0 = model._R0(soc_t, I_norm, u_norm, I_np).numpy()            # (T,) — _R0 expects u_norm
    R1 = model._R1(soc_t, I_norm, u_norm).numpy()            # (T,) — _R0 expects u_norm

    if V_mode in ('static', 'static_no_R0'):
        C1 = None       # C1 not used when V is algebraic — don't evaluate it
    else:
        C1 = get_C1(model, scalar=False, soc=soc_t, I_norm=I_norm, u_exp=u_norm)

    return dict(V=V, soc=soc, U1=U1, R1=R1, Fr=Fr, k=k, s=s, C1=C1, R0=R0, I=I_np)


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

    When V_mode is None (default), it is derived from config['style_V']
    (legacy 'style') via vmode_from_style().
    All trajectories in `trajs` should be the same kind (all CC or all pulse).
    """
    if V_mode is None:
        V_mode = vmode_from_style(_style_V(config))
    n = min(n_show, len(trajs))
    if n == 0:
        raise ValueError("trajs is empty")
    pulse = 'I_seq' in trajs[0]

    # Determine kind from first trajectory; assume the rest are the same.
    if pulse and V_mode in ('static', 'static_no_R0'):
            # Pulse evaluated under an algebraic V — C1 is unused, omit its panel.
        rows = ['I', 'V', 'soc', 'eta', 'R', 'Fr', 'k', 's']
    elif pulse:
        rows = ['I', 'V', 'soc', 'eta', 'R', 'C1', 'Fr', 'k', 's']
    elif V_mode in ('static', 'static_no_R0'):
        rows = ['V', 'eta', 'R', 'Fr', 'k', 's']
    elif V_mode == 'dynamic':
        rows = ['V', 'eta', 'R', 'C1', 'Fr', 'k', 's']
    else:
        raise ValueError(
            f"V_mode must be 'static', 'static_no_R0' or 'dynamic', got {V_mode!r}")

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(5 * n, 3.3 * n_rows), squeeze=False)
    model.eval()

    for j in range(n):
        inds = [2, 5 , 12]
        tr = trajs[inds[j]]   # Hard code skip the first 3 trajs to show more interesting ones
        T  = tr['T']
        out = predict_np(model, config, tr, V_mode=V_mode)
        V, soc_np, U1 = out['V'], out['soc'], out['U1']
        R1, Fr, k_pred, s_pred = out['R1'], out['Fr'], out['k'], out['s']
        C1, R0, I_np = out['C1'], out['R0'], out['I']

        # x-axis: time index when time=True OR for pulse data (SOC isn't monotonic
        # under repeated charge/discharge pulses, so SOC-on-x makes no sense).
        use_time_x = time or pulse
        x = np.arange(T) if use_time_x else soc_np

        # Trajectory header — used in the title of the topmost row
        if pulse:
            traj_header = f'{title}pulse traj {j}, u={tr["u_per"]:.1f}'
        else:
            traj_header = f'{title}I={tr["I"]:.1f}, u={tr["u_per"]:.1f}'

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
                #if pulse:
                    #print(f'Crate = {max(tr['I_seq'] *3600 / Q0):.1f} | u_per = {tr['u_per']:.1f}')
                #else: 
                #    print(f'Crate = {tr["I"] *3600 / Q0:.1f} | u_per = {tr["u_per"]:.1f}')
                #ax.set_ylim([2.2, 4.4])  # zoom in on discharge or charge region
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
                # ax.plot(x, s_true, '--', color=COLORS[1], label=r'True $s$', lw=2, alpha=0.7)
                ax.plot(x, s_pred/100, '-',  color=COLORS[0], label=r'Predicted $s$', lw=2)
                ax.set_ylabel(r'$s$ [mm]'); ax.legend()

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




def plot_report(model, config, trajs, time=False, title='', n_show=3,
                     V_mode=None, inds = [0,1]):
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

    When V_mode is None (default), it is derived from config['style_V']
    (legacy 'style') via vmode_from_style().
    All trajectories in `trajs` should be the same kind (all CC or all pulse).
    """
    if V_mode is None:
        V_mode = vmode_from_style(_style_V(config))
    n = min(n_show, len(trajs))
    if n == 0:
        raise ValueError("trajs is empty")
    pulse = 'I_seq' in trajs[0]

    # Determine kind from first trajectory; assume the rest are the same.
    if pulse and V_mode in ('static', 'static_no_R0'):
            # Pulse evaluated under an algebraic V — C1 is unused, omit its panel.
        rows = ['I', 'V', 'Fr' ]
    elif pulse:
        rows = ['V', 'Fr']
    elif V_mode in ('static', 'static_no_R0'):
        rows = ['I', 'V', 'Fr' ]
    elif V_mode == 'dynamic':
        rows = ['I', 'V', 'Fr' ]
    else:
        raise ValueError(
            f"V_mode must be 'static', 'static_no_R0' or 'dynamic', got {V_mode!r}")

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(5 * n, 3.3 * n_rows), squeeze=False)
    model.eval()

    for j in range(n):
        tr = trajs[inds[j]]   # Hard code skip the first 3 trajs to show more interesting ones
        T  = tr['T']
        out = predict_np(model, config, tr, V_mode=V_mode)
        V, soc_np, U1 = out['V'], out['soc'], out['U1']
        R1, Fr, k_pred, s_pred = out['R1'], out['Fr'], out['k'], out['s']
        C1, R0, I_np = out['C1'], out['R0'], out['I']

        # x-axis: time index when time=True OR for pulse data (SOC isn't monotonic
        # under repeated charge/discharge pulses, so SOC-on-x makes no sense).
        use_time_x = time or pulse
        x = np.arange(T) if use_time_x else soc_np

        # Trajectory header — used in the title of the topmost row
        if pulse:
            traj_header = f'{title}pulse traj {j}, {max(tr['I_seq']*3600/Q0):.1f},  u={tr["u_per"]:.1f}'
        else:
            traj_header = f'{title}I={tr["I"]:.1f}, u={tr["u_per"]:.1f}'

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
                #if pulse:
                    #print(f'Crate = {max(tr['I_seq'] *3600 / Q0):.1f} | u_per = {tr['u_per']:.1f}')
                #else: 
                #    print(f'Crate = {tr["I"] *3600 / Q0:.1f} | u_per = {tr["u_per"]:.1f}')
                ax.set_ylim([2.2, 4.4])  # zoom in on discharge or charge region
                #if not pulse:        # for CC, V is the topmost row
                    #ax.set_title(traj_header)

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
                ax.plot(x, tr['F'].numpy()*1000, '--', color=COLORS[1], label=r'True $F$', lw=2)
                ax.plot(x, Fr*1000,              '-',  color=COLORS[0], label=r'Predicted $F$', lw=2)
                ax.set_ylabel(r'$F$ [MN]'); ax.legend()

            elif name == 'k':
                # Empirical stiffness directly from the data: k_true = -F/u
                k_true = -tr['F'] / tr['u'] * 1e2 # Convert u from 1e-5m to 1e-5*1e2 = mm
                k_pred = k_pred * 1e2   # convert back from GN/1e-5m to GN/mm for plotting. 1e2 GN / (1e-2*1e-3 m) = 1e2GN/mm
                # ax.plot(x, k_true, '--', color=COLORS[1], label=r'INVALID! True $k = -F/u$', lw=2, alpha=0.7)
                ax.plot(x, k_pred, '-',  color=COLORS[0], label=r'Predicted $k$', lw=2)
                ax.set_ylabel(r'$k$ [GN/mm]'); ax.legend()

            elif name == 's':
                # ax.plot(x, s_true, '--', color=COLORS[1], label=r'True $s$', lw=2, alpha=0.7)
                ax.plot(x, s_pred/100, '-',  color=COLORS[0], label=r'Predicted $s$', lw=2)
                ax.set_ylabel(r'$s$ [mm]'); ax.legend()

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
    # """Plot train/test RMSE curves on a log scale."""
    # fig, ax = plt.subplots(figsize=(7, 4.2))
    # epochs = np.arange(1, len(history['train_rmse']) + 1)

    # ax.semilogy(epochs, history['train_rmse_V'],  color=COLORS[0], lw=2,
    #             label=r'Train $V$  (final {:.4f} V)'.format(history['train_rmse_V'][-1]))
    # ax.semilogy(epochs, history['train_rmse_Fr'], color=COLORS[1], lw=2,
    #             label=r'Train $F_r$ (final {:.4f} GN)'.format(history['train_rmse_Fr'][-1]))
    # ax.semilogy(epochs, history['test_rmse_V'],   color=COLORS[2], lw=2, ls='--',
    #             label=r'Test $V$  (final {:.4f} V)'.format(history['test_rmse_V'][-1]))
    # ax.semilogy(epochs, history['test_rmse_F'],   color=COLORS[3], lw=2, ls='--',
    #             label=r'Test $F_r$  (final {:.4f} GN)'.format(history['test_rmse_F'][-1]))

    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('RMSE')
    # ax.grid(True, which='both', ls=':', color='0.8')
    # ax.legend(loc='lower left')
    # fig.tight_layout()

    fig, ax = plt.subplot_mosaic(
            [
                # ["loss", "combined"],
                ["V_loss", "combined"],
                ["F_loss", "combined"]
            ],
            figsize=(12, 5),
            sharex=True
            )

    # Individual plots
    # ax["loss"].semilogy(history["train_rmse"], label="Train loss", color=COLORS[0])
    ax["V_loss"].semilogy(history["train_rmse_V"], label=r"Train $V$ loss", color=COLORS[1])
    ax["F_loss"].semilogy(history["train_rmse_Fr"], label=r"Train $F$ loss", color=COLORS[2])

    # Combined plot (all together)
    # ax["combined"].semilogy(history["train_rmse"], label="Loss", color=COLORS[0])
    ax["combined"].semilogy(history["train_rmse_V"], label=r"$V$ loss", color=COLORS[1])
    ax["combined"].semilogy(history["test_rmse_V"], label=r"$V$ test loss", color='black', alpha=0.5, ls='--')
    ax["combined"].semilogy(history["train_rmse_Fr"], label=r"$F$ loss", color=COLORS[2])
    ax["combined"].semilogy(history["test_rmse_F"], label=r"$F$ test loss", color='black', alpha=0.5, ls='--')

    # Labels and styling
    ax["F_loss"].set_xlabel("epoch")
    ax["combined"].set_xlabel("epoch")

    for key in ["V_loss", "F_loss", "combined"]:
        ax[key].set_ylabel("RMSE")
        ax[key].legend()
        ax[key].grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout()
    return fig


# =════════════════════════════════════════════════════════════════
# RMSE CALC FOR PULSES
# =════════════════════════════════════════════════════════════════

def rmse_pulse(model, pulse_trajs):
    rmse_V = []
    rmse_F = []
    C = []
    d = []
    for tr in pulse_trajs:
        out = predict_np(model, model.config, tr)
        rmse_V.append(float(np.sqrt(np.mean((out['V'] - tr['V'].numpy())**2))))
        rmse_F.append(float(np.sqrt(np.mean((out['Fr'] - tr['F'].numpy())**2))))
        C.append(float(tr['C']))
        d.append(float(tr['u_per']))
    return np.array(rmse_V), np.array(rmse_F), np.array(C), np.array(d)


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
            
            elif param == 'tau':
                C1 = model._C1(soc, I_norm, u_norm).numpy()
                R1 = model._R1(soc, I_norm, u_norm).numpy()
                y = R1 * C1  # tau = R1*C1 in seconds
                ylabel = r'$\tau$ [s]'

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
                # _s_diag returns s [1e-5 m] in static mode and ds/dt at s=0 in
                # dynamic mode (preserving prior lib_4 plotting convention).
                y = model._s_diag(soc, I_norm, u_norm).numpy() / 100.0    # 1e-5 m – mm
                ylabel = r'$s$ [mm]'

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
            s = model._s_diag(soc, I_norm, u_norm).numpy()    # 1e-5 m  (or ds/dt at s=0 in dynamic mode)

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

            # s_net output is in [1e-5 m] in static mode, or ds/dt at s=0 in dynamic
            # mode (preserving prior lib_4 plotting convention).  _s_diag dispatches
            # on CONFIG['style_F'].
            s = model._s_diag(soc, I_norm, u_norm).numpy()  # 1e-5 m
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

    Returns (R1, C1, R0, k, s) numpy arrays of the broadcast shape, or a single
    array if `element` ('R0' / 'R1' / 'C1' / 'k' / 's') is given.

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
        u_real = u_per * L0 / 100                   # cell displacement [1e-5 m]
        u_norm = u_real / model.u_ref          # what the networks were trained on

        R1 = model._R1(soc, I_norm, u_norm).numpy()              # Ohm
        C1 = model._C1(soc, I_norm, u_norm).numpy()              # F
        R0 = model._R0(soc, I_norm, u_norm, I_real).numpy()      # Ohm   (I_seq = real I, not normalised)
        k  = model.k_net(soc, I_norm, u_norm).numpy()                         # GN/1e-5m
        s = model._s_diag(soc, I_norm, u_norm).numpy()           # [1e-5 m] (or ds/dt at s=0 in dynamic mode)

    out = {'R1': R1, 'C1': C1, 'R0': R0, 'k': k, 's': s}
    return out[element] if element is not None else (R1, C1, R0, k, s)


def data_param(model, trajs):
    """
    Return a long-form DataFrame of R0, R1, C1, and k across SOC for all given
    trajectories — one row per (trajectory, soc-sample).
    """
    model.eval()
    trajs_sorted = sorted(trajs, key=lambda tr: tr['C'])
    frames = []

    with torch.no_grad():
        for i, tr in enumerate(trajs_sorted):
            soc    = tr['soc']
            T      = tr['T']
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
            k  = model.k_net(soc, I_norm, u_norm).numpy()                         # GN/1e-5m
            
            # s = model._s_diag(soc, I_norm, u_norm).numpy()           # [1e-5 m] (or ds/dt at s=0 in dynamic mode)

            # List for dynamic s rollout
            s_ref = torch.zeros_like(soc)
            s_steps = [torch.zeros((), dtype=soc.dtype, device=soc.device)]                     # (B,) initial step
            ds_steps = []
            dt = 1.0
            for n in range(T - 1):
                # print(s_steps[n], soc[n], I_norm[n], u_norm[n])
                ds = model.ds_net(s_steps[n], soc[n], I_norm[n], u_norm[n])  # (B,)
                ds_steps.append(ds)
                s_next = s_steps[n] + ds.squeeze(-1) * dt
                s_steps.append(s_next)
            # ds at the final step, so len(ds_steps) == T
            ds_steps.append(model.ds_net(s_steps[-1], soc[-1], I_norm[-1], u_norm[-1]))



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
                's': np.array(s_steps),
                'sdot': np.array(ds_steps)}))

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

def input_map(model, pulse_trajs, rmse_scales):

    rmse_V, rmse_F, C,d = rmse_pulse(model, pulse_trajs)
    rmse_V = np.array(rmse_V) / rmse_scales['V']
    rmse_F = np.array(rmse_F) /  rmse_scales['F']

    rmse_NN = np.vstack((rmse_V, rmse_F)).T
    obs_inpt = np.vstack((C, d)).T
    rmse_SR = rmse_NN.copy()  # Placeholder for symbolic regression RMSE values, to be filled in when available.

    f, ax = plt.subplots(2, 2, figsize=(10, 6), sharex='col', sharey='col')

    for a in ax.flat:
        a.vlines(0.5,0,30, color='gray', linestyle='--')
        a.vlines(5,0,30, color='gray', linestyle='--')
        a.hlines(0,0.5,5, color='gray', linestyle='--')
        a.hlines(30,0.5,5, color='gray', linestyle='--')
        a.fill_between([0.5, 5], 0, 30, color='gray', alpha=0.1)

    ax[0,0].set_ylabel(r'$\Delta u / L_{tot}$ [a.u.]')
    ax[1,0].set_ylabel(r'$\Delta u / L_{tot}$ [a.u.]')
    ax[1,0].set_xlabel('C-rate [Ah]')
    ax[1,1].set_xlabel('C-rate [Ah]')
    for a in ax[0, :]:
        a.tick_params(labelbottom=False)  # hide top row x labels

    for a in ax[:, 1]:
        a.tick_params(labelleft=False)    # hide right column y labels

    # Scatter plots with different colormaps
    sc0 = ax[0,0].scatter(
        obs_inpt[:,0],
        obs_inpt[:,1],
        c=rmse_NN[:,1] ,
        cmap='copper',   # changed color
        label='Test'
    )

    sc1 = ax[0,1].scatter(
        obs_inpt[:,0],
        obs_inpt[:,1],
        c=rmse_SR[:,0],
        cmap='copper',  # changed color
        label='Train'
    )

    sc2 = ax[1,0].scatter(
        obs_inpt[:,0],
        obs_inpt[:,1],
        c=rmse_NN[:,1],
        cmap='copper',   # changed color
        label='Test'
    )

    sc3 = ax[1,1].scatter(
        obs_inpt[:,0],
        obs_inpt[:,1],
        c=rmse_SR[:,0],
        cmap='copper',  # changed color
        label='Train'
    )

    # Attach each colorbar to its own axis
    cb0 = plt.colorbar(sc0, ax=ax[0,0], label = 'RMSE for $F$')
    cb1 = plt.colorbar(sc1, ax=ax[0,1], label = 'RMSE for $V$')
    cb2 = plt.colorbar(sc2, ax=ax[1,0], label = 'RMSE for $F$')
    cb3 = plt.colorbar(sc3, ax=ax[1,1], label = 'RMSE for $V$')

    plt.tight_layout()
    plt.show()


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
    N_HIDDEN = ckpt['N_HIDDEN']
    EPOCHS   = ckpt.get('EPOCHS', 0)
    return history, CONFIG, N_HIDDEN, EPOCHS

def rmse_scale(df,variables = ['V','F']):
    range_val = {}
    for var in variables:
        min_val = df[var].min()
        max_val = df[var].max()
        range_val[var] = (max_val - min_val)
    return range_val
