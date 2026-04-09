"""
run_brucker_improved.py
========================
Brucker-style grey-box battery model — improved Stage 2.

WHAT EACH STAGE DOES AND WHY
─────────────────────────────

Stage 1 (static):
    U1 = R1(SOC, I, u) · I          (algebraic — no dynamics)
    V  = Ue(SOC) − I·R0 − R1·I

    → R1 is trained to make V match data at every time step.
    → Since U1_true varies with SOC (starts ~0 at SOC=1, grows to ~0.9V),
      R1 absorbs this into its SOC dependence.
    → R1 at high SOC ≈ 0 (because U1_true ≈ 0 at the start).
    → This R1 is an "effective" resistance, NOT the physical one.

Stage 2 (dynamic):
    dU1/dt = I/C1 − U1/(R1 · C1)    (RC ODE — starts at U1(0)=0)
    V  = Ue(SOC) − I·R0 − U1

    → U1 now ramps up from zero via RC dynamics.
    → R1 must become the PHYSICAL resistance (nonzero at high SOC).
    → C1 controls the ramp-up speed (time constant τ = R1·C1).
    → The meaning of R1 changes: "effective" → "physical".

HOW IS THIS DIFFERENT FROM THE FLAT MODEL?
───────────────────────────────────────────
The flat model (JH notebook) predicts U1 directly: U1 = f(SOC, I, u).
It uses soc_force_func to force U1→0 at SOC=1 (a near step function).
Within a trajectory it sees each point independently — no time concept.

Brucker Stage 1 is essentially equivalent: R1·I ≈ U1 at each point.
The R1 factorization is a mild prior (U1 ∝ I), but otherwise the same.

Brucker Stage 2 is DIFFERENT: it integrates an ODE, enforcing:
  - U1 starts at zero (physical initial condition)
  - U1 ramps up exponentially with τ = R1·C1
  - The transient is captured dynamically, not via a shape function

This matters because τ ≈ 700–5800s in your data — U1 often does NOT
reach steady state before the trajectory ends. Stage 1 fits an effective
R1 that absorbs the transient; Stage 2 separates transient from steady.

WHY DID STAGE 2 FAIL BEFORE?
─────────────────────────────
1. Batch GD: accumulated loss over ALL trajectories, one step per epoch.
   Stage 1 uses stochastic GD (step per trajectory). Inconsistent.
2. Only 10 epochs of joint training (warmup=20, total=30).
3. No learning rate adaptation.

FIX: Use stochastic GD in Stage 2 (matching Stage 1), more epochs,
     and estimate C1 from the observed transient timescale.
"""

# %%══════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from torchdiffeq import odeint
import time as _time

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()

# %%══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, '2_merged_data.txt')
Q0_INIT     = 17921.57581
TRAIN_SPLIT = 0.8

# Stage 1
S1_EPOCHS   = 1
S1_LR       = 1e-2
S1_LR_DROP  = 1e-3      # lr after epoch 50

# Stage 2
S2_EPOCHS     = 1      # was 30 — doubled
S2_C1_WARMUP  = 15      # train only C1 for these many epochs
S2_LR_C1      = 1e-2    # C1 needs a high lr (it's one scalar)
S2_LR_JOINT   = 5e-4    # slower for joint training (protect Stage 1 weights)

# Network
N_HIDDEN = 100
SOLVER   = 'euler'
RTOL     = 1e-3
ATOL     = 1e-5


# %%══════════════════════════════════════════════════════════
#  KNOWN PHYSICS
# ══════════════════════════════════════════════════════════

def R0_func(u, I):
    return u * (-0.0001887521) - 7.049519e-5 * I + 0.008446693


# %%══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
data['eta'] = -data['eta']

_soc_all = data['soc'].values
_ue_all  = data['Ue'].values
_idx     = np.argsort(_soc_all)
Ue_interp = interp1d(_soc_all[_idx], _ue_all[_idx],
                      kind='linear', fill_value='extrapolate')

n_traj = data['trajectory'].nunique()
print(f"  {len(data)} points, {n_traj} trajectories")


# %%══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES
# ══════════════════════════════════════════════════════════

trajectories = []
for tid, grp in data.sort_values(['trajectory', 't']).groupby('trajectory'):
    grp = grp.reset_index(drop=True)
    I_val  = grp['I'].iloc[0]
    u_val  = grp['u'].iloc[0]
    R0_val = R0_func(u_val, I_val)

    trajectories.append(dict(
        I       = torch.tensor([I_val], dtype=torch.float64),
        u       = torch.tensor([u_val], dtype=torch.float64),
        soc0    = torch.tensor([grp['soc'].iloc[0]], dtype=torch.float64),
        t       = torch.tensor(grp['t'].values - grp['t'].values[0],
                               dtype=torch.float64),
        V       = torch.tensor(grp['V'].values, dtype=torch.float64),
        Ue      = torch.tensor(grp['Ue'].values, dtype=torch.float64),
        soc     = torch.tensor(grp['soc'].values, dtype=torch.float64),
        R0      = R0_val,
        U1_true = torch.tensor(
            grp['Ue'].values - I_val * R0_val - grp['V'].values,
            dtype=torch.float64),
    ))

split       = int(len(trajectories) * TRAIN_SPLIT)
train_trajs = trajectories[:split]
test_trajs  = trajectories[split:]
print(f"  Train: {len(train_trajs)}  |  Test: {len(test_trajs)}")


# %%══════════════════════════════════════════════════════════
#  ESTIMATE C1 FROM DATA  (new — not in original Brucker)
# ══════════════════════════════════════════════════════════

def estimate_C1(trajs, R0_func):
    """
    Estimate C1 from the observed transient timescale.

    For constant-current RC: U1(t) = R1·I · (1 − exp(−t/τ))
    At t = τ,  U1 = 0.632 · U1_ss.
    So find where U1_true crosses 63.2% of its final value.
    Then C1 ≈ τ / (U1_ss / I)  =  τ / R1_ss.
    """
    C1_estimates = []
    for tr in trajs:
        I_val = tr['I'].item()
        U1    = tr['U1_true'].numpy()
        t     = tr['t'].numpy()

        U1_ss  = np.mean(U1[-max(20, len(U1)//20):])  # average of last ~5%
        R1_ss  = U1_ss / I_val if I_val > 0 else np.nan

        target = 0.632 * U1_ss
        idx    = np.argmax(U1 > target)
        if idx > 0 and R1_ss > 1e-6:
            tau = t[idx]
            C1_estimates.append(tau / R1_ss)

    if C1_estimates:
        C1_med = np.median(C1_estimates)
        print(f"  C1 estimate from data: {C1_med:.0f} F  "
              f"(range {np.min(C1_estimates):.0f}–{np.max(C1_estimates):.0f})")
        return C1_med
    else:
        print("  C1 estimate failed, using default 30000 F")
        return 30000.0

C1_INIT = estimate_C1(train_trajs, R0_func)


# %%══════════════════════════════════════════════════════════
#  STAGE 1: MODULES
# ══════════════════════════════════════════════════════════

class SOCFunc(nn.Module):
    """dSOC/dt = −I / (3600·C_bat).  Learnable C_bat."""
    def __init__(self, C_init=Q0_INIT / 3600):
        super().__init__()
        self.C_bat = nn.Parameter(torch.tensor([C_init], dtype=torch.float64))

    def forward(self, t, soc):
        return -1.0 / (3600.0 * self.C_bat) * self._I

    def set_current(self, I):
        self._I = I


class R1Net(nn.Module):
    """R1(SOC, I, u) → positive resistance [Ohm]."""
    def __init__(self, n_hidden=100, I_ref=25.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(n_hidden, 1, dtype=torch.float64),
        )

    def forward(self, soc, I, u):
        x = torch.stack([soc.reshape(-1),
                         I.reshape(-1) / self.I_ref,
                         u.reshape(-1)], dim=-1)
        return torch.abs(self.net(x)) / 100.0


# %%══════════════════════════════════════════════════════════
#  STAGE 1: FORWARD PASS
# ══════════════════════════════════════════════════════════

def run_static(soc_func, r1_net, tr):
    """Static forward pass:  U1 = R1·I  (no capacitor dynamics)."""
    soc_func.set_current(tr['I'])
    soc0 = tr['soc0'].reshape(1, -1)

    pred_SOC = odeint(soc_func, soc0, tr['t'],
                      method=SOLVER, rtol=RTOL, atol=ATOL).squeeze()

    T = len(tr['t'])
    pred_R1   = r1_net(pred_SOC.detach(), tr['I'].expand(T),
                       tr['u'].expand(T)).squeeze()
    pred_vRC1 = pred_R1 * tr['I']

    pred_Ue = torch.tensor(Ue_interp(pred_SOC.detach().numpy()),
                           dtype=torch.float64)
    V_pred  = pred_Ue - tr['I'] * tr['R0'] - pred_vRC1

    return V_pred, pred_SOC, pred_vRC1, pred_Ue


# %%══════════════════════════════════════════════════════════
#  STAGE 1: TRAINING
# ══════════════════════════════════════════════════════════

def soc_penalty(pred_SOC):
    """Brucker's SOC boundary penalty."""
    pen = torch.tensor(0.0, dtype=torch.float64)
    over  = pred_SOC[pred_SOC > 1]
    under = pred_SOC[pred_SOC < 0]
    if len(over) > 0:
        pen = pen + 100 * over.sum()
    if len(under) > 0:
        pen = pen - 100 * (under - 1).sum()
    return pen


print("\n" + "=" * 60)
print("  STAGE 1: Static model")
print("=" * 60)

soc_func = SOCFunc()
r1_net   = R1Net(n_hidden=N_HIDDEN)

optimizer = optim.Adam(r1_net.parameters(), lr=S1_LR)
s1_history = []
t0 = _time.time()

for epoch in range(1, S1_EPOCHS + 1):

    if epoch == 50:
        all_params = list(soc_func.parameters()) + list(r1_net.parameters())
        optimizer = optim.Adam(all_params, lr=S1_LR_DROP)

    order = np.random.permutation(len(train_trajs))
    epoch_loss = 0.0

    for idx in order:
        tr = train_trajs[idx]
        optimizer.zero_grad()

        V_pred, pred_SOC, _, _ = run_static(soc_func, r1_net, tr)
        loss = torch.sqrt(nn.functional.mse_loss(V_pred, tr['V']))
        loss = loss + soc_penalty(pred_SOC)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_trajs)
    s1_history.append(epoch_loss)

    if epoch % 20 == 0 or epoch == 1:
        eta = (_time.time() - t0) / epoch * (S1_EPOCHS - epoch) / 60
        print(f"  Epoch {epoch:4d}/{S1_EPOCHS} | RMSE {epoch_loss:.4f} "
              f"| C_bat {soc_func.C_bat.item():.2f} Ah | ETA {eta:.1f}m")

torch.save(soc_func.state_dict(), '_s1_soc.pt')
torch.save(r1_net.state_dict(),   '_s1_r1.pt')
print(f"  Stage 1 done. C_bat = {soc_func.C_bat.item():.2f} Ah")


# %%══════════════════════════════════════════════════════════
#  STAGE 1: PLOTS
# ══════════════════════════════════════════════════════════

def dU1dt_from_data(tr, dt=1.0):
    """dU1/dt from finite differences on U1_true."""
    U1 = tr['U1_true'].numpy()
    return np.gradient(U1, dt)


def dU1dt_from_rc(r1_net, complete_ode, soc, U1, tr):
    """
    dU1/dt from the RC equation:  I/C1 − U1/(R1·C1)

    Evaluates the learned dynamics at each point.
    soc, U1: numpy arrays from a forward pass.
    """
    T   = len(soc)
    I_v = tr['I']
    u_v = tr['u']
    C1  = complete_ode.C1.item()

    soc_t = torch.tensor(soc, dtype=r1_net.net[0].weight.dtype)
    I_t   = I_v.expand(T).to(soc_t.dtype)
    u_t   = u_v.expand(T).to(soc_t.dtype)

    with torch.no_grad():
        R1 = r1_net(soc_t, I_t, u_t).squeeze().numpy()

    I_val = I_v.item()
    return I_val / C1 - U1 / (R1 * C1)


def plot_results(trajs, run_func, title_prefix='', n_show=3,
                 complete_ode=None, r1_net_ref=None):
    """
    4-row plot: V, U1, R1, dU1/dt.
    If complete_ode is provided, also shows RC-computed dU1/dt.
    """
    n = min(n_show, len(trajs))
    fig, axes = plt.subplots(4, n, figsize=(5 * n, 12), squeeze=False)

    with torch.no_grad():
        for j in range(n):
            tr = trajs[j]
            V_pred, SOC_pred, vRC1_pred, Ue_pred = run_func(tr)
            soc_np = SOC_pred.numpy()
            U1_pred_np = vRC1_pred.numpy()

            # Row 0: Voltage
            ax = axes[0, j]
            ax.plot(soc_np, tr['V'].numpy(), '--', color='tab:blue',
                    label='data', lw=1.5)
            ax.plot(soc_np, V_pred.numpy(), '-', color='tab:orange',
                    label='model', lw=1.5)
            ax.set_xlabel('SOC'); ax.set_ylabel('V [V]')
            ax.set_title(f'{title_prefix}I={tr["I"].item():.1f}, '
                         f'u={tr["u"].item():.3f}')
            ax.legend(); ax.invert_xaxis()

            # Row 1: U1
            ax2 = axes[1, j]
            ax2.plot(soc_np, tr['U1_true'].numpy(), '--',
                     label='U1 true', lw=1.5)
            ax2.plot(soc_np, U1_pred_np, '-',
                     label='U1 model', lw=1.5)
            ax2.set_xlabel('SOC'); ax2.set_ylabel('U1 [V]')
            ax2.legend(); ax2.invert_xaxis()

            # Row 2: Resistances
            ax3 = axes[2, j]
            # R1_np = (vRC1_pred / tr['I']).numpy()
            R1_np = r1_net_ref(soc_pred=SOC_pred, I=tr['I'], u=tr['u']).detach().numpy()
            ax3.plot(soc_np, np.full_like(soc_np, tr['R0']),
                     '--', label='R0', lw=1.5)
            ax3.plot(soc_np, R1_np, '-', label='R1', lw=1.5)
            ax3.set_xlabel('SOC'); ax3.set_ylabel('R [Ohm]')
            ax3.legend(); ax3.invert_xaxis()

            # Row 3: dU1/dt
            ax4 = axes[3, j]
            dU1_data = dU1dt_from_data(tr)
            dU1_model = np.gradient(U1_pred_np, 1.0)
            ax4.plot(soc_np, dU1_data, '--', color='tab:blue',
                     label='dU1/dt data', lw=1.2, alpha=0.7)
            ax4.plot(soc_np, dU1_model, '-', color='tab:orange',
                     label='dU1/dt model', lw=1.2)

            # If we have the RC model, show the analytical dU1/dt too
            if complete_ode is not None and r1_net_ref is not None:
                dU1_rc = dU1dt_from_rc(
                    r1_net_ref, complete_ode,
                    soc_np, U1_pred_np, tr)
                ax4.plot(soc_np, dU1_rc, '-', color='tab:red',
                         label='dU1/dt (RC eq)', lw=1.2, alpha=0.8)

            ax4.set_xlabel('SOC'); ax4.set_ylabel('dU1/dt [V/s]')
            ax4.legend(fontsize=7); ax4.invert_xaxis()

    fig.tight_layout()
    return fig


def _run_static(tr):
    return run_static(soc_func, r1_net, tr)

fig = plot_results(train_trajs, _run_static, 'S1 Train: ')
plt.savefig('s1_train.pdf', bbox_inches='tight'); plt.show()
fig = plot_results(test_trajs, _run_static, 'S1 Test: ')
plt.savefig('s1_test.pdf', bbox_inches='tight'); plt.show()


# %%══════════════════════════════════════════════════════════
#  STAGE 2: COMPLETE ODE
# ══════════════════════════════════════════════════════════

class CompleteBatteryODE(nn.Module):
    """
    state = [vRC1, SOC]

    dSOC/dt  = −I / (3600·C_bat)
    dvRC1/dt = (1/C1) · (I − vRC1/R1)
    """
    def __init__(self, r1_net, C_bat_init, C1_init):
        super().__init__()
        self.r1_net = r1_net
        self.C_bat    = nn.Parameter(torch.tensor([C_bat_init], dtype=torch.float64))
        self.C1_param = nn.Parameter(torch.tensor([C1_init / 1e5], dtype=torch.float64))

    def set_inputs(self, I, u):
        self._I = I
        self._u = u

    @property
    def C1(self):
        """Actual C1 in Farads."""
        return torch.abs(self.C1_param) * 1e5

    def forward(self, t, state):
        vRC1 = state[0, :]
        soc  = state[1, :]

        dSOC  = -1.0 / (3600.0 * self.C_bat) * self._I
        R1    = self.r1_net(soc, self._I, self._u).squeeze()
        C1    = self.C1
        dvRC1 = (1.0 / C1) * (self._I - vRC1 / R1)

        return torch.stack([dvRC1.reshape(-1), dSOC.reshape(-1)], dim=0)


def run_complete(complete_ode, tr):
    """Dynamic forward pass: integrate the full RC ODE."""
    complete_ode.set_inputs(tr['I'], tr['u'])

    state0 = torch.zeros(2, 1, dtype=torch.float64)
    state0[1, 0] = tr['soc0']

    pred = odeint(complete_ode, state0, tr['t'],
                  method=SOLVER, rtol=RTOL, atol=ATOL)

    pred_vRC1 = pred[:, 0, 0]
    pred_SOC  = pred[:, 1, 0]
    pred_Ue   = torch.tensor(Ue_interp(pred_SOC.detach().numpy()),
                             dtype=torch.float64)
    V_pred = pred_Ue - tr['I'] * tr['R0'] - pred_vRC1

    return V_pred, pred_SOC, pred_vRC1, pred_Ue


# %%══════════════════════════════════════════════════════════
#  STAGE 2: TRAINING  (fixed: stochastic GD + more epochs)
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STAGE 2: Dynamic model (RC ODE)")
print("=" * 60)
print(f"  C1 init = {C1_INIT:.0f} F")
print(f"  Warmup: {S2_C1_WARMUP} epochs (C1 only), "
      f"then {S2_EPOCHS - S2_C1_WARMUP} epochs (joint)")

complete_ode = CompleteBatteryODE(
    r1_net     = r1_net,
    C_bat_init = soc_func.C_bat.item(),
    C1_init    = C1_INIT,
)

# ── Phase 1: C1 only ──
optimizer = optim.Adam([complete_ode.C1_param], lr=S2_LR_C1)

s2_history = {'train': [], 'test': []}
t0 = _time.time()

for epoch in range(1, S2_EPOCHS + 1):

    # ── Switch to joint training after warmup ──
    if epoch == S2_C1_WARMUP + 1:
        optimizer = optim.Adam(complete_ode.parameters(), lr=S2_LR_JOINT)
        print(f"  Epoch {epoch}: unfreezing R1Net, lr → {S2_LR_JOINT:.1e}")

    # ── STOCHASTIC GD: step per trajectory (matching Stage 1) ──
    order = np.random.permutation(len(train_trajs))
    epoch_loss = 0.0

    for idx in order:
        tr = train_trajs[idx]
        optimizer.zero_grad()

        V_pred, pred_SOC, _, _ = run_complete(complete_ode, tr)
        loss = torch.sqrt(nn.functional.mse_loss(V_pred, tr['V']))
        loss = loss + soc_penalty(pred_SOC)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_trajs)
    s2_history['train'].append(epoch_loss)

    # ── Test loss ──
    test_loss = 0.0
    with torch.no_grad():
        for tr in test_trajs:
            V_pred, _, _, _ = run_complete(complete_ode, tr)
            test_loss += torch.sqrt(
                nn.functional.mse_loss(V_pred, tr['V'])).item()
    test_loss /= max(len(test_trajs), 1)
    s2_history['test'].append(test_loss)

    if epoch % 10 == 0 or epoch == 1:
        C1_val = complete_ode.C1.item()
        eta = (_time.time() - t0) / epoch * (S2_EPOCHS - epoch) / 60
        print(f"  Epoch {epoch:3d}/{S2_EPOCHS} | "
              f"train {epoch_loss:.4f} | test {test_loss:.4f} | "
              f"C1={C1_val:.0f}F | ETA {eta:.1f}m")

print(f"\n  Stage 2 done.")
print(f"    C_bat = {complete_ode.C_bat.item():.2f} Ah")
print(f"    C1    = {complete_ode.C1.item():.0f} F")


# %%══════════════════════════════════════════════════════════
#  STAGE 2: PLOTS
# ══════════════════════════════════════════════════════════

def _run_complete(tr):
    return run_complete(complete_ode, tr)

fig = plot_results(train_trajs, _run_complete, 'S2 Train: ',
                   complete_ode=complete_ode, r1_net_ref=r1_net)
plt.savefig('s2_train.pdf', bbox_inches='tight'); plt.show()
fig = plot_results(test_trajs, _run_complete, 'S2 Test: ',
                   complete_ode=complete_ode, r1_net_ref=r1_net)
plt.savefig('s2_test.pdf', bbox_inches='tight'); plt.show()


# %%══════════════════════════════════════════════════════════
#  COMPARISON: Stage 1 vs Stage 2  (side by side)
# ══════════════════════════════════════════════════════════

def compute_dU1dt(U1, t, smooth_window=5):
    """Central finite differences of U1 w.r.t. time, with optional smoothing."""
    U1_np = U1.numpy() if hasattr(U1, 'numpy') else np.array(U1)
    t_np  = t.numpy()  if hasattr(t, 'numpy')  else np.array(t)

    # smooth U1 before differentiating to reduce noise
    if smooth_window > 1 and len(U1_np) > 2 * smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        U1_smooth = np.convolve(U1_np, kernel, mode='same')
        # fix edges
        U1_smooth[:smooth_window] = U1_np[:smooth_window]
        U1_smooth[-smooth_window:] = U1_np[-smooth_window:]
    else:
        U1_smooth = U1_np

    dU1dt = np.gradient(U1_smooth, t_np)
    return dU1dt


def compute_dU1dt_RC(U1_model, r1_net, complete_ode, tr):
    """
    dU1/dt from the RC equation: I/C1 − U1/(R1·C1).

    Evaluated along the model trajectory using the learned R1 and C1.
    This is what the ODE solver actually computes at each step.
    """
    I_val = tr['I'].item()
    T     = len(tr['t'])
    dtype = r1_net.net[0].weight.dtype

    # get SOC along trajectory (use model SOC from integration)
    _, SOC_model, _, _ = run_complete(complete_ode, tr)
    soc = SOC_model.detach()

    R1 = r1_net(soc, tr['I'].expand(T), tr['u'].expand(T)).squeeze()
    C1 = complete_ode.C1

    dU1dt = (tr['I'] / C1 - U1_model / (R1 * C1)).numpy()
    return dU1dt


def plot_comparison(trajs, soc_func, r1_net, complete_ode,
                    n_show=3, title_prefix=''):
    """
    Side-by-side comparison: static (S1) vs dynamic (S2).

    Row 0: V(SOC)      — voltage prediction
    Row 1: U1(SOC)     — overpotential
    Row 2: dU1/dt(SOC) — rate of change (data vs RC model)
    """
    n = min(n_show, len(trajs))
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 10), squeeze=False)

    with torch.no_grad():
        for j in range(n):
            tr = trajs[j]

            # Stage 1 (static)
            V_s1, SOC_s1, vRC1_s1, _ = run_static(soc_func, r1_net, tr)
            # Stage 2 (dynamic)
            V_s2, SOC_s2, vRC1_s2, _ = run_complete(complete_ode, tr)

            soc_s1 = SOC_s1.numpy()
            soc_s2 = SOC_s2.numpy()
            t_np   = tr['t'].numpy()

            # ── Row 0: Voltage ──
            ax = axes[0, j]
            ax.plot(soc_s1, tr['V'].numpy(), '--', color='tab:blue',
                    label='data', lw=1.5)
            ax.plot(soc_s1, V_s1.numpy(), '-', color='tab:orange',
                    label='S1 (static)', lw=1.2, alpha=0.8)
            ax.plot(soc_s2, V_s2.numpy(), '-', color='tab:green',
                    label='S2 (dynamic)', lw=1.5)
            ax.set_xlabel('SOC'); ax.set_ylabel('V [V]')
            ax.set_title(f'{title_prefix}I={tr["I"].item():.1f}')
            ax.legend(fontsize=8); ax.invert_xaxis()

            # ── Row 1: U1 ──
            ax2 = axes[1, j]
            ax2.plot(soc_s1, tr['U1_true'].numpy(), '--', color='tab:blue',
                     label='U1 true', lw=1.5)
            ax2.plot(soc_s1, vRC1_s1.numpy(), '-', color='tab:orange',
                     label='S1 (R1·I)', lw=1.2, alpha=0.8)
            ax2.plot(soc_s2, vRC1_s2.numpy(), '-', color='tab:green',
                     label='S2 (RC)', lw=1.5)
            ax2.set_xlabel('SOC'); ax2.set_ylabel('U1 [V]')
            ax2.legend(fontsize=8); ax2.invert_xaxis()

            # ── Row 2: dU1/dt ──
            ax3 = axes[2, j]

            # from data
            dU1dt_data = compute_dU1dt(tr['U1_true'], tr['t'])
            ax3.plot(soc_s1, dU1dt_data, '--', color='tab:blue',
                     label='data', lw=1.2, alpha=0.7)

            # from S1 (finite diff of R1·I — shows how the algebraic
            # approximation implies a derivative)
            dU1dt_s1 = compute_dU1dt(vRC1_s1, tr['t'])
            ax3.plot(soc_s1, dU1dt_s1, '-', color='tab:orange',
                     label='S1 (∆R1·I/∆t)', lw=1.0, alpha=0.6)

            # from S2 RC equation: I/C1 − U1/(R1·C1)
            dU1dt_s2 = compute_dU1dt_RC(vRC1_s2, r1_net, complete_ode, tr)
            ax3.plot(soc_s2, dU1dt_s2, '-', color='tab:green',
                     label='S2 (I/C1−U1/R1C1)', lw=1.5)

            ax3.set_xlabel('SOC'); ax3.set_ylabel('dU1/dt [V/s]')
            ax3.legend(fontsize=7); ax3.invert_xaxis()
            ax3.axhline(0, color='k', lw=0.5, alpha=0.3)

    fig.suptitle('Stage 1 (static) vs Stage 2 (dynamic)', fontsize=12)
    fig.tight_layout()
    return fig


fig = plot_comparison(test_trajs, soc_func, r1_net, complete_ode,
                      title_prefix='Test: ')
plt.savefig('comparison.pdf', bbox_inches='tight'); plt.show()


# %%══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(s1_history)
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('RMSE')
axes[0].set_title('Stage 1: Static')

axes[1].semilogy(s2_history['train'], label='train')
axes[1].semilogy(s2_history['test'],  label='test')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('RMSE')
axes[1].set_title('Stage 2: Dynamic')
axes[1].legend()
fig.tight_layout()
plt.savefig('loss_curves.pdf', bbox_inches='tight'); plt.show()


# %%══════════════════════════════════════════════════════════
#  DIAGNOSTICS  (from battery_diagnostics.py)
# ══════════════════════════════════════════════════════════

try:
    from battery_diagnostics import (
        print_summary, plot_R1_landscape,
        plot_tau_landscape, plot_U1ss_landscape,
        plot_trajectory_physics,
    )

    print_summary(r1_net, complete_ode, R0_func, test_trajs)

    I_vals = sorted(data['I'].unique().tolist())
    plot_R1_landscape(r1_net, I_values=I_vals)
    plt.savefig('R1_landscape.pdf', bbox_inches='tight'); plt.show()

    plot_tau_landscape(r1_net, complete_ode, I_values=I_vals)
    plt.savefig('tau_landscape.pdf', bbox_inches='tight'); plt.show()

    plot_trajectory_physics(r1_net, complete_ode, test_trajs, R0_func,
                            title_prefix='Test: ')
    plt.savefig('traj_physics.pdf', bbox_inches='tight'); plt.show()

except ImportError:
    print("  (battery_diagnostics.py not found — skipping diagnostics)")


# %%══════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════

MODEL_NAME = f'brucker_h{N_HIDDEN}_s1e{S1_EPOCHS}_s2e{S2_EPOCHS}'

torch.save({
    'complete_ode': complete_ode.state_dict(),
    'soc_func':     soc_func.state_dict(),
    'r1_net':       r1_net.state_dict(),
    's1_history':   s1_history,
    's2_history':   s2_history,
    'config': {
        'N_HIDDEN': N_HIDDEN, 'SOLVER': SOLVER,
        'S1_EPOCHS': S1_EPOCHS, 'S2_EPOCHS': S2_EPOCHS,
        'Q0_INIT': Q0_INIT,
        'C_bat_final': complete_ode.C_bat.item(),
        'C1_final': complete_ode.C1.item(),
        'C1_init': C1_INIT,
    },
}, f'{MODEL_NAME}.pt')

print(f"\nSaved: {MODEL_NAME}.pt")
print("Done.")