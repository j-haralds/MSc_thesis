"""
Battery Neural ODE – Voltage prediction (V only, discharge)
=============================================================

Physics (ECM-inspired decomposition):
    dSOC/dt  = −I / Q0                        (known, constant I)
    dU1/dt   = NN(U1, SOC, I, u; θ)           (learned)
    V        = Ue(SOC) − I·R0(I, u) − U1      (algebraic output)

States  z = [SOC, U1]
Inputs  (constant per trajectory): I, u

Mirrors the Brucker et al. (2022) grey-box approach but simplified:
instead of learning R1(SOC, I) inside the RC equation, we let a
small network directly output dU1/dt.

Usage
-----
Paste after your data-loading cell.  Requires:
    data      – DataFrame with columns [I, u, soc, Ue, V, F, eta, trajectory, t]
    Q0        – total charge capacity in As  (e.g. 17921.57581)
    R0_func   – known R0(u, I)
    DT        – time step (1 s)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ╔══════════════════════════════════════════════════════════╗
# ║  1. KNOWN PHYSICS                                       ║
# ╚══════════════════════════════════════════════════════════╝

class UeLookup:
    """
    Ue(SOC) via linear interpolation from data.

    Uses scipy (non-differentiable w.r.t. SOC).  This is fine because
    dSOC/dt has no learnable parameters, so no gradient needs to flow
    through Ue back to NN weights.

    ► If you later make Q0 learnable, replace this with a torch spline
      or a small fitted network so gradients can propagate.
    """
    def __init__(self, soc_arr, ue_arr):
        idx = np.argsort(soc_arr)
        self._f = interp1d(soc_arr[idx], ue_arr[idx],
                           kind='linear', fill_value='extrapolate')

    def __call__(self, soc):
        ue = self._f(soc.detach().cpu().numpy())
        return torch.tensor(ue, dtype=soc.dtype, device=soc.device)


# ╔══════════════════════════════════════════════════════════╗
# ║  2. NEURAL NETWORK  dU1/dt = NN(U1, SOC, I, u)          ║
# ╚══════════════════════════════════════════════════════════╝

class U1Net(nn.Module):
    """
    Small feedforward network:  (U1, SOC, I, u) → dU1/dt

    Design choices
    ──────────────
    • Tanh activation  → C∞ vector field → smooth ODE trajectories.
      (ReLU works too but gives a piecewise-linear field; less smooth.)
    • Small-weight init → dU1/dt ≈ 0 at epoch 0, so the model starts
      at V ≈ Ue − I·R0  (the static ECM) and gradually learns the
      dynamic correction.

    ► To add depth later:  insert more (Linear + Tanh) pairs.
    ► To add force channel: widen the output to 2 and return [dU1, dU2]
      where U2 feeds into a force algebraic equation.
    """
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        # Start near zero so initial prediction ≈ static ECM
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, U1, SOC, I, u):
        """All inputs: (B, 1) → output: (B, 1)"""
        x = torch.cat([U1, SOC, I, u], dim=-1)
        return self.net(x)


# ╔══════════════════════════════════════════════════════════╗
# ║  3. ODE RIGHT-HAND SIDE                                  ║
# ╚══════════════════════════════════════════════════════════╝

class BatteryODE(nn.Module):
    """
    dz/dt   for   z = [SOC, U1]

    Constant inputs I, u are set once per trajectory via .set_inputs()
    before calling the integrator — same pattern as Brucker's func0.i.

    ► To add a force state later, extend z to [SOC, U1, U_F] and
      append a third row to dz/dt.
    """
    def __init__(self, u1_net, Q0):
        super().__init__()
        self.u1_net = u1_net
        self.Q0     = torch.tensor(Q0, dtype=torch.float32)
        self._I = None      # set per trajectory
        self._u = None

    def set_inputs(self, I, u):
        """Call before each integration. I, u: (B, 1)."""
        self._I = I
        self._u = u

    def forward(self, t, z):
        """
        t : scalar   (current time — required by odeint signature)
        z : (B, 2)   →  [SOC, U1]
        """
        SOC = z[:, 0:1]
        U1  = z[:, 1:2]

        # ── physics ──
        dSOC = -self._I / self.Q0

        # ── learned ──
        dU1 = self.u1_net(U1, SOC, self._I, self._u)

        return torch.cat([dSOC, dU1], dim=-1)      # (B, 2)


# ╔══════════════════════════════════════════════════════════╗
# ║  4. INTEGRATORS                                          ║
# ╚══════════════════════════════════════════════════════════╝

def euler_integrate(func, z0, t_eval, **_ignored):
    """
    Forward Euler.  Same call signature as torchdiffeq.odeint,
    so you can swap freely:

        z = euler_integrate(ode, z0, t)   # simple, transparent
        z = odeint(ode, z0, t)            # adaptive, accurate

    ► For higher order: copy this and add an RK4 stage.
    """
    zs = [z0]
    z  = z0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        z  = z + dt * func(t_eval[i], z)
        zs.append(z)
    return torch.stack(zs)          # (T, B, 2)


def rk4_integrate(func, z0, t_eval, **_ignored):
    """Classical RK4.  Same signature as above."""
    zs = [z0]
    z  = z0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        ti = t_eval[i]
        k1 = func(ti,          z)
        k2 = func(ti + dt/2,   z + dt/2 * k1)
        k3 = func(ti + dt/2,   z + dt/2 * k2)
        k4 = func(ti + dt,     z + dt   * k3)
        z  = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        zs.append(z)
    return torch.stack(zs)


# ╔══════════════════════════════════════════════════════════╗
# ║  5. FULL MODEL: INTEGRATE + ALGEBRAIC OUTPUT             ║
# ╚══════════════════════════════════════════════════════════╝

class BatteryModel(nn.Module):
    """
    Wires together:  ODE integration  →  algebraic output V.

    Parameters
    ----------
    ode_func   : BatteryODE instance
    Ue_lookup  : UeLookup instance
    R0_func    : callable(u, I) → R0
    integrator : 'euler' | 'rk4' | any torchdiffeq method string ('dopri5', …)
    """
    def __init__(self, ode_func, Ue_lookup, R0_func, integrator='dopri5'):
        super().__init__()
        self.ode        = ode_func
        self.Ue         = Ue_lookup
        self.R0         = R0_func
        self.integrator = integrator

    def forward(self, I, u, soc0, t_eval):
        """
        I, u, soc0 : (B, 1)   — constant per trajectory
        t_eval     : (T,)     — time grid (shared, padded to longest)

        Returns
        -------
        V_pred  : (B, T)
        SOC_pred: (B, T)
        U1_pred : (B, T)
        """
        B = I.shape[0]
        self.ode.set_inputs(I, u)

        # Initial state:  [SOC₀,  U1₀ = 0]
        z0 = torch.cat([soc0, torch.zeros(B, 1, dtype=torch.float32)], dim=-1)     # (B, 2)

        # ── Integrate ──
        if self.integrator == 'euler':
            z = euler_integrate(self.ode, z0, t_eval)
        elif self.integrator == 'rk4':
            z = rk4_integrate(self.ode, z0, t_eval)
        else:
            from torchdiffeq import odeint
            z = odeint(self.ode, z0, t_eval,
                       method=self.integrator, rtol=1e-5, atol=1e-7)

        SOC = z[:, :, 0]                                       # (T, B)
        U1  = z[:, :, 1]

        # ── Algebraic output ──
        Ue = self.Ue(SOC)                                       # (T, B)
        R0 = self.R0(u.squeeze(-1), I.squeeze(-1))              # (B,)
        V  = Ue - I.squeeze(-1) * R0 - U1                       # (T, B)

        return V.T, SOC.T, U1.T                                 # all (B, T)


# ╔══════════════════════════════════════════════════════════╗
# ║  6. DATA EXTRACTION  (from your DataFrame)               ║
# ╚══════════════════════════════════════════════════════════╝

def prepare_trajectories(data):
    """
    Convert the DataFrame into a list of per-trajectory dicts,
    each containing the tensors needed for training.

    Returns
    -------
    trajs : list of dict with keys
        'I', 'u', 'soc0'  : (1, 1) tensors
        't'                : (T_i,) tensor
        'V', 'eta'         : (T_i,) tensors   (targets)
    """
    trajs = []
    for _, grp in data.sort_values(['trajectory', 't']).groupby('trajectory'):
        grp = grp.reset_index(drop=True)
        trajs.append(dict(
            I    = torch.tensor([[grp['I'].iloc[0]]], dtype=torch.float32),
            u    = torch.tensor([[grp['u'].iloc[0]]], dtype=torch.float32),
            soc0 = torch.tensor([[grp['soc'].iloc[0]]], dtype=torch.float32),
            t    = torch.arange(len(grp), dtype=torch.float32),
            V    = torch.tensor(grp['V'].values, dtype=torch.float32),
            eta  = torch.tensor(grp['eta'].values, dtype=torch.float32),
        ))
    return trajs


# ╔══════════════════════════════════════════════════════════╗
# ║  7. TRAINING LOOP                                        ║
# ╚══════════════════════════════════════════════════════════╝

def train(model, train_trajs, test_trajs,
          n_epochs=300, lr=1e-3, print_every=20):
    """
    Train loop iterating over individual trajectories (à la Brucker).

    Each trajectory is a separate forward pass through the ODE solver,
    since they have different lengths and different constant inputs.
    Gradients accumulate over all training trajectories before a step
    (full-batch per epoch).

    ► For stochastic updates (like Brucker), move optimizer.step()
      inside the inner loop.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5
    )

    history = {'train': [], 'test': []}

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        optimizer.zero_grad()

        train_loss = 0.0
        order = np.random.permutation(len(train_trajs))

        for idx in order:
            tr = train_trajs[idx]
            V_pred, _, _ = model(tr['I'], tr['u'], tr['soc0'], tr['t'])
            loss = torch.mean((V_pred.squeeze(0) - tr['V']) ** 2)
            loss.backward()                     # accumulate gradients
            train_loss += loss.item()

        optimizer.step()                         # one step per epoch
        train_loss /= len(train_trajs)
        history['train'].append(train_loss)

        # # ── Test ──
        # model.eval()
        # test_loss = 0.0
        # with torch.no_grad():
        #     for tr in test_trajs:
        #         V_pred, _, _ = model(tr['I'], tr['u'], tr['soc0'], tr['t'])
        #         loss = torch.mean((V_pred.squeeze(0) - tr['V']) ** 2)
        #         test_loss += loss.item()
        # test_loss /= max(len(test_trajs), 1)
        # history['test'].append(test_loss)
        test_loss = 0

        scheduler.step(train_loss)

        if epoch % print_every == 0 or epoch == 1:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | train MSE {train_loss:.6f} "
                  f"| test MSE {test_loss:.6f} | lr {cur_lr:.1e}")

    return history


# ╔══════════════════════════════════════════════════════════╗
# ║  8. PLOTTING                                             ║
# ╚══════════════════════════════════════════════════════════╝

def plot_predictions(model, trajs, title_prefix=''):
    """Plot V(t) predictions vs data for a list of trajectories."""
    n = len(trajs)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)

    model.eval()
    with torch.no_grad():
        for j, tr in enumerate(trajs):
            V_pred, SOC_pred, U1_pred = model(
                tr['I'], tr['u'], tr['soc0'], tr['t']
            )
            t_np   = tr['t'].numpy()
            V_p    = V_pred.squeeze(0).numpy()
            V_d    = tr['V'].numpy()
            U1_np  = U1_pred.squeeze(0).numpy()
            SOC_np = SOC_pred.squeeze(0).numpy()

            I_val = tr['I'].item()
            u_val = tr['u'].item()

            # ── Voltage ──
            ax = axes[0, j]
            ax.plot(t_np, V_d, '--', label='data',    linewidth=1.5)
            ax.plot(t_np, V_p, '-',  label='NODE',    linewidth=1.5)
            ax.set_xlabel('t  [s]')
            ax.set_ylabel('V  [V]')
            ax.set_title(f'{title_prefix}I={I_val:.1f}, u={u_val:.3f}')
            ax.legend()

            # ── Learned U1 ──
            ax2 = axes[1, j]
            ax2.plot(t_np, U1_np, label='U1 (learned)')
            ax2.set_xlabel('t  [s]')
            ax2.set_ylabel('U1  [V]')
            ax2.legend()

    fig.tight_layout()
    return fig


def plot_history(history):
    """Plot training / test loss curves."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history['train'], label='train')
    ax.semilogy(history['test'],  label='test')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    fig.tight_layout()
    return
