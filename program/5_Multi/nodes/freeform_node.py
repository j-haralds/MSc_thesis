# %% ══════════════════════════════════════════════════════════
#  FREE-FORM NODE — for comparison with ECM lite
# ══════════════════════════════════════════════════════════════
#
#  Same structure as battery_ecm_lite.py but with:
#
#    ECM lite:   dU1/dt = I/C1 − U1/(R1_net(SOC,I,u) · C1)
#    Free-form:  dU1/dt = NN(U1, SOC, I, u)
#
#  Everything else identical:
#    - Same Euler integration (no odeint)
#    - Same training loop (stochastic GD, RMSE on V)
#    - Same plotting (V, U1, dU1/dt)
#    - Same hidden size (32)
#    - Same data, same split
#
#  The comparison answers: how much does the RC physics prior help?
#
#  What the free-form NN must learn from scratch:
#    - That U1 should decay (the −U1/τ term)
#    - That U1 should grow with I (the I/C1 term)
#    - That τ depends on SOC (large at low SOC)
#    - The overall scale of dU1/dt (~1e-4 V/s)
#
#  What ECM lite gets for free from the RC equation:
#    - All of the above. The NN only learns how R1 varies with SOC.

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as _time

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()

# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION  (match ECM lite exactly)
# ══════════════════════════════════════════════════════════════

DATA_DIR    = os.path.abspath(os.path.join(FILE_PATH, '..', 'Multi_data'))
DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/2_merged_data.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
os.makedirs(FIGS_DIR, exist_ok=True)

Q0          = 17921.57581
TRAIN_SPLIT = 0.8
N_HIDDEN    = 32
EPOCHS      = 50          # no warmup/dynamic split — just one phase
LR          = 1e-3

# %% ══════════════════════════════════════════════════════════
#  KNOWN PHYSICS  (same as ECM lite)
# ══════════════════════════════════════════════════════════════

def R0_func(u, I):
    return u * (-0.0001887521) - 7.049519e-5 * I + 0.008446693

# %% ══════════════════════════════════════════════════════════
#  dU1/dt NETWORK  (replaces R1Net + RC equation)
# ══════════════════════════════════════════════════════════════

class dU1Net(nn.Module):
    """
    (U1, SOC, I, u) → dU1/dt     (unconstrained scalar)

    This is the free-form equivalent of:
        ECM lite:  dU1/dt = I/C1 − U1/(R1_net · C1)

    4 inputs (vs 3 for R1Net) because U1 is now an input —
    the network must learn the feedback (decay) term itself.

    Small-weight init → dU1/dt ≈ 0 at epoch 0 → U1 stays near 0
    → V ≈ Ue − I·R0  (static ECM baseline).
    """
    def __init__(self, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(4, n_hidden),     # 4 inputs: U1, SOC, I, u
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        # small init so dU1/dt ≈ 0 initially
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, U1, soc, I_norm, u):
        """All inputs (T,). Returns dU1/dt (T,)."""
        x = torch.stack([U1.reshape(-1),
                         soc.reshape(-1),
                         I_norm.reshape(-1),
                         u.reshape(-1)], dim=-1)
        return self.net(x).squeeze(-1)

# %% ══════════════════════════════════════════════════════════
#  FREE-FORM MODEL  (same Euler loop, NN replaces RC equation)
# ══════════════════════════════════════════════════════════════

class BatteryNODE(nn.Module):
    """
    Free-form NODE:
        1. SOC = SOC0 − I·t/Q0                       (analytical)
        2. dU1/dt = NN(U1, SOC, I, u)                 (learned — no RC prior)
        3. U1[n+1] = U1[n] + dU1/dt · dt              (Euler, dt=1)
        4. V = Ue(SOC) − I·R0 − U1                    (algebraic)

    No C1, no R1. The NN must learn the entire dynamics.
    """
    def __init__(self, du1_net, Ue_interp, R0_func, Q0):
        super().__init__()
        self.du1_net   = du1_net
        self.Ue_interp = Ue_interp
        self.R0_func   = R0_func
        self.Q0        = Q0

    def forward(self, I_val, u_val, soc0_val, T):
        I_t = torch.tensor(I_val, dtype=torch.float32)
        R0  = self.R0_func(u_val, I_val)

        # SOC: analytical
        t_idx = torch.arange(T, dtype=torch.float32)
        soc   = soc0_val - I_val / self.Q0 * t_idx

        # Euler integrate U1
        I_norm = torch.full((T,), I_val / self.du1_net.I_ref)
        u_t    = torch.full((T,), u_val)

        U1_list  = [torch.zeros(1)]
        dU1_list = []

        for n in range(T - 1):
            dU1 = self.du1_net(U1_list[n], soc[n:n+1],
                               I_norm[n:n+1], u_t[n:n+1])
            dU1_list.append(dU1)
            U1_list.append(U1_list[n] + dU1)     # dt = 1

        # last step dU1 (for plotting)
        dU1_last = self.du1_net(U1_list[-1], soc[-1:],
                                I_norm[-1:], u_t[-1:])
        dU1_list.append(dU1_last)

        U1   = torch.cat(U1_list)
        dU1  = torch.cat(dU1_list)

        # Ue from interpolation
        with torch.no_grad():
            Ue = torch.tensor(
                self.Ue_interp(soc.detach().numpy()),
                dtype=torch.float32)

        V = Ue - I_t * R0 - U1
        return V, soc, U1, dU1

# %% ══════════════════════════════════════════════════════════
#  DATA FUNCTIONS  (identical to ECM lite)
# ══════════════════════════════════════════════════════════════

def prepare_data(data, R0_func):
    trajs = []
    for _, grp in data.sort_values(['trajectory', 't']).groupby('trajectory'):
        grp = grp.reset_index(drop=True)
        I_val, u_val = float(grp['I'].iloc[0]), float(grp['u'].iloc[0])
        R0_val = R0_func(u_val, I_val)
        trajs.append(dict(
            I=I_val, u=u_val, soc0=float(grp['soc'].iloc[0]), T=len(grp),
            V=torch.tensor(grp['V'].values, dtype=torch.float32),
            soc=torch.tensor(grp['soc'].values, dtype=torch.float32),
            U1_true=torch.tensor(
                grp['Ue'].values - I_val * R0_val - grp['V'].values,
                dtype=torch.float32),
        ))
    return trajs

# %% ══════════════════════════════════════════════════════════
#  TRAINING  (same stochastic GD as ECM lite, no warmup)
# ══════════════════════════════════════════════════════════════

def train_model(model, train_trajs, test_trajs,
                n_epochs=200, lr=1e-3, print_every=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=40, factor=0.5)

    history = {'train': [], 'test': []}
    t0 = _time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_trajs))
        epoch_loss = 0.0

        for idx in order:
            tr = train_trajs[idx]
            optimizer.zero_grad()
            V_pred, _, _, _ = model(tr['I'], tr['u'], tr['soc0'], tr['T'])
            loss = torch.sqrt(torch.mean((V_pred - tr['V']) ** 2))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_trajs)
        history['train'].append(epoch_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for tr in test_trajs:
                V_pred, _, _, _ = model(
                    tr['I'], tr['u'], tr['soc0'], tr['T'])
                test_loss += torch.sqrt(
                    torch.mean((V_pred - tr['V']) ** 2)).item()
        test_loss /= max(len(test_trajs), 1)
        history['test'].append(test_loss)
        scheduler.step(epoch_loss)

        if epoch % print_every == 0 or epoch == 1:
            eta = (_time.time() - t0) / epoch * (n_epochs - epoch) / 60
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  {epoch:4d}/{n_epochs} | train {epoch_loss:.4f} "
                  f"| test {test_loss:.4f} "
                  f"| lr {lr_now:.1e} | ETA {eta:.1f}m")

    return history

# %% ══════════════════════════════════════════════════════════
#  PLOTTING  (same layout as ECM lite, row 2 = dU1/dt from NN)
# ══════════════════════════════════════════════════════════════

def plot_predictions(model, trajs, title='', n_show=3):
    n = min(n_show, len(trajs))
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 9), squeeze=False)

    model.eval()
    with torch.no_grad():
        for j in range(n):
            tr = trajs[j]
            V, soc, U1, dU1 = model(
                tr['I'], tr['u'], tr['soc0'], tr['T'])
            soc_np = soc.numpy()

            # Row 0: V
            axes[0, j].plot(soc_np, tr['V'].numpy(), '--', color=COLORS[1], label=r'True $V$', lw=2)
            axes[0, j].plot(soc_np, V.numpy(), '-', color=COLORS[0], label=r'Predicted $V$', lw=2)
            axes[0, j].set_ylabel(r'$V$ [V]'); axes[0, j].legend()
            axes[0, j].invert_xaxis()
            axes[0, j].set_title(
                f'{title}I={tr["I"]:.1f}, u={tr["u"]:.3f}')

            # Row 1: U1
            axes[1, j].plot(soc_np, tr['U1_true'].numpy(), '--', color=COLORS[1], label=r'True $U_1$', lw=2)
            axes[1, j].plot(soc_np, U1.numpy(), '-', color=COLORS[0], label=r'Predicted $U_1$', lw=2)
            axes[1, j].set_ylabel(r'$U_1$ [V]'); axes[1, j].legend()
            axes[1, j].invert_xaxis()

            # Row 2: dU1/dt  (NN output directly, no RC equation)
            dU1_data = np.gradient(tr['U1_true'].numpy(), 1.0)
            axes[2, j].plot(soc_np, dU1_data, '--', color=COLORS[1], label=r'True $dU_1/dt$', lw=2, alpha=0.7)
            axes[2, j].plot(soc_np, dU1.numpy(), '-', color=COLORS[0], label=r'NN $dU_1/dt$', lw=2)
            axes[2, j].axhline(0, color='k', lw=0.5, alpha=0.3)
            axes[2, j].set_ylabel('dU1/dt [V/s]')
            axes[2, j].set_xlabel('SOC')
            axes[2, j].legend(fontsize=8); axes[2, j].invert_xaxis()

    fig.tight_layout()
    return fig

# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
print(data.columns)
data['eta'] = -data['eta']
I_MAX = data['I'].max()

_s, _u = data['soc'].values, data['Ue'].values
_i = np.argsort(_s)
Ue_interp = interp1d(_s[_i], _u[_i], kind='linear',
                      fill_value='extrapolate')

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")

# %% ══════════════════════════════════════════════════════════
#  PREPARE DATA
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data, R0_func)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")

# %% ══════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════

du1_net = dU1Net(n_hidden=N_HIDDEN, I_ref=I_MAX)
model   = BatteryNODE(du1_net, Ue_interp, R0_func, Q0)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")
print(f"  (ECM lite with same hidden size has ~130 params + C1)")

# %% ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

print(f"\nTraining ({EPOCHS} epochs)...")
history = train_model(model, train_trajs, test_trajs,
                      n_epochs=EPOCHS, lr=LR, print_every=20)

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVE
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(history['train'], color=COLORS[0], label='train')
ax.semilogy(history['test'],  color=COLORS[1], label='test')
ax.set_xlabel('epoch'); ax.set_ylabel('RMSE'); ax.legend()
ax.set_title('Free-form NODE: dU1/dt = NN(U1, SOC, I, u)')
fig.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'freeform_loss.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TRAIN
# ══════════════════════════════════════════════════════════════

plot_predictions(model, train_trajs, 'Train: ')
# plt.savefig(os.path.join(FIGS_DIR, 'freeform_train.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

plot_predictions(model, test_trajs, 'Test: ')
plt.savefig(os.path.join(FIGS_DIR, 'freeform_test.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

torch.save({
    'model': model.state_dict(),
    'history': history,
    'N_HIDDEN': N_HIDDEN,
    'EPOCHS': EPOCHS,
}, os.path.join(FILE_PATH, f'freeform_node_{N_HIDDEN}h_{EPOCHS}eps.pt'))

print(f"Saved: freeform_node.pt")
# %%
