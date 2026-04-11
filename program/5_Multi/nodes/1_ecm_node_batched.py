# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM NODE — BATCHED VERSION
# ══════════════════════════════════════════════════════════════
#
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
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as _time

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# FILE_PATH = os.getcwd()
print(FILE_PATH)
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()

# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, '2_merged_data.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
Q0          = 17921.57581
TRAIN_SPLIT = 0.8
N_HIDDEN    = 32
EPOCHS      = 20
LR          = 1e-3
BATCH_SIZE  = 1        # ← NEW: trajectories per batch

# %% ══════════════════════════════════════════════════════════
#  KNOWN PHYSICS
# ══════════════════════════════════════════════════════════════

def R0_func(u, I):
    return u * (-0.0001887521) - 7.049519e-5 * I + 0.008446693

# %% ══════════════════════════════════════════════════════════
#  R1 NETWORK  (unchanged — already handles arbitrary batch dims)
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
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def forward(self, soc, I_norm, u):
        # Works for any shape — just needs matching last dims
        x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.01 + 1e-5

# %% ══════════════════════════════════════════════════════════
#  BATCHED ECM MODEL
# ══════════════════════════════════════════════════════════════

class BatteryECM(nn.Module):
    """
    Batched ECM:  forward() now accepts tensors of shape (B,) for
    scalar-per-trajectory quantities and returns (B, T_max) tensors.

    The Euler loop steps ALL trajectories simultaneously at each
    timestep — no per-trajectory Python loop.
    """
    def __init__(self, r1_net, Ue_interp, R0_func, Q0, C1_init=30000.0):
        super().__init__()
        self.r1_net    = r1_net
        self.Ue_interp = Ue_interp
        self.R0_func   = R0_func
        self.Q0        = Q0
        # self.log_C1    = nn.Parameter(torch.tensor(np.log(C1_init), dtype=torch.float32))
        # self.log_C1    = nn.Parameter(torch.tensor(np.log(C1_init - 8000.0), dtype=torch.float32))
        self.log_C1    = nn.Parameter(torch.tensor(np.log(C1_init), dtype=torch.float32))

    @property
    def C1(self):
        # return torch.exp(self.log_C1)
        # return torch.exp(self.log_C1) + 8000.0      # Ensure C1 > 1000 F for stability
        return torch.exp(self.log_C1)

    def forward(self, I_batch, u_batch, soc0_batch, T_max):
        """
        Parameters
        ----------
        I_batch    : (B,) tensor — current per trajectory
        u_batch    : (B,) tensor — aging factor per trajectory
        soc0_batch : (B,) tensor — initial SOC per trajectory
        T_max      : int — padded sequence length

        Returns
        -------
        V    : (B, T_max)
        soc  : (B, T_max)
        U1   : (B, T_max)
        R1   : (B, T_max)
        """
        B  = I_batch.shape[0]
        C1 = self.C1

        # SOC: analytical  (B, T_max)
        t_idx = torch.arange(T_max, dtype=torch.float32).unsqueeze(0)  # (1, T)
        soc = soc0_batch.unsqueeze(1) - I_batch.unsqueeze(1) / self.Q0 * t_idx

        # R1 at every (batch, time) point  →  (B, T_max)
        I_norm = (I_batch / self.r1_net.I_ref).unsqueeze(1).expand(B, T_max)
        u_exp  = u_batch.unsqueeze(1).expand(B, T_max)
        R1 = self.r1_net(soc, I_norm, u_exp)   # (B, T_max)

        # Vectorised Euler integration of U1 across the batch
        # Each timestep: U1[n+1] = U1[n] + I/C1 − U1[n]/(R1[n]·C1)
        U1_steps = [torch.zeros(B)]
        for n in range(T_max - 1):
            dU1 = I_batch / C1 - U1_steps[n] / (R1[:, n] * C1)
            U1_steps.append(U1_steps[n] + dU1)   # dt = 1 s
        U1 = torch.stack(U1_steps, dim=1)         # (B, T_max)

        # Ue from interpolation  (B, T_max)
        with torch.no_grad():
            soc_np = soc.detach().numpy()
            Ue = torch.tensor(
                self.Ue_interp(soc_np), dtype=torch.float32)

        # R0  (B,)  →  (B, 1) for broadcasting
        R0 = torch.tensor(
            [self.R0_func(u_batch[b].item(), I_batch[b].item())
             for b in range(B)],
            dtype=torch.float32).unsqueeze(1)

        V = Ue - I_batch.unsqueeze(1) * R0 - U1
        return V, soc, U1, R1

    # Keep single-trajectory forward for inference / plotting
    def forward_single(self, I_val, u_val, soc0_val, T):
        """Convenience wrapper matching the original forward() signature."""
        I_b    = torch.tensor([I_val], dtype=torch.float32)
        u_b    = torch.tensor([u_val], dtype=torch.float32)
        soc0_b = torch.tensor([soc0_val], dtype=torch.float32)
        V, soc, U1, R1 = self.forward(I_b, u_b, soc0_b, T)
        return V[0], soc[0], U1[0], R1[0]

# %% ══════════════════════════════════════════════════════════
#  DATA FUNCTIONS
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

# %% ══════════════════════════════════════════════════════════
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
    mask       : (B, T_max)    True where data exists
    T_max      : int
    """
    B = len(trajs)
    T_max = max(tr['T'] for tr in trajs)

    I_batch    = torch.tensor([tr['I']    for tr in trajs], dtype=torch.float32)
    u_batch    = torch.tensor([tr['u']    for tr in trajs], dtype=torch.float32)
    soc0_batch = torch.tensor([tr['soc0'] for tr in trajs], dtype=torch.float32)

    V_batch = torch.zeros(B, T_max)
    mask    = torch.zeros(B, T_max, dtype=torch.bool)

    for i, tr in enumerate(trajs):
        T = tr['T']
        V_batch[i, :T] = tr['V']
        mask[i, :T]    = True

    return I_batch, u_batch, soc0_batch, V_batch, mask, T_max

# %% ══════════════════════════════════════════════════════════
#  TRAINING FUNCTION  (batched)
# ══════════════════════════════════════════════════════════════

def train_model(model, train_trajs, test_trajs,
                n_epochs=200, lr=1e-3, batch_size=16, print_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=40, factor=0.5)

    history = {'train': [], 'test': [], 'time': []}
    t0 = _time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_trajs))
        epoch_loss = 0.0
        n_batches  = 0

        # ── mini-batch loop ──
        for start in range(0, len(train_trajs), batch_size):
            idxs  = order[start : start + batch_size]
            batch = [train_trajs[i] for i in idxs]

            I_b, u_b, soc0_b, V_true, mask, T_max = collate_batch(batch)

            optimizer.zero_grad()
            V_pred, _, _, _ = model(I_b, u_b, soc0_b, T_max)

            # Masked RMSE — only score real (non-padded) timesteps
            sq_err = (V_pred - V_true) ** 2
            loss = torch.sqrt(sq_err[mask].mean())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= n_batches
        history['train'].append(epoch_loss)


        # # ── test loss (one big batch) ──
        # model.eval()
        # with torch.no_grad():
        #     I_b, u_b, soc0_b, V_true, mask, T_max = collate_batch(test_trajs)
        #     V_pred, _, _, _ = model(I_b, u_b, soc0_b, T_max)
        #     test_loss = torch.sqrt(((V_pred - V_true)**2)[mask].mean()).item()
        # history['test'].append(test_loss)
        # scheduler.step(epoch_loss)

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for tr in test_trajs:
                I_b    = torch.tensor([tr['I']],    dtype=torch.float32)
                u_b    = torch.tensor([tr['u']],    dtype=torch.float32)
                soc0_b = torch.tensor([tr['soc0']], dtype=torch.float32)
                V_pred, _, _, _ = model(I_b, u_b, soc0_b, tr['T'])
                test_loss += torch.sqrt(torch.mean((V_pred[0] - tr['V'])**2)).item()
            test_loss /= len(test_trajs)

        if epoch % print_every == 0 or epoch == 1:
            C1 = model.C1.item()
            eta = (_time.time() - t0) / epoch * (n_epochs - epoch) / 60
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  {epoch:4d}/{n_epochs} | train {epoch_loss:.4f} "
                  f"| test {test_loss:.4f} | C1={C1:.0f}F "
                  f"| lr {lr_now:.1e} | ETA {eta:.1f}m"
                  f"| min(R1*C1): {(model.r1_net(soc0_b, I_b / model.r1_net.I_ref, u_b) * model.C1).min().item():.2f} s")

    history['time'] = (_time.time() - t0) / 60

    return history

# %% ══════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS  (standalone numpy predict — no batched forward)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict_np(model, I_val, u_val, soc0, T):
    """Pure-numpy single-trajectory predict for plotting."""
    C1 = model.C1.item()
    t  = np.arange(T, dtype=np.float32)
    soc = (soc0 - I_val / model.Q0 * t).astype(np.float32)

    soc_t  = torch.from_numpy(soc)
    I_norm = torch.full((T,), I_val / model.r1_net.I_ref)
    u_t    = torch.full((T,), u_val)
    R1 = model.r1_net(soc_t, I_norm, u_t).numpy()

    U1 = np.zeros(T, dtype=np.float64)
    for n in range(T - 1):
        U1[n+1] = U1[n] + I_val / C1 - U1[n] / (R1[n] * C1)

    Ue = model.Ue_interp(soc)
    R0 = model.R0_func(u_val, I_val)
    V  = Ue - I_val * R0 - U1
    return V, soc, U1, R1


def plot_predictions(model, trajs, title='', n_show=3):
    n = min(n_show, len(trajs))
    fig, axes = plt.subplots(5, n, figsize=(5 * n, 17), squeeze=False)

    model.eval()
    for j in range(n):
        tr = trajs[j]
        V, soc_np, U1, R1 = _predict_np(
            model, tr['I'], tr['u'], tr['soc0'], tr['T'])

        axes[0, j].plot(soc_np, tr['V'].numpy(), '--', color = COLORS[1], label=r'True $V$', lw=2)
        axes[0, j].plot(soc_np, V, '-', color = COLORS[0], label=r'Predicted $V$', lw=2)
        axes[0, j].set_ylabel(r'$V$ [V]'); axes[0, j].legend()
        axes[0, j].invert_xaxis()
        axes[0, j].set_xlabel('State of Charge');
        axes[0, j].set_title(f'{title}I={tr["I"]:.1f}, u={tr["u"]:.3f}')

        axes[1, j].plot(soc_np, tr['U1_true'].numpy(), '--', color = COLORS[1], label=r'True $U_1$', lw=2)
        axes[1, j].plot(soc_np, U1, '-', color = COLORS[0], label=r'Predicted $U_1$', lw=2)
        axes[1, j].set_ylabel(r'$U_1$ [V]'); axes[1, j].legend()
        axes[1, j].set_xlabel('State of Charge');
        axes[1, j].invert_xaxis()

        C1 = model.C1.item()
        dU1_data = np.gradient(tr['U1_true'].numpy(), 1.0)
        dU1_rc   = tr['I'] / C1 - U1 / (R1 * C1)
        axes[2, j].plot(soc_np, dU1_data, '--', color = COLORS[1], label=r'True $dU1/dt$', lw=2, alpha=0.7)
        axes[2, j].plot(soc_np, dU1_rc, '-', color = COLORS[0], label=r'Predicted $dU1/dt$', lw=2)
        axes[2, j].set_ylabel(r'$dU1/dt$ [V/s]'); axes[2, j].set_xlabel('SOC')
        axes[2, j].legend(); axes[2, j].invert_xaxis()

        R0_val = R0_func(tr['u'], tr['I'])
        axes[3, j].axhline(R0_val * 1000, ls='--', color = COLORS[0], label=r'$R0$' + f' = {R0_val * 1000:.1f} m$\Omega$', lw=2)
        axes[3, j].plot(soc_np, R1 * 1000, '-', color = COLORS[0], label=r'$R1$', lw=2)
        axes[3, j].set_ylabel(r'$R$ [m$\Omega$]'); axes[3, j].legend()
        axes[3, j].invert_xaxis()

        axes[4, j].axhline(C1, ls='--', color = COLORS[0], label=r'$C_1=$' + f'{C1:.0f} F', lw=2)
        axes[4, j].set_ylabel(r'$C_1$ [F]'); axes[4, j].legend()
        axes[4, j].invert_xaxis()


    fig.tight_layout()
    return fig


def plot_R1_landscape(model, I_values, u_val=-0.5):
    fig, ax = plt.subplots(figsize=(6, 4))
    soc = torch.linspace(0.02, 1.0, 200)
    model.eval()
    with torch.no_grad():
        for Iv in I_values:
            I_norm = torch.full((200,), Iv / model.r1_net.I_ref)
            u_t    = torch.full((200,), u_val)
            R1 = model.r1_net(soc, I_norm, u_t).numpy() * 1000
            ax.plot(soc.numpy(), R1, label=f'I={Iv:.0f}A')
    ax.set_xlabel('SOC'); ax.set_ylabel(r'$R$ [m$\Omega$]')
    ax.set_title(r'Charge-transfer resistance $R_1$(SOC)')
    ax.legend(); ax.invert_xaxis(); fig.tight_layout()
    return fig

# %% ══════════════════════════════════════════════════════════
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
    C1 = model.C1.item()
    R0 = R0_func(u_val, I_val)
    return dict(soc=np.array(soc_points), R0=np.full(T, R0),
                R1=R1, C1=C1, tau=R1 * C1, U1_ss=R1 * I_val)

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
Ue_interp = interp1d(_s[_i], _u[_i], kind='linear', fill_value='extrapolate')

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")

# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES + ESTIMATE C1
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data, R0_func)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")

C1_init = estimate_C1(train_trajs)
print(f"  C1 estimate: {C1_init:.0f} F")

# %% ══════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════

r1_net = R1Net(n_hidden=N_HIDDEN, I_ref=I_MAX)
model  = BatteryECM(r1_net, Ue_interp, R0_func, Q0, C1_init=C1_init)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")

# %% ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

print(f"\nTraining ({EPOCHS} epochs, batch_size={BATCH_SIZE})...")
history = train_model(model, train_trajs, test_trajs,
                      n_epochs=EPOCHS, lr=LR,
                      batch_size=BATCH_SIZE, print_every=10)

C1_final = model.C1.item()
TOTAL_TIME = history['time'][0]
print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")
print(f"\n  C1: {C1_init:.0f} → {C1_final:.0f} F")

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TRAIN
# ══════════════════════════════════════════════════════════════

plot_predictions(model, train_trajs, 'Train: ')
# plt.savefig('nodes_figs/ecm_node_train.pdf', bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

plot_predictions(model, test_trajs, 'Test: ')
plt.savefig(os.path.join(FIGS_DIR, f'ecm_node_test_{TOTAL_TIME:.1f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  R1 LANDSCAPE
# ══════════════════════════════════════════════════════════════

# I_vals = sorted(data['I'].unique())
# plot_R1_landscape(model, I_vals, u_val=float(data['u'].median()))
# # plt.savefig('nodes_figs/ecm_node_R1.pdf', bbox_inches='tight')
# plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(history['train'], color=COLORS[0], label='train')
ax.semilogy(history['test'], color=COLORS[1], label='test \n last RMSE: {:.4f} V'.format(history['test'][-1]))
ax.set_xlabel('epoch'); ax.set_ylabel('RMSE'); ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, f'ecm_node_loss_{TOTAL_TIME:.1f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  EXTRACT ECM PARAMETERS
# ══════════════════════════════════════════════════════════════

soc_pts = [0.95, 0.80, 0.50, 0.20, 0.10, 0.05]
ecm = extract_ecm_params(model, soc_pts, I_val=11.0, u_val=-0.06)

print(f"ECM parameters at I=11A:")
print(f"  C1 = {ecm['C1']:.0f} F")
print(f"  {'SOC':>5s}  {'R0 mΩ':>7s}  {'R1 mΩ':>7s}  "
      f"{'τ s':>7s}  {'U1ss V':>7s}")
for i, s in enumerate(soc_pts):
    print(f"  {s:5.2f}  {ecm['R0'][i]*1e3:7.2f}  "
          f"{ecm['R1'][i]*1e3:7.2f}  {ecm['tau'][i]:7.0f}  "
          f"{ecm['U1_ss'][i]:7.4f}")

# %% ══════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

torch.save({
    'model': model.state_dict(),
    'history': history,
    'C1_init': C1_init,
    'C1_final': C1_final,
    'N_HIDDEN': N_HIDDEN,
    'EPOCHS': EPOCHS,
}, os.path.join(MODEL_DIR, f'ecm_node_{TOTAL_TIME:.1f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps.pt'))

print(f"Saved: ecm_node_{TOTAL_TIME:.1f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps.pt")
# %%
