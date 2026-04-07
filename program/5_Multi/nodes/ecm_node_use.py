# %% ECM Lite — Load & Use
#
#  Usage:
#      V, soc, U1, R1 = predict(I=11.0, u=-0.06, soc0=0.95, T=3000)

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ──
DATA_FILE  = '2_merged_data.txt'
MODEL_FILE = 'ecm_lite.pt'
Q0         = 17921.57581

# ── R0 (known) ──
def R0_func(u, I):
    return u * (-0.0001887521) - 7.049519e-5 * I + 0.008446693

# ── R1 network ──
class R1Net(nn.Module):
    def __init__(self, n_hidden=32, I_ref=20.0):
        super().__init__()
        self.I_ref = I_ref
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 1))

    def forward(self, soc, I_norm, u):
        x = torch.stack([soc, I_norm, u], dim=-1)
        return nn.functional.softplus(self.net(x)).squeeze(-1) * 0.01 + 1e-5

# ── Load ──
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
_s, _u = data['soc'].values, data['Ue'].values
_i = np.argsort(_s)
Ue_interp = interp1d(_s[_i], _u[_i], kind='linear', fill_value='extrapolate')

ckpt  = torch.load(MODEL_FILE, weights_only=False)
r1net = R1Net(n_hidden=ckpt['N_HIDDEN'], I_ref=data['I'].max())

r1_state = {k.replace('r1_net.', ''): v
             for k, v in ckpt['model'].items() if k.startswith('r1_net.')}
r1net.load_state_dict(r1_state)

log_C1 = ckpt['model']['log_C1']
C1 = torch.exp(log_C1).item()

r1net.eval()
print(f"Loaded model — C1 = {C1:.0f} F, hidden = {ckpt['N_HIDDEN']}")

# ── Predict ──
@torch.no_grad()
def predict(I, u, soc0, T):
    """Run one discharge trajectory. Returns numpy arrays."""
    t = np.arange(T)
    soc = soc0 - I / Q0 * t

    soc_t  = torch.tensor(soc, dtype=torch.float32)
    I_norm = torch.full((T,), I / r1net.I_ref)
    u_t    = torch.full((T,), u)
    R1 = r1net(soc_t, I_norm, u_t).numpy()

    U1 = np.zeros(T)
    for n in range(T - 1):
        U1[n+1] = U1[n] + I / C1 - U1[n] / (R1[n] * C1)

    Ue = Ue_interp(soc)
    R0 = R0_func(u, I)
    V  = Ue - I * R0 - U1

    return V, soc, U1, R1

# %% ── Example: pick a real trajectory and compare ──
traj_id  = data['trajectory'].unique()[0]
grp      = data[data['trajectory'] == traj_id].sort_values('t').reset_index(drop=True)

I_val    = float(grp['I'].iloc[0])
u_val    = float(grp['u'].iloc[0])
soc0     = float(grp['soc'].iloc[0])
T        = len(grp)
V_true   = grp['V'].values
soc_true = grp['soc'].values

V_pred, soc_pred, U1, R1 = predict(I_val, u_val, soc0, T)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(soc_true, V_true, '--', label='data', lw=1.5)
ax.plot(soc_pred, V_pred, '-',  label='model', lw=1.5)
ax.set_xlabel('SOC'); ax.set_ylabel('V [V]')
ax.set_title(f'Trajectory {traj_id} — I={I_val:.1f}A, u={u_val:.3f}')
ax.legend(); ax.invert_xaxis(); fig.tight_layout()
plt.show()

rmse = np.sqrt(np.mean((V_pred - V_true)**2))
print(f"RMSE: {rmse*1000:.1f} mV")
# %%
