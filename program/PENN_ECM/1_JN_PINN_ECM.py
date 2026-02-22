import sys
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

 # Robust path: directory where this file lives
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))  # allow import from program/
print(f"Current working directory: {BASE_DIR}")
import plot_settings
plot_settings.apply()
# -----------------------------
# 0) Load data
# -----------------------------

I = np.load(BASE_DIR / 'NO/data_NO/I.npz')['arr_0']   # [N,1,1000]
V = np.load(BASE_DIR / 'NO/data_NO/V.npz')['arr_0']   # [N,1,1000]

# Ensure shapes [N, T]
I = I.reshape(I.shape[0], -1)
V = V.reshape(V.shape[0], -1)
N, T = I.shape

test_size = 0.1
split = int(test_size * N)

I_test = I[:split]
V_test = V[:split]
I_train = I[split:]
V_train = V[split:]

# -----------------------------
# 1) Normalize (fit on train)
# -----------------------------
scaler_I = StandardScaler()
scaler_V = StandardScaler()

I_train_s = scaler_I.fit_transform(I_train)  # [Ntrain, T]
V_train_s = scaler_V.fit_transform(V_train)
I_test_s  = scaler_I.transform(I_test)
V_test_s  = scaler_V.transform(V_test)

# Store scaler statistics as torch tensors (CPU)
V_mean_t = torch.tensor(scaler_V.mean_, dtype=torch.float32)
V_std_t  = torch.tensor(scaler_V.scale_, dtype=torch.float32)
I_mean_t = torch.tensor(scaler_I.mean_, dtype=torch.float32)
I_std_t  = torch.tensor(scaler_I.scale_, dtype=torch.float32)

# Reshape to sequences for RNN: [N, T, 1]
I_train_s = I_train_s[..., None].astype(np.float32)
V_train_s = V_train_s[..., None].astype(np.float32)
I_test_s  = I_test_s[..., None].astype(np.float32)
V_test_s  = V_test_s[..., None].astype(np.float32)

train_dataset = TensorDataset(
    torch.from_numpy(I_train_s),  # [Ntrain, T, 1]
    torch.from_numpy(V_train_s)   # [Ntrain, T, 1]
)
loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

# -----------------------------
# 2) Causal/Recurrent model
# -----------------------------
class GRUVoltageModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, I_seq):
        # I_seq: [B, T, 1]
        h_seq, _ = self.gru(I_seq)         # [B, T, H] (causal w.r.t. sequence)
        V_seq = self.head(h_seq)           # [B, T, 1]
        return V_seq

model = GRUVoltageModel(hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 3) Losses
# -----------------------------
mse = nn.MSELoss()

def data_loss(V_pred, V_true):
    return mse(V_pred, V_true)

# ---- ECM physics (discrete residual) ----
# You MUST set these to match your ECM / data generation:
ECM = {
    "R0": 0.15,     # Ohmic resistance [Ohm]
    "R1": 0.5,     # RC resistance [Ohm]
    "C1": 800.0,   # RC capacitance [F]
    "Q": 2448.0,    # Capacity [A*s] (e.g. 1Ah = 3600 As)
    "dt": 1.0,      # Sampling period [s] (your data looks like 1s)
    "z0": 1.0       # Initial SOC (or treat as learnable later)
}

# Highest power first
VOC_COEFFS = torch.tensor([3.15845444e+02, -1.55679646e+03, 3.16018260e+03, -3.32336545e+03,
  1.78576170e+03, -2.79658643e+02, -1.95143832e+02, 1.14747786e+02,
 -2.33669284e+01, 2.01423200e+00, 3.63176026e+00], dtype=torch.float32)

def voc_poly(z, coeffs):
    # z: [B, T] in [0,1]
    # coeffs: [P] highest power first
    y = torch.zeros_like(z)
    for c in coeffs.to(z.device):
        y = y * z + c
    return y

def physics_loss_ecm(V_pred, I_seq):
    """
    Enforce 1RC ECM residual:
      V = Voc(z) - R0*I - U1
      dU1/dt = -(1/(R1*C1)) U1 + (1/C1) I
      dz/dt = -(1/Q) I   (Coulomb counting)
      Using discrete-time finite differences (causal, meaningful).
    Physics is enforced in physical units by undoing the StandardScaler normalization.
    Inputs:
      V_pred: [B, T, 1] (scaled voltage!)
      I_seq:  [B, T, 1] (scaled current!)
    """
    # Move to [B, T]
    Vp_s = V_pred.squeeze(-1)   # scaled
    I_s  = I_seq.squeeze(-1)    # scaled

    # Unscale using stored scaler statistics
    Vp = Vp_s * V_std_t + V_mean_t
    I  = I_s  * I_std_t + I_mean_t

    # SOC integration (discrete): z[k] = z0 - dt/Q * sum_{j<=k} I[j]
    dt = ECM["dt"]
    Q  = ECM["Q"]
    z0 = ECM["z0"]

    # cumulative sum over time
    z = z0 - (dt / Q) * torch.cumsum(I, dim=1)
    # optional clamp (softly) to keep within [0,1]
    z = torch.clamp(z, 0.0, 1.0)

    Voc = voc_poly(z, VOC_COEFFS)  # [B, T]

    R0 = ECM["R0"]
    # Reconstruct U1 from predicted voltage
    U1 = Voc - R0 * I - Vp  # [B, T]

    # Finite difference dU1/dt over time: [B, T-1]
    dU1_dt = (U1[:, 1:] - U1[:, :-1]) / dt
    U1_k   = U1[:, :-1]
    I_k    = I[:, :-1]

    R1 = ECM["R1"]
    C1 = ECM["C1"]

    # Residual: dU1/dt + (1/(R1*C1))U1 - (1/C1)I = 0
    resid = dU1_dt + (1.0 / (R1 * C1)) * U1_k - (1.0 / C1) * I_k

    return torch.mean(resid ** 2)

# -----------------------------
# 4) Training
# -----------------------------
def training(epochs=200, tol=2.5e-4, lambda_data=0, lambda_phys=1.0):
    history = {'loss': [], 'data_loss': [], 'phys_loss': []}

    for epoch in range(epochs):
        model.train()
        last_loss = None

        for I_batch, V_batch in loader:
            I_batch = I_batch
            V_batch = V_batch

            V_pred = model(I_batch)

            d_loss = data_loss(V_pred, V_batch)
            p_loss = physics_loss_ecm(V_pred, I_batch)

            loss = lambda_data * d_loss + lambda_phys * p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

        history['loss'].append(last_loss)
        history['data_loss'].append(d_loss.item())
        history['phys_loss'].append(p_loss.item())

        # if last_loss < tol:
        #     print(f'Epoch {epoch+1} loss: {last_loss:.6f} - Early stopping')
        #     break

        # print(f"Epoch {epoch+1} | total: {last_loss:.6f} | data: {d_loss.item():.6f}")
        print(f"Epoch {epoch+1} | total: {last_loss:.6f} | data: {d_loss.item():.6f} | phys: {p_loss.item():.6f}")

    return model, history

model, history = training(epochs=20, lambda_phys=0.1)  # start small (0.01–0.1), then increase

# -----------------------------
# 5) Test + Plot
# -----------------------------
model.eval()
test = 1

i_s = I_test_s[test:test+1]  # [1, T, 1] scaled
v_s = V_test_s[test:test+1]  # [1, T, 1] scaled

i_tensor = torch.from_numpy(i_s)

with torch.no_grad():
    y_pred_s = model(i_tensor).detach().numpy()  # scaled, [1,T,1]

# Inverse transform back to physical units for plotting
# scaler expects [N, T], so squeeze last dim
i_phys = scaler_I.inverse_transform(i_s.squeeze(-1))
v_phys = scaler_V.inverse_transform(v_s.squeeze(-1))
y_phys = scaler_V.inverse_transform(y_pred_s.squeeze(-1))

# Plot
f, axes = plt.subplots(2, 2, figsize=(10, 7))

ax1, ax2, ax3, ax4 = axes.flatten()
ax1.plot(i_phys.ravel(), label='Input $I(t)$', color='black')
ax2.plot(v_phys.ravel(), label='True $V(t)$', color='tab:red', linestyle='dashed')
ax2.plot(y_phys.ravel(), label='Predicted $V(t)$', color='tab:blue')

ax2.legend(fontsize=12)
ax1.legend(fontsize=12)
ax2.set_xlabel('Time step')
ax1.set_ylabel('Current [A]')
ax2.set_ylabel('Voltage [V]')

ax3.plot(history['loss'], label='total')
ax3.plot(history['data_loss'], label='data')
ax4.plot(history['phys_loss'], label='phys')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend(fontsize=12)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Physics Loss')

plt.tight_layout()
plt.savefig(BASE_DIR / 'PENN_ECM/figs/1_JN_PINN_ECM_lamd0.pdf')
plt.show()