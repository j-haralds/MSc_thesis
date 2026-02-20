"""
2_PENN_ECM.py

Parameter-inference model:
  I(t) sequence  --->  {R0, R1, C1, (optional z0)}  --->  ECM ODE solver  --->  V_hat(t)

Key points:
- "Continuous" ECM dynamics: we integrate the continuous-time ODE with RK4.
  (Any continuous model is still numerically integrated on your sampled grid.)
- OCV(SOC) polynomial fitting shown at the bottom (NumPy), with a ridge-regularized fit
  that is often more stable than plain np.polyfit for higher degrees.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 0) Load I/V data
# -----------------------------
BASE_DIR = Path.cwd()
I = np.load(BASE_DIR.parent / 'program/NO/data_NO/I.npz')['arr_0']   # [N,1,T]
V = np.load(BASE_DIR.parent / 'program/NO/data_NO/V.npz')['arr_0']   # [N,1,T]

I = I.reshape(I.shape[0], -1)  # [N, T]
V = V.reshape(V.shape[0], -1)  # [N, T]
N, T = I.shape

test_size = 0.1
split = int(test_size * N)

I_test = I[:split]
V_test = V[:split]
I_train = I[split:]
V_train = V[split:]


# -----------------------------
# 1) Scaling (IMPORTANT)
# -----------------------------
# For physics/ODE models you usually want GLOBAL scaling,
# not a separate mean/std per time index.
scaler_I = StandardScaler()
scaler_V = StandardScaler()

I_train_s = scaler_I.fit_transform(I_train.reshape(-1, 1)).reshape(-1, T)  # global
V_train_s = scaler_V.fit_transform(V_train.reshape(-1, 1)).reshape(-1, T)  # global
I_test_s  = scaler_I.transform(I_test.reshape(-1, 1)).reshape(-1, T)
V_test_s  = scaler_V.transform(V_test.reshape(-1, 1)).reshape(-1, T)

# To keep the ECM solver in PHYSICAL UNITS we will unscale inside the forward,
# using these constants as torch scalars:
I_mean_t = torch.tensor(float(scaler_I.mean_[0]), dtype=torch.float32)
I_std_t  = torch.tensor(float(scaler_I.scale_[0]), dtype=torch.float32)
V_mean_t = torch.tensor(float(scaler_V.mean_[0]), dtype=torch.float32)
V_std_t  = torch.tensor(float(scaler_V.scale_[0]), dtype=torch.float32)

# To sequences [N, T, 1]
I_train_s = I_train_s[..., None].astype(np.float32)
V_train_s = V_train_s[..., None].astype(np.float32)
I_test_s  = I_test_s[..., None].astype(np.float32)
V_test_s  = V_test_s[..., None].astype(np.float32)

train_dataset = TensorDataset(torch.from_numpy(I_train_s), torch.from_numpy(V_train_s))
loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)


# -----------------------------
# 2) OCV model (use fitted polynomial)
# -----------------------------
# Placeholder coefficients: Voc(z) ≈ 4.0 V.
# Replace with fitted coefficients from your OCV/SOC data (see section at bottom).
# Coeff format: highest power first, like np.polyfit.
VOC_COEFFS = torch.tensor([4.0], dtype=torch.float32)  # degree 0 constant
# If you have, for example, a 7th degree fit, this would be length 8.


def voc_poly_torch(z: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate polynomial Voc(z) with Horner's method.
    z: [B, T] in [0,1]
    coeffs: [P] highest power first
    returns Voc: [B, T]
    """
    y = torch.zeros_like(z)
    for c in coeffs.to(z.device):
        y = y * z + c
    return y


# -----------------------------
# 3) Parameter network: sequence -> parameters
# -----------------------------
class ECMParameterNet(nn.Module):
    """
    Encodes the current sequence and outputs parameters for a 1RC ECM.
    Output parameterization is chosen to enforce positivity.

    Outputs:
      R0 > 0
      R1 > 0
      C1 > 0
      z0 in (0,1)  (optional; helps if initial SOC varies across samples)
    """
    def __init__(self, hidden_dim=128, learn_z0=True):
        super().__init__()
        self.learn_z0 = learn_z0
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        out_dim = 4 if learn_z0 else 3
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self.softplus = nn.Softplus()

    def forward(self, I_seq_scaled: torch.Tensor):
        # I_seq_scaled: [B, T, 1] (scaled)
        _, h_last = self.gru(I_seq_scaled)          # h_last: [1, B, H]
        h_last = h_last.squeeze(0)                  # [B, H]
        raw = self.head(h_last)                     # [B, out_dim]

        # Positivity constraints
        R0 = self.softplus(raw[:, 0]) + 1e-6
        R1 = self.softplus(raw[:, 1]) + 1e-6
        C1 = self.softplus(raw[:, 2]) + 1e-6

        if self.learn_z0:
            z0 = torch.sigmoid(raw[:, 3])  # (0,1)
            return R0, R1, C1, z0
        return R0, R1, C1, None


# -----------------------------
# 4) Continuous-time ECM ODE solved with RK4
# -----------------------------
# Deleted function ecm_ode_rhs(...) per instructions.


def ecm_forward_rk4(I_seq_scaled: torch.Tensor,
                    params,
                    dt: float,
                    Q: float,
                    voc_coeffs: torch.Tensor):
    """
    Simpler ECM forward using explicit Euler integration.

    This directly mirrors the discrete-time ECM:

        U1_{k+1} = U1_k + dt * ( -U1_k/(R1*C1) + I_k/C1 )
        z_{k+1}  = z_k  - dt * I_k / Q
        V_k      = Voc(z_k) - R0*I_k - U1_k
    """

    R0, R1, C1, z0 = params

    # Unscale current to physical units
    I_s = I_seq_scaled.squeeze(-1)              # [B, T]
    I_phys = I_s * I_std_t + I_mean_t          # [B, T]

    B, T = I_phys.shape

    # Initial states
    U1 = torch.zeros(B, dtype=torch.float32)   # [B]
    z  = z0 if z0 is not None else torch.ones(B)

    V_out = []
    U1_traj = []
    z_traj = []

    for k in range(T):
        Ik = I_phys[:, k]  # [B]

        # Save states before update
        U1_traj.append(U1)
        z_traj.append(z)

        # Voltage at current state
        Voc = voc_poly_torch(z.view(B, 1), voc_coeffs).view(B)
        Vk = Voc - R0 * Ik - U1
        V_out.append(Vk)

        # No update after last sample
        if k == T - 1:
            break

        # Explicit Euler updates
        dU1 = -U1 / (R1 * C1) + Ik / C1
        dz  = -Ik / Q

        U1 = U1 + dt * dU1
        z  = z  + dt * dz

        # Keep SOC reasonable
        z = torch.clamp(z, 0.0, 1.0)

    V_phys = torch.stack(V_out, dim=1)      # [B, T]
    U1_traj = torch.stack(U1_traj, dim=1)   # [B, T]
    z_traj  = torch.stack(z_traj, dim=1)    # [B, T]

    # Scale voltage back to normalized space
    V_scaled = (V_phys - V_mean_t) / V_std_t
    V_scaled = V_scaled.unsqueeze(-1)

    return V_scaled, V_phys, {"U1": U1_traj, "z": z_traj, "I": I_phys}

# # -----------------------------
# # 4) Continuous-time ECM ODE solved with RK4
# # -----------------------------
# def ecm_ode_rhs(U1: torch.Tensor, z: torch.Tensor, I: torch.Tensor,
#                R1: torch.Tensor, C1: torch.Tensor, Q: float):
#     """
#     RHS of the continuous-time ECM ODEs:
#       dU1/dt = -(1/(R1*C1)) * U1 + (1/C1) * I
#       dz/dt  = -(1/Q) * I

#     Shapes:
#       U1, z, I: [B]
#       R1, C1: [B] (sample-specific parameters)
#     """
#     dU1 = -(1.0 / (R1 * C1)) * U1 + (1.0 / C1) * I
#     dz  = -(1.0 / Q) * I
#     return dU1, dz


# def ecm_forward_rk4(I_seq_scaled: torch.Tensor,
#                     params,
#                     dt: float,
#                     Q: float,
#                     voc_coeffs: torch.Tensor):
#     """
#     Integrate ECM ODE with RK4 over the sampled grid.

#     Inputs:
#       I_seq_scaled: [B, T, 1] (scaled current)
#       params: tuple(R0, R1, C1, z0)
#       dt, Q: physical constants
#       voc_coeffs: polynomial coefficients (torch tensor)

#     Returns:
#       V_hat_scaled: [B, T, 1]  (scaled voltage, for MSE in scaled space)
#       V_hat_phys:   [B, T]     (physical volts, for debugging)
#       states: dict with U1, z trajectories in physical units
#     """
#     R0, R1, C1, z0 = params

#     # Unscale current to physical units
#     I_s = I_seq_scaled.squeeze(-1)                  # [B, T]
#     I_phys = I_s * I_std_t + I_mean_t               # [B, T]

#     B, T = I_phys.shape

#     # Initial states
#     U1 = torch.zeros(B, dtype=torch.float32)        # [B]
#     z  = (z0 if z0 is not None else torch.ones(B) * 1.0).to(U1.dtype)  # [B]

#     U1_traj = []
#     z_traj = []
#     V_traj = []

#     for k in range(T):
#         Ik = I_phys[:, k]  # [B]

#         # save current state -> voltage
#         U1_traj.append(U1)
#         z_traj.append(z)

#         Voc = voc_poly_torch(z.view(B, 1), voc_coeffs).view(B)  # [B]
#         Vk = Voc - R0 * Ik - U1
#         V_traj.append(Vk)

#         # RK4 step to next state (skip last update after final sample)
#         if k == T - 1:
#             break

#         def rhs(u1, zz, ii):
#             return ecm_ode_rhs(u1, zz, ii, R1, C1, Q)

#         k1_u, k1_z = rhs(U1, z, Ik)
#         k2_u, k2_z = rhs(U1 + 0.5 * dt * k1_u, z + 0.5 * dt * k1_z, I_phys[:, k])
#         k3_u, k3_z = rhs(U1 + 0.5 * dt * k2_u, z + 0.5 * dt * k2_z, I_phys[:, k])
#         k4_u, k4_z = rhs(U1 + dt * k3_u,       z + dt * k3_z,       I_phys[:, k])

#         U1 = U1 + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
#         z  = z  + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)

#         # keep z in a reasonable range (soft clamp-ish)
#         z = torch.clamp(z, 0.0, 1.0)

#     U1_traj = torch.stack(U1_traj, dim=1)  # [B, T]
#     z_traj  = torch.stack(z_traj, dim=1)   # [B, T]
#     V_phys  = torch.stack(V_traj, dim=1)   # [B, T]

#     # Scale predicted voltage back to scaled space so it matches V_train_s
#     V_scaled = (V_phys - V_mean_t) / V_std_t          # [B, T]
#     V_scaled = V_scaled.unsqueeze(-1)                 # [B, T, 1]

#     return V_scaled, V_phys, {"U1": U1_traj, "z": z_traj, "I": I_phys}

# def ecm_forward_rk4(I_seq_scaled, params, dt, Q, voc_coeffs):
#     R0, R1, C1, z0 = params

#     I_phys = I_seq_scaled.squeeze(-1) * I_std_t + I_mean_t  # [B,T]
#     B, T = I_phys.shape

#     U = torch.zeros(B)
#     z = z0

#     V_out = []

#     for k in range(T):
#         Ik = I_phys[:, k]

#         Voc = voc_poly_torch(z.view(B,1), voc_coeffs).view(B)
#         V_out.append(Voc - R0*Ik - U)

#         if k == T-1:
#             break

#         def f(U_, z_):
#             dU = -U_/(R1*C1) + Ik/C1
#             dz = -Ik/Q
#             return dU, dz

#         k1_u, k1_z = f(U, z)
#         k2_u, k2_z = f(U + 0.5*dt*k1_u, z + 0.5*dt*k1_z)
#         k3_u, k3_z = f(U + 0.5*dt*k2_u, z + 0.5*dt*k2_z)
#         k4_u, k4_z = f(U + dt*k3_u,     z + dt*k3_z)

#         U = U + dt/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)
#         z = z + dt/6*(k1_z + 2*k2_z + 2*k3_z + k4_z)

#     V_phys = torch.stack(V_out, dim=1)
#     V_scaled = (V_phys - V_mean_t)/V_std_t

#     return V_scaled.unsqueeze(-1)


# -----------------------------
# 5) Train
# -----------------------------
DT = 1.0     # seconds (adjust to your experiment)
Q_AS = 3600.0  # capacity in A*s (adjust to your battery)

model = ECMParameterNet(hidden_dim=128, learn_z0=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

def train(epochs=50):
    hist = {"loss": [], "R0": [], "R1": [], "C1": []}
    for ep in range(epochs):
        model.train()
        last = None

        for I_batch_s, V_batch_s in loader:
            # predict parameters
            R0, R1, C1, z0 = model(I_batch_s)

            # forward ECM
            Vhat_s, _, _ = ecm_forward_rk4(I_batch_s, (R0, R1, C1, z0),
                                           dt=DT, Q=Q_AS, voc_coeffs=VOC_COEFFS)

            loss = mse(Vhat_s, V_batch_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last = float(loss.item())

        hist["loss"].append(last)
        hist["R0"].append(float(R0.mean().item()))
        hist["R1"].append(float(R1.mean().item()))
        hist["C1"].append(float(C1.mean().item()))

        print(f"Epoch {ep+1:3d} | loss {last:.6f} | "
              f"R0 {hist['R0'][-1]:.5g} | R1 {hist['R1'][-1]:.5g} | C1 {hist['C1'][-1]:.5g}")

    return hist


history = train(epochs=30)


# -----------------------------
# 6) Test + plot
# -----------------------------
model.eval()
idx = 1

I_seq_s = torch.from_numpy(I_test_s[idx:idx+1])  # [1, T, 1]
V_seq_s = torch.from_numpy(V_test_s[idx:idx+1])  # [1, T, 1]

with torch.no_grad():
    R0, R1, C1, z0 = model(I_seq_s)
    Vhat_s, Vhat_phys, states = ecm_forward_rk4(I_seq_s, (R0, R1, C1, z0),
                                                dt=DT, Q=Q_AS, voc_coeffs=VOC_COEFFS)

# back to physical for plotting
I_phys = scaler_I.inverse_transform(I_seq_s.squeeze(-1).numpy().reshape(-1, 1)).reshape(1, T)
V_phys = scaler_V.inverse_transform(V_seq_s.squeeze(-1).numpy().reshape(-1, 1)).reshape(1, T)
Vhat_phys_np = Vhat_phys.numpy()

print("Predicted params:")
print("R0:", float(R0.item()), "R1:", float(R1.item()), "C1:", float(C1.item()), "z0:", float(z0.item()))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
ax1.plot(I_phys.ravel(), color="black", label="I(t) [A]")
ax2.plot(V_phys.ravel(), color="tab:red", linestyle="dashed", label="V true [V]")
ax2.plot(Vhat_phys_np.ravel(), color="tab:blue", label="V ECM(pred params) [V]")
ax2.legend()
ax1.legend()
ax2.set_xlabel("time step")
ax1.set_ylabel("Current [A]")
ax2.set_ylabel("Voltage [V]")

ax3.plot(history["loss"], label="train loss")
ax3.set_xlabel("epoch")
ax3.set_ylabel("MSE (scaled)")
ax3.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 7) OCV(SOC) polynomial fitting (how to do it properly)
# =========================================================
"""
You said: "We have the data but the fit seems to be wrong".

Common reasons an OCV fit looks wrong:
1) Mixing charge/discharge curves (hysteresis) -> you need separate curves or a model with hysteresis.
2) SOC not normalized to [0,1] or has drift/offset.
3) Degree too high -> oscillatory polynomial ("Runge phenomenon").
4) Fitting in raw SOC without scaling -> numerically ill-conditioned.
5) Outliers/noise at the ends (SOC~0 or ~1) dominate.

Below is a robust-ish way:
- scale SOC to [-1, 1] for conditioning
- ridge-regularized polynomial fit (stable)
- choose degree by validation / visual sanity

How to use:
- Put your SOC and OCV numpy arrays into soc_data, ocv_data
- Call fit_ocv_poly_ridge(...)
- Copy the returned coeffs into VOC_COEFFS above
"""

def fit_ocv_poly_ridge(soc: np.ndarray, ocv: np.ndarray, degree: int = 7, ridge: float = 1e-6):
    soc = np.asarray(soc).reshape(-1)
    ocv = np.asarray(ocv).reshape(-1)

    # clean NaNs
    m = np.isfinite(soc) & np.isfinite(ocv)
    soc = soc[m]
    ocv = ocv[m]

    # ensure SOC in [0,1]
    # If your SOC is percent (0..100), normalize it:
    if soc.max() > 1.5:
        soc = soc / 100.0

    # scale SOC to [-1,1] to improve conditioning
    x = 2.0 * soc - 1.0

    # Vandermonde matrix for polynomial: [x^deg ... x^0]
    X = np.vander(x, N=degree+1, increasing=False)

    # Ridge regression solve: (X^T X + λI) a = X^T y
    A = X.T @ X + ridge * np.eye(degree + 1)
    b = X.T @ ocv
    coeffs = np.linalg.solve(A, b)  # highest power first

    # return coefficients in terms of x = 2*soc-1
    return coeffs


def eval_ocv_poly_soc01(soc01: np.ndarray, coeffs_x: np.ndarray):
    # evaluate poly in x=2*soc-1
    x = 2.0 * soc01 - 1.0
    return np.polyval(coeffs_x, x)


def demo_ocv_fit_plot(soc_data: np.ndarray, ocv_data: np.ndarray, degree: int = 7, ridge: float = 1e-6):
    coeffs = fit_ocv_poly_ridge(soc_data, ocv_data, degree=degree, ridge=ridge)
    soc_grid = np.linspace(0, 1, 400)
    ocv_fit = eval_ocv_poly_soc01(soc_grid, coeffs)

    plt.figure(figsize=(6, 4))
    plt.plot(soc_data, ocv_data, ".", label="data")
    plt.plot(soc_grid, ocv_fit, "-", label=f"poly deg={degree} ridge={ridge:g}")
    plt.xlabel("SOC")
    plt.ylabel("OCV [V]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Coeffs (in x=2*soc-1 basis), highest power first:")
    print(coeffs)

    return coeffs


# Example usage (uncomment and point to your data):
# soc_data = np.load(BASE_DIR.parent / "program/OCV/soc.npy")
# ocv_data = np.load(BASE_DIR.parent / "program/OCV/ocv.npy")
# coeffs_x = demo_ocv_fit_plot(soc_data, ocv_data, degree=7, ridge=1e-6)
# Then set VOC_COEFFS = torch.tensor(coeffs_x, dtype=torch.float32)
