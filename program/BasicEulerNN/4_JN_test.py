"""
Battery Concentration Dynamics — Neural Network Surrogate
=========================================================
Learns:  dc_n/dt, dc_p/dt = f(I(t), ΔI(t), c_n(t), c_p(t))
Stepper: c(t+dt) = c(t) + f(...) * dt   [Euler, fixed dt]
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ─────────────────────────────────────────────
# 0.  CONSTANTS  (edit these)
# ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
T_HORIZON = 3600.0          # [s]  total simulation horizon
T         = 1000            # number of time steps
DT        = T_HORIZON / (T - 1)   # ≈ 3.6036 s  — fixed for all sims

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 1.  DATA GENERATION  (your code, corrected)
# ─────────────────────────────────────────────
def get_vals_fixed_dt(I0, T=T, t_horizon=T_HORIZON):
    """
    Run PyBaMM at constant current I0, resample onto fixed physical grid.
    Returns arrays of length T and a boolean valid mask.
    """
    import pybamm

    model = pybamm.lithium_ion.SPM()
    param = model.default_parameter_values
    param["Nominal cell capacity [A.h]"] = 1.0
    param["Current function [A]"] = I0
    sim = pybamm.Simulation(model, parameter_values=param)

    # solve on dense grid so interpolation is accurate
    t_dense = np.linspace(0, t_horizon, 1000)
    try:
        solution = sim.solve(t_dense)
    except Exception:
        return None                    # solver failure → skip

    t_end   = float(solution["Time [s]"].entries[-1])
    t_fixed = np.linspace(0, t_horizon, T)   # fixed physical grid
    t_query = np.clip(t_fixed, 0.0, t_end)   # clamp past depletion
    valid   = t_fixed <= t_end               # (T,)  bool

    def obs(var):
        return solution.observe(model.variables[var])(t_query)

    I_arr  = obs("Current [A]")                                              # (T,)
    V_arr  = obs("Terminal voltage [V]")                                     # (T,)
    cn_arr = obs("X-averaged negative particle concentration [mol.m-3]")[-1, :]
    cp_arr = obs("X-averaged positive particle concentration [mol.m-3]")[-1, :]

    return t_fixed, DT, V_arr, I_arr, cn_arr, cp_arr, valid


def gen_data(B, T=T, I_low=0.25, I_high=1.0, seed=None):
    """
    Generate B trajectories with random constant currents.
    Returns numpy arrays shaped (B, T, 1) except valid which is (B, T).
    """
    from tqdm import trange
    rng = np.random.default_rng(seed)

    V      = np.zeros((B, T, 1), dtype=np.float32)
    I      = np.zeros((B, T, 1), dtype=np.float32)
    dI     = np.zeros((B, T, 1), dtype=np.float32)   # ΔI feature
    cn     = np.zeros((B, T, 1), dtype=np.float32)
    cp     = np.zeros((B, T, 1), dtype=np.float32)
    cn_dot = np.zeros((B, T, 1), dtype=np.float32)
    cp_dot = np.zeros((B, T, 1), dtype=np.float32)
    valid  = np.zeros((B, T),    dtype=bool)

    generated = 0
    attempts  = 0
    pbar = trange(B, desc="Generating simulations")

    while generated < B:
        attempts += 1
        I0     = float(rng.uniform(I_low, I_high))
        result = get_vals_fixed_dt(I0, T=T)
        if result is None:
            continue                   # skip failed solves

        t, dt, V_i, I_i, cn_i, cp_i, valid_i = result
        i = generated

        V[i, :, 0]  = V_i
        I[i, :, 0]  = I_i
        cn[i, :, 0] = cn_i
        cp[i, :, 0] = cp_i
        valid[i, :] = valid_i

        # ΔI(t): backward difference,  ΔI[0] = 0
        dI[i, 1:, 0] = np.diff(I_i)
        dI[i, 0,  0] = 0.0

        # dc/dt via forward difference; pad last step
        cn_dot[i, :-1, 0] = np.diff(cn_i) / dt
        cp_dot[i, :-1, 0] = np.diff(cp_i) / dt
        cn_dot[i, -1,  0] = cn_dot[i, -2, 0]
        cp_dot[i, -1,  0] = cp_dot[i, -2, 0]

        generated += 1
        pbar.update(1)

    pbar.close()
    print(f"Done. {attempts} attempts for {B} successful sims.")
    return I, dI, V, cn, cn_dot, cp, cp_dot, valid


def save_data(path, *arrays_and_names):
    """Save generated arrays to .npz for reuse."""
    np.savez(path, **dict(arrays_and_names))
    print(f"Saved to {path}")


def load_data(path):
    data = np.load(path)
    return (data["I"], data["dI"], data["V"],
            data["cn"], data["cn_dot"],
            data["cp"], data["cp_dot"],
            data["valid"])


# ─────────────────────────────────────────────
# 2.  NORMALISATION
# ─────────────────────────────────────────────
class Normalizer:
    """Per-feature mean/std normalisation computed on training data."""

    def __init__(self):
        self.stats = {}   # name → (mean, std)

    def fit(self, **named_arrays):
        """named_arrays: name=array  where array is (B, T, 1) or (B, T)."""
        for name, arr in named_arrays.items():
            flat = arr[np.isfinite(arr)]
            self.stats[name] = (float(flat.mean()), float(flat.std()) + 1e-8)

    def transform(self, name, arr):
        mu, sigma = self.stats[name]
        return (arr - mu) / sigma

    def inverse_transform(self, name, arr):
        mu, sigma = self.stats[name]
        return arr * sigma + mu

    def save(self, path):
        np.savez(path, **{f"{k}_mean": v[0] for k, v in self.stats.items()},
                       **{f"{k}_std":  v[1] for k, v in self.stats.items()})

    def load(self, path):
        d = np.load(path)
        keys = [k.replace("_mean", "") for k in d.files if k.endswith("_mean")]
        for k in keys:
            self.stats[k] = (float(d[f"{k}_mean"]), float(d[f"{k}_std"]))


# ─────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────
class BatteryDataset(Dataset):
    """
    One sample = one full trajectory.
    x  : (T, 4)  [I, dI, cn, cp]  — normalised inputs
    y  : (T, 2)  [cn_dot, cp_dot] — normalised targets
    mask: (T,)   bool valid steps
    """

    def __init__(self, I, dI, cn, cn_dot, cp, cp_dot, valid, norm: Normalizer):
        B, T, _ = I.shape

        # normalise
        I_n      = norm.transform("I",      I)
        dI_n     = norm.transform("dI",     dI)
        cn_n     = norm.transform("cn",     cn)
        cp_n     = norm.transform("cp",     cp)
        cn_dot_n = norm.transform("cn_dot", cn_dot)
        cp_dot_n = norm.transform("cp_dot", cp_dot)

        # build input tensor (B, T, 4)
        self.x    = torch.tensor(
            np.concatenate([I_n, dI_n, cn_n, cp_n], axis=-1),
            dtype=torch.float32)
        self.y    = torch.tensor(
            np.concatenate([cn_dot_n, cp_dot_n], axis=-1),
            dtype=torch.float32)
        self.mask = torch.tensor(valid, dtype=torch.bool)   # (B, T)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]


# ─────────────────────────────────────────────
# 4.  MODEL
# ─────────────────────────────────────────────
class ConcentrationStepper(nn.Module):
    """
    MLP that predicts normalised [dc_n/dt, dc_p/dt]
    from normalised [I, ΔI, c_n, c_p].

    Apply independently at each timestep (no sequence state).
    Upgrade to GRU by swapping the body.
    """

    def __init__(self, input_dim=4, hidden=128, n_layers=3, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (..., 4)  →  (..., 2)   works for (B,T,4) or (T,4) or (4,)"""
        return self.net(x)


class GRUStepper(nn.Module):
    """
    GRU-based stepper — better for capturing long-range SOC drift.
    Drop-in replacement for ConcentrationStepper in training loop.
    """

    def __init__(self, input_dim=4, hidden=128, n_layers=2, dropout=0.0):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h0=None):
        """x: (B, T, 4)  →  (B, T, 2)"""
        out, h = self.gru(x, h0)
        return self.head(out), h


# ─────────────────────────────────────────────
# 5.  TRAINING
# ─────────────────────────────────────────────
def masked_mse(pred, target, mask):
    """MSE only over valid timesteps."""
    # pred, target: (B, T, 2),  mask: (B, T)
    err  = (pred - target) ** 2                    # (B, T, 2)
    m    = mask.unsqueeze(-1).float()              # (B, T, 1)
    return (err * m).sum() / (m.sum() * 2 + 1e-8)


def train(model, train_loader, val_loader,
          epochs=100, lr=1e-3, patience=15,
          save_path="best_model.pt"):

    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, patience=patience//2, factor=0.5)

    best_val  = float("inf")
    no_improve = 0
    history   = {"train": [], "val": []}
    is_gru    = isinstance(model, GRUStepper)

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()

            pred = model(x)[0] if is_gru else model(x)   # (B, T, 2)
            loss = masked_mse(pred, y, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                pred = model(x)[0] if is_gru else model(x)
                val_loss += masked_mse(pred, y, mask).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.4e} | val {val_loss:.4e}")

        # ── early stopping ──
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return history


# ─────────────────────────────────────────────
# 6.  INFERENCE / ROLLOUT
# ─────────────────────────────────────────────
@torch.no_grad()
def rollout(model, norm: Normalizer,
            I_profile: np.ndarray,
            c0_n: float, c0_p: float,
            dt: float = DT,
            v_cutoff: float = 2.5) -> dict:
    """
    Autoregressive Euler rollout.

    Parameters
    ----------
    model      : trained ConcentrationStepper or GRUStepper
    norm       : fitted Normalizer
    I_profile  : (T,) array of current values [A]
    c0_n, c0_p : initial concentrations [mol/m³]
    dt         : physical timestep [s]
    v_cutoff   : (optional) stop early if voltage proxy hits cutoff

    Returns
    -------
    dict with keys "cn", "cp", "cn_dot", "cp_dot", "t", "steps_run"
    """
    model.eval()
    is_gru = isinstance(model, GRUStepper)
    T_run  = len(I_profile)

    cn_traj = np.zeros(T_run, dtype=np.float32)
    cp_traj = np.zeros(T_run, dtype=np.float32)
    cn_dot_traj = np.zeros(T_run, dtype=np.float32)
    cp_dot_traj = np.zeros(T_run, dtype=np.float32)

    cn_traj[0] = c0_n
    cp_traj[0] = c0_p

    h = None  # GRU hidden state

    def make_input(I_val, dI_val, cn_val, cp_val):
        x = np.array([[I_val, dI_val, cn_val, cp_val]], dtype=np.float32)  # (1,4)
        x[:, 0] = norm.transform("I",  x[:, 0:1])[:, 0]
        x[:, 1] = norm.transform("dI", x[:, 1:2])[:, 0]
        x[:, 2] = norm.transform("cn", x[:, 2:3])[:, 0]
        x[:, 3] = norm.transform("cp", x[:, 3:4])[:, 0]
        return torch.tensor(x, dtype=torch.float32, device=DEVICE)   # (1, 4)

    steps_run = 1
    for t in range(T_run - 1):
        dI = I_profile[t] - I_profile[t - 1] if t > 0 else 0.0
        x  = make_input(I_profile[t], dI, cn_traj[t], cp_traj[t])

        if is_gru:
            x_seq = x.unsqueeze(1)                   # (1, 1, 4)
            out, h = model(x_seq, h)                 # (1, 1, 2)
            pred = out[0, 0]                         # (2,)
        else:
            pred = model(x)[0]                       # (2,)

        # inverse-transform derivatives
        dc_n = float(norm.inverse_transform("cn_dot",
                     pred[0].cpu().numpy().reshape(1, 1, 1))[0, 0, 0])
        dc_p = float(norm.inverse_transform("cp_dot",
                     pred[1].cpu().numpy().reshape(1, 1, 1))[0, 0, 0])

        cn_dot_traj[t] = dc_n
        cp_dot_traj[t] = dc_p

        # Euler step
        cn_traj[t + 1] = cn_traj[t] + dc_n * dt
        cp_traj[t + 1] = cp_traj[t] + dc_p * dt

        steps_run += 1

    t_axis = np.arange(T_run) * dt
    return dict(t=t_axis, cn=cn_traj, cp=cp_traj,
                cn_dot=cn_dot_traj, cp_dot=cp_dot_traj,
                steps_run=steps_run)


# ─────────────────────────────────────────────
# 7.  PLOTTING HELPERS
# ─────────────────────────────────────────────
def plot_history(history):
    plt.figure(figsize=(7, 3))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"],   label="val")
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_history.pdf"))
    plt.show()


def plot_rollout(result_nn, result_pybamm=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in zip(axes, ["cn", "cp"],
                               ["c_n (negative)", "c_p (positive)"]):
        ax.plot(result_nn["t"], result_nn[key], label="NN rollout")
        if result_pybamm is not None:
            ax.plot(result_pybamm["t"], result_pybamm[key],
                    "--", label="PyBaMM ground truth")
        ax.set_xlabel("time [s]"); ax.set_ylabel("c [mol/m^3]")
        ax.set_title(label); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rollout_comparison.pdf"))
    plt.show()


# ─────────────────────────────────────────────
# 8.  MAIN — end-to-end example
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── 8a. Generate (or load) data ──────────────────────────────
    DATA_FILE = os.path.join(RESULTS_DIR, "battery_data.npz")
    if Path(DATA_FILE).exists():
        print("Loading cached data …")
        I, dI, V, cn, cn_dot, cp, cp_dot, valid = load_data(DATA_FILE)
    else:
        print("Generating data …")
        I, dI, V, cn, cn_dot, cp, cp_dot, valid = gen_data(
            B=500, T=T, I_low=0.25, I_high=1.0, seed=42)
        save_data(DATA_FILE,
                  ("I", I), ("dI", dI), ("V", V),
                  ("cn", cn), ("cn_dot", cn_dot),
                  ("cp", cp), ("cp_dot", cp_dot),
                  ("valid", valid))

    # ── 8b. Train/val split ───────────────────────────────────────
    B     = I.shape[0]
    idx   = np.random.default_rng(0).permutation(B)
    split = int(0.8 * B)
    tr, va = idx[:split], idx[split:]

    # ── 8c. Fit normaliser on training data only ──────────────────
    norm = Normalizer()
    norm.fit(I=I[tr], dI=dI[tr], cn=cn[tr], cp=cp[tr],
             cn_dot=cn_dot[tr], cp_dot=cp_dot[tr])
    norm.save(os.path.join(RESULTS_DIR, "normalizer.npz"))

    # ── 8d. Build datasets ────────────────────────────────────────
    def make_ds(idx_arr):
        return BatteryDataset(I[idx_arr], dI[idx_arr],
                              cn[idx_arr], cn_dot[idx_arr],
                              cp[idx_arr], cp_dot[idx_arr],
                              valid[idx_arr], norm)

    train_ds = make_ds(tr)
    val_ds   = make_ds(va)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    # ── 8e. Build & train model ───────────────────────────────────
    # Swap ConcentrationStepper → GRUStepper if you want recurrence
    model = ConcentrationStepper(input_dim=4, hidden=128, n_layers=3).to(DEVICE)
    # model = GRUStepper(input_dim=4, hidden=128, n_layers=2).to(DEVICE)

    history = train(model, train_loader, val_loader,
                    epochs=200, lr=1e-3, patience=20,
                    save_path=os.path.join(RESULTS_DIR, "best_model.pt"))
    plot_history(history)

    # ── 8f. Inference on a held-out simulation ────────────────────
    # Run a new PyBaMM sim as ground truth
    test_I0  = 0.6
    result_gt = get_vals_fixed_dt(test_I0, T=T)
    if result_gt is not None:
        t_gt, dt_gt, V_gt, I_gt, cn_gt, cp_gt, valid_gt = result_gt

        # NN rollout from same initial conditions
        result_nn = rollout(
            model, norm,
            I_profile=I_gt,
            c0_n=cn_gt[0],
            c0_p=cp_gt[0],
            dt=DT)

        plot_rollout(result_nn,
                     result_pybamm={"t": t_gt, "cn": cn_gt, "cp": cp_gt})

        # Quantitative error
        n_valid = int(valid_gt.sum())
        rmse_n  = np.sqrt(np.mean((result_nn["cn"][:n_valid] - cn_gt[:n_valid])**2))
        rmse_p  = np.sqrt(np.mean((result_nn["cp"][:n_valid] - cp_gt[:n_valid])**2))
        print(f"RMSE c_n: {rmse_n:.4f} mol/m^3")
        print(f"RMSE c_p: {rmse_p:.4f} mol/m^3")