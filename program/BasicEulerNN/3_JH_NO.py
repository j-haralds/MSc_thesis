"""
Battery concentration rollout model.
f(I, dI, cn, cp) -> (cn_dot, cp_dot)

Training phases:
  1. Warmup  : standard supervised MSE on cn_dot/cp_dot (ground-truth inputs)
  2. Rollout : curriculum rollout loss; Euler-integrate cn/cp and compare to GT
"""

import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_HORIZON = 3600.0
T         = 1000
DT        = T_HORIZON / (T - 1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path: str):
    path = os.path.join(os.path.dirname(__file__), path)
    d = np.load(path)
    return (d["I"], d["dI"], d["V"],
            d["cn"], d["cn_dot"],
            d["cp"], d["cp_dot"],
            d["valid"])


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
class Normalizer:
    """Z-score normalizer for numpy arrays and torch tensors."""

    def __init__(self):
        self.stats = {}

    def fit(self, **arrays):
        for name, arr in arrays.items():
            mu    = float(arr.mean())
            sigma = float(max(arr.std(), 1e-8))
            self.stats[name] = (mu, sigma)

    def norm(self, arr, name):
        mu, sigma = self.stats[name]
        return (arr - mu) / sigma

    def denorm(self, arr, name):
        mu, sigma = self.stats[name]
        return arr * sigma + mu

    # torch — keeps autograd graph intact
    def norm_t(self, x, name):
        mu, sigma = self.stats[name]
        return (x - mu) / sigma

    def denorm_t(self, x, name):
        mu, sigma = self.stats[name]
        return x * sigma + mu


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, hidden=256, depth=3):
        super().__init__()
        layers, dim = [], 4
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
            dim = hidden
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def masked_mse(pred, target, mask):
    err = (pred - target) ** 2
    m   = mask.unsqueeze(-1)
    return (err * m).sum() / (m.sum() * 2 + 1e-8)


def rollout_loss(model, nrm, x_b, mask_b, n_steps):
    """
    x_b    : (B, T, 4)  normalised [I, dI, cn, cp]
    mask_b : (B, T)

    Randomly picks start s, unrolls n_steps via Euler integration in
    physical space (pure torch, gradients flow), compares to GT cn/cp.
    """
    B, T_seq, _ = x_b.shape
    cn_gt = x_b[:, :, 2:3]
    cp_gt = x_b[:, :, 3:4]

    max_s = max(T_seq - n_steps - 1, 1)
    s     = int(torch.randint(0, max_s, (1,)).item())

    cn_cur = x_b[:, s, 2:3].clone()
    cp_cur = x_b[:, s, 3:4].clone()

    total_loss  = torch.zeros(1, device=x_b.device)
    valid_steps = 0

    for k in range(n_steps):
        t = s + k
        if t + 1 >= T_seq:
            break

        x_t  = torch.cat([x_b[:, t, 0:1], x_b[:, t, 1:2], cn_cur, cp_cur], dim=-1)
        pred = model(x_t)

        # Euler step in physical space — gradient flows through
        cn_phys = nrm.denorm_t(cn_cur,       "cn")
        cp_phys = nrm.denorm_t(cp_cur,       "cp")
        dc_n    = nrm.denorm_t(pred[:, 0:1], "cn_dot")
        dc_p    = nrm.denorm_t(pred[:, 1:2], "cp_dot")

        cn_cur = nrm.norm_t(cn_phys + dc_n * DT, "cn")
        cp_cur = nrm.norm_t(cp_phys + dc_p * DT, "cp")

        step_mask  = mask_b[:, t + 1].unsqueeze(-1)
        cn_err     = (cn_cur - cn_gt[:, t + 1, :]) ** 2
        cp_err     = (cp_cur - cp_gt[:, t + 1, :]) ** 2
        total_loss = total_loss + ((cn_err + cp_err) * step_mask).sum() / (step_mask.sum() + 1e-8)
        valid_steps += 1

    return total_loss / max(valid_steps, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, nrm, loader, epochs=200, warmup=30,
          steps_start=1, steps_end=50, lr=1e-3, grad_clip=1.0):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-5)
    history = {"loss": [], "phase": []}

    for ep in range(epochs):
        model.train()
        phase = "warmup" if ep < warmup else "rollout"

        if phase == "rollout":
            frac    = (ep - warmup) / max(epochs - warmup - 1, 1)
            n_steps = int(steps_start + (steps_end - steps_start) * frac)
        else:
            n_steps = 1

        ep_loss = 0.0
        for x_b, y_b, mask_b in loader:
            if phase == "warmup":
                pred = model(x_b.reshape(-1, 4))
                loss = masked_mse(pred, y_b.reshape(-1, 2), mask_b.reshape(-1))
            else:
                loss = rollout_loss(model, nrm, x_b, mask_b, n_steps)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ep_loss += loss.item()

        ep_loss /= len(loader)
        scheduler.step(ep_loss)
        history["loss"].append(ep_loss)
        history["phase"].append(phase)

        if ep % 10 == 0 or ep == epochs - 1:
            print(f"Ep {ep+1:3d}/{epochs} | {phase:7s} | n_steps={n_steps:3d} "
                  f"| loss={ep_loss:.3e} | lr={optimizer.param_groups[0]['lr']:.1e}")

    return history


# ---------------------------------------------------------------------------
# Inference rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_inference(model, nrm, I_traj, dI_traj, cn0, cp0, cn_min=None):
    model.eval()
    N = len(I_traj)
    cn_out  = np.zeros(N, np.float32);  cp_out  = np.zeros(N, np.float32)
    dcn_out = np.zeros(N, np.float32);  dcp_out = np.zeros(N, np.float32)
    cn_out[0] = cn0;  cp_out[0] = cp0
    steps = 1

    for t in range(N - 1):
        I_n  = float(nrm.norm(np.array([[[I_traj[t]]]]),  "I") [0, 0, 0])
        dI_n = float(nrm.norm(np.array([[[dI_traj[t]]]]), "dI")[0, 0, 0])
        cn_n = float(nrm.norm(np.array([[[cn_out[t]]]]),  "cn")[0, 0, 0])
        cp_n = float(nrm.norm(np.array([[[cp_out[t]]]]),  "cp")[0, 0, 0])

        x_t  = torch.tensor([[I_n, dI_n, cn_n, cp_n]], dtype=torch.float32)
        pred = model(x_t)

        dc_n = float(nrm.denorm(pred[0, 0].item(), "cn_dot"))
        dc_p = float(nrm.denorm(pred[0, 1].item(), "cp_dot"))

        cn_out[t + 1]  = cn_out[t] + dc_n * DT
        cp_out[t + 1]  = cp_out[t] + dc_p * DT
        dcn_out[t]     = dc_n
        dcp_out[t]     = dc_p
        steps         += 1

        if cn_min is not None and cn_out[t + 1] < cn_min:
            break

    return cn_out, cp_out, dcn_out, dcp_out, steps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_history(history):
    fig, ax = plt.subplots(figsize=(6, 3))
    losses, phases = history["loss"], history["phase"]
    wu = [i for i, p in enumerate(phases) if p == "warmup"]
    ro = [i for i, p in enumerate(phases) if p == "rollout"]
    if wu: ax.semilogy(wu, [losses[i] for i in wu], "b-", label="warmup")
    if ro: ax.semilogy(ro, [losses[i] for i in ro], "r-", label="rollout")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    plt.tight_layout()
    return fig


def plot_rollout_result(model, nrm, idx,
                        cn_d, cp_d, cn_dot_d, cp_dot_d,
                        I_d, dI_d, mask_d, title=""):
    n_gt = int(mask_d[idx].sum())
    cn_pred, cp_pred, dcn_pred, _, steps = rollout_inference(
        model, nrm,
        I_traj  = I_d[idx, :, 0],
        dI_traj = dI_d[idx, :, 0],
        cn0     = float(cn_d[idx, 0, 0]),
        cp0     = float(cp_d[idx, 0, 0]),
        cn_min  = float(cn_d[idx, :n_gt, 0].min()),
    )
    t_gt = np.arange(n_gt)  * DT
    t_nn = np.arange(steps) * DT

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    if title: fig.suptitle(title)

    axes[0].plot(t_gt, cn_d[idx, :n_gt, 0], "r--", label="True $c_n$")
    axes[0].plot(t_nn, cn_pred[:steps],       "b-",  label="Pred $c_n$")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("Conc. [mol/m³]"); axes[0].legend()

    axes[1].plot(t_gt, cp_d[idx, :n_gt, 0], "r--", label="True $c_p$")
    axes[1].plot(t_nn, cp_pred[:steps],       "b-",  label="Pred $c_p$")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("Conc. [mol/m³]"); axes[1].legend()

    axes[2].plot(t_gt, cn_dot_d[idx, :n_gt, 0], "r--", label=r"True $\dot{c}_n$")
    axes[2].plot(t_nn, dcn_pred[:steps],           "b-",  label=r"Pred $\dot{c}_n$")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("Conc. rate [mol/m³/s]"); axes[2].legend()

    axes[3].plot(t_gt, I_d[idx, :n_gt, 0], "k-", label="$I$")
    axes[3].set_xlabel("t [s]"); axes[3].set_ylabel("Current [A]"); axes[3].legend(loc="upper left")
    ax_twin = axes[3].twinx()
    ax_twin.plot(t_gt, dI_d[idx, :n_gt, 0], "r--", label=r"$\dot{I}$")
    ax_twin.set_ylabel("Current rate [A/s]"); ax_twin.legend(loc="upper right")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    I, dI, V, cn, cn_dot, cp, cp_dot, valid = load_data("data_c_GRFI.npz")

    nrm = Normalizer()
    nrm.fit(I=I, dI=dI, V=V, cn=cn, cn_dot=cn_dot, cp=cp, cp_dot=cp_dot)

    x = torch.tensor(
        np.concatenate([nrm.norm(I, "I"),  nrm.norm(dI, "dI"),
                        nrm.norm(cn, "cn"), nrm.norm(cp, "cp")], axis=-1),
        dtype=torch.float32)

    y = torch.tensor(
        np.concatenate([nrm.norm(cn_dot, "cn_dot"),
                        nrm.norm(cp_dot, "cp_dot")], axis=-1),
        dtype=torch.float32)

    mask  = torch.tensor(valid, dtype=torch.float32)
    B     = x.shape[0]
    split = int(0.8 * B)

    loader = DataLoader(
        TensorDataset(x[:split], y[:split], mask[:split]),
        batch_size=max(1, split // 4), shuffle=True)

    model = Net(hidden=256, depth=3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, nrm, loader,
                    epochs=200, warmup=30,
                    steps_start=1, steps_end=50,
                    lr=1e-3, grad_clip=1.0)

    torch.save({"model_state": model.state_dict(), "norm_stats": nrm.stats}, "model.pt")
    print("Saved model.pt")

    plot_history(history).savefig("training_history.png", dpi=150)

    for idx in range(min(3, B - split)):
        plot_rollout_result(
            model, nrm, idx,
            cn[split:], cp[split:], cn_dot[split:], cp_dot[split:],
            I[split:], dI[split:], valid[split:],
            title=f"Test trajectory {idx}",
        ).savefig(f"rollout_test_{idx}.png", dpi=150)

    plt.show()