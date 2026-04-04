"""
Usage example — paste into notebook cells after your data-loading code
======================================================================

Assumes you already have:
    data   : DataFrame (with eta already negated)
    Q0     : 17921.57581
    R0_func: defined
"""

# ── Cell: imports ──
from JN_node_test import (
    UeLookup, U1Net, BatteryODE, BatteryModel,
    prepare_trajectories, train, plot_predictions, plot_history,
    R0_func,
)

# ╔══════════════════════════════════════════════════════════╗
# ║  Build Ue(SOC) lookup from your data                     ║
# ╚══════════════════════════════════════════════════════════╝

Ue_lookup = UeLookup(data['soc'].values, data['Ue'].values)

# Quick sanity check: Ue at SOC = 0.5
import torch
print(f"Ue(0.5) = {Ue_lookup(torch.tensor([0.5])).item():.4f} V")


# ╔══════════════════════════════════════════════════════════╗
# ║  Extract per-trajectory data                             ║
# ╚══════════════════════════════════════════════════════════╝

trajs = prepare_trajectories(data)
print(f"Total trajectories: {len(trajs)}")
for i, tr in enumerate(trajs[:3]):
    print(f"  traj {i}: I={tr['I'].item():.2f},  "
          f"u={tr['u'].item():.4f},  T={len(tr['t'])}")

# ── Train / test split ──
split = int(len(trajs) * 0.8)
train_trajs = trajs[:split]
test_trajs  = trajs[split:]
print(f"Train: {len(train_trajs)}  |  Test: {len(test_trajs)}")


# ╔══════════════════════════════════════════════════════════╗
# ║  Build the model                                         ║
# ╚══════════════════════════════════════════════════════════╝

Q0 = 17921.57581

u1_net   = U1Net(n_hidden=32)       # the learned part
ode_func = BatteryODE(u1_net, Q0)   # ODE right-hand side

model = BatteryModel(
    ode_func,
    Ue_lookup,
    R0_func,
    integrator='rk4',    # start here; swap to 'dopri5' once it works
)

# Check parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Learnable parameters: {n_params}")


# ╔══════════════════════════════════════════════════════════╗
# ║  Sanity check: forward pass before training              ║
# ╚══════════════════════════════════════════════════════════╝

# Pick one trajectory and check that V_pred ≈ Ue − I·R0
# (because U1 ≈ 0 at init thanks to small-weight initialisation)
tr0 = train_trajs[0]
with torch.no_grad():
    V_pred, SOC_pred, U1_pred = model(tr0['I'], tr0['u'], tr0['soc0'], tr0['t'])

print(f"\nBefore training (traj 0):")
print(f"  V_pred[0] = {V_pred[0, 0].item():.4f}   V_data[0] = {tr0['V'][0].item():.4f}")
print(f"  U1[0]     = {U1_pred[0, 0].item():.6f}   (should be ≈ 0)")
print(f"  V_pred[-1]= {V_pred[0, -1].item():.4f}   V_data[-1]= {tr0['V'][-1].item():.4f}")


# ╔══════════════════════════════════════════════════════════╗
# ║  Train                                                   ║
# ╚══════════════════════════════════════════════════════════╝

history = train(
    model,
    train_trajs,
    test_trajs,
    n_epochs=300,       # increase if loss is still dropping
    lr=1e-3,
    print_every=25,
)

plot_history(history)


# ╔══════════════════════════════════════════════════════════╗
# ║  Evaluate                                                ║
# ╚══════════════════════════════════════════════════════════╝

# Plot a few training trajectories
fig_train = plot_predictions(model, train_trajs[:3], title_prefix='Train: ')

# Plot test trajectories
fig_test = plot_predictions(model, test_trajs[:3], title_prefix='Test: ')


# ╔══════════════════════════════════════════════════════════╗
# ║  Inspect the learned vector field (optional)             ║
# ╚══════════════════════════════════════════════════════════╝
# This helps you check whether dU1/dt has a sensible shape.
# For a fixed (I, u), sweep U1 and SOC and plot dU1/dt.

import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(model, I_val, u_val, n_grid=30):
    """Visualise dU1/dt as a function of (SOC, U1) at fixed I, u."""
    soc_grid = np.linspace(0.05, 0.95, n_grid)
    u1_grid  = np.linspace(-0.1, 1.0, n_grid)
    SOC_g, U1_g = np.meshgrid(soc_grid, u1_grid)

    SOC_t = torch.tensor(SOC_g.ravel(), dtype=torch.float32).unsqueeze(-1)
    U1_t  = torch.tensor(U1_g.ravel(),  dtype=torch.float32).unsqueeze(-1)
    I_t   = torch.full_like(SOC_t, I_val)
    u_t   = torch.full_like(SOC_t, u_val)

    with torch.no_grad():
        dU1 = model.ode.u1_net(U1_t, SOC_t, I_t, u_t).numpy().reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(SOC_g, U1_g, dU1, levels=30, cmap='RdBu_r')
    fig.colorbar(c, ax=ax, label='dU₁/dt')
    ax.set_xlabel('SOC')
    ax.set_ylabel('U₁')
    ax.set_title(f'Learned dU₁/dt  |  I={I_val:.1f}, u={u_val:.3f}')
    fig.tight_layout()
    return fig


# Example: plot for the first training trajectory's (I, u)
plot_vector_field(model,
                  I_val=train_trajs[0]['I'].item(),
                  u_val=train_trajs[0]['u'].item())
