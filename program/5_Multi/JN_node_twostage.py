"""
Two-stage training for Battery NODE
====================================

Stage 1 (collocation):  Train NN on pointwise dU1/dt targets.
                        Fast, no ODE solve, stable from epoch 1.

Stage 2 (shooting):     Fine-tune through the ODE solver.
                        Enforces trajectory consistency.

This mirrors Brucker's strategy of static pre-training followed
by dynamic training — but adapted to your formulation.

Paste after your data-loading cell and battery_node.py import.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from JN_node_test import (
    UeLookup, U1Net, BatteryODE, BatteryModel,
    prepare_trajectories, train, plot_predictions, plot_history,
)

# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 0: PREPARE COLLOCATION TARGETS                   ║
# ╚══════════════════════════════════════════════════════════╝

def build_collocation_data(data, Q0, R0_func):
    """
    From the DataFrame, compute the pointwise targets for dU1/dt.

    Physics:
        V     = Ue(SOC) − I·R0 − U1
        dV/dt = dUe/dSOC · dSOC/dt  −  dU1/dt      (I·R0 is constant)

    Rearranging:
        dU1/dt = dUe/dSOC · (−I/Q0)  −  dV/dt

    We also reconstruct U1(t) by cumulative integration of the data's
    dV column, so the NN sees the right input during collocation.

    Returns
    -------
    inputs  : (N, 4)  →  [U1, SOC, I, u]
    targets : (N, 1)  →  dU1/dt
    """
    all_inputs  = []
    all_targets = []

    for _, grp in data.sort_values(['trajectory', 't']).groupby('trajectory'):
        grp = grp.reset_index(drop=True)

        I_val   = grp['I'].values[0]
        u_val   = grp['u'].values[0]
        soc     = grp['soc'].values
        Ue      = grp['Ue'].values
        V       = grp['V'].values
        dV      = grp['dV'].values        # dV/dt from data

        # ── dUe/dSOC via finite differences on the Ue(SOC) curve ──
        # (safer than differentiating the interpolant analytically)
        dUe_dSOC = np.gradient(Ue, soc)

        # ── dU1/dt target ──
        dSOC_dt  = -I_val / Q0
        dU1_dt   = dUe_dSOC * dSOC_dt - dV

        # ── U1(t) by reconstruction ──
        # V = Ue - I*R0 - U1  →  U1 = Ue - I*R0 - V
        R0_val = R0_func(u_val, I_val)
        U1_vals = Ue - I_val * R0_val - V

        # ── Assemble ──
        N = len(grp)
        inp = np.stack([U1_vals,
                        soc,
                        np.full(N, I_val),
                        np.full(N, u_val)], axis=1)

        all_inputs.append(inp)
        all_targets.append(dU1_dt.reshape(-1, 1))

    inputs  = torch.tensor(np.concatenate(all_inputs),  dtype=torch.float32)
    targets = torch.tensor(np.concatenate(all_targets), dtype=torch.float32)

    return inputs, targets


# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 1: COLLOCATION PRE-TRAINING                      ║
# ╚══════════════════════════════════════════════════════════╝

def pretrain_collocation(u1_net, inputs, targets,
                         n_epochs=500, lr=1e-3, batch_size=2048,
                         print_every=50):
    """
    Train the NN on pointwise (input → dU1/dt) pairs.

    No ODE solver involved — this is just standard supervised learning.
    Very fast: each epoch is a single forward + backward on a batch
    of scalar samples, rather than integrating trajectories.
    """
    optimizer = torch.optim.Adam(u1_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5, verbose=False
    )

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    history = []

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for x_batch, y_batch in loader:
            U1  = x_batch[:, 0:1]
            SOC = x_batch[:, 1:2]
            I   = x_batch[:, 2:3]
            u   = x_batch[:, 3:4]

            pred = u1_net(U1, SOC, I, u)
            loss = torch.mean((pred - y_batch) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"[Collocation] Epoch {epoch:4d} | "
                  f"MSE {avg_loss:.3e} | lr {cur_lr:.1e}")

    return history


# ╔══════════════════════════════════════════════════════════╗
# ║  DIAGNOSTICS                                             ║
# ╚══════════════════════════════════════════════════════════╝

def plot_collocation_fit(u1_net, inputs, targets, n_show=5000):
    """Check the NN vs targets after collocation pre-training."""
    idx = np.random.choice(len(inputs), min(n_show, len(inputs)), replace=False)
    x = inputs[idx]
    y = targets[idx].numpy().ravel()

    with torch.no_grad():
        pred = u1_net(x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4])
    pred = pred.numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Parity plot
    ax = axes[0]
    ax.scatter(y, pred, s=2, alpha=0.3)
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel('dU₁/dt  (data)')
    ax.set_ylabel('dU₁/dt  (NN)')
    ax.set_title('Collocation parity')
    ax.set_aspect('equal')

    # Residuals vs SOC
    ax = axes[1]
    soc = x[:, 1].numpy()
    ax.scatter(soc, pred - y, s=2, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('SOC')
    ax.set_ylabel('residual (pred − target)')
    ax.set_title('Residual vs SOC')

    fig.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════╗
# ║  PUTTING IT ALL TOGETHER                                 ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # ── Assumes `data`, `Q0`, `R0_func` are defined ──

    # 1. Build collocation data
    print("Building collocation targets...")
    inputs, targets = build_collocation_data(data, Q0, R0_func)
    print(f"  Collocation samples: {len(inputs)}")
    print(f"  dU1/dt range: [{targets.min():.4e}, {targets.max():.4e}]")

    # 2. Build model components
    Ue_lookup = UeLookup(data['soc'].values, data['Ue'].values)
    u1_net    = U1Net(n_hidden=32)
    ode_func  = BatteryODE(u1_net, Q0)
    model     = BatteryModel(ode_func, Ue_lookup, R0_func,
                             integrator='rk4')

    # 3. Stage 1: collocation pre-training
    print("\n=== Stage 1: Collocation pre-training ===")
    col_history = pretrain_collocation(
        u1_net, inputs, targets,
        n_epochs=500, lr=1e-3, print_every=50
    )

    fig = plot_collocation_fit(u1_net, inputs, targets)
    plt.show()

    # 4. Stage 2: ODE fine-tuning (optional but recommended)
    print("\n=== Stage 2: ODE fine-tuning ===")
    trajs = prepare_trajectories(data)
    split = int(len(trajs) * 0.8)
    train_trajs = trajs[:split]
    test_trajs  = trajs[split:]

    # Lower learning rate for fine-tuning — we're already close
    ode_history = train(
        model, train_trajs, test_trajs,
        n_epochs=200,         # fewer epochs needed after pre-training
        lr=1e-4,              # smaller lr to not destroy pre-trained weights
        print_every=25,
    )

    plot_history(ode_history)
    plot_predictions(model, train_trajs[:3], title_prefix='Train: ')
    plot_predictions(model, test_trajs[:3],  title_prefix='Test: ')
    plt.show()
