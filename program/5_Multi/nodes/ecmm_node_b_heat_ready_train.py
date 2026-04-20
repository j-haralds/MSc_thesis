# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM NODE — BATCHED VERSION – READY FOR HEATS
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
import importlib

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# FILE_PATH = os.getcwd()
print(FILE_PATH)
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()


from ecmm_node_b_heat_ready_lib import *
importlib.reload(sys.modules['ecmm_node_b_heat_ready_lib'])


# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, '2_merged_data.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
SAVE_FIGS   = True
SAVE_MODELS = True

Q0          = 17921.57581
TRAIN_SPLIT = 0.8
N_HIDDEN    = 32
EPOCHS      = 100
LR          = 1e-3
BATCH_SIZE  = 1        # Trajectories per batch

CONFIG = {
    'R1_mode': 'net',   # 'net' or 'const'
    'C1_mode': 'net',   # 'net' or 'const' or 'param'
    'R0_mode': 'net',  # 'net', 'func', or 'const'
    'n_hidden': N_HIDDEN,
        'R1_constrained': 'true', 'R1_min': 0.006, 'R1_max': 0.1,      # Ohm
        'C1_constrained': 'true', 'C1_min': 100.0, 'C1_max': 25000.0,  # F
        'R0_constrained': 'true', 'R0_min': 0.001, 'R0_max': 0.50,     # Ohm
}



# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
print(data.columns)
data['eta'] = -data['eta']
I_MAX = data['I'].max()

# TODO: Replace with existing GP 
_s, _u = data['soc'].values, data['Ue'].values
_i = np.argsort(_s)
Ue_interp = interp1d(_s[_i], _u[_i], kind='linear', fill_value='extrapolate')

F_first = data.groupby('u')['F'].first()
FORCE_CONST = abs(F_first / data.groupby('u')['u'].first()).values[0]  # GN/mm
print(f'Force constant: {100 * FORCE_CONST:.2f} GN/mm')

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")

# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES + ESTIMATE C1
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data, R0_func)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")

# C1_init = estimate_C1(train_trajs)
C1_init = 1500.0 # By hand
print(f"  C1 estimate: {C1_init:.0f} F")

# %% ══════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════

model  = BatteryECMM(CONFIG, Ue_interp, R0_func, Q0, C1_init=C1_init, I_ref=I_MAX, k=FORCE_CONST)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")

# %% ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

print(f"\nTraining ({EPOCHS} epochs, batch_size={BATCH_SIZE})...")
history = train_model(model, train_trajs, test_trajs,
                      n_epochs=EPOCHS, lr=LR,
                      batch_size=BATCH_SIZE, print_every=10)

C1_final = get_C1(model, scalar=True)
TOTAL_TIME = history['time']
print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")
print(f"\n  C1: {C1_init:.0f} → {C1_final:.0f} F")

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TRAIN
# ══════════════════════════════════════════════════════════════

# plot_predictions(model, CONFIG, train_trajs, 'Train: ')
# # plt.savefig('nodes_figs/ecm_node_train.pdf', bbox_inches='tight')
# plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════


if CONFIG.get('C1_constrained', 'false') == 'true' and CONFIG.get('R0_constrained', 'false') == 'false' and CONFIG.get('R1_constrained', 'false') == 'false':
    SAVE_NAME = f'{CONFIG["R0_mode"]}R0_{CONFIG["C1_mode"]}C1_Constr_{TOTAL_TIME:.4f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps'
elif CONFIG.get('R0_constrained', 'false') == 'true' and CONFIG.get('R1_constrained', 'false') == 'false' and CONFIG.get('C1_constrained', 'false') == 'false':
    SAVE_NAME = f'{CONFIG["R0_mode"]}R0_Constr_{CONFIG["C1_mode"]}C1_{TOTAL_TIME:.4f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps'
elif CONFIG.get('R1_constrained', 'false') == 'true' and CONFIG.get('R0_constrained', 'false') == 'true' and CONFIG.get('C1_constrained', 'false') == 'true':
    SAVE_NAME = f'{CONFIG["R0_mode"]}R0_Constr_{CONFIG["C1_mode"]}C1_Constr_{CONFIG["R1_mode"]}R1_Constr_{TOTAL_TIME:.4f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps'
else:
    SAVE_NAME = f'{CONFIG["R0_mode"]}R0_{CONFIG["C1_mode"]}C1_{TOTAL_TIME:.4f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps'
print(SAVE_NAME)

plot_predictions(model, CONFIG, test_trajs, time=False, title='Test: ')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_test_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════════

plot_loss(history)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_loss_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PARAMS
# ═════════════════════════════════════════════════════════════

plot_param(model, trajs, param='R0')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_R0_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(model, trajs, param='R1')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_R1_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(model, trajs, param='C1')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_C1_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PREDICTS
# ═════════════════════════════════════════════════════════════

sort = 'u_per'  # 'C_rate' or 'u_per'
plot_predicts(model, CONFIG, test_trajs, predict='F', sort=sort)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_F_{sort}_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

if SAVE_MODELS:
    torch.save({
        'model': model.state_dict(),
        'config': CONFIG,
        'history': history,
        'C1_init': C1_init,
        'C1_final': C1_final,
        'N_HIDDEN': N_HIDDEN,
        'EPOCHS': EPOCHS,
    }, os.path.join(MODEL_DIR, f'ecm_node_{SAVE_NAME}.pt'))

    print(f"Saved: ecm_node_{SAVE_NAME}.pt")
# %% ══════════════════════════════════════════════════════════
#  EXTRACT ECM PARAMETERS
# ══════════════════════════════════════════════════════════════

# soc_pts = [0.95, 0.80, 0.50, 0.20, 0.10, 0.05]
# ecm = extract_ecm_params(model, soc_pts, I_val=11.0, u_val=-0.06)

# print(f"ECM parameters at I=11A:")
# print(f"  C1 = {ecm['C1']:.0f} F")
# print(f"  {'SOC':>5s}  {'R0 mΩ':>7s}  {'R1 mΩ':>7s}  "
#       f"{'τ s':>7s}  {'U1ss V':>7s}")
# for i, s in enumerate(soc_pts):
#     print(f"  {s:5.2f}  {ecm['R0'][i]*1e3:7.2f}  "
#           f"{ecm['R1'][i]*1e3:7.2f}  {ecm['tau'][i]:7.0f}  "
#           f"{ecm['U1_ss'][i]:7.4f}")
# %%
