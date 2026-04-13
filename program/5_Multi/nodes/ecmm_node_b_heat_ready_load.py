# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM NODE — LOAD SAVED MODEL
# ══════════════════════════════════════════════════════════════
#
#  Mirror of ecmm_node_b_heat_ready_train.py, but instead of
#  training a fresh model it loads a checkpoint from MODEL_DIR
#  and reproduces the same plots / parameter tables.

import os
import sys
import glob
import torch
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# FILE_PATH = os.getcwd()
print(FILE_PATH)
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()

importlib.reload(sys.modules['ecmm_node_b_heat_ready_lib'])

from ecmm_node_b_heat_ready_lib import (
    prepare_data, BatteryECMM,
    plot_predictions, plot_R1_landscape, extract_ecm_params, R0_func, plot_loss, _scalar_C1
)


# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, '2_merged_data.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
MODEL_NAME  = 'ecm_node_51.2min_1b_32h_100eps.pt'
SAVE_FIGS   = False

Q0          = 17921.57581
TRAIN_SPLIT = 0.8

# Checkpoint
CKPT_FILE = os.path.join(MODEL_DIR, MODEL_NAME)
print(f"Checkpoint: {os.path.basename(CKPT_FILE)}")

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
#  PREPARE TRAJECTORIES (same split as training)
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data, R0_func)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")

# %% ══════════════════════════════════════════════════════════
#  REBUILD + LOAD MODEL
# ══════════════════════════════════════════════════════════════

ckpt     = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
C1_init  = ckpt['C1_init']
C1_final = ckpt['C1_final']
N_HIDDEN = ckpt['N_HIDDEN']
EPOCHS   = ckpt['EPOCHS']
history  = ckpt['history']
# config   = ckpt['config']
# print(f"Loaded checkpoint with config: {config}")
config = {
    'R1_mode': 'net',   # 'net' or 'const'
    'C1_mode': 'const',   # 'net' or 'const'
    'R0_mode': 'func',  # 'net', 'func', or 'const'
    'n_hidden': N_HIDDEN,
}

model = BatteryECMM(config, Ue_interp, R0_func, Q0,
                    C1_init=C1_init, I_ref=I_MAX, k=FORCE_CONST)
model.load_state_dict(ckpt['model'])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")
print(f"  C1: {C1_init:.0f} → {_scalar_C1(model):.0f} F  (saved final: {C1_final:.0f})")
print(f"  Trained {EPOCHS} epochs, {history.get('time', float('nan')):.1f} min total")

TOTAL_TIME = history.get('time', 0.0)
BATCH_SIZE = 1  # only used in filenames below

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TRAIN
# ══════════════════════════════════════════════════════════════

plot_predictions(model, train_trajs, 'Train: ')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

plot_predictions(model, test_trajs, 'Test: ')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_test_{TOTAL_TIME:.1f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps_loaded.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES (from saved history)
# ══════════════════════════════════════════════════════════════

plot_loss(history)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_loss_{TOTAL_TIME:.1f}min_{BATCH_SIZE}b_{N_HIDDEN}h_{EPOCHS}eps_loaded.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  R1 LANDSCAPE
# ══════════════════════════════════════════════════════════════

# I_vals = sorted(data['I'].unique())
# plot_R1_landscape(model, I_vals, u_val=float(data['u'].median()))
# plt.show()

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

# %%