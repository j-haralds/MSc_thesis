# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM NODE — LOAD SAVED MODEL
# ══════════════════════════════════════════════════════════════
#
#  Mirror of ecmm_node_staged_clean_train.py, but instead of
#  training a fresh model it loads a checkpoint from MODEL_DIR
#  and reproduces the same plots / parameter tables.

import os
import sys
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

# --- Import library (reload-safe for repeated cell runs in Jupyter) ---
import ecmm_node_cleaner_lib as _lib
importlib.reload(_lib)
from ecmm_node_cleaner_lib import *


# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/merged_DC_hyper.txt')
PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulses/merged_pulse_hyper.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
MODEL_NAME  = 'ecm_node_0429_0924_staged_funcR0_unconstr_pulse_15.31min_32h_1000_70_S2b10eps.pt'
SAVE_FIGS   = False
SAVE_PULSE_FIGS = False

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
I_MAX = data['I'].max()

pulse_raw = pd.read_csv(PULSE_FILE, sep=';', comment='%')
print(pulse_raw.columns)

# TODO: Replace with existing GP
_s, _u = data['soc'].values, data['Ue'].values
_i = np.argsort(_s)
Ue_interp = interp1d(_s[_i], _u[_i], kind='linear', fill_value='extrapolate')

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")

# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES (same split as training)
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data, R0_func)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")

pulse_trajs = prepare_pulse_data(pulse_raw)
split_p = int(len(pulse_trajs) * TRAIN_SPLIT)
pulse_train, pulse_test = pulse_trajs[:split_p], pulse_trajs[split_p:]
print(f"  Pulse train: {len(pulse_train)} | Pulse test: {len(pulse_test)}")

# %% ══════════════════════════════════════════════════════════
#  REBUILD + LOAD MODEL
# ══════════════════════════════════════════════════════════════

ckpt    = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
CONFIG  = ckpt['config']
history = ckpt['history']

# Pull metadata with fallbacks so this script handles checkpoints saved by
# both the legacy training script (single 'EPOCHS' key) and the staged
# training script ('EPOCHS_STATIC' / '_DYNAMIC' / '_UNFREEZE', 'USE_PULSE').
N_HIDDEN        = ckpt.get('N_HIDDEN', CONFIG.get('n_hidden', 32))
C1_init         = ckpt.get('C1_init',  None)
C1_final        = ckpt.get('C1_final', None)
EPOCHS_STATIC   = ckpt.get('EPOCHS_STATIC',   ckpt.get('EPOCHS', 0))
EPOCHS_DYNAMIC  = ckpt.get('EPOCHS_DYNAMIC',  0)
EPOCHS_UNFREEZE = ckpt.get('EPOCHS_UNFREEZE', 0)
USE_PULSE       = ckpt.get('USE_PULSE',       False)

print(f"Loaded checkpoint with config: {CONFIG}")

model = BatteryECMM(CONFIG, Ue_interp, R0_func, Q0, I_ref=I_MAX)
model.load_state_dict(ckpt['model'])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")
if C1_init is not None and C1_final is not None:
    print(f"  C1: {C1_init:.0f} → {get_C1(model):.0f} F  (saved final: {C1_final:.0f})")
else:
    print(f"  C1: {get_C1(model):.0f} F")
print(f"  Stages: S1={EPOCHS_STATIC}, S2={EPOCHS_DYNAMIC}"
      f"{f', S2b={EPOCHS_UNFREEZE}' if EPOCHS_UNFREEZE else ''}"
      f"  | total time {history.get('time', float('nan')):.1f} min")

TOTAL_TIME = history.get('time', 0.0)

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST  (build save-name from CONFIG flags)
# ══════════════════════════════════════════════════════════════

constr_tags = []
if CONFIG.get('R0_constrained', 'false') == 'true': constr_tags.append('R0c')
if CONFIG.get('R1_constrained', 'false') == 'true': constr_tags.append('R1c')
if CONFIG.get('C1_constrained', 'false') == 'true': constr_tags.append('C1c')
constr = '_'.join(constr_tags) if constr_tags else 'unconstr'

SAVE_NAME = (f'staged_{CONFIG["R0_mode"]}R0_{constr}'
             f'_{"pulse" if USE_PULSE else "CC"}'
             f'_{TOTAL_TIME:.2f}min_{N_HIDDEN}h'
             f'_{EPOCHS_STATIC}_{EPOCHS_DYNAMIC}'
             f'{f"_S2b{EPOCHS_UNFREEZE}" if EPOCHS_UNFREEZE > 0 else ""}'
             f'eps')
print(SAVE_NAME)

plot_predictions(model, CONFIG, test_trajs, time=False, title='Test: ')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_test_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES (from saved history)
# ═════════════════════════════════════════════════════════════

plot_loss(history)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_loss_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
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
plot_param(model, trajs, param='k')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_k_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

plot_force(model, trajs)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_F_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
# DATA PARAMS
# ═════════════════════════════════════════════════════════════

# df = data_param(model, trajs)
# print(df.head())
# PARAM_DATA_DIR = os.path.join(FILE_PATH, '..', 'symbols/data')
# df.to_csv(os.path.join(PARAM_DATA_DIR, f'ecm_elements_{SAVE_NAME}.txt'), index=False)

# %% ══════════════════════════════════════════════════════════
# PLOT PULSES  (plot_predictions auto-detects pulse trajectories)
# ══════════════════════════════════════════════════════════════

plot_predictions(model, CONFIG, pulse_test, title='Pulse test: ',
                 n_show=min(4, len(pulse_test)))
if SAVE_PULSE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_pulse_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
plt.show()

# Numeric RMSE summary across the pulse test set
rmses = rmse_pulse(model, pulse_test)
print(f"Pulse test RMSE (V):  mean {np.mean(rmses):.4f} V | "
      f"median {np.median(rmses):.4f} V | max {np.max(rmses):.4f} V "
      f"({len(rmses)} trajs)")

# %% ══════════════════════════════════════════════════════════
# RMSE SCATTER — CC test trajectories
# ══════════════════════════════════════════════════════════════

rmse, I_list, u_list = [], [], []
for tr in test_trajs:
    out = predict_np(model, CONFIG, tr)
    rmse.append(float(np.sqrt(np.mean((out['V'] - tr['V'].numpy())**2))))
    I_list.append(tr['C'])
    u_list.append(tr['u_per'])

plt.figure()
plt.scatter(I_list, u_list, marker='o', c=rmse, cmap='copper', s=50)
plt.xlabel('C-rate [a.u.]')
plt.ylabel(r'$u$ [%]')
plt.colorbar(label=r'RMSE [V]')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_rmse_test_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
plt.show()

print(f'Mean RMSE for test trajectories: {np.mean(rmse):.3f} V')

# %% ══════════════════════════════════════════════════════════
# RMSE SCATTER — pulse trajectories (with one outlier removable)
# ══════════════════════════════════════════════════════════════

# rmse_p, I_p, u_p = [], [], []
# for tr in pulse_trajs:
#     out = predict_np(model, CONFIG, tr)
#     rmse_p.append(float(np.sqrt(np.mean((out['V'] - tr['V'].numpy())**2))))
#     I_p.append(tr['I_seq'].max().item() / Q0 * 3600)   # peak C-rate
#     u_p.append(tr['u'])
#
# plt.figure()
# plt.scatter(I_p, u_p, marker='o', c=rmse_p, cmap='copper', s=50)
# plt.xlabel('Peak C-rate [a.u.]')
# plt.ylabel(r'$u$ [a.u.]')
# plt.colorbar(label=r'RMSE [V]')
# plt.tight_layout()
# plt.show()

# %%
