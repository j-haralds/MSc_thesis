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

from ecmm_node_b_heat_ready_lib import *


# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, '2_merged_data.txt')
PULSE_FILE  = os.path.join(DATA_DIR, 'data_pulse1.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
MODEL_NAME  = 'ecm_node_netR0_netC1_56.9min_1b_32h_100eps.pt'
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
data['eta'] = -data['eta']
I_MAX = data['I'].max()

pulse_raw = pd.read_csv(PULSE_FILE, sep=',', comment='%')
print(pulse_raw.columns)
pulse_raw['eta'] = -pulse_raw['eta']


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

def prepare_pulse_data(pulse_raw):
    pulse_trajs = []
    for _, grp in pulse_raw.sort_values(['trajectory', 't']).groupby('trajectory'):
        grp = grp.reset_index(drop=True)
        pulse_trajs.append(dict(
            I_seq = torch.tensor(grp['I'].values,   dtype=torch.float32),  # sequence!
            u     = float(grp['u'].iloc[0]),
            soc0  = float(grp['soc'].iloc[0]),
            T     = len(grp),
            t     = torch.tensor(grp['t'].values,   dtype=torch.float32),
            V     = torch.tensor(grp['V'].values,   dtype=torch.float32),
            F     = torch.tensor(grp['F'].values,   dtype=torch.float32),
            soc   = torch.tensor(grp['soc'].values, dtype=torch.float32)
        ))
    return pulse_trajs

pulse_trajs = prepare_pulse_data(pulse_raw)

# %% ══════════════════════════════════════════════════════════
#  REBUILD + LOAD MODEL
# ══════════════════════════════════════════════════════════════

ckpt     = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
C1_init  = ckpt['C1_init']
C1_final = ckpt['C1_final']
N_HIDDEN = ckpt['N_HIDDEN']
EPOCHS   = ckpt['EPOCHS']
history  = ckpt['history']
CONFIG   = ckpt['config']
# CONFIG = {
#     'R1_mode': 'net',   # 'net' or 'const'
#     'C1_mode': 'const',   # 'net' or 'const'
#     'R0_mode': 'func',  # 'net', 'func', or 'const'
#     'n_hidden': N_HIDDEN,
# }
print(f"Loaded checkpoint with config: {CONFIG}")

model = BatteryECMM(CONFIG, Ue_interp, R0_func, Q0,
                    C1_init=C1_init, I_ref=I_MAX, k=FORCE_CONST)
model.load_state_dict(ckpt['model'])
model.eval()


n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")
print(f"  C1: {C1_init:.0f} → {get_C1(model):.0f} F  (saved final: {C1_final:.0f})")
print(f"  Trained {EPOCHS} epochs, {history.get('time', float('nan')):.1f} min total")

TOTAL_TIME = history.get('time', 0.0)
BATCH_SIZE = 1  # only used in filenames below

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TRAIN
# ══════════════════════════════════════════════════════════════

# plot_predictions(model, CONFIG, train_trajs, 'Train: ')
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

plot_predictions(model, CONFIG, test_trajs, noise=False, noise_lvl=0.05, title='Test: ')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_test_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES (from saved history)
# ═════════════════════════════════════════════════════════════

# plot_loss(history)
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_loss_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
# plt.show()

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
# DATA PARAMS
# ═════════════════════════════════════════════════════════════

# Only when 'net' for all three elements
df = data_param(model, trajs)
print(df.head())

plt.plot(df['soc'][df['trajectory'] == 0], df['R0'][df['trajectory'] == 0], label='R0')


PARAM_DATA_DIR = os.path.join(FILE_PATH, '..', 'symbols/data')
df.to_csv(os.path.join(PARAM_DATA_DIR, f'ecm_elements_{SAVE_NAME}.txt'), index=False)

# %% ══════════════════════════════════════════════════════════
# Plot PULSES
# ══════════════════════════════════════════════════════════════

# plot_predictions_pulse(model, pulse_trajs, time=True, noise=True, noise_lvl=0.01)
# if SAVE_PULSE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_pulse_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')
# plt.show()

# rmse_pulses = rmse_pulse(model, pulse_trajs)

# %% ══════════════════════════════════════════════════════════

# rmse_pulses = rmse_pulses[:19] + rmse_pulses[20:]  # Remove spec 20, which is very high and messes up the plot
# # plt.hist(rmse_pulses, bins=50, color=COLORS[0], alpha=0.7);

# # plot_predictions_pulse(model, pulse_trajs, time=True, n_show=1, spec=19)

# rmse = []
# I = []
# u = []
# for i,tr in enumerate(pulse_trajs):
#     V, soc, U1, R1, Fs, Fr, ks_pred, C1_t = predict_pulse_np(model, tr['I_seq'], tr['u'], tr['soc0'], tr['T'])
#     rmse.append(np.sqrt(np.mean((V - tr['V'].numpy())**2)))
#     I.append(tr['I_seq'].max() / Q0 * 3600)
#     u.append(tr['u'])


# rmse = rmse[:19] + rmse[20:]  # Remove spec 20, which is very high and messes up the plot
# I = I[:19] + I[20:]
# u = u[:19] + u[20:]
# plt.scatter(I,u, marker='o', c = rmse, cmap='copper',s = 50)
# plt.xlabel('C-rate [Ah]')
# plt.ylabel(r'$\Delta u$ [a.u.]')
# plt.colorbar(label = r'RMSE [V]')


# %% =══════════════════════════════════════════════════════════
rmse = []
I = []
u = []
for i,tr in enumerate(test_trajs):
    V, soc_np, U1, R1, Fs, Fr, ks, C1, R0 = predict_np(model, CONFIG, tr['I'], tr['u'], tr['soc0'], tr['T'])
    rmse.append(np.sqrt(np.mean((V - tr['V'].numpy())**2)))
    I.append(tr['C'])
    u.append(tr['u_per'])


rmse = rmse[:19] + rmse[20:]  # Remove spec 20, which is very high and messes up the plot
I = I[:19] + I[20:]
u = u[:19] + u[20:]
plt.scatter(I,u, marker='o', c = rmse, cmap='copper',s = 50)
plt.xlabel('C-rate [a.u.]')
plt.ylabel(r'$\Delta u$ [a.u.]')
plt.colorbar(label = r'RMSE [V]')
plt.tight_layout()
# plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_rmse_train_{SAVE_NAME}_loaded.pdf'), bbox_inches='tight')

print(f'Mean RMSE for test trajectories: {np.mean(rmse):.3f} V')

# %% ══════════════════════════════════════════════════════════
# NOISY
# ══════════════════════════════════════════════════════════════

# plot_noisy_inputs(trajs, noise_lvl=0.01)

# plt.show()

# %% ══════════════════════════════════════════════════════════
# NOISE ROBUSTNESS
# ══════════════════════════════════════════════════════════════

# %%