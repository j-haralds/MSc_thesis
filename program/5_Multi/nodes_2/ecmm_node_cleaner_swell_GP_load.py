# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM NODE — with kdot=NN
# ══════════════════════════════════════════════════════════════

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as _time
import importlib

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# FILE_PATH = os.getcwd()
print(FILE_PATH)
sys.path.append(os.path.join(FILE_PATH, '..', '..'))    # Up two steps
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()


# --- Import library (reload-safe for repeated cell runs in Jupyter) ---
import ecmm_node_cleaner_swell_GP_lib as _lib
importlib.reload(_lib)
from ecmm_node_cleaner_swell_GP_lib import *

from datetime import datetime
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')




# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/merged_DC_hyper.txt')
PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulses/merged_pulse_hyper.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
SAVE_FIGS   = False
SAVE_MODELS = False
SAVE_DATA   = False

MODEL_NAME= 'ecm_node_0430_1313_staged_swelling_netR0_25.28min_32h_2000_100_S2b20eps.pt'

Q0          = 17921.57581



# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
print(data.columns)
I_MAX = data['I'].max()
L_CELL = -(data['u'] / (data['u_par']/100))[0]
print(f'Cell lengths: {L_CELL:.5f} 1e-5m | I max: {I_MAX:.2f} A')

# Ue(SOC) is now provided by the module-level GP (loaded lazily inside the
# library from JN_GP) — no Ue_interp construction needed here anymore.

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")

# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES + ESTIMATE C1
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data, R0_func)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")


# %% ══════════════════════════════════════════════════════════
#  PREPARE PULSE TRAJECTORIES  (for Stage 2 / 2b)
# ══════════════════════════════════════════════════════════════

pulse_data = pd.read_csv(PULSE_FILE, sep=';', comment='%')
pulse_trajs = prepare_pulse_data(pulse_data)
split_p = int(len(pulse_trajs) * TRAIN_SPLIT)
pulse_train, pulse_test = pulse_trajs[:split_p], pulse_trajs[split_p:]
print(f"  Pulse train: {len(pulse_train)} | Pulse test: {len(pulse_test)} "
        f"(T per traj: {pulse_trajs[0]['T']})")



# %% ══════════════════════════════════════════════════════════
#  LOAD MODEL  (no Ue argument — GP loaded internally by lib)
# ══════════════════════════════════════════════════════════════


# Pass I_ref explicitly because older checkpoints don't have it stored.
# New checkpoints saved by the updated train script will carry it, in which
# case the I_ref argument can be omitted.
bat_model, ckpt = load_nn_model(MODEL_NAME, I_ref=I_MAX)
history, CONFIG, C1_final, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE = load_checkpoint(ckpt)

n_params = sum(p.numel() for p in bat_model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")



# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

# Build save name from active flags — much cleaner than the prior 6-branch chain.
constr_tags = []
if CONFIG.get('R0_constrained', 'false') == 'true': constr_tags.append('R0c')
if CONFIG.get('R1_constrained', 'false') == 'true': constr_tags.append('R1c')
if CONFIG.get('C1_constrained', 'false') == 'true': constr_tags.append('C1c')
constr = '_'.join(constr_tags) if constr_tags else 'unconstr'

TOTAL_TIME = history.get('time', 0.0)

SAVE_NAME = (f'staged_swelling'
             f'{CONFIG["R0_mode"]}R0_{constr}'
             f'_{TOTAL_TIME:.2f}min_{N_HIDDEN}h'
             f'_{EPOCHS_STATIC}_{EPOCHS_DYNAMIC}'
             f'{f"_S2b{EPOCHS_UNFREEZE}" if EPOCHS_UNFREEZE > 0 else ""}'
             f'eps')
print(SAVE_NAME)

plot_predictions(bat_model, CONFIG, test_trajs, time=False, title='Test: ')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_test_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════════

plot_loss(history)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_loss_{TIMESTAMP}_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PARAMS
# ═════════════════════════════════════════════════════════════

plot_param(bat_model, test_trajs, param='R0')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_R0_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='R1')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_R1_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='C1')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_C1_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='k')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_k_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='s')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_s_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()



plot_force(bat_model, test_trajs)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_F_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_swelling(bat_model, test_trajs)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_su_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PREDICTS
# ═════════════════════════════════════════════════════════════

sort = 'u_per'  # 'C_rate' or 'u_per'
plot_predicts(bat_model, CONFIG, test_trajs, predict='F', sort=sort)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_F_{sort}_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — PULSE TEST
# ══════════════════════════════════════════════════════════════

# plot_predictions auto-detects pulse trajectories (they carry 'I_seq');
plot_predictions(bat_model, CONFIG, pulse_test, title='Pulse test: ',
                 n_show=min(3, len(pulse_test)))
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_pulse_{SAVE_NAME}.pdf'),
                bbox_inches='tight')
plt.show()

# Numeric RMSE summary across the pulse test set
rmses = rmse_pulse(bat_model, pulse_test)
print(f"\nPulse test RMSE (V):  mean {np.mean(rmses):.4f} V | "
        f"median {np.median(rmses):.4f} V | max {np.max(rmses):.4f} V "
        f"({len(rmses)} trajs)")

# %% ══════════════════════════════════════════════════════════
# ELEMENT SAVER
# ═════════════════════════════════════════════════════════════

element_data = data_param(bat_model, trajs)
if SAVE_DATA:
    element_data.to_csv(os.path.join('..', 'sr/symbol_data', f'ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt'), index=False)
    print(f"Saved element data: ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt")



# %% ══════════════════════════════════════════════════════════
