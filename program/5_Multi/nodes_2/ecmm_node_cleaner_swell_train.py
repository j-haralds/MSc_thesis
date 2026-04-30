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


# --- Import library (reload-safe for repeated cell runs in Jupyter) ---
import ecmm_node_cleaner_swell_lib as _lib
importlib.reload(_lib)
from ecmm_node_cleaner_swell_lib import *

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

Q0          = 17921.57581
TRAIN_SPLIT = 0.8
N_HIDDEN          = 32
EPOCHS_STATIC     = 100   # Stage 1 : V static, train R1 and k
EPOCHS_DYNAMIC    = 2       # Stage 2 : V dynamic, train C1 and k (R1 frozen)
EPOCHS_UNFREEZE   = 2        # Stage 2b: V dynamic, R1 unfrozen (0 = skip)
LR_STATIC         = 1e-3
LR_DYNAMIC        = 1e-3
LR_UNFREEZE       = 1e-3     # smaller LR once R1 is being refined

# Use pulse trajectories for Stage 2 (and 2b).  Stage 1 always uses CC trajs.
USE_PULSE         = True

CONFIG = {
    'R1_mode': 'net',   # 'net'
    'C1_mode': 'net',   # 'net'
    'R0_mode': 'net',           # 'func', 'net', 'param', 'net_no_soc'
    'n_hidden': N_HIDDEN,
    'R1_constrained': 'true', 'R1_min': 0.005, 'R1_max': 0.2,      # Ohm
    'C1_constrained': 'false', 'C1_min': 500.0, 'C1_max': 50000.0,  # F
    'R0_constrained': 'true', 'R0_min': 0.008, 'R0_max': 0.015,    # Ohm
}



# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
print(data.columns)
I_MAX = data['I'].max()

# TODO: Replace with existing GP
_s, _u = data['soc'].values, data['Ue'].values
_i = np.argsort(_s)
Ue_interp = interp1d(_s[_i], _u[_i], kind='linear', fill_value='extrapolate')

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

# print(f"\nLoading pulse data from {os.path.basename(PULSE_FILE)} ...")
pulse_data = pd.read_csv(PULSE_FILE, sep=';', comment='%')
pulse_trajs = prepare_pulse_data(pulse_data)
split_p = int(len(pulse_trajs) * TRAIN_SPLIT)
pulse_train, pulse_test = pulse_trajs[:split_p], pulse_trajs[split_p:]
print(f"  Pulse train: {len(pulse_train)} | Pulse test: {len(pulse_test)} "
        f"(T per traj: {pulse_trajs[0]['T']})")


# %% ══════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════

bat_model = BatteryECMM(CONFIG, Ue_interp, R0_func, Q0, I_ref=I_MAX)

n_params = sum(p.numel() for p in bat_model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")

# %% ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

print(f"\nStaged training: S1={EPOCHS_STATIC}ep static, S2={EPOCHS_DYNAMIC}ep dynamic"
      f"{f', S2b={EPOCHS_UNFREEZE}ep R1-unfrozen' if EPOCHS_UNFREEZE > 0 else ''}"
      f"{' (pulse Stage 2)' if USE_PULSE else ''}")

def _post_stage1(bat_model, history):
    """Plot test predictions at the end of Stage 1, before Stage 2 starts.

    V_mode='static' so the plot reflects how Stage 1 was actually trained
    (U1 = I·R1, no C1) — and the C1 panel is omitted entirely since C1
    has not been trained yet.
    """
    plot_predictions(bat_model, CONFIG, test_trajs, time=False,
                     title='Post-Stage-1: ', V_mode='static')
    plt.suptitle(f'After Stage 1 ({history["stage1_epochs"]} static epochs) — '
                 f'C1 not yet trained, omitted', y=1.0)
    plt.show()

history = train_staged(bat_model, train_trajs, test_trajs,
                       n_epochs_static=EPOCHS_STATIC,
                       n_epochs_dynamic=EPOCHS_DYNAMIC,
                       lr_static=LR_STATIC,
                       lr_dynamic=LR_DYNAMIC,
                       print_every=1,
                       on_stage1_done=_post_stage1,
                       pulse_train_trajs=pulse_train,
                       pulse_test_trajs=pulse_test,
                       n_epochs_unfreeze=EPOCHS_UNFREEZE,
                       lr_unfreeze=LR_UNFREEZE)

C1_final = get_C1(bat_model, scalar=True)
TOTAL_TIME = history['time']
print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")


# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

# Build save name from active flags — much cleaner than the prior 6-branch chain.
constr_tags = []
if CONFIG.get('R0_constrained', 'false') == 'true': constr_tags.append('R0c')
if CONFIG.get('R1_constrained', 'false') == 'true': constr_tags.append('R1c')
if CONFIG.get('C1_constrained', 'false') == 'true': constr_tags.append('C1c')
constr = '_'.join(constr_tags) if constr_tags else 'unconstr'

SAVE_NAME = (f'staged_swelling'
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

# TODO: switch param to element
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
#  SAVE
# ═════════════════════════════════════════════════════════════

if SAVE_MODELS:
    torch.save({
        'model': bat_model.state_dict(),
        'config': CONFIG,
        'history': history,
        'C1_final': C1_final,
        'N_HIDDEN': N_HIDDEN,
        'EPOCHS_STATIC': EPOCHS_STATIC,
        'EPOCHS_DYNAMIC': EPOCHS_DYNAMIC,
        'EPOCHS_UNFREEZE': EPOCHS_UNFREEZE,
        'USE_PULSE': USE_PULSE,
    }, os.path.join(MODEL_DIR, f'ecm_node_{TIMESTAMP}_{SAVE_NAME}.pt'))

    print(f"Saved: ecm_node_{TIMESTAMP}_{SAVE_NAME}.pt")

# %% ══════════════════════════════════════════════════════════
# ELEMENT SAVER
# ═════════════════════════════════════════════════════════════

element_data = data_param(bat_model, trajs)
element_data.to_csv(os.path.join('..', 'sr/symbol_data', f'ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt'), index=False)
print(f"Saved element data: ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt")

