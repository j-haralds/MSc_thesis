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
import ecmm_vnode_vstatic_snode_sstatic_lib_6 as _lib
importlib.reload(_lib)
from ecmm_vnode_vstatic_snode_sstatic_lib_6 import *

from datetime import datetime
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')




# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '..', 'Multi_data')
DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/merged_DC_hyper.txt')
ALL_DATA =  os.path.join(DATA_DIR, 'merged_combo.txt')
HALF_COMBO = os.path.join(DATA_DIR, 'combo_half.txt'); OTHER_HALF_COMBO = os.path.join(DATA_DIR, 'combo_other_half.txt')
PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulses/merged_pulse_hyper.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
#MODEL_DIR   = os.path.join(FILE_PATH, 'final_models')
SAVE_FIGS   = True
SAVE_MODELS = False
SAVE_DATA   = False

MODEL_NAME= '0508_2228_DC_DC_V-dynamic_F-dynamic_436.43min_16h_650eps.pt'

Q0          = 17921.57581



# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
print(data.columns)

RMSE_scales = rmse_scale(pd.read_csv(ALL_DATA, sep=';', comment='%'))

    
I_MAX = data['I'].max()
U_MAX = abs(data['u'].min())
L_CELL = 14.37325   #-(data['u'] / (data['u_par']/100))[0]
print(f'Cell lengths: {L_CELL:.5f} 1e-5m | I max: {I_MAX:.4f} A | u max: {U_MAX:.4f} 1e-5m')

# Ue(SOC) is now provided by the module-level GP (loaded lazily inside the
# library from JN_GP) — no Ue_interp construction needed here anymore.

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")

# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES + ESTIMATE C1
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data)
print(trajs[0].keys())
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")


# %% ══════════════════════════════════════════════════════════
#  PREPARE PULSE TRAJECTORIES  (for Stage 2 / 2b)
# ══════════════════════════════════════════════════════════════

pulse_data = pd.read_csv(PULSE_FILE, sep=';', comment='%')
pulse_trajs = prepare_pulse_data(pulse_data)
print(pulse_trajs[0].keys())
split_p = int(len(pulse_trajs) * TRAIN_SPLIT)
pulse_train, pulse_test = pulse_trajs[:split_p], pulse_trajs[split_p:]
print(f"  Pulse train: {len(pulse_train)} | Pulse test: {len(pulse_test)} "
        f"(T per traj: {pulse_trajs[0]['T']})")

# %% ══════════════════════════════════════════════════════════
# PREPARE COMBO TRAJECTORIES
# ══════════════════════════════════════════════════════════════

combo_data = pd.read_csv(OTHER_HALF_COMBO, sep=';', comment='%')
combo_trajs = prepare_pulse_data(combo_data)
print(combo_trajs[0].keys())
split_c = int(len(combo_trajs) * TRAIN_SPLIT)
combo_train, combo_test = combo_trajs[:split_c], combo_trajs[split_c:]
print(f"  Combo train: {len(combo_train)} | Combo test: {len(combo_test)}")



# %% ══════════════════════════════════════════════════════════
#  LOAD MODEL  (no Ue argument — GP loaded internally by lib)
# ══════════════════════════════════════════════════════════════


# Pass I_ref explicitly because older checkpoints don't have it stored.
# New checkpoints saved by the updated train script will carry it, in which
# case the I_ref argument can be omitted.
bat_model, ckpt = load_nn_model(MODEL_NAME, I_ref=I_MAX)   # U_MAX when loading lib_3
history, CONFIG, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE = load_checkpoint(ckpt)

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
#rmseV,rmseF = rmse_pulse(bat_model, pulse_trajs)
#print(f"\nPulse test RMSE (V):  mean {np.mean(rmses):.4f} V | "
#        f"median {np.median(rmses):.4f} V | max {np.max(rmses):.4f} V "
#        f"({len(rmses)} trajs)")

# %% ══════════════════════════════════════════════════════════
# ELEMENT SAVER
# ═════════════════════════════════════════════════════════════

element_data = data_param(bat_model, trajs)
if SAVE_DATA:
    element_data.to_csv(os.path.join('..', 'sr/symbol_data', f'ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt'), index=False)
    print(f"Saved element data: ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt")



# %% ══════════════════════════════════════════════════════════
# PLOT PREDICTIONS 
# ══════════════════════════════════════════════════════════════

import ecmm_vnode_vstatic_snode_sstatic_lib_6 as _lib
importlib.reload(_lib)
from ecmm_vnode_vstatic_snode_sstatic_lib_6 import *


plot_report(bat_model, CONFIG, test_trajs, title='Pulse test: ',
                 n_show=min(2, len(pulse_test)), time = True)
plt.show()

#plt.savefig(os.path.join(FIGS_DIR, f'static_VF_pulse.pdf'), bbox_inches='tight')

# %% ══════════════════════════════════════════════════════════
# INPUT ERROR MAP
# ══════════════════════════════════════════════════════════════

# import ecmm_vnode_vstatic_snode_sstatic_lib_6 as _lib
# importlib.reload(_lib)
# from ecmm_vnode_vstatic_snode_sstatic_lib_6 import *

# input_map(bat_model, test_trajs,rmse_scales=RMSE_scales)
# print(rmse_pulse(bat_model, pulse_trajs)[0].mean()/RMSE_scales['V'], rmse_pulse(bat_model, pulse_trajs)[1].mean()/RMSE_scales['F'])
# plt.show()



# %% ══════════════════════════════════════════════════════════
# INPUT ERROR MAPS COMPARISON
# ══════════════════════════════════════════════════════════════



# %% ══════════════════════════════════════════════════════════
# DISCHARGE DIFFERENT C-RATES
# ══════════════════════════════════════════════════════════════

import ecmm_vnode_vstatic_snode_sstatic_lib_6 as _lib
importlib.reload(_lib)
from ecmm_vnode_vstatic_snode_sstatic_lib_6 import *

MODEL_NAME_STAT = '0508_1444_snode_DC_V-static_no_R0_F-static_netR0_R0c_R1c_C1c_2.97min_16h_650eps_0stat_0dyneps.pt'
MODEL_NAME_DYNA = '0508_2228_DC_DC_V-dynamic_F-dynamic_436.43min_16h_650eps.pt'
MODEL_NAME_FULL = '0509_0102_combo_full_combo_V-dynamic_F-dynamic_359.65min_16h_650eps.pt'

bat_model_static_DC, ckpt_stat = load_nn_model(MODEL_NAME_STAT, I_ref=I_MAX)
bat_model_dynamic_DC, ckpt_dyna = load_nn_model(MODEL_NAME_DYNA, I_ref=I_MAX)
bat_model_full, ckpt_full = load_nn_model(MODEL_NAME_FULL, I_ref=I_MAX)
plot_predicts_report(bat_model_dynamic_DC, CONFIG, test_trajs, predict='V', sort='C_rate', n_show=5, time=False)
# plot_predicts_report(bat_model_dynamic_DC, CONFIG, pulse_test, predict='V', sort='C_rate', n_show=3, time=True)

plot_param(bat_model_full, test_trajs, param='R0')
plot_param(bat_model_full, test_trajs, param='R1')
plot_param(bat_model_full, test_trajs, param='C1')
plot_param(bat_model_full, test_trajs, param='tau')

plot_predicts_report(bat_model_full, CONFIG, test_trajs, predict='F', sort='u_per', n_show=10, time=False)
plot_force_report(bat_model_full, CONFIG, test_trajs, n_show=3)
plt.show()






















# %% ══════════════════════════════════════════════════════════
# MODEL COMPARISON AVGNRMSE
# ══════════════════════════════════════════════════════════════
import ecmm_vnode_vstatic_snode_sstatic_lib_6 as _lib
importlib.reload(_lib)
from ecmm_vnode_vstatic_snode_sstatic_lib_6 import *
# ========================================================================









# ========================================================================










# ========================================================================

MODEL_NAME_STAT = '0508_1444_snode_DC_V-static_no_R0_F-static_netR0_R0c_R1c_C1c_2.97min_16h_650eps_0stat_0dyneps.pt'
MODEL_NAME_DYNA = '0508_2228_DC_DC_V-dynamic_F-dynamic_436.43min_16h_650eps.pt'

bat_model_static_DC, ckpt_stat = load_nn_model(MODEL_NAME_STAT, I_ref=I_MAX)
bat_model_dynamic_DC, ckpt_dyna = load_nn_model(MODEL_NAME_DYNA, I_ref=I_MAX)

# USE FOR REPORT. COMPARISON BAR PLOT STATIC AND DYNAMIC TRAINED ON CC
plot_nrmse_bars(models = {'Static':  bat_model_static_DC,
                        'Dynamic': bat_model_dynamic_DC},
    trajs_by_set = {'CC': test_trajs, 'Pulse': pulse_test},
    rmse_scales  = RMSE_scales,
ECM_fix = True, metric_names = ['Force'])
plt.savefig(os.path.join(FIGS_DIR, f'0508_2228dyna_0508_1444stat_CCtrained_nrmse_comparison.pdf'), bbox_inches='tight')
plt.show()
# %%
# USE FOR REPORT. STATIC TRAINED ON CC
#

history, CONFIG, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE = load_checkpoint(ckpt_stat)
plot_report(bat_model_static_DC, CONFIG, test_trajs, title='CC test: ',
                 n_show=min(2, len(pulse_test)), time = True)
plt.savefig(os.path.join(FIGS_DIR, f'0508_1444_static_ccPred.pdf'), bbox_inches='tight')
plt.show()
plot_report(bat_model_static_DC, CONFIG, pulse_test, title='Pulse test: ',
                 n_show=min(2, len(pulse_test)), time = True)
plt.savefig(os.path.join(FIGS_DIR, f'0508_1444_static_pulsePred.pdf'), bbox_inches='tight')
plt.show()

# %%
# USE FOR REPORT. DYNAMIC TRAINED ON CC
#

history, CONFIG, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE = load_checkpoint(ckpt_dyna)
plot_report(bat_model_dynamic_DC, CONFIG, test_trajs, title='CC test: ',
                 n_show=min(2, len(pulse_test)), time = True)
plt.savefig(os.path.join(FIGS_DIR, f'0508_2228_dynamic_ccPred.pdf'), bbox_inches='tight')
plt.show()
plot_report(bat_model_dynamic_DC, CONFIG, pulse_test, title='Pulse test: ',
                 n_show=min(2, len(pulse_test)), time = True)
plt.savefig(os.path.join(FIGS_DIR, f'0508_2228_dynamic_pulsePred.pdf'), bbox_inches='tight')
plt.show()

# %%
# USE FOR REPORT. INPUT ERROR MAP 
# 
 
MODEL_NAME_LOW = '0509_0103_combo_low_c_d_combo_V-dynamic_F-dynamic_netR0_R0c_R1c_C1c_428.58min_16h_650eps_0stat_0dyneps.pt'
MODEL_NAME_HIGH = '0509_0102_combo_full_combo_V-dynamic_F-dynamic_359.65min_16h_650eps.pt'


bat_model_low, ckpt_low = load_nn_model(MODEL_NAME_LOW, I_ref=I_MAX)   # U_MAX when loading lib_3
history_low, CONFIG, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE = load_checkpoint(ckpt_low)

bat_model_high, ckpt_high = load_nn_model(MODEL_NAME_HIGH, I_ref=I_MAX)   # U_MAX when loading lib_3
history_high, CONFIG, N_HIDDEN, EPOCHS_STATIC, EPOCHS_DYNAMIC, EPOCHS_UNFREEZE = load_checkpoint(ckpt_high)

fig,ax = input_map_comparison(bat_model_low, bat_model_high,combo_trajs  , rmse_scales=RMSE_scales)
plt.show()
 