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
import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6_full as _lib
importlib.reload(_lib)
from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6_full import *

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
SAVE_FIGS   = False
SAVE_MODELS = False
SAVE_DATA   = False

MODEL_NAME= '0516_1937_FULL_b4_integration_softplus_combo_V-dynamic_F-dynamic_unconstr_685.96min_16h_2500eps.pt'

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

combo_data = pd.read_csv(HALF_COMBO, sep=';', comment='%')
combo_trajs = prepare_pulse_data(combo_data)
print(combo_trajs[0].keys())
split_c = int(len(combo_trajs) * TRAIN_SPLIT)
combo_train, combo_test = combo_trajs[:split_c], combo_trajs[split_c:]
print(f"  Combo train: {len(combo_train)} | Combo test: {len(combo_test)}")


other_combo_data = pd.read_csv(OTHER_HALF_COMBO, sep=';', comment='%')
other_combo_trajs = prepare_pulse_data(other_combo_data)


# %% ══════════════════════════════════════════════════════════
#  LOAD MODEL  (no Ue argument — GP loaded internally by lib)
# ══════════════════════════════════════════════════════════════


# Pass I_ref explicitly because older checkpoints don't have it stored.
# New checkpoints saved by the updated train script will carry it, in which
# case the I_ref argument can be omitted.
bat_model, ckpt = load_nn_model(MODEL_NAME, I_ref=I_MAX)   # U_MAX when loading lib_3
history, CONFIG, N_HIDDEN, EPOCHS = load_checkpoint(ckpt)

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

SAVE_NAME = (f'_{MODEL_NAME}_')
print(SAVE_NAME)

# plot_predictions(bat_model, CONFIG, test_trajs, time=False, title='Test: ')
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_test_{SAVE_NAME}.pdf'), bbox_inches='tight')
#     print('Saved figure')
# plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════════

plot_loss(history)

# plt.savefig(os.path.join(FIGS_DIR, f'loss_{MODEL_NAME}.pdf'), bbox_inches='tight')
# print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PARAMS
# ═════════════════════════════════════════════════════════════

# plot_param(bat_model, test_trajs, param='R0')
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_R0_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plot_param(bat_model, test_trajs, param='R1')
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_R1_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plot_param(bat_model, test_trajs, param='C1')
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_C1_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plot_param(bat_model, test_trajs, param='k')
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_k_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plot_param(bat_model, test_trajs, param='s')
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_s_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plt.show()



# plot_force(bat_model, test_trajs)
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_F_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plot_swelling(bat_model, test_trajs)
# if SAVE_FIGS:
#     plt.savefig(os.path.join(FIGS_DIR, f'ecmm_node_su_{SAVE_NAME}.pdf'), bbox_inches='tight')
# plt.show()

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

#import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 as _lib
#importlib.reload(_lib)
#from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 import *

# element_data = data_param(bat_model, trajs)
# if SAVE_DATA:
#     element_data.to_csv(os.path.join('..', 'sr/symbol_data', f'ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt'), index=False)
#     print(f"Saved element data: ecm_node_elements_{TIMESTAMP}_{SAVE_NAME}.txt")


def R0_nn(c_rate, u_per, soc):
    return element_predict(bat_model, c_rate, u_per, soc, element='R0')

def R1_nn(c_rate, u_per, soc):
    return element_predict(bat_model, c_rate, u_per, soc, element='R1')

def C1_nn(c_rate, u_per, soc):
    return element_predict(bat_model, c_rate, u_per, soc, element='C1')

def k_nn(c_rate, u_per, soc):
    return element_predict(bat_model, c_rate, u_per, soc, element='k')

# def s_nn(c_rate, u_per, soc):
#     return element_predict(bat_model, c_rate, u_per, soc, element='s')

# def sdot_nn(c_rate, u_per, soc):
#     return element_predict(bat_model, c_rate, u_per, soc, element='sdot')

# for c_rate in [0.5, 1, 2, 3, 4, 5]:
#     plt.plot(np.linspace(1, 0, 1000), sdot_nn(c_rate, 10, np.linspace(1, 0, 1000)), label=f'sdot ({c_rate} C)')
# plt.gca().invert_xaxis()

# plot_param(bat_model, test_trajs, param='C1')
# plt.show()


# %% ══════════════════════════════════════════════════════════
# PLOT PREDICTIONS 
# ══════════════════════════════════════════════════════════════

#import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 as _lib
#importlib.reload(_lib)
#from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 import *


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

#import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 as _lib
#importlib.reload(_lib)
#from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 import *

MODEL_NAME_STAT = '0508_1444_snode_DC_V-static_no_R0_F-static_netR0_R0c_R1c_C1c_2.97min_16h_650eps_0stat_0dyneps.pt'
MODEL_NAME_DYNA = '0508_2228_DC_DC_V-dynamic_F-dynamic_436.43min_16h_650eps.pt'
MODEL_NAME_FULL = '0510_2034_b4_combo_full_combo_V-dynamic_F-dynamic_642.62min_16h_2500eps.pt'

# bat_model_static_DC, ckpt_stat = load_nn_model(MODEL_NAME_STAT, I_ref=I_MAX)
# bat_model_dynamic_DC, ckpt_dyna = load_nn_model(MODEL_NAME_DYNA, I_ref=I_MAX)
bat_model_full, ckpt_full = load_nn_model(MODEL_NAME_FULL, I_ref=I_MAX)
# plot_predicts_report(bat_model_dynamic_DC, CONFIG, test_trajs, predict='V', sort='C_rate', n_show=5, time=False)
plot_predicts_report(bat_model_full, CONFIG, pulse_test, predict='V', sort='C_rate', n_show=4, time=True, pulse=True)


# plot_predicts_report(bat_model_full, CONFIG, test_trajs, predict='F', sort='u_per', n_show=10, time=False)
plot_force_report(bat_model_full, CONFIG, test_trajs, n_show=3)
# plt.savefig(os.path.join(FIGS_DIR, f'Fd.pdf'), bbox_inches='tight')
plt.show()


# %% SCARCITY PLOT

#import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 as _lib
#importlib.reload(_lib)
#from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 import *

def plot_data_scarcity_loss(names, trajs):
    nrmsesV = []
    nrmsesF = []
    nrmseV_fix = []
    rmse_scale = RMSE_scales['V']
    fracs = [2, 10, 20, 40,60, 80, 100]
    rmseV_fix = rmse_fix(trajs, RMSE_scales)
    for name in names:
        model, ckpt = load_nn_model(name, I_ref=I_MAX)
        rmseV, rmseF, _, _ = rmse_pulse(model, trajs)
        #rmseV_fix = rmse_fix(trajs, rmse_scale).mean()
        nrmseV = rmseV.mean() / RMSE_scales['V']
        nrmseF = rmseF.mean() / RMSE_scales['F']
        print(f"{name} | Pulse test nRMSE (V): {nrmseV:.4f} | nRMSE (F): {nrmseF:.4f}")
        nrmsesV.append(nrmseV)
        nrmsesF.append(nrmseF)

    fig = plt.figure(figsize=(6,4))
    plt.semilogy(fracs, nrmsesV, label='Voltage NRMSE', marker='o', color = plt.cm.Blues(0.95))
    plt.plot(fracs, rmseV_fix.mean()*np.ones_like(fracs), label='Voltage NRMSE (fixed ECM)', color = plt.cm.Blues(0.95), linestyle='--')
    plt.semilogy(fracs, nrmsesF, label='Force NRMSE', marker='s', color = plt.cm.Reds(0.9))
    #plt.semilogy(nrmseV_fix, label='Voltage NRMSE (fixed ECM)', color = 'gray', linestyle='--')

    plt.xlabel(r'Percentage of training data [\%]')
    plt.ylabel('NRMSE')
    plt.legend()
    plt.grid(True)
    return fig



def plot_data_scarcity_loss_volt(names, trajs, batch_num = 1, batch_names = None):

    fracs = [20, 40,60, 80, 100]
    colors = [plt.cm.Blues(0.95), plt.cm.Reds(0.9)]
    nrmses_V = np.zeros((len(fracs),batch_num))
    fig = plt.figure(figsize=(6,4))
    
    for batch in range(batch_num):
        for j in range(len(names[batch])):
            name = names[batch][j]
            model, ckpt = load_nn_model(name, I_ref=I_MAX)
            rmseV, _, _, _ = rmse_pulse(model, trajs)
     
            nrmseV = rmseV.mean() / RMSE_scales['V']
     
            nrmses_V[j,batch] = nrmseV

        plt.semilogy(fracs, nrmses_V[:,batch], label=f'Voltage NRMSE {batch_names[batch]}', marker='o', color = colors[batch])

    #plt.plot(fracs, rmseV_fix.mean()*np.ones_like(fracs), label='Voltage NRMSE (fixed ECM)', color = plt.cm.Blues(0.95), linestyle='--')
    #plt.semilogy(nrmseV_fix, label='Voltage NRMSE (fixed ECM)', color = 'gray', linestyle='--')

    plt.xlabel(r'Percentage of training data [\%]')
    plt.ylabel('NRMSE [a.u.]')
    plt.legend()
    plt.grid(True)
    return fig

names = [
         #'0512_1640_b4_combo_0.02__combo_V-dynamic_F-dynamic_R0c_C1c_27.13min_16h_2500eps.pt',
         #'0512_1459_b4_combo_0.1__combo_V-dynamic_F-dynamic_R0c_C1c_94.18min_16h_2500eps.pt',
         '0511_1252_b4_combo_0.2__combo_V-dynamic_F-dynamic_246.31min_16h_2500eps.pt',
         '0511_1249_b4_combo_0.4__combo_V-dynamic_F-dynamic_280.31min_16h_2500eps.pt',
         '0512_2326_b4_combo_0.6__combo_V-dynamic_F-dynamic_R0c_C1c_379.07min_16h_2500eps.pt',
         '0511_2138_b4_combo_0.8__combo_V-dynamic_F-dynamic_R0c_C1c_453.64min_16h_2500eps.pt',
         '0510_2034_b4_combo_full_combo_V-dynamic_F-dynamic_642.62min_16h_2500eps.pt',
         ]
BB_names = [
            '0516_1128_20_percent_b4_combo_combo_V-back_in_black_F-static_R0c_C1c_4.29min_16h_2500eps.pt',
            '0516_1120_40_percent_b4_combo_combo_V-back_in_black_F-static_R0c_C1c_6.12min_16h_2500eps.pt',
            '0516_1106_60_percent_b4_combo_combo_V-back_in_black_F-static_R0c_C1c_8.11min_16h_2500eps.pt',
            '0516_1049_80_percent_b4_combo_combo_V-back_in_black_F-static_R0c_C1c_9.54min_16h_2500eps.pt',
            '0516_1619_100_percent_b4_combo_V-back_in_black_F-static_unconstr_13.52min_16h_2500eps.pt',
            ]

name_batch = [names, BB_names]
print(name_batch[1])
plot_data_scarcity_loss_volt(name_batch, other_combo_trajs, batch_num = 2,batch_names=['NODE', 'Black-Box'])
plt.savefig(os.path.join(FIGS_DIR, f'data_scarcity_no_early_stop_BB_comp.pdf'), bbox_inches='tight')

plt.show()
##











# %% ══════════════════════════════════════════════════════════
# MODEL COMPARISON AVGNRMSE
# ══════════════════════════════════════════════════════════════
# import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 as _lib
# importlib.reload(_lib)
# from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6 import *

# dataparammasss = data_param(bat_model_full, test_trajs)
# ========================================================================









# ========================================================================










# %% ========================================================================

MODEL_NAME_STAT = '0513_1037_b4_combo_1.0__DC_V-static_no_R0_F-static_R0c_C1c_3.04min_16h_500eps.pt'
MODEL_NAME_DYNA = '0513_1037_b4_combo_1__DC_V-dynamic_F-dynamic_R0c_C1c_105.62min_16h_500eps.pt'

bat_model_static_DC, ckpt_stat = load_nn_model(MODEL_NAME_STAT, I_ref=I_MAX)
bat_model_dynamic_DC, ckpt_dyna = load_nn_model(MODEL_NAME_DYNA, I_ref=I_MAX)
history_stat, config_stat, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_stat)
history_dyna, config_dyna, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_dyna)

# %% USE FOR REPORT. COMPARISON BAR PLOT STATIC AND DYNAMIC TRAINED ON CC
plot_nrmse_bars(models = {'Static':  bat_model_static_DC,
                        'Dynamic': bat_model_dynamic_DC},
    trajs_by_set = {'CC': test_trajs, 'Pulse': pulse_test},
    rmse_scales  = RMSE_scales,
ECM_fix = True, metric_names = ['Voltage'])
# plt.savefig(os.path.join(FIGS_DIR, f'Vnrmse_0513_1037dyna_0513_1037stat_CCtrained.pdf'), bbox_inches='tight')
plt.show()
# %%
# USE FOR REPORT. STATIC TRAINED ON CC
#

plot_report(bat_model_static_DC, config_stat, test_trajs, title='CC test: ',
                 n_show=min(2, len(pulse_test)), time = True)
# plt.savefig(os.path.join(FIGS_DIR, f'0513_1037_static_ccPred.pdf'), bbox_inches='tight')
plt.show()
plot_report(bat_model_static_DC, config_stat, pulse_test, title='Pulse test: ',
                 n_show=min(2, len(pulse_test)), time = True)
# plt.savefig(os.path.join(FIGS_DIR, f'0513_1037_static_pulsePred.pdf'), bbox_inches='tight')
plt.show()

# %%
# USE FOR REPORT. DYNAMIC TRAINED ON CC
#

plot_report(bat_model_dynamic_DC, config_dyna, test_trajs, title='CC test: ',
                 n_show=min(2, len(pulse_test)), time = True)
# plt.savefig(os.path.join(FIGS_DIR, f'0513_1037_dynamic_ccPred.pdf'), bbox_inches='tight')
plt.show()
plot_report(bat_model_dynamic_DC, config_dyna, pulse_test, title='Pulse test: ',
                 n_show=min(2, len(pulse_test)), time = True)
# plt.savefig(os.path.join(FIGS_DIR, f'0513_1037_dynamic_pulsePred.pdf'), bbox_inches='tight')
plt.show()




# %%
# USE FOR REPORT. INPUT ERROR MAP 
# 
 
MODEL_NAME_LOW = '0510_2034_combo_low_c_d_combo_V-dynamic_F-dynamic_702.38min_16h_2500eps.pt'
MODEL_NAME_FULL = '0510_2034_b4_combo_full_combo_V-dynamic_F-dynamic_642.62min_16h_2500eps.pt'


bat_model_low, ckpt_low = load_nn_model(MODEL_NAME_LOW, I_ref=I_MAX)   # U_MAX when loading lib_3
history_low, config_low, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_low)

bat_model_full, ckpt_full = load_nn_model(MODEL_NAME_FULL, I_ref=I_MAX)
history_full, config_full, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_full)



# %% –––––––– INPUT ERROR MAP COMPARISON ––––––––––––––––––––––––––––––––––––––––
fig,ax = input_map_comparison(bat_model_low, bat_model_full, other_combo_trajs  , rmse_scales=RMSE_scales)
# plt.savefig(os.path.join(FIGS_DIR, f'0510_2034_low_0510_2034_full_otherCombo_input_error_comparison.pdf'), bbox_inches='tight')
plt.show()


# %%

plot_report(bat_model_full, config_full, test_trajs, title='CC test: ', n_show=min(2, len(pulse_test)), time = True)
# plt.savefig(os.path.join(FIGS_DIR, f'ccPred_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
plt.show()
plot_report(bat_model_full, config_full, pulse_test, title='Pulse test: ', n_show=min(2, len(pulse_test)), time = True)
# plt.savefig(os.path.join(FIGS_DIR, f'pulsePred_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
plt.show()

# %% USE FOR REPORT

import ecmm_vnode_vstatic_snode_sstatic_batch_lib_6_full as _lib
importlib.reload(_lib)
from ecmm_vnode_vstatic_snode_sstatic_batch_lib_6_full import *

MODEL_NAME_FULL = MODEL_NAME
#MODEL_NAME_FULL = '0510_2034_b4_combo_full_combo_V-dynamic_F-dynamic_642.62min_16h_2500eps.pt'
# MODEL_NAME_LOW = '0510_2034_b4_combo_low_c_d_combo_V-dynamic_F-dynamic_702.38min_16h_2500eps.pt'

# bat_model_static_DC, ckpt_stat = load_nn_model(MODEL_NAME_STAT, I_ref=I_MAX)
# bat_model_dynamic_DC, ckpt_dyna = load_nn_model(MODEL_NAME_DYNA, I_ref=I_MAX)
# plot_param(bat_model_dynamic_DC, test_trajs, param='tau')
# plt.savefig(os.path.join(FIGS_DIR, f'tau_{MODEL_NAME_DYNA}.pdf'), bbox_inches='tight')

bat_model_full, ckpt_full = load_nn_model(MODEL_NAME_FULL, I_ref=I_MAX)
config_full, _, _, _ = load_checkpoint(ckpt_full)

# plot_predicts_report(bat_model_full, config_full, test_trajs, predict='V', sort='C_rate', n_show=5, time=True)
# # plt.savefig(os.path.join(FIGS_DIR, f'V_Crates_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
# plot_predicts_report(bat_model_full, config_full, pulse_test, predict='F', sort='C_rate', n_show=4, time=True, pulse=False)
# # plt.savefig(os.path.join(FIGS_DIR, f'V_Crates_pulse_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')

# plot_predicts_report(bat_model_full, config_full, test_trajs, predict='F', sort='C_rate', n_show=3, time=False)
# plot_predicts_report(bat_model_full, config_full, pulse_test, predict='F', sort='C_rate', n_show=10, time=True)

# plot_param(bat_model_low, test_trajs, param='R0')
# # plt.savefig(os.path.join(FIGS_DIR, f'R0_{MODEL_NAME_LOW}.pdf'), bbox_inches='tight')
# plot_param(bat_model_low, test_trajs, param='R1')
# # plt.savefig(os.path.join(FIGS_DIR, f'R1_{MODEL_NAME_LOW}.pdf'), bbox_inches='tight')
# plot_param(bat_model_low, test_trajs, param='C1')
# # plt.savefig(os.path.join(FIGS_DIR, f'C1_{MODEL_NAME_LOW}.pdf'), bbox_inches='tight')
# plot_param(bat_model_low, test_trajs, param='tau')
# # plt.savefig(os.path.join(FIGS_DIR, f'tau_{MODEL_NAME_LOW}.pdf'), bbox_inches='tight')

plot_param(bat_model_full, test_trajs, param='R0')
# # plt.savefig(os.path.join(FIGS_DIR, f'R0_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
plot_param(bat_model_full, test_trajs, param='R1')
# # plt.savefig(os.path.join(FIGS_DIR, f'R1_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
plot_param(bat_model_full, test_trajs, param='C1')
# # plt.savefig(os.path.join(FIGS_DIR, f'C1_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
# plot_param(bat_model_full, test_trajs, param='tau')
# # plt.savefig(os.path.join(FIGS_DIR, f'tau_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')

### –––– Mechanical elements ––––
plot_param(bat_model_full, test_trajs, param='s')
# # plt.savefig(os.path.join(FIGS_DIR, f's_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
plot_param(bat_model_full, test_trajs, param='sdot')
# # plt.savefig(os.path.join(FIGS_DIR, f'sdot_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
# plot_param(bat_model_full, test_trajs, param='k')
# # plt.savefig(os.path.join(FIGS_DIR, f'k_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
plot_param(bat_model_full, test_trajs, param='ku')
# plt.savefig(os.path.join(FIGS_DIR, f'ku_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
 
# plot_param_pulse(bat_model_full, pulse_test, param='R0', n_show=4)
# # plt.savefig(os.path.join(FIGS_DIR, f'R0_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
# plot_param_pulse(bat_model_full, pulse_test, param='R1', n_show=4)
# # plt.savefig(os.path.join(FIGS_DIR, f'R1_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
# plot_param_pulse(bat_model_full, pulse_test, param='C1', n_show=4)
# # plt.savefig(os.path.join(FIGS_DIR, f'C1_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')
# plot_param_pulse(bat_model_full, pulse_test, param='tau', n_show=4)
# # plt.savefig(os.path.join(FIGS_DIR, f'tau_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')


plot_force_report(bat_model_full, config_full, test_trajs, n_show=3)
# plt.savefig(os.path.join(FIGS_DIR, f'F_CC_{MODEL_NAME_FULL}.pdf'), bbox_inches='tight')

plt.show()

# # %% USE FOR REPORT INPUT ERROR MAPS

# input_map_single(bat_model_full, other_combo_trajs, rmse_scales=RMSE_scales, observable='V')
# input_map_single(bat_model_full, other_combo_trajs, rmse_scales=RMSE_scales, observable='F')

# %%


# ––––––– Prepare data and plotting  ––––––––––––––––––––––––––––––––––––––––––––––––––

other_combo_pulse = prepare_pulse_data(other_combo_data[other_combo_data['pulse'] == True])
plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_pulse, sort='C_rate', predict='V',
                             n_show=4, pulse=True, fixed=False, start=18)
plt.savefig(os.path.join('pred_figs', f'0510_2034_full_otherCombo_pulse_V.pdf'), bbox_inches='tight')
plt.show()

other_combo_cc = prepare_pulse_data(other_combo_data[other_combo_data['pulse'] == False])
plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_cc, sort='C_rate', predict='V',
                             n_show=8, pulse=False, fixed=False, start=2)
plt.savefig(os.path.join('pred_figs', f'0510_2034_full_otherCombo_cc_V.pdf'), bbox_inches='tight')
plt.show()

# ––––– Pulse force
other_combo_pulse_d0 = prepare_pulse_data(other_combo_data[(other_combo_data['u_par']==0) & (other_combo_data['pulse'] == True)])
plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_pulse_d0, sort='C_rate', predict='F',
                             n_show=5, pulse=True)
plt.savefig(os.path.join('pred_figs', f'0510_2034_full_otherCombo_pulse_F.pdf'), bbox_inches='tight')
plt.show()

# new generation data c_rate 2.5
crate_usweep_pulse = pd.read_csv(os.path.join(DATA_DIR, 'crate2.5_usweep_pulse.txt'), sep=';', comment='%')
pulse_c25_usweep = prepare_pulse_data(crate_usweep_pulse)
plot_mosaic_predicts_report(bat_model_full, config_full, pulse_c25_usweep, sort='u_per', predict='F',
                             n_show=5, pulse=True)
plt.savefig(os.path.join('pred_figs', f'0510_2034_full_c25usweep_pulse_F.pdf'), bbox_inches='tight')
plt.show()
# –––––

# ––––– CC force
other_combo_cc_d0 = prepare_pulse_data(other_combo_data[(other_combo_data['u_par']==0) & (other_combo_data['pulse'] == False)])
plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_cc_d0, sort='C_rate', predict='F',
                             n_show=5, pulse=False)
plt.savefig(os.path.join('pred_figs', f'0510_2034_full_otherCombo_cc_F.pdf'), bbox_inches='tight')
plt.show()

crate_usweep_cc = pd.read_csv(os.path.join(DATA_DIR, 'crate2.5_usweep.txt'), sep=';', comment='%')
cc_c25_usweep = prepare_pulse_data(crate_usweep_cc)
plot_mosaic_predicts_report(bat_model_full, config_full, cc_c25_usweep, sort='u_per', predict='F',
                             n_show=5, pulse=False)
plt.savefig(os.path.join('pred_figs', f'0510_2034_full_c25usweep_cc_F.pdf'), bbox_inches='tight')
plt.show()
# –––––

# ––––––––––––– pulse dynamic o static ––––––––––––––
crate_usweep_pulse = pd.read_csv(os.path.join(DATA_DIR, 'crate2.5_usweep_pulse.txt'), sep=';', comment='%')
pulse_c25_usweep = prepare_pulse_data(crate_usweep_pulse)
plot_mosaic_predicts_report(bat_model_dynamic_DC, config_dyna, other_combo_pulse, sort='C_rate', predict='V', 
                            n_show=1, start=39, pulse=True, fixed=False, bar=False)
plt.savefig(os.path.join('pred_figs', f'CC_dynamic_pulse_V.pdf'), bbox_inches='tight')
plt.show()

plot_mosaic_predicts_report(bat_model_static_DC, config_stat, other_combo_pulse, sort='C_rate', predict='V', 
                            n_show=1, start=39, pulse=True, fixed=False, bar=False)
plt.savefig(os.path.join('pred_figs', f'CC_static_pulse_V.pdf'), bbox_inches='tight')
plt.show()
