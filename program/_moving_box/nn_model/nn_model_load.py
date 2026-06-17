# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM LOADING SCRIPT
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
sys.path.append(os.path.join(FILE_PATH, '..'))
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()




# --- Import library (reload-safe for repeated cell runs in Jupyter) ---
import nn_model_lib as _lib
importlib.reload(_lib)
from nn_model_lib import *

from datetime import datetime
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')




# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(FILE_PATH, '../data')
PYBAMM_DATA_DIR = os.path.join(FILE_PATH, '../data_pybamm')
DATA_FILE   = os.path.join(DATA_DIR, 'polished_CC/merged_CC_hyper.txt')
ALL_DATA    =  os.path.join(DATA_DIR, 'polished_combo/merged_combo.txt')
HALF_COMBO  = os.path.join(DATA_DIR, 'polished_combo/combo_half.txt'); 
OTHER_HALF_COMBO = os.path.join(DATA_DIR, 'polished_combo/combo_other_half.txt')
PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulse/merged_pulse_hyper.txt')
FIGS_DIR    = os.path.join(FILE_PATH, 'figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'saved_NN_models')



# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
print(DATA_FILE)
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

TRAIN_SPLIT = 0.8
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


wombo_combo = combo_test + other_combo_trajs

# %% ══════════════════════════════════════════════════════════
#  LOAD MODEL  (no Ue argument — GP loaded internally by lib)
# ═════════════════════════════════════════════════════════════
 
MODEL_NAME_STAT = '0508_1444_snode_DC_V-static_no_R0_F-static_netR0_R0c_R1c_C1c_2.97min_16h_650eps_0stat_0dyneps.pt'
bat_model_static_DC, ckpt_stat = load_nn_model(MODEL_NAME_STAT)
history_stat, config_stat, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_stat)

MODEL_NAME_DYNA = '0508_2228_DC_DC_V-dynamic_F-dynamic_436.43min_16h_650eps.pt'
bat_model_dynamic_DC, ckpt_dyna = load_nn_model(MODEL_NAME_DYNA)
history_dyna, config_dyna, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_dyna)

MODEL_NAME_FULL = '0515_0840_b4_combo_softplus_combo_V-dynamic_F-dynamic_unconstr_700.40min_16h_2500eps.pt'
bat_model_full, ckpt_full = load_nn_model(MODEL_NAME_FULL)
history_full, config_full, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_full)

MODEL_NAME_PYBAMM = '0617_1234_pybamm_CC_V-static_F-static_unconstr_0.82min_1000eps.pt'
bat_model_pybamm, ckpt_pybamm = load_nn_model(MODEL_NAME_PYBAMM)
history_pybamm, config_pybamm, N_HIDDEN, EPOCHS = load_checkpoint(ckpt_pybamm)


# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ═════════════════════════════════════════════════════════════

plot_loss(history_full)
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTION REPORTS
# ═════════════════════════════════════════════════════════════

other_combo_pulse = prepare_pulse_data(other_combo_data[other_combo_data['pulse'] == True])
other_combo_cc = prepare_pulse_data(other_combo_data[(other_combo_data['pulse'] == False) & (other_combo_data['u_par'] == 0)])

# # new generation data c_rate 2.5
crate_usweep_cc = pd.read_csv(os.path.join(DATA_DIR, 'polished_CC/crate2.5_usweep.txt'), sep=';', comment='%')
cc_c25_usweep = prepare_data(crate_usweep_cc)

crate_usweep_pulse = pd.read_csv(os.path.join(DATA_DIR, 'polished_pulse/crate2.5_usweep_pulse.txt'), sep=';', comment='%')
pulse_c25_usweep = prepare_pulse_data(crate_usweep_pulse)



plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_pulse, sort='C_rate', predict='V', n_show=4, pulse=True, fixed=False, start=18)

plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_cc, sort='C_rate', predict='V', n_show=4, pulse=False, fixed=False, start=1)

plot_mosaic_predicts_report_data(bat_model_full, config_full, other_combo_cc, sort='C_rate', predict='V', n_show=3, pulse=False, show_current=False, fixed=False, start=3, bar=False)

plot_mosaic_predicts_report_data(bat_model_full, config_full, cc_c25_usweep, sort='u_per', predict='V', n_show=3, bar=False, pulse=False, show_current=False, fixed=False, start=2)
plt.show()

##### ––––– For report –––––
# # ––––– Pulse force
other_combo_pulse_d0 = prepare_pulse_data(other_combo_data[(other_combo_data['u_par']==0) & (other_combo_data['pulse'] == True)])
plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_pulse_d0, sort='C_rate', predict='F', n_show=4, pulse=True)

plot_mosaic_predicts_report(bat_model_full, config_full, pulse_c25_usweep, sort='u_per', predict='F', n_show=4, pulse=True)
# # –––––

# # ––––– CC force
plot_mosaic_predicts_report(bat_model_full, config_full, cc_c25_usweep, sort='u_per', predict='F', n_show=4, bar=True, pulse=False, fixed=True)

other_combo_cc_d0 = prepare_pulse_data(other_combo_data[(other_combo_data['u_par']==0) & (other_combo_data['pulse'] == False)])
plot_mosaic_predicts_report(bat_model_full, config_full, other_combo_cc_d0, sort='C_rate', predict='F', n_show=4, bar=True, pulse=False, fixed=True)

plt.show()
##### –––––––


##### ––– For presentation –––
T_ref = max(tr['T'] for tr in cc_c25_usweep)        # ~1500; or just set T_ref = 1500
other_combo_cc_d0 = prepare_pulse_data(other_combo_data[(other_combo_data['u_par']==0) & (other_combo_data['pulse'] == False)])
short = [tr for tr in other_combo_cc_d0 if tr['T'] <= T_ref]

plot_mosaic_predicts_report(bat_model_full, config_full, short, sort='C_rate', predict='F', n_show=4, bar=True, pulse=False, fixed=False)
plot_mosaic_predicts_report_data(bat_model_full, config_full, short, sort='C_rate', predict='F', n_show=3, bar=False, pulse=False, fixed=False)

plot_mosaic_predicts_report(bat_model_full, config_full, cc_c25_usweep, sort='u_per', predict='F', n_show=3, bar=False, pulse=False, fixed=False)
plot_mosaic_predicts_report_data(bat_model_full, config_full, cc_c25_usweep, sort='u_per', predict='F', n_show=3, bar=False, pulse=False, fixed=False)

plt.show()
##### –––––




# %% ––––––––––––– pulse dynamic o static ––––––––––––––

## –––– Change colors in plotter ––––
plot_mosaic_predicts_report(bat_model_dynamic_DC, config_dyna, other_combo_pulse, sort='C_rate', predict='V', 
                            n_show=1, start=39, pulse=True, fixed=False, bar=False)
plt.show()

plot_mosaic_predicts_report(bat_model_static_DC, config_stat, other_combo_pulse, sort='C_rate', predict='V', 
                            n_show=1, start=39, pulse=True, fixed=False, bar=False)
plt.show()



# %% ══════════════════════════════════════════════════════════
#  PYBAMM MODEL TEST
# ═════════════════════════════════════════════════════════════

pybamm_cc = prepare_data(pd.read_csv(os.path.join(PYBAMM_DATA_DIR, 'CC/pybamm_CC.txt'), sep=';', comment='%'))

plot_mosaic_predicts_report(bat_model_pybamm, config_pybamm, pybamm_cc, sort='C_rate', predict='V', n_show=4, pulse=False, fixed=False, start=1)
plt.show()