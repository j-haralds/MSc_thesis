# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM TRAINING SCRIPT
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
from nn_model_lib import _traj_inputs
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()


# --- Import library (reload-safe for repeated cell runs in Jupyter) ---
import nn_model_lib as _lib
importlib.reload(_lib)
from nn_model_lib import *

from datetime import datetime


# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

TIMESTAMP   = datetime.now().strftime('%m%d_%H%M')
PYBAMM_DATA_DIR = os.path.abspath(os.path.join(FILE_PATH, '..', 'data_pybamm'))
DATA_DIR    = os.path.abspath(os.path.join(FILE_PATH, '..', 'data'))
FIGS_DIR    = os.path.join(FILE_PATH, 'figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'saved_NN_models')

SAVE_FIGS     = False
SAVE_MODELS   = True 

TRAIN_SPLIT = 0.8
N_HIDDEN    = 16
LR_STATIC   = 1e-3
LR_DYNAMIC  = 1e-3

# Batched-training controls.  Trajectories of any length mix freely — each
# mini-batch is padded to its own max-T and the loss is masked to ignore padded positions.
BATCH_SIZE  = 4     # mini-batch size
EVAL_EVERY  = 1      # epochs between test-set evals; raise for cheaper eval

# Choose trajectories to train on. 
# (Framework/constraints optimized for comsol data with force. Pybamm data is without force data, yet force is predicted)
HF_MODEL    = 'pybamm'  # 'comsol' or 'pybamm' (both have same inputs and outputs)
# Change softplus scaling when running pybamm

if HF_MODEL == 'comsol':
    Q0          = 17921.57581     # As
    DATA_FILE   = os.path.join(DATA_DIR, 'polished_CC/merged_CC_hyper.txt')         # Full from hyperelastic material in Comsol
    PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulse/merged_pulse_hyper.txt')
    COMBO_FILE  = os.path.join(DATA_DIR, 'polished_combo/combo_half.txt')
    #COMBO_FILE  = os.path.join(DATA_DIR, 'polished_combo/combo_other_half.txt')
elif HF_MODEL == 'pybamm':
    Q0          = 3600.0           # As, nominal capacity for 1 A.h cell, defined in sim_PyBaMM.py
    DATA_FILE   = os.path.join(PYBAMM_DATA_DIR, 'CC/pybamm_CC.txt')        
    # PULSE_FILE  = os.path.join(PYBAMM_DATA_DIR, 'pybamm_pulse.txt')
    # COMBO_FILE  = os.path.join(PYBAMM_DATA_DIR, 'pybamm_combo.txt')
    PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulse/merged_pulse_hyper.txt')
    COMBO_FILE  = os.path.join(DATA_DIR, 'polished_combo/combo_half.txt')

USE_PULSE   = 'CC'   # 'pulse', 'CC', 'combo' (combo = both CC and pulse)


# Configure networks, constraints
CONFIG = {
    'R1_mode': 'net',   # 'net'
    'C1_mode': 'net',   # 'net'
    'R0_mode': 'net',   # 'func', 'net', 'param', 'net_no_soc'
    'n_hidden': N_HIDDEN,

    # 'false' constraints uses softplus * magnitude value.
    # 'true' constraints uses sigmoid between min and max values.
    'R1_constrained': 'false', 'R1_min': 0.005,  'R1_max': 0.25,      # Ohm
    'C1_constrained': 'false', 'C1_min': 5000.0, 'C1_max': 14000.0,  # F
    'R0_constrained': 'false', 'R0_min': 0.007,  'R0_max': 0.015,     # Ohm
    'k_constrained':  'false', 'k_min':  0.02,   'k_max':  0.04,          # [≤ 0.04]  GN/1e-5m
    # ── F-branch swelling constraints — split per style_F mode ──
    # style_F='static'  uses  s_constrained / s_min / s_max     — bounds on s itself        [1e-5 m]
    # style_F='dynamic' uses  sdot_constrained / sdot_min / sdot_max — bounds on ds/dt  [1e-5 m / s]
    # Backward compatable: if the sdot_* keys are absent, sdotNet falls back to the s_* keys
    's_constrained':    'false', 's_min':    0.0, 's_max':    0.005*100,   # static sNet  [1e-5 m]
    'sdot_constrained': 'false', 'sdot_min': 0.0, 'sdot_max': 0.001,   # dynamic sdotNet [1e-5 m / s]

    # ── style_V (V branch): 'static_no_R0' | 'static' | 'dynamic' ──
    # 'static_no_R0' : V = Ue − I·R1,                (algebraic, no R0)
    # 'dynamic'      : V = Ue − I·R0 − U1,           with U1 integrated by semi-implicit Euler
    'style_V': 'static_no_R0',  # 'static_no_R0', 'dynamic', 'back_in_black' (Full black box model)

    # ── style_F (F branch): 'static' (algebraic sNet) | 'dynamic' ──
    # 'static':  s = sNet(soc, I_norm)              — no time integration, F is fully algebraic
    # 'dynamic': ds/dt = sdotNet(s, soc, I_norm, u) — Euler-rolled from s(0)=0
    'style_F': 'static',  # 'static', 'dynamic'

    'HF_model': HF_MODEL,  # 'comsol' or 'pybamm'
}

EPOCHS  = 2500  # Total training epochs
split_percentage = 1 # Out of 100% of the training data, how much to use (for quick tests)



# %% ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
data = pd.read_csv(DATA_FILE, sep=';', comment='%')
print(data.columns)

# Normalization factors
I_MAX = data['I'].max()
U_MIN = abs(data['u'].min())
L_CELL = 14.37325   #-(data['u'] / (data['u_par']/100))[0]
F_max = data['F'].min()
V_max = data['V'].max()

if HF_MODEL == 'pybamm':
    U_MIN = 1   # Avoid divide by zero
    F_max = 1

print(f'Cell lengths: {L_CELL:.5f} 1e-5m | I max: {I_MAX:.4f} A | u min: {U_MIN:.4f} 1e-5m'
      f'\nF max: {F_max:.4f} GN | k max: {F_max/(0-U_MIN)}|')

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")



# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")


# %% ══════════════════════════════════════════════════════════
#  PREPARE PULSE TRAJECTORIES
# ══════════════════════════════════════════════════════════════

pulse_data = pd.read_csv(PULSE_FILE, sep=';', comment='%')
print(pulse_data.columns)
pulse_trajs = prepare_pulse_data(pulse_data)
split_p = int(len(pulse_trajs) * TRAIN_SPLIT)
pulse_train, pulse_test = pulse_trajs[:split_p], pulse_trajs[split_p:]
print(f"  Pulse train: {len(pulse_train)} | Pulse test: {len(pulse_test)} "
        f"(T per traj: {pulse_trajs[0]['T']})")

# %% ══════════════════════════════════════════════════════════
#  PREPARE COMBINED TRAJECTORIES
# ══════════════════════════════════════════════════════════════

combo_data = pd.read_csv(COMBO_FILE, sep=';', comment='%')
print(combo_data.columns)
combo_trajs = prepare_pulse_data(combo_data)
split_c = int(len(combo_trajs) * TRAIN_SPLIT)
combo_train, combo_test = combo_trajs[:split_c], combo_trajs[split_c:]

#### Split the training data ####
#split_percentage = 0.2 # 0.6, 0.4, 0.2  # Out of 100% of the training data
second_split = int(len(combo_train) * split_percentage)
combo_train = combo_train[:second_split]
####

print(f"  Combo train: {len(combo_train)} | Combo test: {len(combo_test)} "
        f"(T per traj: {combo_trajs[0]['T']})")



# %% ══════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════

bat_model = BatteryECMM(CONFIG, Q0=Q0, I_ref=I_MAX, u_ref=U_MIN)

n_params = sum(p.numel() for p in bat_model.parameters())
print(f"  Model: {n_params} parameters, {N_HIDDEN} hidden neurons")


# %% ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

if USE_PULSE == 'pulse':
    print("\nUsing pulse trajectories for training.")
    _train_trajs = pulse_train
    _test_trajs = pulse_test
elif USE_PULSE == 'CC':
    print("\nUsing CC trajectories for training.")
    _train_trajs = train_trajs
    _test_trajs = test_trajs
elif USE_PULSE == 'combo':
    print("\nUsing combined CC + pulse trajectories for training.")
    _train_trajs = combo_train
    _test_trajs = combo_test


if CONFIG['style_V'] == 'dynamic':
    print(f"\nTraining: {EPOCHS} epochs with dynamic V"
          f"  (style_F={CONFIG['style_F']!r})")
    history = train_model(bat_model, _train_trajs, _test_trajs,
                n_epochs=EPOCHS, lr=LR_DYNAMIC, print_every=1,
                V_mode='dynamic', freeze=('black_net',),
                batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)

    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")

elif CONFIG['style_V'] == 'static_no_R0':
    print(f"\nTraining: {EPOCHS} epochs with static V: V = Ue - I·R1 (no R0)"
          f"  (style_F={CONFIG['style_F']!r})")
    history = train_model(bat_model, _train_trajs, _test_trajs,
                n_epochs=EPOCHS, lr=LR_STATIC, print_every=1,
                V_mode='static_no_R0', freeze=('black_net', 'R0_net', 'C1_net'),
                batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)

    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")

elif CONFIG['style_V'] == 'static':
    print(f"\nTraining: {EPOCHS} epochs with static V: V = Ue - I·R0 - I·R1"
          f"  (style_F={CONFIG['style_F']!r})")
    history = train_model(bat_model, _train_trajs, _test_trajs,
                n_epochs=EPOCHS, lr=LR_STATIC, print_every=1,
                V_mode='static', freeze=('C1_net', 'black_net'),
                batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)

    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")

elif CONFIG['style_V'] == 'back_in_black':
    print(f"\nTraining: {EPOCHS} epochs with back-in-black V: V = R1"
          f"  (style_V={CONFIG['style_V']!r})")
    history = train_model(bat_model, _train_trajs, _test_trajs,
                n_epochs=EPOCHS, lr=LR_STATIC, print_every=1,
                V_mode='back_in_black', freeze=('R1_net','R0_net', 'C1_net'),
                batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)

    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")

else:
    raise ValueError(
        f"Unknown style_V: {CONFIG['style_V']!r}.  Use 'dynamic', 'static_no_R0', or 'static'.")


# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

# Build save name from active flags — much cleaner than the prior 6-branch chain.
# Includes both style_V (V branch) and style_F (F branch) so checkpoints for
# the four combinations (static/dynamic × static/dynamic) don't collide.
constr_tags = []
if CONFIG.get('R0_constrained', 'false') == 'true': constr_tags.append('R0c')
if CONFIG.get('R1_constrained', 'false') == 'true': constr_tags.append('R1c')
if CONFIG.get('C1_constrained', 'false') == 'true': constr_tags.append('C1c')
constr = '_'.join(constr_tags) if constr_tags else 'unconstr'

# Tag style as e.g. 'V-dynamic_F-static' to make the F branch visible in the filename.
style_tag = f'V-{CONFIG["style_V"]}_F-{CONFIG["style_F"]}'

SAVE_NAME = (f'{HF_MODEL}_{USE_PULSE}_{style_tag}'
             f'_{constr}'
             f'_{TOTAL_TIME:.2f}min'
             f'_{EPOCHS}eps')

print(SAVE_NAME)

plot_predictions(bat_model, CONFIG, test_trajs, time=False, title='Test: ', n_show=3)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'test_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — PULSE TEST
# ══════════════════════════════════════════════════════════════

# plot_predictions auto-detects pulse trajectories (they carry 'I_seq');
plot_predictions(bat_model, CONFIG, pulse_test, title='Pulse test: ', 
                 n_show=min(3, len(pulse_test)), time =True) if HF_MODEL == 'comsol' else None

if SAVE_FIGS and HF_MODEL == 'comsol':
    plt.savefig(os.path.join(FIGS_DIR, f'pulse_{SAVE_NAME}.pdf'),bbox_inches='tight')
plt.show()

# Numeric RMSE summary across the pulse test set
rmses = rmse_pulse(bat_model, pulse_test)
print(f"\nPulse test RMSE (V):  mean {np.mean(rmses):.4f} V | "
        f"median {np.median(rmses):.4f} V | max {np.max(rmses):.4f} V "
        f"({len(rmses)} trajs)")

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════════

plot_loss(history)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'loss_{TIMESTAMP}_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════

if SAVE_MODELS:
    torch.save({
        'model': bat_model.state_dict(),
        'config': CONFIG,
        'history': history,
        'I_ref': float(I_MAX),
        'u_ref': float(U_MIN),
        'Q0': float(Q0),
        'N_HIDDEN': N_HIDDEN,
        'EPOCHS': EPOCHS,
        'USE_PULSE': USE_PULSE,
    }, os.path.join(MODEL_DIR, f'{TIMESTAMP}_{SAVE_NAME}.pt'))

    print(f"Saved: {TIMESTAMP}_{SAVE_NAME}.pt")

