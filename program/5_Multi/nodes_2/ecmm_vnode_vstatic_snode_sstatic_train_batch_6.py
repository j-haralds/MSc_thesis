# %% ══════════════════════════════════════════════════════════
#  BATTERY ECM + EMM NODE
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
# import ecmm_node_cleaner_swell_GP_lib as _lib
# importlib.reload(_lib)
# from ecmm_node_cleaner_swell_GP_lib import *

from datetime import datetime


# %% ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
DATA_DIR    = os.path.abspath(os.path.join(FILE_PATH, '..', 'Multi_data'))
# Choose data file
DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/merged_DC_hyper.txt') # Full
# # DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/DC_high_comp.txt') # high compression
# DATA_FILE   = os.path.join(DATA_DIR, 'polished_DC/DC_low_comp.txt') # low compression

PULSE_FILE  = os.path.join(DATA_DIR, 'polished_pulses/merged_pulse_hyper.txt')

COMBO_FILE  = os.path.join(DATA_DIR, 'combo_half.txt')
#COMBO_FILE  = os.path.join(DATA_DIR, 'combo_low_c_d.txt')

FIGS_DIR    = os.path.join(FILE_PATH, 'nodes_figs')
MODEL_DIR   = os.path.join(FILE_PATH, 'models')
#MODEL_DIR   = os.path.join(FILE_PATH, 'final_models')
SAVE_FIGS   = False
SAVE_MODELS = True 
SAVE_ELEMENTS = False

Q0          = 17921.57581     # As
TRAIN_SPLIT = 0.8
N_HIDDEN          = 16
LR_STATIC         = 1e-3
LR_DYNAMIC        = 1e-3
LR_UNFREEZE       = 1e-3     # smaller LR once R1 is being refined

# Batched-training controls (used when train trajectories are length-uniform,
# i.e. pulse / combo data).  CC data falls back to per-trajectory SGD.
BATCH_SIZE  = 16     # mini-batch size for the batched path; ignored for CC
EVAL_EVERY  = 1      # epochs between test-set evals in the batched path;
                     # set to 10 for a further ~10× cut on test-eval cost

# Use pulse trajectories for Stage 2 (and 2b).  Stage 1 always uses CC trajs.
USE_PULSE         = 'combo'   # 'pulse', 'DC', 'combo' (combo = both CC and pulse for training)

CONFIG = {
    'R1_mode': 'net',   # 'net'
    'C1_mode': 'net',   # 'net'
    'R0_mode': 'net',           # 'func', 'net', 'param', 'net_no_soc'
    'n_hidden': N_HIDDEN,
    'R1_constrained': 'true', 'R1_min': 0.005, 'R1_max': 0.2,      # Ohm
    'C1_constrained': 'true', 'C1_min': 500.0, 'C1_max': 50000.0,  # F
    'R0_constrained': 'true', 'R0_min': 0.008, 'R0_max': 0.015,    # Ohm
    # OBS: k increased for for low u? F_min/u_min ~ 0.002/0.009 = 0.22
    'k_constrained': 'true', 'k_min': 0.02, 'k_max': 0.04,    # [≤ 0.04]  GN/1e-5m
    # ── F-branch swelling constraints — split per style_F mode ──
    # style_F='static'  uses  s_constrained / s_min / s_max     — bounds on s itself        [1e-5 m]
    # style_F='dynamic' uses  sdot_constrained / sdot_min / sdot_max — bounds on ds/dt  [1e-5 m / s]
    # Backward compat: if the sdot_* keys are absent, sdotNet falls back to the s_* keys
    # (so old checkpoints which only had s_* keys still load and behave identically).
    's_constrained':    'true', 's_min':    0.0, 's_max':    0.005*100,   # static sNet  [1e-5 m]
    'sdot_constrained': 'true', 'sdot_min': 0.0, 'sdot_max': 0.001,   # dynamic sdotNet [1e-5 m / s]

    # ── style_V (V branch): 'static_no_R0' | 'static' | 'dynamic' | 'staged' ──
    # OBS: for only static training U1 = I*R1, use only stage 1 with 'staged' style. It freezes C1.
    # OBS 'staged' uses CC for static and pulse for dynamic regardless of USE_PULSE
    'style_V': 'dynamic',  # 'static_no_R0', 'dynamic', 'staged'

    # ── style_F (F branch): 'static' (lib_3 algebraic sNet) | 'dynamic' (lib_4 sdotNet NODE) ──
    # 'static':  s = sNet(soc, I_norm)              — no time integration, F is fully algebraic
    # 'dynamic': ds/dt = sdotNet(s, soc, I_norm, u) — Euler-rolled from s(0)=0 (the snode lib_4 default)
    'style_F': 'dynamic',  # 'static' (lib_3 algebraic sNet) | 'dynamic' (lib_4 sdotNet NODE)

    # 'freeze_static_no_R0': ('R0_net', 'C1_net'),  # mainly for 'static_no_R0' style
}

# OBS: Specifiy only used training epochs, set rest to 0.
EPOCHS  = 650  # For single-stage training (style_V='static_no_R0' or 'dynamic')

# Only for staged
EPOCHS_STATIC     = 0  # Stage 1 : V static, train R1 and k
EPOCHS_DYNAMIC    = 0      # Stage 2 : V dynamic, train C1 and k (R1 frozen)
EPOCHS_UNFREEZE   = 0     # Stage 2b: V dynamic, R1 unfrozen (0 = skip)

NAME_START = f'combo_full'



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
# F_upar = data['F'][data['u_par'] == 26.5].values
# F_diff = F_upar[-1] - F_upar[0]
# print(F_diff)
V_max = data['V'].max()
print(f'Cell lengths: {L_CELL:.5f} 1e-5m | I max: {I_MAX:.4f} A | u min: {U_MIN:.4f} 1e-5m'
      f'\nF max: {F_max:.4f} GN | k max: {F_max/(0-U_MIN)}|')

# Ue(SOC) is now provided by the module-level GP (loaded lazily inside the
# library from JN_GP) — no Ue_interp construction needed here anymore.

print(f"  {len(data)} pts, {data['trajectory'].nunique()} trajectories")



# %% ══════════════════════════════════════════════════════════
#  PREPARE TRAJECTORIES + ESTIMATE C1
# ══════════════════════════════════════════════════════════════

trajs = prepare_data(data)
split = int(len(trajs) * TRAIN_SPLIT)
train_trajs, test_trajs = trajs[:split], trajs[split:]
print(f"  Train: {len(train_trajs)} | Test: {len(test_trajs)}")


# %% ══════════════════════════════════════════════════════════
#  PREPARE PULSE TRAJECTORIES  (for Stage 2 / 2b)
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
elif USE_PULSE == 'DC':
    print("\nUsing CC trajectories for training.")
    _train_trajs = train_trajs
    _test_trajs = test_trajs
elif USE_PULSE == 'combo':
    print("\nUsing combined CC + pulse trajectories for training.")
    _train_trajs = combo_train
    _test_trajs = combo_test


if CONFIG['style_V'] == 'dynamic':
    print(f"\nSingle-stage training: {EPOCHS} epochs with dynamic V"
          f"  (style_F={CONFIG['style_F']!r})")
    history = train_model(bat_model, _train_trajs, _test_trajs,
                n_epochs=EPOCHS, lr=LR_DYNAMIC, print_every=1,
                V_mode='dynamic', freeze=None,
                batched=None, batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)

    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")

elif CONFIG['style_V'] == 'static_no_R0':
    print(f"\nSingle-stage training: {EPOCHS} epochs with static V1: VB = Ue - iR1 (no R0)"
          f"  (style_F={CONFIG['style_F']!r})")
    history = train_model(bat_model, _train_trajs, _test_trajs,
                n_epochs=EPOCHS, lr=LR_STATIC, print_every=1,
                V_mode='static_no_R0', freeze=('R0_net', 'C1_net'),
                batched=None, batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)          #CONFIG['freeze_static_no_R0'])

    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")



elif CONFIG['style_V'] == 'staged':
    print(f"\nStaged training: S1={EPOCHS_STATIC}ep static, S2={EPOCHS_DYNAMIC}ep dynamic"
        f"{f', S2b={EPOCHS_UNFREEZE}ep R1-unfrozen' if EPOCHS_UNFREEZE > 0 else ''}"
        f"{' (pulse Stage 2)'}  (style_F={CONFIG['style_F']!r})")

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
                        lr_unfreeze=LR_UNFREEZE,
                        batched=None, batch_size=BATCH_SIZE, eval_every=EVAL_EVERY)

    C1_final = get_C1(bat_model, scalar=True)
    TOTAL_TIME = history['time']
    print(f"\nTraining completed in {TOTAL_TIME:.1f} minutes.")


# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — TEST
# ══════════════════════════════════════════════════════════════

# Build save name from active flags — much cleaner than the prior 6-branch chain.
# Includes both style_V (V branch) and style_F (F branch) so checkpoints for
# the four combinations (static/dynamic × static/dynamic) don't collide.
# constr_tags = []
# if CONFIG.get('R0_constrained', 'false') == 'true': constr_tags.append('R0c')
# if CONFIG.get('R1_constrained', 'false') == 'true': constr_tags.append('R1c')
# if CONFIG.get('C1_constrained', 'false') == 'true': constr_tags.append('C1c')
# constr = '_'.join(constr_tags) if constr_tags else 'unconstr'

# Tag style as e.g. 'V-dynamic_F-static' to make the F branch visible in the filename.
style_tag = f'V-{CONFIG["style_V"]}_F-{CONFIG["style_F"]}'

SAVE_NAME = (f'{NAME_START}_{USE_PULSE}_{style_tag}'
            #  f'_{constr}'
             f'_{TOTAL_TIME:.2f}min_{N_HIDDEN}h'
             f'_{EPOCHS}eps')

print(SAVE_NAME)

plot_predictions(bat_model, CONFIG, test_trajs, time=False, title='Test: ', n_show=3)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'test_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  LOSS CURVES
# ══════════════════════════════════════════════════════════════

plot_loss(history)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'loss_{TIMESTAMP}_{SAVE_NAME}.pdf'), bbox_inches='tight')
    print('Saved figure')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PARAMS
# ═════════════════════════════════════════════════════════════

plot_param(bat_model, test_trajs, param='R0')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'R0_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='R1')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'R1_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='C1')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'C1_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='k')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'k_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='ku')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'ku_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_param(bat_model, test_trajs, param='s')
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f's_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()



plot_force(bat_model, test_trajs)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'F_{SAVE_NAME}.pdf'), bbox_inches='tight')
plot_swelling(bat_model, test_trajs)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'su_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
# PLOT PREDICTS
# ═════════════════════════════════════════════════════════════

sort = 'u_per'  # 'C_rate' or 'u_per'
plot_predicts(bat_model, CONFIG, test_trajs, predict='F', sort=sort)
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'F_{sort}_{SAVE_NAME}.pdf'), bbox_inches='tight')
plt.show()

# %% ══════════════════════════════════════════════════════════
#  PREDICTIONS — PULSE TEST
# ══════════════════════════════════════════════════════════════

# plot_predictions auto-detects pulse trajectories (they carry 'I_seq');
plot_predictions(bat_model, CONFIG, pulse_test, title='Pulse test: ',
                 n_show=min(3, len(pulse_test)))
if SAVE_FIGS:
    plt.savefig(os.path.join(FIGS_DIR, f'pulse_{SAVE_NAME}.pdf'),
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
        # 'C1_final': C1_final,
        'I_ref': float(I_MAX),
        'u_ref': float(U_MIN),  
        'N_HIDDEN': N_HIDDEN,
        'EPOCHS': EPOCHS,
        'EPOCHS_STATIC': EPOCHS_STATIC,
        'EPOCHS_DYNAMIC': EPOCHS_DYNAMIC,
        'EPOCHS_UNFREEZE': EPOCHS_UNFREEZE,
        'USE_PULSE': USE_PULSE,
    }, os.path.join(MODEL_DIR, f'{TIMESTAMP}_{SAVE_NAME}.pt'))

    print(f"Saved: {TIMESTAMP}_{SAVE_NAME}.pt")

# %% ══════════════════════════════════════════════════════════
# ELEMENT SAVER
# ═════════════════════════════════════════════════════════════

element_data = data_param(bat_model, trajs)
if SAVE_ELEMENTS:
    element_data.to_csv(os.path.join('..', 'sr/symbol_data', f'elements_{TIMESTAMP}_{SAVE_NAME}.txt'), index=False)
    print(f"Saved element data: elements_{TIMESTAMP}_{SAVE_NAME}.txt")



# %% ══════════════════════════════════════════════════════════

