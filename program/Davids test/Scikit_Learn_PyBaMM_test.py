
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyBaMM → Dataset → scikit-learn MLP baseline
---------------------------------------------
Goal: Learn mapping from current profile u(t) to voltage profile V(t).

- Simulates randomized pulse discharge experiments via PyBaMM (SPM default).
- Resamples to a fixed time grid.
- Trains a scikit-learn MLPRegressor (multi-output regression).

Requirements:
    pip install pybamm scikit-learn matplotlib numpy

Notes:
- Use DFN() instead of SPM() for higher fidelity (slower).
- Voltage variable name may differ by PyBaMM version ("Terminal voltage [V]" is typical).
- Simulation may terminate early due to cutoffs; we clamp the tail in resampling.
"""


import numpy as np
import matplotlib.pyplot as plt

# LaTeX font
plt.style.use('default')
plt.rc('text', usetex = True)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

font_size = 16
plt.rcParams['font.size'] = font_size
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

import pybamm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# =========================
# 0) Configuration
# =========================
RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

# Simulation horizon and sampling
HORIZON_S = 1800.0      # total length of each experiment [s]
PERIOD_S = 1.0          # sampling period [s] (also PyBaMM experiment period)

# Dataset size
N_SAMPLES = 300         # increase to improve training
TEST_SIZE = 0.2
RANDOM_STATE = 1

# Pulse generation ranges
C_MIN = 0.5
C_MAX = 2.5
PULSE_DUR_RANGE_S = (30, 300)   # discharge pulse duration range
REST_DUR_RANGE_S = (0, 120)     # rest duration range

# PyBaMM model/params
MODEL = pybamm.lithium_ion.SPM()         # change to pybamm.lithium_ion.DFN() if desired
PARAM_SET_NAME = "Chen2020"              # commonly used set


# =========================
# 1) Experiment helpers
# =========================
def build_pulse_experiment(pulses, period_s=1.0):
    """
    Build a PyBaMM Experiment from a list of pulses.
    pulses: list[(c_rate, dur_s, rest_s)]
    period_s: output sampling period for PyBaMM experiment.
    """
    steps = []
    for c_rate, dur_s, rest_s in pulses:
        steps.append(f"Discharge at {c_rate}C for {int(dur_s)} seconds")
        if rest_s and rest_s > 0:
            steps.append(f"Rest for {int(rest_s)} seconds")
    experiment = pybamm.Experiment(steps, period=f"{period_s} seconds")
    return experiment


def run_simulation(pulses, model=None, params_name="Chen2020", period_s=1.0):
    """
    Run a single PyBaMM simulation with given pulses.
    Returns:
        t (np.ndarray): time [s]
        V (np.ndarray): terminal voltage [V]
        I (np.ndarray): current [A] (discharge positive)
    """
    if model is None:
        model = pybamm.lithium_ion.SPM()

    params = model.default_parameter_values
    experiment = build_pulse_experiment(pulses, period_s=period_s)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)

    sol = sim.solve()
    t = sol.t

    # Voltage variable name can differ by PyBaMM version; try a few
    voltage_keys = [
        "Terminal voltage [V]",
        "Voltage [V]",
        "Measured voltage [V]",
    ]
    V = None
    for k in voltage_keys:
        try:
            V = sol[k].entries
            break
        except KeyError:
            continue
    if V is None:
        available = list(sol.variables.keys())
        raise KeyError(
            f"Could not find a voltage variable among {voltage_keys}. "
            f"Available variables include (truncated): {available[:20]}"
        )

    # Current key
    current_keys = [
        "Current [A]",
        "Total current density [A.m-2]",
    ]
    I = None
    for k in current_keys:
        try:
            I = sol[k].entries
            break
        except KeyError:
            continue
    if I is None:
        # If not present, we can reconstruct sign from pulses if needed; for now raise.
        available = list(sol.variables.keys())
        raise KeyError(
            f"Could not find a current variable among {current_keys}. "
            f"Available variables include (truncated): {available[:20]}"
        )

    return t, V, I


# =========================
# 2) Dataset generation
# =========================
def random_pulse_sequence(
    horizon_s=1800.0,
    c_min=0.5,
    c_max=2.5,
    pulse_dur_s_range=(30, 300),
    rest_dur_s_range=(0, 120),
    rng=None,
):
    """
    Create a random pulse sequence that roughly fills horizon_s seconds.
    Returns list of (c_rate, dur_s, rest_s).
    """
    if rng is None:
        rng = np.random.default_rng()

    pulses = []
    elapsed = 0.0
    while elapsed < horizon_s:
        c = float(rng.uniform(c_min, c_max))
        dur = int(rng.integers(pulse_dur_s_range[0], pulse_dur_s_range[1] + 1))
        rest = int(rng.integers(rest_dur_s_range[0], rest_dur_s_range[1] + 1))
        if elapsed + dur + rest > horizon_s:
            dur = max(1, int(horizon_s - elapsed))
            rest = 0
        pulses.append((c, dur, rest))
        elapsed += dur + rest
    return pulses


def resample_to_fixed_grid(t, x, t_eval):
    """
    Resample x(t) onto t_eval using linear interpolation.
    - Clamps values beyond last simulation time to the last simulated value.
    - If t_eval starts before t[0], clamp to x[0].
    """
    x_res = np.interp(t_eval, t, x, left=x[0], right=x[-1])
    return x_res


def generate_dataset(
    n_samples=500, horizon_s=7200.0, period_s=1.0, model=None, params_name="Default",
    c_min=0.7, c_max=0.8, pulse_dur_s_range=(100, 500), rest_dur_s_range=(100, 500)
):
    """
    Generate synthetic dataset:
      X: current C-rate profile (length N)
      y: voltage profile (length N)
    Returns:
      t_eval, X, y
    """
    t_eval = np.arange(0.0, horizon_s + 1e-9, period_s)  # fixed grid
    currents_c = []  # list of length-N arrays
    voltages = []    # list of length-N arrays

    rng = np.random.default_rng(RANDOM_SEED)

    for i in range(n_samples):
        pulses = random_pulse_sequence(
            horizon_s=horizon_s,
            c_min=c_min,
            c_max=c_max,
            pulse_dur_s_range=pulse_dur_s_range,
            rest_dur_s_range=rest_dur_s_range,
            rng=rng,
        )

        # Run PyBaMM simulation
        t, V, I_A = run_simulation(
            pulses, model=model, params_name=params_name, period_s=period_s
        )

        # Build the intended C-rate time series directly from the pulse plan
        c_profile = np.zeros_like(t_eval, dtype=float)
        cursor = 0.0
        for c, dur, rest in pulses:
            mask_dis = (t_eval >= cursor) & (t_eval < cursor + dur)
            c_profile[mask_dis] = c
            cursor += dur
            if rest > 0:
                mask_rest = (t_eval >= cursor) & (t_eval < cursor + rest)
                c_profile[mask_rest] = 0.0
                cursor += rest

        # Resample voltage to fixed grid
        V_res = resample_to_fixed_grid(t, V, t_eval)

        currents_c.append(c_profile)
        voltages.append(V_res)

    X = np.stack(currents_c, axis=0)  # shape: (n_samples, N)
    y = np.stack(voltages, axis=0)    # shape: (n_samples, N)
    return t_eval, X, y


# =========================
# 3) Train scikit-learn MLP
# =========================
def train_mlp_multioutput(X, y, test_size=0.2, random_state=42, verbose=True):
    """
    Train a multi-output MLPRegressor: X -> y (vectors).
    Applies feature scaling and target scaling for stability.
    Returns:
        pipeline, (X_test, y_test, y_pred), metrics_dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base_mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        batch_size=64,
        random_state=random_state,
        verbose=verbose,
    )

    pipeline = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", TransformedTargetRegressor(
            regressor=base_mlp,
            transformer=StandardScaler(with_mean=True, with_std=True)
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    return pipeline, (X_test, y_test, y_pred), metrics


# =========================
# 4) Main: run & visualize
# =========================
def main():
    print("Generating synthetic dataset with PyBaMM ...")
    t_eval, X, y = generate_dataset(
        n_samples=N_SAMPLES,
        horizon_s=HORIZON_S,
        period_s=PERIOD_S,
        model=MODEL,
        params_name=PARAM_SET_NAME,
        c_min=C_MIN,
        c_max=C_MAX,
        pulse_dur_s_range=PULSE_DUR_RANGE_S,
        rest_dur_s_range=REST_DUR_RANGE_S,
    )

    print("Training scikit-learn MLP (multi-output regression) ...")
    model_pipe, (X_test, y_test, y_pred), metrics = train_mlp_multioutput(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, verbose=True
    )
    print("Test metrics:", metrics)

    # Plot a random held-out example
    idx = np.random.randint(0, X_test.shape[0])
    plt.figure(figsize=(9, 4))
    plt.plot(t_eval, y_test[idx], label="PyBaMM (true)", lw=2, color="tab:red", ls='--')
    plt.plot(t_eval, y_pred[idx], label="MLP (pred)", lw=2, color="tab:blue")
    plt.xlabel("Time [s]")
    plt.ylabel("Terminal voltage [V]")
    plt.title("Voltage profile prediction (test example)")
    plt.legend()
    #plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('NN_pred_V.pdf', bbox_inches='tight')
    plt.show()

    # f, ax = plt.subplots(2,1,figsize=(9,6.5), gridspec_kw={'height_ratios': [.5, 1]}, sharex=True)
    # ax[0].plot(X_test[idx], label='Input $I(t)$', color='black')
    # ax[1].plot(y_test[idx], label='True $V(t)$', color='tab:red', linestyle='dashed')
    # ax[1].plot(y_pred[idx], label='Predicted $V(t)$', color='tab:blue')
    # ax[1].legend()
    # ax[0].legend()
    # ax[1].set_xlabel('Time step')
    # ax[0].set_ylabel('Current [A]')
    # ax[1].set_ylabel('Voltage [V]')
    # plt.savefig('NN_pred_I.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()
