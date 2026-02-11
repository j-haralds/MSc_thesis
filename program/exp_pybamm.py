import numpy as np
import matplotlib.pyplot as plt

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
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

# Simulation horizon and sampling
HORIZON_S = 1800.0      # total length of each experiment [s]
PERIOD_S = 1.0          # sampling period [s] (also PyBaMM experiment period)

# Dataset size
N_SAMPLES = 1         # increase to improve training
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

    params = pybamm.ParameterValues(params_name)
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
    n_samples=500,
    horizon_s=1800.0,
    period_s=1.0,
    model=None,
    params_name="Chen2020",
    c_min=0.5,
    c_max=2.5,
    pulse_dur_s_range=(30, 300),
    rest_dur_s_range=(0, 120),
):
    """
    Generate synthetic dataset:
      X: current C-rate profile (length N)
      y: voltage profile (length N)
    Returns:
      t_eval, X, y
    """
    t_eval = np.arange(0.0, horizon_s + 1e-9, period_s)  # fixed grid
    currents = []    # list of length-N arrays
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

        # Resample current
        I_res = resample_to_fixed_grid(t, I_A, t_eval)

        currents.append(I_res)
        currents_c.append(c_profile)
        voltages.append(V_res)

    X = np.stack(currents_c, axis=0)  # shape: (n_samples, N)
    Xc = np.stack(currents, axis=0)    # shape: (n_samples, N)
    y = np.stack(voltages, axis=0)    # shape: (n_samples, N)
    return t_eval, X, Xc, y



# =========================
# 4) Main: run & visualize
# =========================
def main():
    print("Generating synthetic dataset with PyBaMM ...")
    t_eval, X, Xc, y = generate_dataset(
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


    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))
    ax0.plot(t_eval, X[0], label="C-rate profile [C]")
    ax0.set_title("C-rate Profile")
    ax0.set_xlabel("Time [s]")
    ax1.plot(t_eval, Xc[0], label="Current [A]")
    ax1.set_title("Current Profile")
    ax1.set_xlabel("Time [s]")
    ax2.plot(t_eval, y[0], label="Voltage [V]")
    ax2.set_title("Voltage Profile")
    ax2.set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
