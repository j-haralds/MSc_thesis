import numpy as np
import matplotlib.pyplot as plt
import pybamm

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
        model = pybamm.lithium_ion.DFN()

    # params = pybamm.ParameterValues(params_name)
    params = model.default_parameter_values
    experiment = build_pulse_experiment(pulses, period_s=period_s)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)

    sol = sim.solve()

    return sol, model


# =========================
# 2) Dataset generation
# =========================
def random_pulse_sequence(
    horizon_s=3600.0,
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



def generate_data(
    horizon_t=7200.0, dt=1.0, model=None, params_name="Default",
    c_min=0.7, c_max=0.8, pulse_dur_s_range=(100, 500), rest_dur_s_range=(100, 500)
):
    """
    Generate synthetic dataset:
      X: current profile (length N)
      y: voltage profile (length N)
    Returns:
      t_eval, X, y
    """

    t_eval = np.arange(0, horizon_t, dt)

    rng = np.random.default_rng(1337)
    rng = None

    pulses = random_pulse_sequence(
            horizon_s=horizon_t, c_min=c_min, c_max=c_max,
            pulse_dur_s_range=pulse_dur_s_range, rest_dur_s_range=rest_dur_s_range, rng=rng,
        )

    # Run PyBaMM simulation
    sol, model = run_simulation(
        pulses, model=model, params_name=params_name, period_s=dt
    )



    #return t_, t_eval, I(t_), V(t_), ocv(t_)
    return sol, model, t_eval


def get_Bvolt():
    sol, model, t_eval = generate_data()
    V = sol.observe(model.variables['Battery voltage [V]'])
    I = sol.observe(model.variables['Current variable [A]'])
    t_ = np.linspace(0, sol['Time [s]'].entries[-1], 1000)
    return t_, t_eval, I, V

def get_ocv():
    sol, model, t_eval = generate_data()
    ocv = sol.observe(model.variables['Open circuit voltage [V]'])
    I = sol.observe(model.variables['Current variable [A]'])
    t_ = np.linspace(0, sol['Time [s]'].entries[-1], 1000)
    return t_, t_eval, I, ocv

def get_npc():
    sol, model, t_eval = generate_data()
    npc = sol.observe(model.variables['Negative particle concentration [mol.m-3]'])
    I = sol.observe(model.variables['Current variable [A]'])
    t_ = np.linspace(0, sol['Time [s]'].entries[-1], 1000)
    print('Concentration profile shape (r, x, t)')
    return t_, t_eval, I, npc


def get_ppc():
    sol, model, t_eval = generate_data()
    ppc = sol.observe(model.variables['Positive particle concentration [mol.m-3]'])
    I = sol.observe(model.variables['Current variable [A]'])
    t_ = np.linspace(0, sol['Time [s]'].entries[-1], 1000)
    print('Concentration profile shape (r, x, t)')
    return t_, t_eval, I, ppc

# =========================
# 4) Main: run & visualize
# =========================
# def main():
#     t, t_eval, I, V = get_values()
#     print(t.shape)


# if __name__ == "__main__":
#     main()
