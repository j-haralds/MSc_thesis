import pybamm
import numpy as np

def run_single_simulation(task_payload):
    """
    Runs a single PyBaMM simulation. Designed to be called by a worker process.
    """
    params, current, soc, config = task_payload

    model_options = config['pybamm_settings'].get('model', 'SPM')
    solver_name = config['pybamm_settings'].get('solver', 'CasadiSolver')
    param_key_map = config.get('param_key_map', {})

    if model_options == 'SPM':
        model = pybamm.lithium_ion.SPM()
    else:
        raise NotImplementedError(f"Model {model_options} not implemented.")

    model.events = []
    param_set = pybamm.ParameterValues(config["parameter_set"])
    short_to_full = {name: info[0] for name, info in (config.get("parameter_bounds", {}) or {}).items()}

    for short_name, value in params.items():
        if short_name == "A":
            param_set["Electrode height [m]"] = param_set["Electrode width [m]"] = np.sqrt(value)
        else:
            param_set[short_to_full[short_name]] = value

    t_eval = np.linspace(0, config["pybamm_settings"]["t_max_s"], config["pybamm_settings"]["t_num_points"])
    param_set["Current function [A]"] = pybamm.Interpolant(t_eval, -current, pybamm.t)

    solver = getattr(pybamm, solver_name)()
    sim = pybamm.Simulation(model, parameter_values=param_set, solver=solver)

    try:
        sol = sim.solve(initial_soc=float(soc), t_eval=t_eval)

        # Rename parameters based on the map, but keep all of them.
        # Apply log10 transformation to diffusivity values to match reference format
        mapped_params = {}
        for k, v in params.items():
            new_key = param_key_map.get(k, k)
            if new_key in ['Dan', 'Dca']:  # Apply log10 to diffusivity values
                mapped_params[new_key] = np.log10(v)
            else:
                mapped_params[new_key] = v

        # Add default diffusivity values if they weren't sampled
        # TODO: make D param colum  the same
        if 'Dan' not in mapped_params and 'Dan' in param_key_map.values():
            mapped_params['Dan'] = np.log10(param_set["Negative particle diffusivity [m2.s-1]"])
        if 'Dca' not in mapped_params and 'Dca' in param_key_map.values():
            mapped_params['Dca'] = np.log10(param_set["Positive particle diffusivity [m2.s-1]"])

        return {
            "cn_anode": sol["Negative particle concentration"].entries[:, 0, :],
            "c0_anode": sol["Negative particle concentration"].entries[:, 0, 0],
            "cn_cathode": sol["Positive particle concentration"].entries[:, 0, :],
            "c0_cathode": sol["Positive particle concentration"].entries[:, 0, 0],
            "current": current,
            "soc": soc,
            **mapped_params,
        }
    except (pybamm.SolverError, pybamm.ModelError):
        return None


