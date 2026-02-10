import pybamm
import numpy as np
import matplotlib.pyplot as plt

def simulate(I0):
    model = pybamm.lithium_ion.DFN()
    param = model.default_parameter_values
    def my_current(t, I0=I0):
        return I0
    
    param["Current function [A]"] = my_current
    param["Nominal cell capacity [A.h]"] = 10.
    param['Number of cells connected in series to make a battery'] = 1.


    # param = pybamm.ParameterValues(
    # {
    #     "Nominal cell capacity": 10.,
    #     "Current function [A]": my_current,
    # }
    # )

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    t_eval = [0,3*3600]

    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-3)
    solution = solver.solve(model, t_eval)

    return solution,model


def get_voltage(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Battery voltage [V]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],101)
    return t_,npc(t_)


def get_OC_voltage(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Battery open-circuit voltage [V]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],101)
    return t_,npc(t_)


def get_discharge_capacity(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Discharge capacity [A.h]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],101)
    return t_,npc(t_)

def get_discharge_capacity(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Discharge capacity [A.h]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],101)
    return t_,npc(t_)