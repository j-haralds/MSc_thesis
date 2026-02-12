import pybamm
import numpy as np
import matplotlib.pyplot as plt

def simulate(I0):
    model = pybamm.lithium_ion.DFN()
    param = model.default_parameter_values
    def my_current(t, I0=I0):
        return I0
    
    param["Current function [A]"] = my_current
    param["Nominal cell capacity [A.h]"] = 2.

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    t_eval = np.linspace(0,3600 / I0,100)

    solver = pybamm.IDAKLUSolver(atol=1e-7, rtol=1e-5)
    solution = solver.solve(model, t_eval)

    return solution,model


def current_profile(t, I0):
    times =    np.array([0, 600, 601, 1200, 1201, 1800, 1801, 2400])
    currents = np.array([2.,  2.,   0,    0  ,0.5 , 0.5,  1.,     1.]) * I0  
    return pybamm.Interpolant(times, currents, pybamm.t)



def simulate_variable_current(I0):
    model = pybamm.lithium_ion.DFN()
    param = model.default_parameter_values
    def my_current(t,i0=I0):
        return current_profile(t,i0)
    
    param["Current function [A]"] = my_current
    param["Nominal cell capacity [A.h]"] = 1.

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    t_eval = np.linspace(0,3600 / I0,100)

    solver = pybamm.IDAKLUSolver(atol=1e-7, rtol=1e-5)
    solution = solver.solve(model, t_eval)

    return solution,model


def get_voltage(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Battery voltage [V]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],1000)
    return t_,npc(t_)


def get_volt_variable(I):
    solution, model = simulate_variable_current(I)
    V = solution.observe(model.variables['Battery voltage [V]'])
    It = solution.observe(model.variables['Current variable [A]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],1000)
    return t_,V(t_), It(t_)


def get_OC_voltage(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Battery open-circuit voltage [V]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],1000)
    return t_,npc(t_)


def get_discharge_capacity(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Discharge capacity [A.h]'])
    print(len(solution['Time [s]'].entries))
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],1000)
    return t_,npc(t_)

def get_discharge_capacity_II(I):
    solution, model = simulate(I)
    npc = solution.observe(model.variables['Discharge capacity [A.h]'])
    t_ = np.linspace(0,solution['Time [s]'].entries[-1],1000)
    return t_,npc(t_), solution['Time [s]'].entries