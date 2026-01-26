#!/usr/bin/env python

# run it: python3 1D_model_JH.py


import matplotlib.pyplot as plt
import pybamm 
import numpy as np

model = pybamm.lithium_ion.DFN()
print(model)

geometry = model.default_geometry


param = model.default_parameter_values

param.process_model(model)
param.process_geometry(geometry)

mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)


solver = model.default_solver
t_eval = np.linspace(0, 3600, 100)
solution = solver.solve(model, t_eval)

plot = pybamm.QuickPlot(solution)

plot.dynamic_plot()

