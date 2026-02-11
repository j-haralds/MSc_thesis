
import pybamm
import matplotlib.pyplot as plt
import numpy as np

# Load the DFN model (P2D)
model = pybamm.lithium_ion.DFN()

# create geometry
geometry = model.default_geometry


# Choose a standard parameter set
param = model.default_parameter_values
# load parameter values and process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_finite_volume_methods)
disc.process_model(model);

# solve model
solver = model.default_solver
t_eval = np.linspace(0, 3600, 300)  # time in seconds
solution = solver.solve(model, t_eval)

quick_plot = pybamm.QuickPlot(
    solution, ["Terminal voltage [V]", "Current [A]"]
)
quick_plot.dynamic_plot();

# Create a simulation object
# sim = pybamm.Simulation(model, parameter_values=param)

# Run the simulation for 3600 seconds (1 hour)
# sim.solve([0, 3600])

# Plot the results
# sim.plot()
