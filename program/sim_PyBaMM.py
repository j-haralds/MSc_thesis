import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
import scipy.stats as stats

def simulate_DC(I0):
    model = pybamm.lithium_ion.SPM()
    param = model.default_parameter_values
    def my_DC_current(t, I0=I0):
        return I0

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    param['Nominal cell capacity [A.h]'] =  pybamm.Scalar(1.0)
    param['Current function [A]'] = my_DC_current

    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    t_eval = np.linspace(0,3600 / I0,100)

    solver = pybamm.IDAKLUSolver(atol=1e-7, rtol=1e-5)
    solution = solver.solve(model, t_eval)

    return solution,model,param




def GP_process(alpha, X,y,X_test):
    kernel_GP = gp.kernels.RBF(length_scale=alpha[0]) * gp.kernels.ConstantKernel(constant_value=alpha[1])
    gp_model = gp.GaussianProcessRegressor(kernel=kernel_GP,optimizer=None,normalize_y=False)

    gp_model.fit(X, y)

    gp_mu, gp_cov = gp_model.predict(X_test, return_cov=True)

    return gp_mu, gp_cov, gp_model


def gen_current(t):
    sig = 0.1
    l = 250
    alpha = [l, sig]
    t_0 = np.arange(0,7) * 600
    i0 = np.zeros_like(t_0)
    i0[1] = 0.5#i0[1] = np.random.choice([0.25, 0.75])
    i0[2:] = np.random.uniform(sig * 3,1 - 3 * sig,size=5)
    mu, cov, gp_model = GP_process(alpha, t_0.reshape(-1,1), i0.reshape(-1,1), t.reshape(-1,1))
    sample = stats.multivariate_normal.rvs(mean=mu, cov=cov)
    sample = np.clip(sample, 0, 1)
    I_int = np.trapezoid(sample, t,dx=3.6)
    sample = sample * 3600 / I_int

    return sample
    


def current_profile(t):
    times =    np.linspace(0, 3600, 1000)  
    currents = gen_current(times)
    return pybamm.Interpolant(times, currents, pybamm.t)



def simulate_GRF():
    model = pybamm.lithium_ion.DFN()
    param = model.default_parameter_values
    def my_current(t):
        return current_profile(t)
    param["Current function [A]"] = my_current

    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    param['Nominal cell capacity [A.h]'] =  pybamm.Scalar(1.0)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    t_eval = np.linspace(0,3600)

    solver = pybamm.IDAKLUSolver(atol=1e-7, rtol=1e-5)
    solution = solver.solve(model, t_eval)

    return solution,model, param


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



plt.plot(gen_current(np.linspace(0,3600,1000)))
plt.show()