import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pybamm
import sys
sys.path.append('..')
import sim_PyBaMM
import torch


def pol_fit(c,V_ocv):
    coefs = np.polyfit(c, V_ocv, 10)
    return np.poly1d(coefs)

sol, model,params = sim_PyBaMM.simulate_DC1(1, T=1000, T_horizon=3600*2)
t = np.linspace(0,sol['Time [s]'].entries[-1],1000)
ocp_p = sol.observe(model.variables['X-averaged positive electrode open-circuit potential [V]'])(t)
ocp_n = sol.observe(model.variables['X-averaged negative electrode open-circuit potential [V]'])(t)
cn = sol.observe(model.variables['X-averaged negative particle concentration [mol.m-3]'])(t)[-1,:]
cp = sol.observe(model.variables['X-averaged positive particle concentration [mol.m-3]'])(t)[-1,:]
cn_max = params['Maximum concentration in negative electrode [mol.m-3]']
cp_max = params['Maximum concentration in positive electrode [mol.m-3]']

n_fit = pol_fit(cn/cn_max, ocp_n)
p_fit = pol_fit(cp/cp_max, ocp_p)

def OCP(c_n, c_p,n_fit = n_fit,p_fit= p_fit):
    '''Calculate the open circuit poential from the surface concentrations.
    Parameters:
    cn: surface concentration in the negative electrode non-normalized
    cp: surface concentration in the positive electrode non-normalized
    '''
    Cn = c_n / cn_max
    Cp = c_p / cp_max

    ocp_N = 0
    ocp_P = 0

    for coef_n,coef_p in zip(n_fit.coefficients,p_fit.coefficients):
        ocp_N += coef_n * Cn ** n_fit.order
        ocp_P += coef_p * Cp ** p_fit.order


    return ocp_P, ocp_N

def OCV(cn,cp):
    '''Calculate the open circuit poential from the surface concentrations.
    Parameters:
    cn: surface concentration in the negative electrode non-normalized
    cp: surface concentration in the positive electrode non-normalized
    '''
    return OCP(cn,cp)[0] - OCP(cn,cp)[1]

# def exchange_current_dens(c,k=None,K = [5,2]):
#     c_reduced = c / np.max(c)
#     a = 0.5
#     return (c_reduced * (1-c_reduced)) ** (1 - a)


def exchange_current_dens(cn,cp,params, kn=2e-5,kp=6e-7):
    ce = params['Initial concentration in electrolyte [mol.m-3]']
    cn_max = params['Maximum concentration in negative electrode [mol.m-3]']
    cp_max = params['Maximum concentration in positive electrode [mol.m-3]']
    jn0 = kn * np.sqrt(ce * cn * (cn_max-cn)) 
    jp0 = kp * np.sqrt(ce * cp * (cp_max-cp))
    return jn0, jp0 



def eta_r(i,cp,cn, params, K = [0.05,0.2]):
    Ln = params['Negative electrode thickness [m]']
    Lp = params['Positive electrode thickness [m]']
    A = params['Electrode height [m]'] * params['Electrode width [m]']
    Rk = params['Negative particle radius [m]']
    en = params['Negative electrode active material volume fraction']
    ep = params['Positive electrode active material volume fraction']
    ap = 3 * ep / Rk
    an = 3 * en / Rk
    T = params['Initial temperature [K]']
    
    R = sc.R
    F = sc.physical_constants['Faraday constant'][0]
    jn0,jp0  = exchange_current_dens(cn,cp,params)

    #i_app = i / A
    #jn = i_app / Ln
    #jp = i_app / Lp
    alpha_p = 0.5
    alpha_n = 0.5
    e = R*T/F *( (1/alpha_p) * np.arcsinh(i / (2 * ap * A *Lp*jp0)) - (1/alpha_n) * np.arcsinh(-i / (2 * an * A *Ln*jn0)))
    return e #(2 * R * T / F) * (np.arcsinh(jn/jn0) -  np.arcsinh(jp/(jp0)))

# def eta_r(i,cp,cn, params,K = [0.05,0.2]):
#     Ln = params['Negative electrode thickness [m]']
#     Lp = params['Positive electrode thickness [m]']
    
#     A = params['Electrode width [m]'] * params['Electrode height [m]']
#     T = params['Initial temperature [K]']
    
#     R = sc.R
#     F = sc.physical_constants['Faraday constant'][0]
#     jp0 = exchange_current_dens(cp,k = 'p',K=K)
#     jn0 = exchange_current_dens(cn,k = 'n',K=K)

#     i_app = i / A
#     jn = i_app / Ln
#     jp = -i_app / Lp
#     return 2 * R * T / F * (np.arcsinh(jn/jn0) -  np.arcsinh(jp/(jp0)))

def V_terminal(I,cp,cn, params, K = [0.05,0.2]):
    U = OCV(cn,cp)
    eta = eta_r(I,cp,cn,params=params, K=K)
    return U - eta