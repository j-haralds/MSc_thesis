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

sol, model,params = sim_PyBaMM.simulate_DC(0.1)
t = np.linspace(0,sol['Time [s]'].entries[-1],1000)
ocp_p = sol.observe(model.variables['X-averaged positive electrode open-circuit potential [V]'])(t)
ocp_n = sol.observe(model.variables['X-averaged negative electrode open-circuit potential [V]'])(t)
cn = sol.observe(model.variables['X-averaged negative particle concentration [mol.m-3]'])(t)[-1,:]
cp = sol.observe(model.variables['X-averaged positive particle concentration [mol.m-3]'])(t)[-1,:]
cn_max = params['Maximum concentration in negative electrode [mol.m-3]']
cp_max = params['Maximum concentration in positive electrode [mol.m-3]']

n_fit = pol_fit(cn/cn_max, ocp_n)
p_fit = pol_fit(cp/cp_max, ocp_p)

def OCP(c,which_c = 'n',n_fit = n_fit,p_fit= p_fit):
    '''Calculate the open circuit poential from the surface concentrations.
    Parameters:
    cn: surface concentration in the negative electrode non-normalized
    cp: surface concentration in the positive electrode non-normalized
    '''
    if which_c == 'n':
        cmax = params['Maximum concentration in negative electrode [mol.m-3]']
        c_fit = n_fit
    elif which_c == 'p':
        cmax = params['Maximum concentration in positive electrode [mol.m-3]']
        c_fit = p_fit
    else:
        raise ValueError('which_c must be either "n" or "p"')
    C = c / cmax
    ocp = c_fit(C)
    return ocp



def eta(i,c,params, which_c = 'n'):
    if which_c == 'n':
        L = params['Negative electrode thickness [m]']
        Rs = params['Negative particle radius [m]']
        eps = params['Negative electrode active material volume fraction']
        a = 3 * eps / Rs
        alpha = 0.5
    elif which_c == 'p':
        L = params['Positive electrode thickness [m]']
        Rs = params['Positive particle radius [m]']
        eps = params['Positive electrode active material volume fraction']
        a = 3 * eps / Rs
        alpha = 0.5
    else:
        raise ValueError('which_c must be either "n" or "p"')
    A = params['Electrode height [m]'] * params['Electrode width [m]']
    F = sc.physical_constants['Faraday constant'][0]
    R = sc.R
    T = params['Initial temperature [K]']
    j0 = j(c,params, which_c)

    return R*T/F * np.arcsinh(i / (2 * a * A *L*j0))/alpha


def j(c_surf,params, which_c = 'n'):
    ce = params['Initial concentration in electrolyte [mol.m-3]']
    kn=2e-5;kp=6e-7
    F = sc.physical_constants['Faraday constant'][0]
    R = sc.R
    T = params['Initial temperature [K]']

    if which_c == 'n':
        cmax = params['Maximum concentration in negative electrode [mol.m-3]']
        k = kn
    elif which_c == 'p':
        cmax = params['Maximum concentration in positive electrode [mol.m-3]']
        k = kp
    else:
        raise ValueError('which_c must be either "n" or "p"')
    j0 = k * np.sqrt(ce * c_surf * (cmax-c_surf)) 

    return j0

def j_bar(c_surf,params, which_c = 'n'):
    j0 = j(c_surf,params, which_c)
    F = sc.physical_constants['Faraday constant'][0]
    R = sc.R
    T = params['Initial temperature [K]']
    return 2 * F * j0 * np.sinh(F * eta(1,cp,cn,params) / (2*R*T))    

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
    jn0,jp0  = j(cn,cp,params)

    #i_app = i / A
    #jn = i_app / Ln
    #jp = i_app / Lp
    alpha_p = 0.5
    alpha_n = 0.5
    e = R*T/F *( (1/alpha_p) * np.arcsinh(i / (2 * ap * A *Lp*jp0)) - (1/alpha_n) * np.arcsinh(-i / (2 * an * A *Ln*jn0)))
    return e #(2 * R * T / F) * (np.arcsinh(jn/jn0) -  np.arcsinh(jp/(jp0)))


def omega():
    return 0

def dGdt_am(c,params = params, which_c = 'n'):
    if which_c == 'n':
        Rs = params['Negative particle radius [m]']
    elif which_c == 'p':
        Rs = params['Positive particle radius [m]']
    else:
        raise ValueError('which_c must be either "n" or "p"')
    
    c_surf = c[-1,:]
    
    dG = 4 * np.pi * Rs**2 * j(c,params,which_c) * OCP(c,which_c) - omega()

    return dG


def Gs(c,params = params, which_c = 'n'):
    if which_c == 'n':
        Rs = params['Negative particle radius [m]']
    elif which_c == 'p':
        Rs = params['Positive particle radius [m]']
    else:
        raise ValueError('which_c must be either "n" or "p"')
    c_surf = c[-1,:]
    c_int = np.linspace(np.min(c),np.max(c),1000)
    F = sc.physical_constants['Faraday constant'][0]
    gs = - F * np.trapezoid(OCP(c,which_c),c_surf)
    return G
    

# def V_terminal(I,cp,cn, params, K = [0.05,0.2]):
#     U = OCV(cn,cp)
#     eta = eta_r(I,cp,cn,params=params, K=K)
#     return U - eta

