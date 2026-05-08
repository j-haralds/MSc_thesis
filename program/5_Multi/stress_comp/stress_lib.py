import numpy as np
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
import scipy
import sympy as sp
from scipy.interpolate import interp1d


import sklearn.gaussian_process as gp
from scipy import stats




sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('..')

sys.path.append('../nodes_2')
import plot_settings
import importlib
#import multi_exp_log
#import Param_symbol

import GP
#import ecmm_node_cleaner_swell_GP_lib as _lib


from IPython.display import display, Math

COLORS = plot_settings.colors()

def import_reload():
    plot_settings.apply()
    importlib.reload(GP)
 #   importlib.reload(_lib)
    
import_reload()
GP_FIT = GP.GP_process()

DT = 1
Q0 = 17921.57581 


def ECM_solve_du(u_par,C_rate, U0 = 0, discharge = True, mode = 'DC', NN = False):
    
    t = np.arange(0, 2 * int(3600 / C_rate), DT)
    if discharge:
        print(f"Simulating discharge at C-rate: {C_rate}x")
    else:
        C_rate = -C_rate
        print(f"Simulating charge at C-rate: {C_rate}x")

    # if mode == 'pulse':
    #     def I_interp(t):
    #         return pulse_current_profile(t, C_rate)
    #else:
    i = C_rate * Q0 / 3600 * np.ones_like(t)  # Assuming constant current for simplicity
    I_interp = interp1d(t, i, fill_value="extrapolate")

    i = I_interp(t)
    if not discharge:
        #i = -i
        soc0 = 0
    else:        
        soc0 = 1

    print(f"Initial conditions: SOC={soc0}, U1={U0}")
    def dSOC(SOC, t, C_rate=C_rate, u_par=u_par):
        I_t = I_interp(t)
        return -I_t / Q0
    
    
    def SOC(t,SOC0=soc0):
        SOC_t = scipy.integrate.odeint(dSOC, SOC0, t).flatten()
        return SOC_t
    
    SOC_t = SOC(t)
    #plt.show()
    soc_interp = interp1d(t,SOC_t,bounds_error=False,fill_value=(SOC_t[0], SOC_t[-1]))

    def dU(V_1, t, soc_int = soc_interp, d = u_par, NN = NN):
        I_t = pulse_current_profile(t, C_rate)
        SOC_t = soc_int(t)
        if not discharge:
            SOC_t = 1 - SOC_t
        R = R1(np.abs(C_rate), d, SOC_t, NN = NN)
        C = C1(np.abs(C_rate), d, SOC_t, NN = NN)
        return -V_1 / (R * C) + I_t / C 
    
    def mech(t, soc_int = soc_interp, d = u_par, NN = NN):
        SOC_t = soc_int(t)
        return k(np.abs(C_rate), d, SOC_t, NN = NN)*d

    # Solve ODE
    

    def ECM_system(y, t, u_par, NN = NN):

        SOC, U1 = y
        #if t == 0:
            #print(f"Initial conditions: SOC={SOC}, U1={U1}")

        I_t = I_interp(t)

        if not discharge:

            SOC_eff = 1 - SOC

        else:

            SOC_eff = SOC
        R = R1(np.abs(C_rate), u_par, SOC_eff, NN = NN)
        C = C1(np.abs(C_rate), u_par, SOC_eff, NN = NN)
        dSOC_dt = -I_t / Q0

        dU1_dt = -U1 / (R * C) + I_t / C

        return [dSOC_dt, dU1_dt]

    U1 = scipy.integrate.odeint(dU, U0, t).flatten()
    y0 = [soc0, U0]

    sol = scipy.integrate.odeint(ECM_system, y0, t, args=(u_par,))

    SOC_t = sol[:, 0]

    U1 = sol[:, 1]

    U = U1 #+ U2
    # Voltage model
    R0_vals = []
    for s in SOC_t:
        R0_vals.append(R0(np.abs(C_rate),u_par,s, NN = NN))
    eta_model =  i*R0_vals + U
    V_B = GP.soc_to_Ue(SOC_t, GP_FIT) - eta_model

    if not discharge:
        valid_inds = V_B< 5.5
    else:
        valid_inds = V_B> 2.3
    V_B = V_B[valid_inds]
    t = t[valid_inds]
    soc = SOC_t[valid_inds]
    Ue = GP.soc_to_Ue(soc, GP_FIT)
    i = I_interp(t)

    return V_B,soc, Ue,i ,t#, mech(t)
