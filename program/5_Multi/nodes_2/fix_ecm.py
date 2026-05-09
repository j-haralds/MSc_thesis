import scipy
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


LIMS = {
    'R0': (0.001, 0.1),  
    'R1': (0.001, 0.5),  
    'C1': (500, 2.5e4)   
}



def ECM_solve_du(theta, i, t, U0 = 0):

    I_interp = interp1d(t, i, fill_value="extrapolate")

    def dU(u, t,R,C):
        I_t = I_interp(t)
        return -u / (R * C) + I_t / C 
    
    U1 = scipy.integrate.odeint(dU, U0, t, args=(theta[1],theta[2])).flatten()
    U = U1
    eta_model =  i*theta[0] + U
    return eta_model


def ECM_solve_curvefit(x, R0, R1, C1):
    t, i = x
    eta_model = ECM_solve_du(theta=[R0, R1, C1], i=i, t=t)
    return eta_model

def parameter_estimation_curvefit(tr,I,lims = LIMS, U0 = 0, ret_elements = False):
    bounds = ([lims['R0'][0], lims['R1'][0], lims['C1'][0]], [lims['R0'][1], lims['R1'][1], lims['C1'][1]])
    p0 = [(lims['R0'][0] + lims['R0'][1]) / 2, (lims['R1'][0] + lims['R1'][1]) / 2, (lims['C1'][0] + lims['C1'][1]) / 2]
    # if type(tr['I']) == float:
    #     I = tr['I']*np.ones_like(tr['t'])
    # else:
    #     I = tr['I']
    xdata = [tr['t'], I]
    ydata = tr['eta']
    popt, pcov = scipy.optimize.curve_fit(ECM_solve_curvefit, xdata, ydata, p0=p0, bounds=bounds)
    R0, R1 , C1 = popt
    eta = ECM_solve_curvefit(xdata, R0, R1, C1)
    if ret_elements:
        return eta, R0, R1, C1
    else:
        return eta 