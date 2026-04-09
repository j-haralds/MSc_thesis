import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as TF


def ECM_model(Ue, I, R0, U1):
    #Ue = soc_to_Ue(soc,return_torch=True    )
    return Ue - I * R0 - U1

def Mech_model(u,Fs , k = 0.5316441632537718):
    return -k * u + Fs

def soc_force_func(soc, variable = 'F'):
    
    if variable == 'F': 
        forcing_const = 0.9975
    elif variable == 'U':
        forcing_const = 0.995
    return 1 - np.exp((soc-1) / (1 - forcing_const));

def R0_func(I, u):
    R0_est = -0.0001887521*u - 7.049519e-5*I + 0.00844669
    return TF.relu(R0_est)




class U1Net(nn.Module):
    def __init__(self, input_size=4, hidden_size=64): 
        super().__init__()
        
        # Main network (Fs, U1)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
        
        # Separate network for R0
        self.R0_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Optional fallback constant R0
        self.R0_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, constraints=False, R0_mode="const"):
        out = self.net(x)

        Fs_raw = out[:, 0]
        U1_raw = out[:, 1]

        # --- R0 logic ---
        if R0_mode == "net":
            R0 = TF.softplus(self.R0_net(x)).squeeze(-1) / 100
            
        elif R0_mode == "func":
            R0 = R0_func(x[:, 1], x[:, 0])
            
        elif R0_mode == "const":
            R0 = TF.softplus(self.R0_param) / 100
            
        else:
            raise ValueError("Invalid R0_mode")

        # --- Fs, U1 logic ---
        if constraints:
            soc = x[:, 2]
            Fs = soc_force_func(soc, variable='F') * Fs_raw
            U1 = soc_force_func(soc, variable='U') * TF.softplus(U1_raw)
        else:
            Fs = Fs_raw
            U1 = TF.softplus(U1_raw) / 100

        return Fs, U1, R0


def next_step(model, x, constraints=False, R0_mode="const"):
    print(x.shape)
    Fs, U1, R0 = model(x, constraints=constraints, R0_mode=R0_mode)
    return torch.stack([ECM_model(x[:, 3], x[:, 0], R0, U1), Mech_model(x[:, 1], Fs)], dim=1)