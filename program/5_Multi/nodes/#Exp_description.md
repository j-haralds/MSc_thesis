## Version 1
Basic model, batched and not
Only electrics
Both works equally good and fast with batch_size=1

## Version 2 
ECM and EMM


## 2_ecmm_node_b_heat_ready
Definitions and run, with guards # %%

### ecmm_node_b_heat_ready_lib
Only definitions without guard # %%. Updated and left 2_ecmm_node_b_heat_ready in the dust!!
To be trained by ecmm_node_b_heat_ready_train
Basically 2_ecmm_node_b_heat_ready split in two



## Constrained network tests
Paste the constrained networks here and the name of the model. För att hålla koll på exakt nätstruktur sen när den laddas in

### Exp 1
    - CONFIG = {
        'R1_mode': 'net', 
        'C1_mode': 'net', 
        'R0_mode': 'net', 
        'n_hidden': N_HIDDEN,
            'R1_constrained': 'false',
            'C1_constrained': 'true', 'C1_min': 1000.0, 'C1_max': 20000.0,
            'R0_constrained': 'false',
    }

    class C1NetConstrained(nn.Module):
        """(SOC, I, u) → C1 > 0  [F].  One hidden layer, softplus output, linear constraint."""
        def __init__(self, config, n_hidden=32, I_ref=20.0):
            super().__init__()
            self.I_ref = I_ref
            self.net = nn.Sequential(
                nn.Linear(3, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, 1),
            )
            self.C1_min = config.get('C1_min')
            self.C1_max = config.get('C1_max')
            print(f'C1 constrained to [{self.C1_min}, {self.C1_max}] F')

        def forward(self, soc, I_norm, u):
            x = torch.stack([soc, I_norm, u], dim=-1)   # (..., 3)
            s = torch.sigmoid(self.net(x)).squeeze(-1)  # (0, 1)
            return self.C1_min + s * (self.C1_max - self.C1_min)


### Exp 2
ecm_node_net_no_socR0_netC1_60.4089min_1b_32h_100eps.pt
R0(i, u)
With semi-implicit euler


## staged_lib.py
Stage 2 on CC and not optional to train R1

## staged_lib_2.py
Stage 2 on pulse C, trains k and optional to train R1.
