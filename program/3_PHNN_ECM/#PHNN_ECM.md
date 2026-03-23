

## Version 4
Can evaluate on different dt than data DT. 
This was rollout before:
 ```
def rollout(model, x, I_phys, ocv_phys, dt, evalu=False):
     """
     x0    : (B, 2)   [Q0 in Coulombs, q1_0=0]
     Q_0   : (B,)     initial charge, e.g. from coulomb counting or assumed full
     """
     T_steps = I.shape[1]
     V_pred = []
     C1 =[]

     for t in range(T_steps):
         H_pred = model(x)   (B, 1)
         I_t = I_phys[:, t]          (B,)

         dH_dq = dH_dx(H_pred, x)[:, 1]
         V_t_pred = ocv_phys[:, t] - model.R0 * I_t - dH_dq
         V_pred.append(V_t_pred)

         x = euler_step(H_pred.squeeze(), x, I_t, dt)

         C1.append(2 * H_pred / dH_dq**2)
        
     steps = t+1
     print(steps)
     if evalu:
         return torch.stack(V_pred, dim=1), torch.stack(C1, dim=1)   # (B, T)
     else:
         return torch.stack(V_pred, dim=1)   # (B, T)
```


It still needs to train on DT though
C1 is time dependent

## Version 4.1
Constant C1 from dH_dq calculation instead of autograd
Implicit Euler - Halvt fungerande

## Version 5
Added RC branch q1 and q2, instead of Q

