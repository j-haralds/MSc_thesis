# Description

Simple well structured NNs that usitlize forward Euler integration. The idea is to extend them with constraints.

Data from SPM

# Versions and experimental log

## Version 1
Input I_t, c_t output c_dot_t

## Version 1.1
Input I_t, I_dot_t output c_dot_t

## Version 1.2
K unroll + physics loss

## Version 2
K unroll

## Version 4
Basic working for constant currents. One-step training. Predict on initial condition and roll out

## Version 4.1
Uses 'physical' time T_horizon /(T-1), otherwise wrong units in Euler forward
Valid mask 
Consistent scaling
Time independent training (B*T, 4)

### Cons
- Normal training large integration drift, but decreasing train loss
- Rollout loss (on its own, not good), increasing train loss
- Combining the two does not work

### Pros
- It works for constant currents (predicts equally bad on GRF when trained on DC, but less noisy)
- Works for pulses (normal training + rollout) (10 - 500 steps)

### Log
- When normal masked mse is added within rollout. Loss and predictions becomes worse when prefactor is gradually decreased. Meaning only rollout is bad
- 50 eps w.o. rollout: MSE=0.2677, max relative cn error 0.0253. Looks ok
- 50 eps + 50 eps rollout w.o. normal mse: max relative cn error 0.0154. Looks ok, slightly better relax.
- 100 eps w.o. rollout: MSE=0.2047, max relative cn error 0.012. Looks better but not perfect relax.
- 50 eps + 50 eps rollout w normal mse (0.1 factor): MSE= 0.2461, MSE_roll=0.0247 (not decreasing). max relative cn error 0.0228.
- Rollout from 10 to 1000 steps, is worse

    - Conclusion: Rollout does not add any value. Might be implemented incorrectly

## Version 5
W.o. rollout loss, only one step training.
- 100 eps: MSE=0.1909, max relative cn error 0.015. 
Needs to have the same T in the data as in model, to obtain the same DT

## Version 5.1
Physics loss. Not working


