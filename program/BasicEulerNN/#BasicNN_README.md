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
- Works for pulses (normal training + rollout)
