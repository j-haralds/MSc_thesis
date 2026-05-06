
## clean_lib & clean_train
Similar to staged_2_lib but without all alternatives, that is hopefully the final model structure with R0 func and net for the rest.

Future updated function helpers and plotters will most likely be in here (28/04)

## cleaner
Removed batched
Forward into one forward function
Plot_predictions into one function (a lot of ifs)


## GP 2
tabulates the GP fit and uses interpolation from that grid for faster training
Run with GP_train.py which have started the train single implementation, not quite working. Staged should still work.

## train single
trains only static with eta = R1 i or dynamic with full ode, or staged
