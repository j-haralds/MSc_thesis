
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


## Numbers at the end of name are supposed to note some kind ofchronological version name


version 3 should be the same but with more layers in one

Version 4 extension of 3 but with s node


ecm_node_0507_1638_snode_lib4_combo_dynamic_netR0_R0c_R1c_C1c_89.52min_16h_0_100eps.pt


FINAL MODEL: 0515_0840_b4_combo_softplus_combo_V-dynamic_F-dynamic_unconstr_700.40min_16h_2500eps.pt