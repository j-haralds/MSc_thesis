
This directory consists of two main files for training and loading NN models: nn_model_train.py and nn_model_load.py.

These two files uses functions from the library nn_model_lib.py, which contains the network and model classes, training loop and plotters. This file in turn calls upon fix_ecm.py to generate fixed value ecm fit along with Ue_GP_comsol.py and Ue_GP_pybamm.py to generate a equilibrium voltage function.

Ue_run_comsol.txt is data from a OCV experiment run in Comsol and Ue_run_pybamm.txt is the corresponding data from pybamm generator in the directory data_pybamm.
