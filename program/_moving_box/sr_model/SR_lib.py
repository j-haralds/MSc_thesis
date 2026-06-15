from IPython.display import display, Math
from pysr import PySRRegressor, TemplateExpressionSpec
# import mean
from pysr import jl
jl.seval("using Statistics")
import matplotlib.pyplot as plt
import numpy as np
import plot_settings
plot_settings.apply()
COLORS = plot_settings.colors()
from IPython.display import display, Math
import pandas as pd

# =================================================
# POST PROCESSING
# =================================================

def get_SR_inds(RUN_ID):
    chosen_all_inds = {
        'SP': {
            'R0': 8,   # JN
            'R1': 8,   # JH
            'C1': 7,   # JN
            'k': 7,    # JH
            's': 6,     # JN
            'Ue':7
        }
    }

    best_all_inds = {
        'SP': {
            'R0': 19,  # JN
            'R1': 19,  # JH
            'C1': 24,  # JN
            'k': 22,   # JH
            's': 27,    # JN
            'Ue':15
        }
    }

    return chosen_all_inds[RUN_ID], best_all_inds[RUN_ID]


def get_ref_values():
    '''Reference values for normalization. 
    For s, C and d, these are the maximum values in the dataset. 
    For R0, R1, and k, these are arbitrary.
    Values can be adjusted based on the specific dataset and requirements.'''
    
    REF_VALUES = {
    'R0': 0.01,
    'R1': 0.01,
    'C1': 1000,
    'k':  0.01,
    's':  0.37266314,
    'sdot': 0.0001, 
    'C': 5,
    'd': 30,
    }
    return REF_VALUES
def get_latex_dict():
    '''LaTeX representations for each variable, used in plots and tables.'''

    latex_dict ={
        'R0': r'R_0',
        'R1': r'R_1',
        'C1': r'C_1',
        'k': r'k',
        's': r'\dot{s}',
        'Ue': r'U_{eq}'
    }
    return latex_dict


def get_units_dict():
    ''''Units for each variable, used in plots and tables. 
    Conversion factors are applied to convert to resonable units.'''
    
    units_dict = {
        'R0': r'm$\Omega$',
        'R1': r'm$\Omega$',
        'C1': r'kF',
        'k':  r'MN/µm',
        's':  r'm/s'
        }
    
    unit_conversion = {
        'R0': 1e3,   # from Ohm to mOhm
        'R1': 1e3,  # from Ohm to mOhm
        'C1': 1e-3,  # from F to kF
        'k':  1e-3, 
        's':  1e3
        }
    
    return units_dict, unit_conversion


# ================================================= 
# DATA PREPARATION 
# ================================================= 

def read_data(file_name):
    df = pd.read_csv(
        f"symbol_data/{file_name}.txt",
        sep=',',
        comment="%"
    )
    
    return df


def read_raw_data(file_name):
    df = pd.read_csv(
        f"{file_name}.txt",
        sep=';',
        comment="%"
    )
    return df

def prepare_data(data, index = 0, test = False):
    if test:
        i = index + int(0.8*len(data['trajectory'].unique()))
    else:
        i = index

    trajs = data[data['trajectory'] == i].reset_index(drop=True)

    return trajs


# =================================================
# SYMBOLIC REGRESSION SETUP AND RUN
# =================================================

def get_default_settings():
    bin_ops = ["+", "*", '-','/','^']
    un_ops = ['sqrt','square', 'cube','exp','log', 'sin', 'cos', 'tan']
    nest_const = {'sin':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                          'cos':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                          'tan':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                          'log':  {'sin': 0, 'cos': 0, 'tan': 0},
                          'exp':  {'sin': 0, 'cos': 0, 'tan': 0},
                          'exp':  {'exp': 0}
                          }
    consts = {"^": (-1, 1)}
    op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 1, 'sin': 3, 'cos': 3, 'tan': 3}
    var_names = ['i','d','soc']
    return bin_ops, un_ops, nest_const,consts, op_comps, var_names

def get_var_names(elem):
    if elem in ['R0', 'R1', 'C1','k'] or elem == None:
        return ['C','d','soc']
    elif elem in ['s']:
        return ['C','d','soc','s']
    elif elem in ['Ue']:
        return ['soc']
    else:
        raise ValueError(f"Unknown element: {elem}")

def get_settings(elem):
    bin_ops, un_ops, nest_const,consts, op_comps, var_names = get_default_settings()
    
    if elem == 'R0':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
    if elem == 'R1':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
    if elem == 'C1':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}

    if elem == 'k':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
    if elem == 's':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
    
    if elem == 'Ue':
        bin_ops = ["+", "*", '-','/']
        un_ops = ['sqrt','square', 'cube','exp','log', 'sin', 'cos', 'tan']
        nest_const = {'sin':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                      'cos':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                      'tan':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                      'log':  {'sin': 0, 'cos': 0, 'tan': 0},
                      'exp':  {'sin': 0, 'cos': 0, 'tan': 0},
                      }
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 1, 'sin':3,'cos':3,'tan':3}
    
    var_names = get_var_names(elem)
    
    return bin_ops, un_ops, nest_const,consts, op_comps, var_names

def setup_model(its = int(1e3), pops = 30, selection = "accuracy",run_id = None, elem = None):
    bin_ops, un_ops, nest_const, consts,op_comps, var_names = get_settings(elem)
    
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=bin_ops,
        unary_operators=un_ops,
        populations=pops,
        nested_constraints = nest_const,
        verbosity=0,     
        variable_names=var_names,    
        constraints = consts,
        batching = True,
        complexity_of_operators = op_comps,
        complexity_of_constants = 1,
        maxsize = 30,
        batch_size = 512,
        run_id= run_id
    )
    return model


def run_symbolic_regression(X, y, model = None,run_id = None, its = int(1e3), pops = 30, selection = "best", elem = None):
    var_name = get_var_names(elem)
    print(f"Variable names for element {elem}: {var_name}")
    if model is None:
        model = setup_model(run_id = run_id, its = its, pops = pops, selection = selection, elem = elem)
    print(f"Running symbolic regression for element {elem} with run_id {run_id}...")
    print(f"Settings: iterations={its}, populations={pops}, selection={selection}")
    model.fit(X, y, variable_names = var_name)
    return model

# =================================================
# VISUALIZATION
# =================================================


def print_models(df_model):
    for i in range(len(df_model)):
        print(f"Model {i}: Complexity={df_model['complexity'][i]}, Loss={df_model['loss'][i]}")
        print(f"Equation: {df_model['equation'][i]}\n")

def print_best_model(model, model_index = None, s = False):
    best = model.get_best()
    display(Math(f'\\LARGE {model.latex(model_index)}'))
    display(Math(r'\LARGE $x_0 = \mathrm{{c}}$-rate$'))
    display(Math(r'\LARGE $x_1 = \Delta u / L_{{tot}}$'))
    display(Math(r'\LARGE $x_2 = \mathrm{SOC}$'))
    if s:
        display(Math(r'\LARGE $x_3= s$'))
    print(f'Best model in symbolic form: {model.sympy()}\nLoss: {best["loss"]}\nBest model complexity: {best["complexity"]}')


# Parity plot
def pareto_plot(model, colors = COLORS):
    df_model = model.equations_
    best = model.get_best()
    plt.grid(True, which="both", ls="-", linewidth=0.5)
    plt.plot(df_model['complexity'], df_model['loss'], marker='o', linestyle='-', color=colors[0], label='Models')
    plt.plot(best['complexity'], best['loss'],'o', color=colors[1], label='Best Model')
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Model Complexity vs Loss')
    plt.legend()
    plt.show()

def post_pareto_plot(elem, run_id, colors = COLORS):
    print(f'saved_sr_models/model_{elem}_{run_id}.csv')
    df_model = pd.read_csv(f'saved_sr_models/model_{elem}_{run_id}.csv')
    best = pd.read_csv(f'saved_sr_models/model_{elem}_best_{run_id}.csv')
    best_complexity = float(best.iloc[0, 0])
    best_rows = df_model[np.isclose(df_model['complexity'], best_complexity)]
    best_ind = int(best.iloc[0,0])
    plt.figure(figsize=(9, 4))
    plt.xticks(np.arange(0, df_model['complexity'].max() + 1, 2))
    plt.grid(True, which="both", ls="-", linewidth=0.5)
    plt.plot(df_model['complexity'], df_model['loss'], marker='o', linestyle='-', color=colors[0], label='Models')
    plt.plot(best_rows['complexity'], best_rows['loss'], marker='o', linestyle='', color=colors[1], label='Best Model')
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Model Complexity vs Loss')
    plt.legend()
    plt.show()

def post_pareto_plot(elem, run_id, colors = COLORS):
    print(f'saved_sr_models/model_{elem}_{run_id}.csv')
    df_model = pd.read_csv(f'saved_sr_models/model_{elem}_{run_id}.csv')
    best = pd.read_csv(f'saved_sr_models/model_{elem}_best_{run_id}.csv')
    plt.grid(True, which="both", ls="-", linewidth=0.5)
    plt.plot(df_model['complexity'], df_model['loss'], marker='o', linestyle='-', color=colors[0], label='Models')
    plt.plot(best['complexity'], best['loss'],'o', color=colors[1], label='Best Model')
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Model Complexity vs Loss')
    plt.legend()
    plt.show()

def parity_plot(element,model,X_test,Y_test,X_train,Y_train, colors = COLORS,data_set = 'test'):
    
    if data_set == 'test':
        Y_pred = model.predict(X_test)
        Y = Y_test[element]
    elif data_set == 'train':
        Y_pred = model.predict(X_train)
        Y = Y_train[element]
    elif data_set == 'both':
        Y_pred_test = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        Y = np.concatenate([Y_test[element], Y_train[element]])
    
    if data_set == 'both':
        plt.scatter(Y_test[element], Y_pred_test, color=colors[0],alpha = 0.5, label = 'Test data')
        plt.scatter(Y_train[element], Y_pred_train, color=colors[1],alpha = 0.5, label = 'Train data')
    else:
        plt.scatter(Y, Y_pred, color=colors[0],alpha = 0.5, label = f'{data_set.capitalize()} data')
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
    plt.xlabel(f'Actual ${element}$ [a.u.]')
    plt.ylabel(f'Predicted ${element}$ [a.u.]')
    plt.legend()
    plt.show()

def parity_plot_Ue(model,Y,X,colors = COLORS,data_set = 'test'):
    Y_pred = model.predict(X)
    plt.scatter(Y, Y_pred, color=colors[0],alpha = 0.5)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)

# =================================================
# SAVE MODELS
# =================================================

def save_expressions(df_model, element, run_id):
    # make directory if it doesn't exist
    import os
    if not os.path.exists(f'saved_sr_models/{run_id}'):
        os.makedirs(f'saved_sr_models/{run_id}')
    
    df_model.to_csv(f'saved_sr_models/{run_id}/model_{element}.csv', index=False)