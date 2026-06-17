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
import pandas as pd
import os
from pathlib import Path

# =================================================
# POST PROCESSING
# =================================================

def get_SR_inds(RUN_ID):
    df_best_inds = pd.read_csv("saved_sr_models/best_indices.csv")
    df_chosen_inds = pd.read_csv("saved_sr_models/chosen_indices.csv")

    if RUN_ID not in df_best_inds['run_id'].values:
        raise ValueError(f"Run ID {RUN_ID} not found in best_indices.csv. Please add it before proceeding.")
    row = df_best_inds[df_best_inds['run_id'] == RUN_ID].iloc[0]
    best_inds = {
        'R0': int(row['R0']),
        'R1': int(row['R1']),
        'C1': int(row['C1']),
        'k': int(row['k']),
        'sdot': int(row['sdot']),
        'Ue': int(row['Ue'])
    }
    if RUN_ID not in df_chosen_inds['run_id'].values:
        raise ValueError(f"Run ID {RUN_ID} not found in chosen_indices.csv. Please add it before proceeding.")
    row = df_chosen_inds[df_chosen_inds['run_id'] == RUN_ID].iloc[0]
    chosen_inds = {
        'R0': int(row['R0']),
        'R1': int(row['R1']),
        'C1': int(row['C1']),
        'k': int(row['k']),
        'sdot': int(row['sdot']),
        'Ue': int(row['Ue'])
    }
    return chosen_inds, best_inds


def get_ref_values(run_id = None):
    '''Reference values for normalization. 
    For s, C and d, these are the maximum values in the dataset. 
    For R0, R1, and k, these are arbitrary.
    Values can be adjusted based on the specific dataset and requirements.'''
    


    if run_id is None:
        REF_VALUES = {
        'R0': 0.01,
        'R1': 0.01,
        'C1': 1000,
        'k':  0.01,
        's':  0.37266314,
        'sdot': 0.0001, 
        'C': 5,
        'd': 30,
        'Ue': 1
        }
    else:
        df_existing = pd.read_csv("saved_sr_models/ref_values.csv")
        if run_id not in df_existing['run_id'].values:
            raise ValueError(f"Run ID {run_id} not found in ref_values.csv. Please add it before proceeding.")
        row = df_existing[df_existing['run_id'] == run_id].iloc[0]
        REF_VALUES = {
            'R0': row['R0'],
            'R1': row['R1'],
            'C1': row['C1'],
            'k':  row['k'],
            's':  row['s'],
            'sdot': row['sdot'],
            'C': row['C'],
            'd': row['d'],
            'Ue': row['Ue']
        }
    
    return REF_VALUES

def get_latex_dict():
    '''LaTeX representations for each variable, used in plots and tables.'''

    latex_dict ={
        'R0': r'R_0',
        'R1': r'R_1',
        'C1': r'C_1',
        'k': r'k',
        'sdot': r'\dot{s}',
        'Ue': r'U_{eq}',
        's': r's'
    }
    return latex_dict


def get_units_dict():
    ''''Units for each variable, used in plots and tables. 
    Conversion factors are applied to convert to resonable units.'''
    
    units_dict = {
        'R0': r'm$\Omega$',
        'R1': r'm$\Omega$',
        'C1': r'kF',
        'k':  r'MN/$\mu$m',
        'sdot':  r'$\mu$m/s',
        's': r'$\mu$m',
        'Ue': r'V'
        }
    
    unit_conversion = {
        'R0': 1e3,   # from Ohm to mOhm
        'R1': 1e3,  # from Ohm to mOhm
        'C1': 1e-3,  # from F to kF
        'k':  1e2, 
        'sdot':  1e1,
        's': 1e1,
        'Ue': 1
        }
    
    return units_dict, unit_conversion


def extract_expressions(run_id, elements = ['R0', 'R1', 'C1', 'k', 'sdot', 'Ue']):
    expressions = {}
    for elem in elements:
    
        results_path = Path(f'saved_sr_models/{run_id}/model_{elem}_{run_id}.csv')
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        results_df = pd.read_csv(results_path)
        expr = results_df['sympy_format']# == int(results_df['complexity'].mean())]
        expressions[elem] = expr
    
    return expressions



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
    elif elem in ['sdot']:
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
    
    if elem == 'sdot':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':2, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
    
    if elem == 'Ue':
        bin_ops = ["+", "*", '-','/','^']
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
    plt.plot(best['complexity'], best['loss'],'o', color=colors[1], label='Preferred Model')
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
    plt.xlabel(f'Actual $U_{{eq}}$ [V]')
    plt.ylabel(f'Predicted $U_{{eq}}$ [V]')

# =================================================
# SAVE MODELS
# =================================================

def save_expressions(df_model, element, run_id):
    # make directory if it doesn't exist
    import os
    if not os.path.exists(f'saved_sr_models/{run_id}'):
        os.makedirs(f'saved_sr_models/{run_id}')
    
    df_model.to_csv(f'saved_sr_models/{run_id}/model_{element}.csv', index=False)