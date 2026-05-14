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
# DATA PREPARATION 
# ================================================= 

def read_data(file_name):
    df = pd.read_csv(
        f"symbol_data/{file_name}.txt",
        sep=',',
        comment="%"
    )
    # cols = u_par,C,t,V,I,F,u,Ue,eta,trajectory,dV,dF,soc
    #df.columns = ['u_par','C','t','V','I','F','u','Ue','eta','trajectory','dV','dF','soc']
    # Sort by time within each batch
    
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
    return bin_ops, un_ops, nest_const,consts, op_comps

def get_settings(elem):
    bin_ops, un_ops, nest_const,consts, op_comps = get_default_settings()
    
    if elem == 'R0':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':1, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
    if elem == 'R1':
        un_ops = ['exp', 'log','sqrt','square', 'cube',]
        nest_const = {'exp':  {'exp': 1, 'log': 1},
                      'log':  {'exp': 1, 'log': 1}}
        op_comps = {"+": 1, "*": 1, '-': 1, '/': 1,'^':1, 'sqrt': 1, 'square':1, 'cube': 1,'exp': 1,'log': 2}
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
    
    return bin_ops, un_ops, nest_const,consts, op_comps

def setup_model(its = int(1e3), pops = 30, selection = "accuracy",run_id = None, elem = None):
    

    bin_ops, un_ops, nest_const, consts,op_comps = get_settings(elem)
    
    
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=bin_ops,
        unary_operators=un_ops,
        populations=pops,
        nested_constraints = nest_const,
        verbosity=0,         
        constraints = consts,
        batching = True,
        complexity_of_operators = op_comps,
        complexity_of_constants = 1,
        maxsize = 20,
        batch_size = 1024,
        run_id= run_id
    )
    return model


def run_symbolic_regression(X, y, model = None,run_id = None, its = int(1e3), pops = 100, selection = "best", elem = None):
    if model is None:
        model = setup_model(run_id = run_id, its = its, pops = pops, selection = selection, elem = elem)
    print(f"Running symbolic regression for element {elem} with run_id {run_id}...")
    print(f"Settings: iterations={its}, populations={pops}, selection={selection}")
    model.fit(X, y)
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
    print(f'sr_models/model_{elem}_{run_id}.csv')
    df_model = pd.read_csv(f'sr_models/model_{elem}_{run_id}.csv')
    best = pd.read_csv(f'sr_models/model_{elem}_best_{run_id}.csv')
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
    print(f'sr_models/model_{elem}_{run_id}.csv')
    df_model = pd.read_csv(f'sr_models/model_{elem}_{run_id}.csv')
    best = pd.read_csv(f'sr_models/model_{elem}_best_{run_id}.csv')
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


# =================================================
# SAVE MODELS
# =================================================

def save_expressions(df_model, element):
    df_model.to_csv(f'sr_models/model_{element}.csv', index=False)