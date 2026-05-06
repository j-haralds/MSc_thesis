from IPython.display import display, Math
from pysr import PySRRegressor, TemplateExpressionSpec
# import mean
from pysr import jl
jl.seval("using Statistics")


def setup_model(its = int(1e3), pop = 30, selection = "accuracy",run_id = None):
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=["+", "*", '-','/','^'],
        unary_operators=['sqrt','square', 'cube','exp','log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh'],
        populations=pop,
        nested_constraints = {'sin':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                          'cos':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                          'tan':  {'sin': 0, 'cos': 0, 'tan': 0, 'log': 0, 'exp': 0},
                          'sinh': {'sinh': 1, 'cosh': 1, 'tanh': 1},
                          'cosh': {'sinh': 1, 'cosh': 1, 'tanh': 1},
                          'tanh': {'sinh': 1, 'cosh': 1, 'tanh': 1},
                          'log':  {'sin': 1, 'cos': 1, 'tan': 1},
                          'exp':  {'sin': 1, 'cos': 1, 'tan': 1},
                          },
        verbosity=0,         
        constraints={"^": (-1, 1)},
        batching = True,
        maxsize = 15,
        batch_size = 1024,
        run_id= run_id
    )
    return model


def run_symbolic_regression(X, y, model = None,run_id = None):
    if model is None:
        model = setup_model(run_id = run_id)
    model.fit(X, y)
    return model