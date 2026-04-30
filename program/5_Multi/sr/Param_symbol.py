from IPython.display import display, Math
from pysr import PySRRegressor, TemplateExpressionSpec
# import mean
from pysr import jl
jl.seval("using Statistics")


def setup_model(its = int(1e3), pop = 30, selection = "best",run_id = None):
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=["+", "*", '-','/'],
        unary_operators=['sqrt','square', 'cube','exp','log'],
        populations=pop,
        # nested_constraints={'sin':{'sin':0}, 'sin':{'cos':0}, 'sin':{'tan':0}, 'sin':{'log':0}, 'sin':{'exp':0},
        #                     'cos':{'sin':0}, 'cos':{'cos':0}, 'cos':{'tan':0}, 'cos':{'log':0}, 'cos':{'exp':0},
        #                     'tan':{'sin':0}, 'tan':{'cos':0}, 'tan':{'tan':0}, 'tan':{'log':0}, 'tan':{'exp':0},
        #                     'log':{'sin':0}, 'log':{'cos':0}, 'log':{'tan':0},
        #                     'exp':{'sin':0}, 'exp':{'cos':0}, 'exp':{'tan':0}
        #                     },
        verbosity=0, 
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