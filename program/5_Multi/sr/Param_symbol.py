from IPython.display import display, Math
from pysr import PySRRegressor, TemplateExpressionSpec
# import mean
from pysr import jl
jl.seval("using Statistics")


def setup_model(its = 500, pop = 25, selection = "best",run_id = None):
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=["+", "*", '-','/'],
        unary_operators=['sqrt','square', 'cube','exp','log', 'sin', 'cos', 'tan'],
        elementwise_loss="loss(x, y) = (x - y)^2",
        populations=pop,
        verbosity=0, 
        batching = True,
        maxsize = 15,
        batch_size = 512,
        run_id= run_id
    )
    return model


def run_symbolic_regression(X, y, model = None,run_id = None):
    if model is None:
        model = setup_model(run_id = run_id)
    model.fit(X, y)
    return model