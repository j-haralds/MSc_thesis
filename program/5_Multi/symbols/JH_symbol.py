from IPython.display import display, Math
from pysr import PySRRegressor, TemplateExpressionSpec
# import mean
from pysr import jl
jl.seval("using Statistics")


def setup_model(its = 300, pop = 25, selection = "best",run_id = 'all'):
    template = TemplateExpressionSpec(
        expressions=['f', 'g'],
        variable_names= ['x1', 'x2', 'x3','x4'],   
        combine = '-x1 / (abs(f(x2, x3,x4)) * abs(g(x2, x3,x4))) + x2 / abs(g(x2, x3,x4))'
    )
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=["+", "*", '-','/'],
        unary_operators=['sqrt','square', 'cube','exp'],
        elementwise_loss="loss(x, y) = (x - y)^2",
        populations=pop,
        verbosity=0, 
        expression_spec=template, 
        batching = True,
        maxsize = 15,
        batch_size = 5000,
        run_id = run_id
    )
    return model


def run_symbolic_regression(X, y, model = None,run_id = 'all'):
    if model is None:
        model = setup_model(run_id = run_id)
    model.fit(X, y)
    return model