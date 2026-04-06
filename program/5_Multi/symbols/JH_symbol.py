from IPython.display import display, Math
from pysr import PySRRegressor, TemplateExpressionSpec
# import mean
from pysr import jl
jl.seval("using Statistics")


def setup_model(its = 500, pop = 25, selection = "best",run_id = None):
    template = TemplateExpressionSpec(
        expressions=['R', 'C'],
        variable_names= ['x1', 'x2', 'x3','x4'],   #x1 = dU1, x2 = I, x3 = u, x4 = soc 
        #combine = '-x1 / (exp(f(x2, x3,x4)) * exp(g(x2, x3,x4))) + x2 / exp(g(x2, x3,x4))'
        combine = '(R(x2,x3,x4))^2 * (x2 - (C(x2,x3,x4))^2 * x1)'
    )
    model = PySRRegressor(
        model_selection=selection,
        niterations=its,
        binary_operators=["+", "*", '-','/'],
        unary_operators=['sqrt','square', 'cube','exp','log'],
        elementwise_loss="loss(x, y) = (x - y)^2",
        populations=pop,
        verbosity=0, 
        expression_spec=template, 
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