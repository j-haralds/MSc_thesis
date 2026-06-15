import matplotlib.pyplot as plt
import numpy as np

def apply():
    # LaTeX font
    plt.style.use('default')
    plt.rc('text', usetex = True)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    font_size = 16
    plt.rcParams['font.size'] = font_size
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True


def grid_gen(xlim, ylim, size):  
    x_size =  size[0]; y_size = size[1]

    x = np.linspace(xlim[0], xlim[1], int(x_size))
    y = np.linspace(ylim[0], ylim[1], int(y_size))
    X, Y = np.meshgrid(x, y)
    return X, Y

def colors():
    colors = ['tab:blue', 'tab:red', 'tab:green','black', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan']
    return colors