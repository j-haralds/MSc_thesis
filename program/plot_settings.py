import matplotlib.pyplot as plt


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