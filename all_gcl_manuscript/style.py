import os

import seaborn as sns
from matplotlib import pyplot as plt


def set_rc_params(notebook_dpi=None):
    sns.set_context('paper')
    sns.set_style('ticks')
    plt.style.use(os.path.join(os.path.dirname(__file__), 'paper.mplstyle'))
    if notebook_dpi is not None:
        plt.rcParams['figure.dpi'] = notebook_dpi
