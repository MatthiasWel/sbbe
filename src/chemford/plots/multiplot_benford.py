import seaborn as sns
import math
import matplotlib.pyplot as plt
import pandas as pd

def _multiplot(plot, distribution):
    plot.digit = plot.digit.astype(str)
    n_digits = plot.digit.nunique()
    n_cols = 8
    n_rows = math.ceil(plot.cls.nunique() / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
    for ax, cls in zip(axes.flat, plot.cls.unique()):
        data = plot[plot.cls == cls]
        sns.pointplot(x=[str(key) for key in distribution.keys()], y=[val * len(data) for val in distribution.values()],  ax=ax)
        sns.histplot(x=data.digit, bins=n_digits, discrete=True, ax=ax)
        ax.set_title(cls);
        ax.set_xlabel('Digit')
        if n_digits > 10:
            ax.set_xticklabels([])
            ax.set_xticks([])
        
    plt.tight_layout()
    return fig

def multiplot(plot: pd.DataFrame, distribution):
    if 'cls' not in plot.columns:
        raise ValueError("`cls` must be a column containing the classes")
    if 'digit' not in plot.columns:
        raise ValueError("`digit` must be a column containing the digits")
    return _multiplot(plot, distribution)
