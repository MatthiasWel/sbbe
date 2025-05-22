import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def _multiplot(plot: pd.DataFrame, distribution: dict[int, float]) -> Figure:
    plot.digit = plot.digit.astype(str)
    n_digits = plot.digit.nunique()
    n_cols = 8
    n_rows = math.ceil(plot.cls.nunique() / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    for ax, cls in zip(axes.flat, plot.cls.unique(), strict=False):
        data = plot[plot.cls == cls]
        sns.pointplot(
            x=[str(key) for key in distribution],
            y=[val * len(data) for val in distribution.values()],
            ax=ax,
        )
        sns.histplot(x=data.digit, bins=n_digits, discrete=True, ax=ax)
        ax.set_title(cls)
        ax.set_xlabel("Digit")
        too_many_digits = 10
        if n_digits > too_many_digits:
            ax.set_xticklabels([])
            ax.set_xticks([])

    plt.tight_layout()
    return fig


def multiplot(plot: pd.DataFrame, distribution: dict[int, float]) -> Figure:
    """Plot data that is expected to follow Benford's law."""
    if "cls" not in plot.columns:
        msg = "`cls` must be a column containing the classes"
        raise ValueError(msg)
    if "digit" not in plot.columns:
        msg = "`digit` must be a column containing the digits"
        raise ValueError(msg)
    return _multiplot(plot, distribution)
