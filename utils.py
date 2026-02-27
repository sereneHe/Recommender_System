import functools
import logging

def log_exceptions(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logging.exception(err)
            # re-raise the exception
            raise
    return wrapper


def plot_heatmap(W, names_x, names_y, filename=None, dpi=None):
    import matplotlib.pyplot as plt

    # Remove '_lag0' suffix from names
    names_x = [name.split("_lag")[0] for name in names_x]
    names_y = [name.split("_lag")[0] for name in names_y]

    fig, ax = plt.subplots()

    # Create the heatmap using imshow
    limit = max(abs(W.min()), abs(W.max()))

    cax = ax.imshow(W, cmap='YlGnBu', interpolation='nearest', vmin=-limit, vmax=limit) #cmaps: # YlGnBu # coolwarm

    import numpy as np
    ax.set_xticks(np.arange(len(names_x)))
    ax.set_xticklabels(names_x, rotation=90)
    ax.set_yticks(np.arange(len(names_y)))
    ax.set_yticklabels(names_y)

    # Add a colorbar to the figure
    fig.colorbar(cax, ax=ax)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename,format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()