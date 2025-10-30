import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(mat, xlabel="x coordinate", ylabel="Velocity [-0.7, 0.7]", title=None, show_ticks=True, save_path=None, cmap="turbo"):
    """
    Plot a 2D numpy array `mat` as a heatmap using matplotlib.

    Parameters:
        mat : 2D numpy array
        xlabel, ylabel, title : optional strings for axis labels and title
        show_ticks : whether to show numeric tick labels
        save_path : if provided, saves the figure to this filepath
    """
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)
    if mat.ndim != 2:
        raise ValueError(f"Input must be a 2D array/matrix. Got ndim={mat.ndim}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        rows, cols = mat.shape
        if cols <= 20:
            ax.set_xticks(np.arange(cols))
            ax.set_xticklabels(np.arange(cols))
        else:
            ax.set_xticks([0, cols - 1])
            ax.set_xticklabels([0, cols - 1])
        if rows <= 20:
            ax.set_yticks(np.arange(rows))
            ax.set_yticklabels(np.arange(rows))
        else:
            ax.set_yticks([0, rows - 1])
            ax.set_yticklabels([0, rows - 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


def plot_transitions(P, nx, ny):

    # coordinates for each "to" state
    x_to = np.arange(ny)[None,None,:,None]  # shape (1,1,ny,1)
    y_to = np.arange(nx)[None,None,None,:]  # shape (1,1,1,nx)

    # compute expected next x,y under P
    next_x = (P * x_to).sum(axis=(2,3))  # sum over to_x and to_y axes
    next_y = (P * y_to).sum(axis=(2,3))

    # current grid coordinates
    x_grid, y_grid = np.meshgrid(np.arange(ny), np.arange(nx))

    # displacement vectors
    dx = next_x - x_grid
    dy = next_y - y_grid

    # plot
    plt.figure(figsize=(7,6))
    plt.quiver(x_grid, y_grid, dx, -dy, angles='xy', scale_units='xy', scale=1, color='tab:blue')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title("Transition Vector Field")
    plt.xlabel("X (col index)")
    plt.ylabel("Y (row index)")
    plt.savefig("transitions.png")