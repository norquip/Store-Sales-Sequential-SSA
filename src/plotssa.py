
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math
import copy



def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Plot eigenvalues
def plot_scree(eigenvalues, idx=slice(None), log_scale=True, show=True, label=None):
    """
    Plot the SSA scree plot.

    Parameters
    ----------
    eigenvalues : torch.Tensor or numpy.ndarray
        SSA eigenvalues.

    idx : slice, optional
        Range of components to display.
        Default is slice(None), which plots all components.

    log_scale : bool, optional
        If True, plot eigenvalues on a logarithmic scale.
        This is recommended for SSA because eigenvalues often span
        several orders of magnitude.

    Returns
    -------
    None
        Displays the scree plot.
    """
    plt.figure(figsize=(6,4))

    # SSA eigenvalues
    eigenvalues = to_numpy(eigenvalues[idx])

    # Component indices
    x_axis = np.arange(1, len(eigenvalues) + 1)

    if log_scale:
        plt.semilogy(
            x_axis,
            eigenvalues,
            'o-',
            linewidth=2,
            markersize=4,
            label=label
        )
    else:
        plt.plot(
            x_axis,
            eigenvalues,
            'o-',
            linewidth=2,
            markersize=4,
            label=label
        )
    plt.title('SSA Scree Plot')
    plt.xlabel('Component Index', fontsize=10)
    plt.ylabel('Eigenvalue', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    if show:
        plt.legend()
        plt.tight_layout()
     



# Relative contribution
def relative_contribution(eigenvalues):
    """
    Plot the relative contribution.

    Parameters
    ----------
    eigenvalues: torch.Tensor or numpy.ndarray
       

    Returns
    -------
    None
        Displays the relative contribution
    """

    rel_contrib = (100 * eigenvalues / torch.sum(eigenvalues)).cpu().numpy()


    plt.figure(figsize=(6,4))
    bars = plt.bar(range(1, len(rel_contrib)+1), rel_contrib)
    
    for bar, value in zip(bars, rel_contrib):
        plt.text(
            bar.get_x() + bar.get_width()/2,   
            value + 0.2,                      
            f"{value:.1f}%",                 
            ha="center",
            va="bottom",
            fontsize=8
        )
    
    plt.xlabel("SSA Component", fontsize=10)
    plt.ylabel("Contribution (%)", fontsize=10)
    plt.title("Relative Component Contribution")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Plot Eigenvectors
def plot_eigenvecs(U, n=8, cols=4):
    """
    Plot the first n eigenvectors.

    Parameters
    ----------
    U : torch.Tensor or ndarray
        Matrix of eigenvectors.
    n : int, optional
        Number of eigenvectors to display (default = 8).
    cols : int, optional
        Number of columns in the figure (default = 4).
    """

    n = min(n, U.shape[1])   # avoid to ask above the real number of eigenvectors
    eigenvecs = to_numpy(U[:, :n])

    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4*cols, 2.8*rows))

    axes = np.array(axes).reshape(-1)

    x = np.arange(U.shape[0])

    for i in range(n):
        axes[i].plot(x, eigenvecs[:, i], color="teal")
        axes[i].set_title(f"Eigenvector {i+1}")
        axes[i].grid(True)

    
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Eigenvector plots of elementary components", fontsize=20)
    
    plt.tight_layout()
    plt.show()


# Plot  eigenvector pairs
def plot_eigenvector_pairs(
    U,
    pairs=None,
    n_pairs=5,
    n_samples=150,
    figsize=(18, 8)
):
    """
    Scatter plots of SSA eigenvector pairs.

    Parameters
    ----------
    U : torch.Tensor
        Matrix of left singular vectors.

    pairs : list of tuples, optional
        List of eigenvector index pairs.
        If None, consecutive pairs are used:
        (0,1), (2,3), (4,5), ...

    n_pairs : int, optional
        Number of consecutive pairs to plot
        (ignored if pairs is provided).

    n_samples : int, optional
        Number of samples displayed.

    figsize : tuple
        Figure size.
    """

    if pairs is None:
        pairs = [(2*k, 2*k+1) for k in range(n_pairs)]

    fig, axes = plt.subplots(2, (len(pairs)+1)//2, figsize=figsize)
    axes = np.ravel(axes)

    for ax, (i, j) in zip(axes, pairs):

        ax.plot(
            U[:n_samples, i].numpy(),
            U[:n_samples, j].numpy(),
            lw=0.5,
            alpha=0.7
        )

        ax.set_title(f"EV {i+1} vs EV {j+1}")
        ax.set_xlabel(f"EV {i+1}")
        ax.set_ylabel(f"EV {j+1}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    
    for ax in axes[len(pairs):]:
        ax.remove()
        
    fig.suptitle("Scatter plots of paired eigenvectors", fontsize=22)
    
    plt.tight_layout()
    plt.show()



# Plot Cumulative energy
def plot_cumulative_energy(S, idx=slice(None), threshold=0.9, k=None, L=None):
    """
    Plots the cumulative energy contribution of components.

    Parameters:
    - S (torch.Tensor): Singular values.
    - idx (slice): Slice object to select the range of indices to plot.
    - threshold (float): Target energy threshold line (e.g., 0.90 for 90%).
    - k (int): Reference vertical line indicating the number of components chosen.
    - L (int/str): Total length or context parameter for the title.
    """
    # 1. Compute the global cumulative energy 
    energy_total = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    energy_np = energy_total.cpu().numpy()

    # 2. Create the full array of original component indices (0, 1, 2... N-1)
    components = np.arange(1 , len(energy_np) + 1)

    # 3. Apply the slice (idx) to both axes to zoom in on the desired range
    x_plot = components[idx]
    y_plot = energy_np[idx]

    # 4. Plot the sliced data
    plt.plot(x_plot, y_plot, 'o-', markersize=3)

    # Reference lines (only plotted if values are provided)
    if threshold is not None:
        plt.axhline(threshold, color='r', linestyle='--', label=f'{threshold*100:.0f}%')
    if k is not None:
        plt.axvline(k, color='g', linestyle='--', label=f'k={k}')

    # Labels and Title 
    plt.xlabel('Component Index')
    plt.ylabel('Cumulative Energy')
    plt.title(f'Cumulative Energy (L={L})' if L else 'Cumulative Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)





# Plot W-correlation matrix with matplotlib
def plot_w_corr(W, title=None):
    """
    Plot the SSA W-correlation matrix.

    Parameters
    ----------
    W : torch.Tensor or ndarray
        W-correlation matrix.

    title : str, optional
        Figure title.
    """

    # Convert to NumPy if needed
    if hasattr(W, "cpu"):
        W = W.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        W,
        cmap="viridis",
        origin="upper",
        vmin=0,
        vmax=1
    )

    n = W.shape[0]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    # Number components from 1 instead of 0
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.set_yticklabels(np.arange(1, n + 1))

    ax.set_xlabel("Component")
    ax.set_ylabel("Component")

    if title is not None:
        ax.set_title(f"W-correlation Matrix ({title})")
    else:
        ax.set_title("W-correlation Matrix")

    # Grid between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.colorbar(im, ax=ax, label="|W-correlation|")

    plt.tight_layout()
    plt.show()


    
# plot W matrix with plotly
def plot_w_correlation(W, title="W-Correlation Matrix"):
    """
    Interactive W-correlation matrix using Plotly.

    Parameters
    ----------
    W: torch.Tensor or ndarray
        W-correlation matrix.

    title: str
        Figure title.

    Returns
    -------
    fig: plotly.graph_objects
    """

    # Convert torch tensor to numpy
    if hasattr(W, "cpu"):
        W = W.cpu().numpy()

    n = W.shape[0]

    fig = go.Figure(
        data=go.Heatmap(
            z=W,
            x=np.arange(1, n + 1),
            y=np.arange(1, n + 1),
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="|W|"),
            hovertemplate=(
                "RC %{x} vs RC %{y}<br>"
                "W = %{z:.3f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Component",
        width=700,
        height=700,
        yaxis=dict(autorange="reversed"),
        template="plotly_white"
    )

    return fig



