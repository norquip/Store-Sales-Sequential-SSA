# libreries
import torch
import numpy as np




# SSA tool
def svd(ts, L, q=None):
    """
    Parameters
    ----------
    ts --- the time series (N,)
    L ---  window length
    q ---  int, optional
        Target rank for the low-rank approximation.

        - If q is None, the full SVD is computed.
        - If q is specified, a randomized low-rank SVD is computed,
          returning only the leading q singular components.

    Returns
    -------
    U --- torch tensor, left singular vectors of the Hankel matrix
    S --- torch tensor composed by the singular values
    VT ---torch tensor, transposed right singular vectors

    """
    print(f"Running SSA with L={L}")

    # ts can be a numpy array or a torch tensor
    ts = torch.as_tensor(ts, dtype=torch.float64)

    # Construct the Hankel Matrix
    N = len(ts) # length of the timeseries

    # Check that L < N
    if not (1 <= L <= N):
            raise ValueError(
                f"L must satisfy 1 <= L <= N ({N})"
            )

    # Number of  columns K
    K = N - L + 1

    # Slides a window of size K along dimension 0 in steps of 1 produces shape (L, K).
    X = ts.unfold(0, K, 1)   # float64 en torch


    # SVD decomposition
    if q is None:
        U, S, VT = torch.linalg.svd(
            X,
            full_matrices=False
        )

    else:
        torch.manual_seed(42)

        U, S, V = torch.svd_lowrank(
            X,
            q=min(q, min(X.shape))
        )

        VT = V.T


    triple =  {
        "U":U,
        "S":S,
        "VT": VT
    }
    # free ram memory
    del X
    torch.cuda.empty_cache()

    return triple



# SSA eigenvalues
def eigenvalues(S):
    """
    Compute SSA eigenvalues lambda from singular values sigma.

    Parameters
    ----------
    S : torch.Tensor
        Singular values returned by the SVD.

    Returns
    -------
    torch.Tensor
        SSA eigenvalues.
    """

    return S**2






# Diagonal average
def diagonal_average(Y):
    """
    Hankelization via diagonal averaging.
    Projects an elementary matrix back to the Hankel structure
    by averaging along anti-diagonals, producing the
    corresponding reconstructed time series.

    Parameters
    ----------
    Y:  tensor (L, K) — elementary matrix X_i

    Returns
    -------
    y:  tensor (N,) — reconstructed time series
             where N = L + K - 1
    """
    L, K = Y.shape
    N = L + K - 1

    # count the contribution of antidiagonals
    counts = torch.zeros(N, dtype=Y.dtype, device=Y.device)
    y = torch.zeros(N, dtype=Y.dtype, device=Y.device)

    for i in range(L):
        # The  ith row  contributes to the antidiagonals i, i+1, ..., i+K-1
        indices = torch.arange(i, i + K, device=Y.device)
        y.index_add_(0, indices, Y[i])
        counts.index_add_(0, indices, torch.ones(K, dtype=Y.dtype, device=Y.device))

    return y / counts


# Reconstruct  group
def reconstruct_group(triple, group):
    """
    Reconstruct grouped SSA components.

    Parameters
    ----------
    triple : (U, S, VT) tensors output from SSA.

    group : list[int]
        Indices of the elementary components to combine.

    Returns
    -------
    ts_group : torch.Tensor
        Reconstructed time series associated with the group.
    """
    U, S, VT = triple["U"], triple["S"], triple["VT"]

    L = U.shape[0]
    K = VT.shape[1]

    X_group = torch.zeros((L, K), dtype=U.dtype, device=U.device)

    for i in group:
        X_group += S[i] * torch.outer(U[:, i], VT[i, :])

    ts_group = diagonal_average(X_group)

    del X_group          # free the matrix (L, K)
    torch.cuda.empty_cache()

    return ts_group.cpu()




# Reconstruct  several groups
def reconstruct_all_groups(triple, groups):
    """
    Reconstruct several grouped SSA components.

    Parameters
    ----------
    triple : dict
        Dictionary containing U, S, and VT.

    groups : dict
        Dictionary of groups.
        Example:
        {
            "Trend": [0],
            "Weekly": [2, 3],
            "Harmonic": [4, 5, 6, 7]
        }

    Returns
    -------
    reconstructed: dict
        Dictionary containing the reconstructed time series.
    """

    reconstructed = {}

    for name, group in groups.items():
        reconstructed[name] = reconstruct_group(triple, group)

    return reconstructed



# W-correlation matrix
def w_correlation_matrix(ts_rec, L):
    """
     Parameters
    ----------
    ts_rec: list of r tensors, each of shape (N,)
                output of diagonal_average for each component
    L: window length
    returns: W of shape (r, r)

    Returns
    -------
    """
    r = len(ts_rec)
    N = ts_rec[0].shape[0]
    K = N - L + 1

    # Weights w_n
    n = torch.arange(N, dtype=torch.float64)
    w = torch.minimum(
            torch.minimum(n + 1, torch.tensor(L, dtype=torch.float64)),
            torch.minimum(torch.tensor(K, dtype=torch.float64), N - n)
        )
    # Stack: shape (r, N)
    F = torch.stack(ts_rec, dim=0)

    # Weighted Inner Product: <Fi, Fj>_w
    F_w = F * w.unsqueeze(0)       # (r, N)
    W_inner_product = F_w @ F.T            # (r, r)

    # Normalization
    norms = torch.sqrt(torch.diag(W_inner_product))
    W = W_inner_product / torch.outer(norms, norms)

    return W.abs()








