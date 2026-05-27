import numpy as np
from cgmy_bspline.parameters import CGMYParams


def make_grid(params: CGMYParams, N: int, domain_scale: float = 10.0) -> dict:
    """
    Build the uniform log-price grid for the CGMY solver.

    Parameters
    ----------
    params       : CGMYParams
    N            : int, number of subintervals
    domain_scale : float, controls domain size (default 10.0 gives exp(-10) tail error)

    Returns
    -------
    dict with keys:
      x_min   : float
      x_max   : float
      h       : float = (x_max - x_min) / N
      nodes   : numpy array of length N+1, uniformly spaced from x_min to x_max
      N       : int
      log_K   : float = log(params.K)
    """
    log_K = np.log(params.K)
    target_left  = domain_scale / params.G
    target_right = domain_scale / params.M
    domain = target_left + target_right
    # Snap x_min so that log_K is a grid node for all N = N_base * 2^k.
    # Using N_base=64: j0 = round(64 * target_left / domain), then
    # x_min = log_K - j0 * (domain / 64).  For each N, j = j0*(N/64) is an
    # integer, so grids nest properly (every coarse node is also a fine node).
    # For symmetric grids (G==M) j0 = 32 = 64/2 and x_min = log_K - target_left
    # exactly, so this is a no-op in the symmetric case.
    N_base = 64
    h_base = domain / N_base
    j0 = round(target_left / h_base)
    x_min = log_K - j0 * h_base
    x_max = x_min + domain
    return _build(x_min, x_max, N, log_K)


def make_grid_manual(x_min: float, x_max: float, N: int) -> dict:
    """
    Same return structure but with manually specified endpoints.
    Useful for manufactured-solution tests where the domain is prescribed.
    """
    return _build(x_min, x_max, N, log_K=None)


def _build(x_min: float, x_max: float, N: int, log_K) -> dict:
    h = (x_max - x_min) / N
    nodes = np.linspace(x_min, x_max, N + 1)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "h": h,
        "nodes": nodes,
        "N": N,
        "log_K": log_K,
    }
