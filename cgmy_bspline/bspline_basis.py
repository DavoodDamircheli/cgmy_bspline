import numpy as np


def cardinal_bspline(x, p):
    """
    Evaluate the cardinal B-spline N_p at scalar or array x.
    Cox-de Boor recursion; supports p = 1, 2, 3, 4.
    N_1 uses the half-open convention: 1 if 0 <= x < 1 else 0.
    Returns zeros for p <= 0 (needed by ddphi when p=2).
    """
    x = np.asarray(x, dtype=float)
    if p <= 0:
        return np.zeros_like(x)
    if p == 1:
        return np.where((x >= 0) & (x < 1), 1.0, 0.0)
    return (x / (p - 1)) * cardinal_bspline(x, p - 1) + \
           ((p - x) / (p - 1)) * cardinal_bspline(x - 1, p - 1)


def active_indices(N, p):
    """
    Integer array of active indices k = -(p-1), ..., N-1.
    supp(N_p(xi - k)) = [k, k+p] intersects (0, N) iff k in [-(p-1), N-1].
    Total count: N + p - 1.
    """
    return np.arange(-(p - 1), N, dtype=int)


def phi(x, k, p, h, x_min):
    """Physical basis function: h^{-1/2} * N_p((x - x_min)/h - k)."""
    xi = (np.asarray(x, dtype=float) - x_min) / h
    return h ** (-0.5) * cardinal_bspline(xi - k, p)


def dphi(x, k, p, h, x_min):
    """First derivative (eq. 83): h^{-3/2} * [N_{p-1}(xi-k) - N_{p-1}(xi-k-1)]."""
    xi = (np.asarray(x, dtype=float) - x_min) / h
    return h ** (-1.5) * (cardinal_bspline(xi - k, p - 1) -
                          cardinal_bspline(xi - k - 1, p - 1))


def ddphi(x, k, p, h, x_min):
    """Second derivative (eq. 84): h^{-5/2} * [N_{p-2}(xi-k) - 2*N_{p-2}(xi-k-1) + N_{p-2}(xi-k-2)]."""
    xi = (np.asarray(x, dtype=float) - x_min) / h
    return h ** (-2.5) * (cardinal_bspline(xi - k, p - 2) -
                          2 * cardinal_bspline(xi - k - 1, p - 2) +
                          cardinal_bspline(xi - k - 2, p - 2))
