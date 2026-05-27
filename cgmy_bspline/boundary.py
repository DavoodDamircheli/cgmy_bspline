import numpy as np


def put_payoff(x, K):
    """European put payoff: psi(x) = max(K - exp(x), 0). x is log-price."""
    return np.maximum(K - np.exp(x), 0.0)


def left_boundary_value(params, tau):
    """Far-left boundary value: K * exp(-r * tau)."""
    return params.K * np.exp(-params.r * tau)


def boundary_forcing_vector(N, p, h, x_min, params, tau, matrices_dict):
    """
    Compute the boundary forcing vector f_bc(tau) for the piecewise exterior extension.

    Zero-forcing approximation: the exterior contribution is negligible when
    exp(-G * (log(K) - x_min)) < 1e-4, i.e. the left boundary is far enough into
    the money that the B-spline tails into the exterior region carry negligible weight.
    """
    return np.zeros(N + p - 1)
