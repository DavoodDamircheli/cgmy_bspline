import numpy as np
import pytest
from scipy.integrate import quad
from cgmy_bspline.bspline_basis import cardinal_bspline, active_indices, phi, dphi, ddphi


def test_support():
    """phi(x, k=5, p=3, h=0.1, x_min=0) is zero for x strictly outside (0.5, 0.8)."""
    k, p, h, x_min = 5, 3, 0.1, 0.0
    for x in [-0.1, 0.0, 0.3, 0.499, 0.8001, 0.9, 1.5]:
        assert phi(x, k, p, h, x_min) == pytest.approx(0.0, abs=1e-15), \
            f"phi({x}) should be 0 outside support"


def test_partition_of_unity():
    """
    sum_k N_p(xi - k) == 1 on interior [p-1, N-(p-1)] for N=32, p=3.
    Interior is xi in [2, 30]; points closer to the boundary lose basis functions.
    """
    N, p = 32, 3
    ks = active_indices(N, p)
    xs = np.linspace(2.0, N - 2.0, 500)
    xi_k = xs[:, None] - ks[None, :]      # (500, N+p-1)
    vals = cardinal_bspline(xi_k, p)      # (500, N+p-1)
    sums = vals.sum(axis=1)               # (500,)
    assert np.allclose(sums, 1.0, atol=1e-12)


def test_l2_norm():
    """
    Integral of phi_{h,k}^2 over R equals ||N_p||^2_{L^2} = 11/20 for p=3.

    The change of variables xi = (x - x_min)/h gives:
      integral phi^2 dx = h^{-1} * h * integral N_p(xi)^2 dxi = integral N_p^2 dxi

    For N_3: pieces are x^2/2 on [0,1), (-x^2+3x-3/2) on [1,2), (3-x)^2/2 on [2,3).
    Exact: 1/20 + 9/20 + 1/20 = 11/20.  Independent of h.
    """
    p, h, k, x_min = 3, 0.25, 5, 0.0
    a = x_min + k * h
    b = x_min + (k + p) * h
    result, _ = quad(lambda x: phi(x, k, p, h, x_min) ** 2, a, b)
    assert result == pytest.approx(11 / 20, rel=1e-8)


def test_dphi_finite_difference():
    """dphi matches central finite difference of phi; relative error < 1e-5."""
    p, h, k, x_min = 3, 0.1, 5, 0.0
    x, eps = 0.62, 1e-6
    fd = (phi(x + eps, k, p, h, x_min) - phi(x - eps, k, p, h, x_min)) / (2 * eps)
    analytic = dphi(x, k, p, h, x_min)
    assert abs(analytic - fd) / abs(analytic) < 1e-5


def test_ddphi_finite_difference():
    """ddphi matches central finite difference of dphi; relative error < 1e-5."""
    p, h, k, x_min = 3, 0.1, 5, 0.0
    x, eps = 0.62, 1e-6
    fd = (dphi(x + eps, k, p, h, x_min) - dphi(x - eps, k, p, h, x_min)) / (2 * eps)
    analytic = ddphi(x, k, p, h, x_min)
    assert abs(analytic - fd) / abs(analytic) < 1e-5
