import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate
from scipy.special import gamma as sc_gamma

from cgmy_bspline.bspline_basis import phi, dphi, active_indices
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft
from cgmy_bspline.parameters import omega_stock, lambda0


def _breakpoints(k, kp, p, h, x_min, a, b):
    """Return interior grid breakpoints strictly inside (a, b)."""
    m_lo = max(k, kp) + 1
    m_hi = min(k + p, kp + p)
    pts = []
    for m in range(m_lo, m_hi):
        xm = x_min + m * h
        if a < xm < b:
            pts.append(xm)
    return pts


def mass_matrix(N, p, h, x_min):
    """
    Assemble M_h as a scipy.sparse.csr_matrix.
    (M_h)_{ij} = integral phi_{h,k_j}(x) * phi_{h,k_i}(x) dx over Omega.
    Shape: (N+p-1, N+p-1). Bandwidth: nonzero only for |i-j| < p.
    """
    n = N + p - 1
    ks = active_indices(N, p)
    x_max = x_min + N * h

    rows, cols, vals = [], [], []
    for i, k in enumerate(ks):
        for j, kp in enumerate(ks):
            if abs(i - j) >= p:
                continue
            a = max(x_min + k * h, x_min + kp * h, x_min)
            b = min(x_min + (k + p) * h, x_min + (kp + p) * h, x_max)
            if a >= b:
                continue
            pts = _breakpoints(k, kp, p, h, x_min, a, b)
            val, _ = integrate.quad(
                lambda x, _k=k, _kp=kp: phi(x, _kp, p, h, x_min) * phi(x, _k, p, h, x_min),
                a, b,
                points=pts or None,
                limit=200,
                epsabs=1e-14,
                epsrel=1e-14,
            )
            rows.append(i)
            cols.append(j)
            vals.append(val)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def derivative_matrix(N, p, h, x_min):
    """
    Assemble D_h as a scipy.sparse.csr_matrix.
    (D_h)_{ij} = integral dphi_{h,k_j}(x) * phi_{h,k_i}(x) dx over Omega.
    Same sparsity pattern as M_h.
    """
    n = N + p - 1
    ks = active_indices(N, p)
    x_max = x_min + N * h

    rows, cols, vals = [], [], []
    for i, k in enumerate(ks):
        for j, kp in enumerate(ks):
            if abs(i - j) >= p:
                continue
            a = max(x_min + k * h, x_min + kp * h, x_min)
            b = min(x_min + (k + p) * h, x_min + (kp + p) * h, x_max)
            if a >= b:
                continue
            pts = _breakpoints(k, kp, p, h, x_min, a, b)
            val, _ = integrate.quad(
                lambda x, _k=k, _kp=kp: dphi(x, _kp, p, h, x_min) * phi(x, _k, p, h, x_min),
                a, b,
                points=pts or None,
                limit=200,
                epsabs=1e-12,
                epsrel=1e-12,
            )
            rows.append(i)
            cols.append(j)
            vals.append(val)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def fractional_stiffness_matrices(N, p, h, params, bandwidth=64):
    """
    Assemble K_G_minus and K_M_plus as banded Toeplitz sparse matrices (Version 1).

    (K_G_minus)_{ij} = h^{-Y} * Lambda_G_minus(i-j; G*h)
    (K_M_plus)_{ij}  = h^{-Y} * Lambda_M_plus (i-j; M*h)

    Entries with |i-j| > bandwidth are truncated to zero.
    Shape: (N+p-1, N+p-1).
    """
    n = N + p - 1
    Y = params.Y

    delta_G = params.G * h
    delta_M = params.M * h

    lam_G = tempered_coeffs_via_ifft(delta_G, Y, p, 'minus', L=bandwidth)
    lam_M = tempered_coeffs_via_ifft(delta_M, Y, p, 'plus',  L=bandwidth)

    scale = h ** (-Y)
    lam_G = lam_G * scale
    lam_M = lam_M * scale

    rows_G, cols_G, vals_G = [], [], []
    rows_M, cols_M, vals_M = [], [], []

    half_bw = bandwidth // 2
    for i in range(n):
        for j in range(n):
            m = i - j
            # Only include lags in the alias-free half-spectrum [-half_bw, half_bw-1]
            if m < -half_bw or m > half_bw - 1:
                continue
            idx = m % bandwidth
            rows_G.append(i); cols_G.append(j); vals_G.append(lam_G[idx])
            rows_M.append(i); cols_M.append(j); vals_M.append(lam_M[idx])

    K_G = sp.csr_matrix((vals_G, (rows_G, cols_G)), shape=(n, n))
    K_M = sp.csr_matrix((vals_M, (rows_M, cols_M)), shape=(n, n))

    return K_G, K_M


def operator_matrix(N, p, h, x_min, params, bandwidth=64):
    """
    Assemble the full semidiscrete operator A_h (eq. 71) and its components.

      A_h = -(r - omega_S) * D_h
            - C * Gamma(-Y) * K_M_plus
            - C * Gamma(-Y) * K_G_minus
            + lambda_0 * M_h

    Returns dict with keys: A_h, M_h, D_h, K_minus (K_G_minus), K_plus (K_M_plus).
    """
    M_h = mass_matrix(N, p, h, x_min)
    D_h = derivative_matrix(N, p, h, x_min)
    K_G, K_M = fractional_stiffness_matrices(N, p, h, params, bandwidth)

    r = params.r
    C = params.C
    Y = params.Y

    omega_S = omega_stock(params)
    lam0 = lambda0(params)
    gY = sc_gamma(-Y)

    A_h = (
        -(r - omega_S) * D_h
        - C * gY * K_M
        - C * gY * K_G
        + lam0 * M_h
    )

    return {
        'A_h': A_h.tocsr(),
        'M_h': M_h,
        'D_h': D_h,
        'K_minus': K_G,
        'K_plus':  K_M,
    }
