import numpy as np
import scipy.sparse.linalg as spla
from scipy.integrate import fixed_quad

from cgmy_bspline.bspline_basis import phi, active_indices
from cgmy_bspline.boundary import put_payoff


def assemble_put_rhs(params, grid_info, p):
    """
    Assemble b^0_k = integral_{Omega} psi(x) * phi_{h,k}(x) dx for each active k.

    Integration is performed element-by-element within each basis function's support.
    When log(K) falls strictly inside an element, the integral is split there to
    avoid loss of accuracy from the kink in psi.
    """
    N = grid_info['N']
    h = grid_info['h']
    x_min = grid_info['x_min']
    K = params.K
    log_K = np.log(K)

    ks = active_indices(N, p)
    b0 = np.zeros(N + p - 1)

    for i, k in enumerate(ks):
        j_lo = max(0, k)
        j_hi = min(N - 1, k + p - 1)

        val = 0.0
        for j in range(j_lo, j_hi + 1):
            a_elem = x_min + j * h
            b_elem = x_min + (j + 1) * h
            integrand = lambda x, _k=k: put_payoff(x, K) * phi(x, _k, p, h, x_min)
            if a_elem < log_K < b_elem:
                v1, _ = fixed_quad(integrand, a_elem, log_K, n=10)
                v2, _ = fixed_quad(integrand, log_K, b_elem, n=10)
                val += v1 + v2
            else:
                v, _ = fixed_quad(integrand, a_elem, b_elem, n=10)
                val += v

        b0[i] = val

    return b0


def l2_project_payoff(params, grid_info, p, matrices_dict):
    """
    Compute c^0 by solving M_h * c^0 = b^0.

    Parameters
    ----------
    params        : CGMYParams
    grid_info     : dict from grid.make_grid
    p             : B-spline order
    matrices_dict : dict containing key 'M_h' (from matrices.operator_matrix)

    Returns
    -------
    c0 : numpy array of length N+p-1, the initial coefficient vector
    """
    b0 = assemble_put_rhs(params, grid_info, p)
    M_h = matrices_dict['M_h']
    return spla.spsolve(M_h, b0)


def eval_solution(c, grid_info, p, x_eval):
    """
    Evaluate V_h(x) = sum_k c_k * phi_{h,k}(x) at each x in x_eval.

    Parameters
    ----------
    c         : coefficient vector, length N+p-1
    grid_info : dict from grid.make_grid
    p         : B-spline order
    x_eval    : numpy array of evaluation points

    Returns
    -------
    V : numpy array of same length as x_eval
    """
    h = grid_info['h']
    x_min = grid_info['x_min']
    N = grid_info['N']
    ks = active_indices(N, p)

    x_eval = np.asarray(x_eval, dtype=float)
    V = np.zeros_like(x_eval)
    for i, k in enumerate(ks):
        V += c[i] * phi(x_eval, k, p, h, x_min)
    return V
