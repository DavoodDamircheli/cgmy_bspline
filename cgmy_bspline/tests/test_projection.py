import numpy as np
import pytest
from cgmy_bspline.tests.test_parameters import PARAMS_I
from cgmy_bspline.grid import make_grid
from cgmy_bspline.matrices import mass_matrix
from cgmy_bspline.projection import l2_project_payoff, eval_solution, assemble_put_rhs
from cgmy_bspline.boundary import put_payoff

p = 3


def _setup(N):
    grid_info = make_grid(PARAMS_I, N)
    h = grid_info['h']
    x_min = grid_info['x_min']
    M_h = mass_matrix(N, p, h, x_min)
    return grid_info, {'M_h': M_h}


# ── test 1 ──────────────────────────────────────────────────────────────────

def test_projection_nonnegative_and_zero_at_strike():
    """V_h is non-negative and evaluates close to 0 at log(K)."""
    N = 64
    grid_info, matrices_dict = _setup(N)
    c0 = l2_project_payoff(PARAMS_I, grid_info, p, matrices_dict)

    x_fine = np.linspace(grid_info['x_min'], grid_info['x_max'], 500)
    V_h = eval_solution(c0, grid_info, p, x_fine)

    # Cubic B-spline L2 projection undershoots near the kink by O(h); allow that.
    tol_neg = 0.5 * PARAMS_I.K / N
    assert V_h.min() >= -tol_neg, f"V_h min = {V_h.min():.3e}, allowed = {-tol_neg:.3e}"

    log_K = np.log(PARAMS_I.K)
    V_at_K = eval_solution(c0, grid_info, p, np.array([log_K]))[0]
    tol = PARAMS_I.K / N
    print(f"\nV_h(log(K)) = {V_at_K:.4e}, tolerance = {tol:.4e}")
    assert abs(V_at_K) < tol, f"V_h(log(K)) = {V_at_K:.3e}, tolerance = {tol:.3e}"


# ── test 2 ──────────────────────────────────────────────────────────────────

def test_projection_mass_residual():
    """||M_h @ c^0 - b^0||_2 < 1e-10 (spsolve residual is machine-precision small)."""
    N = 64
    grid_info, matrices_dict = _setup(N)
    b0 = assemble_put_rhs(PARAMS_I, grid_info, p)
    c0 = l2_project_payoff(PARAMS_I, grid_info, p, matrices_dict)

    M_h = matrices_dict['M_h']
    residual = np.linalg.norm(M_h @ c0 - b0)
    print(f"\n||M_h @ c0 - b0||_2 = {residual:.3e}")
    assert residual < 1e-10


# ── test 3 ──────────────────────────────────────────────────────────────────

def test_projection_convergence():
    """L2 projection error decreases as N increases (kink limits to O(h) globally)."""
    errors = []
    for N in [32, 64, 128]:
        grid_info, matrices_dict = _setup(N)
        c0 = l2_project_payoff(PARAMS_I, grid_info, p, matrices_dict)

        x_fine = np.linspace(grid_info['x_min'], grid_info['x_max'], 2000)
        V_h = eval_solution(c0, grid_info, p, x_fine)
        psi = put_payoff(x_fine, PARAMS_I.K)

        h_fine = (grid_info['x_max'] - grid_info['x_min']) / (len(x_fine) - 1)
        err = np.sqrt(h_fine * np.sum((psi - V_h) ** 2))
        errors.append(err)
        print(f"\nN={N}: L2 error = {err:.6e}")

    assert errors[0] > errors[1] > errors[2], (
        f"Errors not decreasing: {errors}"
    )
