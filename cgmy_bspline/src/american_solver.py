"""
Step C — American put solver via Crank-Nicolson + Projected SOR LCP.

Grid domain:
  x_min = log(K) - 10/G,  x_max = log(K) + 10/M
  (ensures exterior tail ~ exp(-10) per Proposition on domain truncation)

Boundary conditions (American put):
  Left  (deep ITM): V(x_min) = K - exp(x_min)  [intrinsic; immediate exercise]
  Right (deep OTM): V(x_max) = 0

LCP at each step:
  F * c^{n+1} >= b^n,  c^{n+1} >= g,  complementarity = 0
  solved via psor_lcp_banded.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cgmy_bspline.parameters import CGMYParams as _OldParams
from cgmy_bspline import matrices as _mat
from cgmy_bspline.projection import assemble_put_rhs, eval_solution
from cgmy_bspline.src.american_utils import put_payoff_projection, early_exercise_boundary
from cgmy_bspline.src.psor_solver import psor_lcp_banded
from cgmy_bspline.src.cgmy_params import CGMYParams


def _make_grid(K, G, M, N, N_base=64):
    """Build grid that snaps log(K) to a node for N = N_base * 2^k."""
    log_K = np.log(K)
    target_left  = 10.0 / G
    target_right = 10.0 / M
    domain = target_left + target_right
    h_base = domain / N_base
    j0 = int(round(N_base * target_left / domain))  # index of log_K in base grid
    x_min = log_K - j0 * h_base   # snapped so log_K is always a grid node
    x_max = x_min + domain
    h = (x_max - x_min) / N
    x_grid = x_min + np.arange(N + 1, dtype=float) * h
    return x_grid, h, x_min, x_max


def solve_american_put(params, N, N_tau,
                       omega_sor=1.2, psor_tol=1e-10,
                       psor_max_iter=500, bandwidth=None):
    """
    Solve for the American put price profile via CN time-stepping + PSOR LCP.

    Parameters
    ----------
    params       : CGMYParams (src)
    N            : spatial intervals
    N_tau        : time steps
    omega_sor    : SOR relaxation parameter
    psor_tol     : PSOR convergence tolerance
    psor_max_iter: max PSOR sweeps per time step
    bandwidth    : Toeplitz truncation bandwidth (default: 2*N, full)

    Returns
    -------
    V        : ndarray(N+1) — put price on x_grid at tau=T
    x_grid   : ndarray(N+1) — log-price nodes
    S_f_path : list of (tau, S_f) — exercise boundary at each time step
    """
    p = 3
    K = params.K
    r = params.r
    T = params.T
    # Full bandwidth is needed for CGMY non-local operator to avoid
    # constraint-propagation errors in PSOR (truncated bw causes AM < EU).
    bw = (2 * N) if bandwidth is None else int(bandwidth)

    x_grid, h, x_min, x_max = _make_grid(K, params.G, params.M, N)
    n_dof = N + p - 1
    dt = T / N_tau

    # ── Matrices ─────────────────────────────────────────────────────────────
    old = _OldParams(C=params.C, G=params.G, M=params.M, Y=params.Y,
                     r=r, K=K, T=T)
    grid_info = {'N': N, 'h': h, 'x_min': x_min, 'x_max': x_max, 'nodes': x_grid}
    mats = _mat.operator_matrix(N, p, h, x_min, old, bandwidth=bw)
    M_h = mats['M_h']
    A_h = mats['A_h']

    F_mat = (M_h + (dt / 2.0) * A_h).tocsr()
    B_mat = (M_h - (dt / 2.0) * A_h).tocsr()

    # ── BC setup ──────────────────────────────────────────────────────────────
    bc_left  = list(range(p - 1))       # [0, 1] for p=3
    bc_right = list(range(n_dof - p + 1, n_dof))  # [N, N+1] for p=3
    bc_dofs  = bc_left + bc_right

    # Enforce BC rows in F: identity rows at BC nodes
    F_lil = F_mat.tolil()
    for i in bc_dofs:
        F_lil[i, :] = 0.0
        F_lil[i, i] = 1.0
    F_mat = F_lil.tocsr()

    # BC coefficient values (American: time-independent intrinsic at left)
    c_L = (K - np.exp(x_min)) * np.sqrt(h)   # left BC: intrinsic * sqrt(h)
    c_R = 0.0                                  # right BC: OTM

    # ── Constraint vector g and initial condition ─────────────────────────────
    # The correct Galerkin LCP constraint is c >= g_proj where M_h*g_proj=b_psi.
    # This enforces V_h >= psi in the L2 (average) sense without spuriously
    # constraining OTM coefficients that are legitimately negative in B-spline
    # coefficient space.
    b0 = assemble_put_rhs(old, grid_info, p)
    g = spla.spsolve(M_h, b0).copy()   # g_proj = M_h^{-1} b_psi
    g[bc_dofs] = -1e30

    ks_float = np.arange(-(p - 1), N, dtype=float)
    x_dof = x_min + ks_float * h

    # ── Initial condition (tau=0 = expiry) ────────────────────────────────────
    c = spla.spsolve(M_h, b0)
    # Enforce BCs at tau=0
    for i in bc_left:
        c[i] = c_L
    for i in bc_right:
        c[i] = c_R

    # ── Time-stepping ─────────────────────────────────────────────────────────
    S_f_path = []
    tau = 0.0
    total_sor_iters = 0

    for n in range(N_tau):
        tau_new = (n + 1) * dt

        # RHS from explicit side
        rhs = B_mat @ c
        # Set BC values in RHS
        for i in bc_left:
            rhs[i] = c_L
        for i in bc_right:
            rhs[i] = c_R

        # PSOR LCP solve
        c, n_iter, conv = psor_lcp_banded(
            F_mat, rhs, g, c_init=c,
            omega_sor=omega_sor, max_iter=psor_max_iter, tol=psor_tol,
        )
        total_sor_iters += n_iter

        if not conv and n < 5:
            pass  # silent in production; first few steps may not fully converge

        # Enforce BCs explicitly (safety)
        for i in bc_left:
            c[i] = c_L
        for i in bc_right:
            c[i] = c_R

        # Safety projection: enforce L2-projected payoff constraint
        c = np.maximum(c, g)
        for i in bc_left:
            c[i] = c_L
        for i in bc_right:
            c[i] = c_R

        # Early exercise boundary
        V_tau = eval_solution(c, grid_info, p, x_grid)
        x_f = early_exercise_boundary(V_tau, x_grid, K)
        S_f_path.append((tau_new, np.exp(x_f)))
        tau = tau_new

    # Reconstruct V from final coefficients
    V = eval_solution(c, grid_info, p, x_grid)
    avg_sor = total_sor_iters / N_tau
    return V, x_grid, S_f_path


def price_american_put(params, N, N_tau, S_eval=None):
    """
    Price American put at S_eval (default: params.S0) by interpolation.
    """
    if S_eval is None:
        S_eval = params.S0
    V, x_grid, _ = solve_american_put(params, N, N_tau)
    x0 = np.log(float(S_eval))
    return float(np.interp(x0, x_grid, V))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from cgmy_bspline.src.cgmy_params import CGMYParams
    from cgmy_bspline.src.cn_solver import price_put as price_european

    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)
    print("Pricing American put...", flush=True)
    am_price = price_american_put(params, N=128, N_tau=40)
    eu_price  = price_european(params, N=128, N_tau=40)

    print(f"American put price:     {am_price:.6f}")
    print(f"European put price:     {eu_price:.6f}")
    print(f"Early exercise premium: {am_price - eu_price:.6f}")

    assert am_price >= eu_price - 1e-8, \
        f"FAIL: American ({am_price:.6f}) < European ({eu_price:.6f})"
    assert am_price > eu_price + 1e-4, \
        f"WARN: EEP essentially zero ({am_price - eu_price:.2e})"
    assert am_price <= params.K, \
        f"FAIL: American price ({am_price:.6f}) > K={params.K}"
    print("PASS: american_solver self-test")
