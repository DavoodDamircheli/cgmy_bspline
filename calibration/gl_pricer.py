"""
Grünwald–Letnikov (GL) finite-difference CN pricer for the CGMY FPDE.

Implements the SAME backward-FPDE operator as the B-Spline Galerkin solver
(cgmy_bspline/solver_cn.py) on a point-value grid, using:

  - Tempered GL weights for the fractional jump operators
    (reuses tempered_coeffs_via_ifft with p=0 — same symbol, no B-spline hat)
  - Central finite differences for the first-derivative drift term
  - Crank–Nicolson time-stepping (unconditionally stable for all Y ∈ (0,2))
  - Dense LU factorisation (scipy.linalg) — valid because the GL operator
    matrix is dense (lower + upper Toeplitz sum) and N is small (32–64)

Accuracy: O(h) for the fractional part (vs O(h^{p-Y/2}) ≈ O(h^{2.5}) for
the B-spline scheme).  Faster per evaluation (no scipy.quad pre-build, dense
LU for small N) but requires larger N for equivalent pricing accuracy.

Interface:
  gl_price_surface(theta, T, K_array, option_type, grid_kwargs)
    — same signature as calibration.pricer.price_surface
  make_gl_cached_pricer(surface_by_maturity, S0, rates_by_T, grid_kwargs)
    — same signature as calibration.pricer.make_cached_pricer

The GL "cached" pricer pre-builds only the grid geometry (instant); the
operator matrix is rebuilt every call because ALL GL components depend on
(G, M, Y).  This is not a limitation: the rebuild costs ~0.5ms per maturity.
"""

import math
import numpy as np
import scipy.linalg as la

from cgmy_bspline.grid import make_grid_manual
from cgmy_bspline.parameters import CGMYParams as SolverP, omega_stock, lambda0
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft
from scipy.special import gamma as sc_gamma


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def _build_gl_operator(C, G, M, Y, r, h, n, bw):
    """
    Build the GL CGMY operator matrix A on an n-point uniform grid.

    A[u] = -(r - ω_S) D u + C Γ(-Y) (K_G + K_M) u + λ_0 I u

    where
      - D  : central-difference first-derivative matrix (n×n)
      - K_G: lower-Toeplitz fractional matrix (negative/left jumps, decay G)
      - K_M: upper-Toeplitz fractional matrix (positive/right jumps, decay M)

    Uses tempered_coeffs_via_ifft with p=0 (identical symbol to the B-spline
    scheme but without the B-spline hat function), so both schemes discretise
    the same continuous operator — only the order of accuracy differs.

    Parameters
    ----------
    bw : int  bandwidth for the Toeplitz approximation (use 2*n for full)
    """
    # ── Tempered GL weights (p=0 → hat_N_0 = 1, symbol = S directly) ─────────
    delta_G = G * h
    delta_M = M * h
    lam_G = tempered_coeffs_via_ifft(delta_G, Y, 0, 'minus', L=bw) * h ** (-Y)
    lam_M = tempered_coeffs_via_ifft(delta_M, Y, 0, 'plus',  L=bw) * h ** (-Y)

    half_bw = bw // 2

    # ── Toeplitz operator matrices (vectorized, no Python loop) ───────────────
    ii = np.arange(n)[:, None]
    jj = np.arange(n)[None, :]
    m_mat = ii - jj                            # lag matrix, shape (n, n)
    mask  = (m_mat >= -half_bw) & (m_mat <= half_bw - 1)
    idx   = m_mat % bw                         # wrap into [0, bw)
    K_G   = np.where(mask, lam_G[idx], 0.0)
    K_M   = np.where(mask, lam_M[idx], 0.0)

    # ── First-derivative matrix (central differences, forward/backward at BCs) ─
    D = np.zeros((n, n))
    for i in range(1, n - 1):
        D[i, i - 1] = -1.0 / (2.0 * h)
        D[i, i + 1] =  1.0 / (2.0 * h)
    # boundary rows: forward/backward
    D[0,  0] = -1.0 / h;  D[0,  1] =  1.0 / h
    D[-1, -1] =  1.0 / h; D[-1, -2] = -1.0 / h

    # ── Assemble A  (same formula as B-spline operator_matrix) ────────────────
    gY   = sc_gamma(-Y)
    sp_  = SolverP(C=C, G=G, M=M, Y=Y, r=r, K=1.0, T=1.0)
    om_S = omega_stock(sp_)
    lam0 = lambda0(sp_)   # = r

    A = (
        -(r - om_S) * D
        - C * gY * K_G
        - C * gY * K_M
        + lam0 * np.eye(n)
    )
    return A


def solve_gl_cn(params, N, N_tau, grid=None, bw=None):
    """
    GL finite-difference + Crank–Nicolson solver for the CGMY backward FPDE.

    Parameters
    ----------
    params : cgmy_bspline.parameters.CGMYParams
    N      : number of spatial intervals (N+1 point values)
    N_tau  : number of CN time steps
    grid   : pre-built grid dict (from make_grid_manual); built internally if None
    bw     : Toeplitz bandwidth (default 2*(N+1) = full coverage)

    Returns
    -------
    dict with keys: u_final (point values), x_nodes, grid
    """
    g = grid if grid is not None else make_grid_manual(
        math.log(params.K) - 2.0, math.log(params.K) + 2.0, N
    )
    x = g['nodes']   # shape (N+1,)
    h = float(g['h'])
    n = len(x)        # N+1
    dt = params.T / N_tau

    if bw is None:
        bw = 2 * n

    # ── Build GL operator and CN system ───────────────────────────────────────
    A = _build_gl_operator(params.C, params.G, params.M, params.Y,
                           params.r, h, n, bw)
    I = np.eye(n)
    F_full = I + (dt / 2.0) * A   # left-hand matrix
    B_mat  = I - (dt / 2.0) * A   # right-hand matrix

    # Enforce BCs by zeroing rows and setting diagonal = 1
    bc_left  = 0
    bc_right = n - 1
    F = F_full.copy()
    F[bc_left,  :] = 0.0;  F[bc_left,  bc_left]  = 1.0
    F[bc_right, :] = 0.0;  F[bc_right, bc_right] = 1.0

    lu_fac = la.lu_factor(F)

    # ── Initial condition: European put payoff ─────────────────────────────────
    u = np.maximum(params.K - np.exp(x), 0.0).copy()

    # ── CN time loop ──────────────────────────────────────────────────────────
    tau = 0.0
    for step in range(N_tau):
        tau_n = (step + 1) * dt
        rhs   = B_mat @ u
        # Left BC: deep ITM put = K·e^{-r·τ} - e^{x_0} ≈ K·e^{-r·τ} for large K
        rhs[bc_left]  = params.K * math.exp(-params.r * tau_n) - math.exp(x[0])
        rhs[bc_right] = 0.0
        u   = la.lu_solve(lu_fac, rhs)
        tau = tau_n

    return {'u_final': u, 'x_nodes': x, 'grid': g}


# ---------------------------------------------------------------------------
# Surface pricer  (same interface as calibration.pricer.price_surface)
# ---------------------------------------------------------------------------

def gl_price_surface(
    theta: tuple,
    T: float,
    K_array,
    option_type: str = "auto",
    grid_kwargs: dict | None = None,
) -> np.ndarray:
    """
    GL finite-difference + CN surface pricer.

    Identical interface to calibration.pricer.price_surface.
    Uses the same homogeneity identity and dividend-adjustment trick.

    Parameters
    ----------
    theta : (C, G, M, Y, r, q, S0)
    """
    C, G, M, Y, r, q, S0 = theta
    K_array = np.asarray(K_array, dtype=float)
    gkw = grid_kwargs or {}

    N     = gkw.get("N",      64)
    N_tau = gkw.get("N_tau",  10)
    bw_fac = gkw.get("bandwidth", None)  # None → full (2*(N+1))
    half_w = gkw.get("domain_half_width", 2.0)

    K_ref    = float(S0)
    log_Kref = math.log(K_ref)
    grid     = make_grid_manual(log_Kref - half_w, log_Kref + half_w, N)

    sp = SolverP(C=C, G=G, M=M, Y=Y, r=r, K=K_ref, T=T)
    bw = bw_fac if bw_fac is not None else 2 * (N + 1)
    res = solve_gl_cn(sp, N, N_tau, grid=grid, bw=bw)

    # Dividend adjustment + evaluation
    log_S0_eff = math.log(S0) - q * T
    x_eval     = log_S0_eff + log_Kref - np.log(K_array)
    x_nodes    = res['x_nodes']
    x_clipped  = np.clip(x_eval, x_nodes[0], x_nodes[-1])

    V_ref      = np.interp(x_clipped, x_nodes, res['u_final'])
    put_prices = (K_array / K_ref) * V_ref

    D = math.exp(-r * T)
    F = S0 * math.exp((r - q) * T)

    if option_type == "put":
        return put_prices
    elif option_type == "call":
        return put_prices + D * (F - K_array)
    else:
        call_prices = put_prices + D * (F - K_array)
        return np.where(K_array < F, put_prices, call_prices)


# ---------------------------------------------------------------------------
# Cached pricer factory  (same interface as calibration.pricer.make_cached_pricer)
# ---------------------------------------------------------------------------

def make_gl_cached_pricer(
    surface_by_maturity: dict,
    S0: float,
    rates_by_T: dict,
    grid_kwargs: dict | None = None,
):
    """
    Return a GL pricer_fn for calibration.

    Unlike the B-spline factory, no expensive pre-build is needed:
    GL has no CGMY-independent slow step (no scipy.quad).  The factory
    pre-computes only the grid geometry and boundary indices (instant).

    The per-call cost is dominated by:
      • tempered weight computation: O(N) via recurrence (~0.1ms)
      • Toeplitz matrix assembly: O(N²) numpy ops (~0.3ms at N=64)
      • Dense LU factorisation: O(N³) LAPACK (~0.05ms at N=64)
      • N_tau CN solves: O(N²) × N_tau LAPACK (~0.1ms total)

    Total: ~0.5–2ms per maturity at N=32–64.

    Parameters
    ----------
    surface_by_maturity : dict  (same as make_cached_pricer)
    S0 : float
    rates_by_T : dict   round(T,6) → (r_eff, q_eff)
    grid_kwargs : dict  N, N_tau, bandwidth, domain_half_width
    """
    gkw    = grid_kwargs or {}
    N      = gkw.get("N",      64)
    N_tau  = gkw.get("N_tau",  10)
    half_w = gkw.get("domain_half_width", 2.0)

    K_ref    = float(S0)
    log_Kref = math.log(K_ref)

    # Pre-compute: grid geometry only (instant)
    _grids: dict = {}
    for df_mat in surface_by_maturity.values():
        T_key = round(float(df_mat["T"].iloc[0]), 6)
        if T_key not in _grids:
            _grids[T_key] = make_grid_manual(log_Kref - half_w,
                                              log_Kref + half_w, N)

    def _pricer_fn(cgmy_tuple, T_mat, K_array, option_type="auto"):
        C, G, M, Y = cgmy_tuple
        T_key = round(T_mat, 6)
        r_e, q_e = rates_by_T[T_key]
        grid = _grids[T_key]

        sp = SolverP(C=C, G=G, M=M, Y=Y, r=r_e, K=K_ref, T=T_mat)
        bw = 2 * (N + 1)
        res = solve_gl_cn(sp, N, N_tau, grid=grid, bw=bw)

        K_array    = np.asarray(K_array, dtype=float)
        log_S0_eff = math.log(S0) - q_e * T_mat
        x_eval     = log_S0_eff + log_Kref - np.log(K_array)
        x_nodes    = res['x_nodes']
        x_clipped  = np.clip(x_eval, x_nodes[0], x_nodes[-1])

        V_ref      = np.interp(x_clipped, x_nodes, res['u_final'])
        put_prices = (K_array / K_ref) * V_ref

        D_disc = math.exp(-r_e * T_mat)
        F_fwd  = S0 * math.exp((r_e - q_e) * T_mat)

        if option_type == "put":
            return put_prices
        elif option_type == "call":
            return put_prices + D_disc * (F_fwd - K_array)
        else:
            call_prices = put_prices + D_disc * (F_fwd - K_array)
            return np.where(K_array < F_fwd, put_prices, call_prices)

    return _pricer_fn
