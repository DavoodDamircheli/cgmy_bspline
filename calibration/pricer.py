"""
European option pricer for the CGMY calibration pipeline (Section 7, Step 6).

price_surface prices a strip of European puts and/or calls at a given maturity T
by calling the existing B-Spline Galerkin + Crank-Nicolson solver ONCE via the
homogeneity property, then evaluating the saved solution at multiple log-moneyness
points.

Homogeneity identity
--------------------
For a CGMY model, the put price P(S, K, T) is homogeneous of degree 1 in (S, K):
    P(S, K, T) = (K / K_ref) * P(S * K_ref / K, K_ref, T)

So we solve the PDE once with strike K_ref = S0 (the current spot), then recover
P(S0, K_j) by evaluating the solution at x_j = log(S0_eff * K_ref / K_j), where:
    S0_eff = S0 * exp(-q * T)   (prepaid-forward / dividend adjustment)

This exact identity reduces q != 0 to the q = 0 PDE:
    P(S0, K; r, q) = P(S0 * exp(-q*T), K; r, q=0)

The existing CN solver uses q = 0 internally (no dividend support in SolverP), so
passing S0_eff through the evaluation point handles dividends without modifying the
solver.  The BC discounting inside the solver uses params.r (correct for q = 0 equiv).

Note on domain width
--------------------
The default make_grid(params, N, domain_scale=10) produces a domain of only ±1/G in
log-space.  For G = M = 10 this is ±0.1, which is too narrow for active CGMY (Y ≈ 1.5).
price_surface therefore builds a wider grid via make_grid_manual with half_width = 2.0
(±2 log-price units around log(K_ref)), which gives < 0.25 % error relative to the
COS pricer at N = 256.
"""

import math

import numpy as np

from cgmy_bspline.parameters import CGMYParams as SolverP
from cgmy_bspline.grid import make_grid_manual
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.projection import eval_solution


# ---------------------------------------------------------------------------
# Domain sizing
# ---------------------------------------------------------------------------

# Default/floor half-width -- adequate for typical (non-heavy-tailed) theta.
_DEFAULT_HALF_WIDTH = 2.0
# Cap on the adaptive widening -- chosen to match the half-width already
# empirically validated in calibration/debug_domain_truncation.py (D4) and
# calibration/debug_short_maturity.py (D3's local-refinement sweep), where
# widening further started to coarsen h enough to hurt short-maturity
# accuracy. This is a partial mitigation for very heavy tails (small G or
# M), not a full fix -- see debug_domain_truncation.py for the measured
# residual gap at this cap.
_MAX_HALF_WIDTH = 3.0
# Target exterior-tail weight exp(-min(G,M)*half_width) the adaptive rule
# aims for, before hitting the cap above.
_TARGET_TAIL_EPS = 0.05


def adaptive_half_width(
    G: float,
    M: float,
    base: float = _DEFAULT_HALF_WIDTH,
    target_eps: float = _TARGET_TAIL_EPS,
    cap: float = _MAX_HALF_WIDTH,
) -> float:
    """Domain half-width sized to theta's tail-decay rates (D4 fix).

    Uses the same exp(-decay*half_width) truncation-factor heuristic as
    cgmy_bspline/grid.py's domain_scale/G, domain_scale/M convention (see
    calibration/debug_domain_truncation.py, item 1), evaluated against the
    SLOWER-decaying (more dangerous) of the two tails, min(G, M).

    Returns `base` unchanged for typical theta (G, M not too small) --
    this is exactly the previous fixed default, so low-vol dates pay no
    extra cost. Widens up to `cap` for heavy-tailed theta (e.g. a crisis-
    period fit with G* ~ 0.1), instead of one shared fixed value across
    all dates.
    """
    decay = min(G, M)
    required = math.log(1.0 / target_eps) / decay
    return float(min(cap, max(base, required)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def price_surface(
    theta: tuple,
    T: float,
    K_array,
    option_type: str = "auto",
    grid_kwargs: dict | None = None,
) -> np.ndarray:
    """Price European options at a strip of strikes via one B-Spline CN solve.

    Parameters
    ----------
    theta : tuple (C, G, M, Y, r, q, S0)
        CGMY parameters plus risk-free rate r, dividend yield q, and spot S0.
    T : float
        Time to expiry in years.
    K_array : array-like
        Strike prices.  All must satisfy 0 < K_j < S0 * exp((r-q)*T) * 1.5
        roughly (well inside the spatial domain).
    option_type : {'put', 'call', 'auto'}
        'put'  — return put prices for all strikes.
        'call' — return call prices for all strikes.
        'auto' — OTM selection: put for K < F, call for K >= F (F = S0*exp((r-q)*T)).
    grid_kwargs : dict, optional
        Override solver defaults:
          N              int   spatial intervals       (default 256)
          p              int   B-spline order          (default 3)
          N_tau          int   CN time steps           (default 50)
          bandwidth      int   Toeplitz truncation     (default 2*N)
          domain_half_width  float  ±log-price extent
                             (default: adaptive_half_width(G, M), see below;
                              equals 2.0 unless G or M is small)

    Returns
    -------
    np.ndarray, shape (len(K_array),)
        Option prices (same order as K_array).
    """
    C, G, M, Y, r, q, S0 = theta
    K_array = np.asarray(K_array, dtype=float)
    gkw = grid_kwargs or {}

    N       = gkw.get("N", 256)
    p       = gkw.get("p", 3)
    N_tau   = gkw.get("N_tau", 50)
    bw      = gkw.get("bandwidth", 2 * N)
    half_w  = gkw.get("domain_half_width", adaptive_half_width(G, M))

    # ── Reference strike = S0 (ATM); grid centred at log(S0) ─────────────────
    K_ref    = float(S0)
    log_Kref = np.log(K_ref)
    grid     = make_grid_manual(log_Kref - half_w, log_Kref + half_w, N)

    # ── Solver params: pass K = K_ref so payoff = max(K_ref - e^x, 0) ────────
    sp = SolverP(C=C, G=G, M=M, Y=Y, r=r, K=K_ref, T=T)

    # ── Single PDE solve ──────────────────────────────────────────────────────
    res = solve_cn(sp, N, p, N_tau, bandwidth=bw, grid=grid)

    # ── Dividend adjustment: P(S0, K; r, q) = P(S0*exp(-q*T), K; r, 0) ──────
    log_S0_eff = np.log(S0) - q * T

    # Evaluation points: x_j = log(S0_eff * K_ref / K_j)
    x_eval = log_S0_eff + log_Kref - np.log(K_array)

    # Guard: clip to domain so eval_solution doesn't extrapolate
    x_min, x_max = grid["x_min"], grid["x_max"]
    x_clipped = np.clip(x_eval, x_min + 1e-10, x_max - 1e-10)
    if not np.allclose(x_clipped, x_eval, atol=1e-8):
        import warnings
        out_mask = (x_eval < x_min) | (x_eval > x_max)
        warnings.warn(
            f"price_surface: {out_mask.sum()} strike(s) map outside domain "
            f"[{x_min:.3f}, {x_max:.3f}]; results may be inaccurate.",
            stacklevel=2,
        )

    # ── Homogeneity scaling ───────────────────────────────────────────────────
    V_ref      = eval_solution(res["c_final"], grid, p, x_clipped)
    put_prices = (K_array / K_ref) * V_ref

    # ── Put-call parity for calls ─────────────────────────────────────────────
    D = np.exp(-r * T)
    F = S0 * np.exp((r - q) * T)

    if option_type == "put":
        return put_prices
    elif option_type == "call":
        return put_prices + D * (F - K_array)
    else:  # 'auto': OTM selection
        call_prices = put_prices + D * (F - K_array)
        return np.where(K_array < F, put_prices, call_prices)


# ---------------------------------------------------------------------------
# Cached pricer factory for calibration (avoids redundant M_h/D_h builds)
# ---------------------------------------------------------------------------


def make_cached_pricer(
    surface_by_maturity: dict,
    S0: float,
    rates_by_T: dict,
    grid_kwargs: dict | None = None,
    theta_hint: tuple | None = None,
):
    """Return a pricer_fn for calibration that pre-builds M_h and D_h once.

    The mass matrix M_h and derivative matrix D_h depend only on the spatial
    grid (N, p, h, x_min) and take ~500 ms each to assemble via scipy.quad.
    During calibration the grid is fixed per maturity, so building them once
    and caching gives a ~100× speedup over calling price_surface repeatedly.

    The fractional stiffness matrices K_G and K_M depend on (G, M, Y) and
    take only ~5 ms to rebuild via IFFT, so they are recomputed per call.

    Parameters
    ----------
    surface_by_maturity : dict
        Keys are arbitrary identifiers; values are DataFrames with columns
        T, K, F, D (at least).  One DataFrame per maturity.
    S0 : float
        Reference spot / homogeneity anchor (= K_ref for the PDE solve).
    rates_by_T : dict
        Mapping round(T, 6) → (r_eff, q_eff) for per-maturity discount rates.
    grid_kwargs : dict, optional
        Same keys as price_surface: N, p, N_tau, bandwidth, domain_half_width.
    theta_hint : tuple (C, G, M, Y), optional
        If given AND grid_kwargs does not explicitly set domain_half_width,
        size the domain via adaptive_half_width(G, M) instead of the fixed
        default (2.0). Intended for post-calibration re-evaluation/plotting
        callers that already know the fitted theta (see
        calibration/debug_domain_truncation.py, D4) -- NOT passed by the
        live DE/L-BFGS optimization loop in calibrate.py, since theta varies
        every call there and the whole point of this cache is to avoid
        rebuilding the grid per call. Omitting it reproduces the exact prior
        behaviour.

    Returns
    -------
    pricer_fn : callable
        pricer_fn(cgmy_tuple, T_mat, K_array, option_type='auto') -> np.ndarray
        where cgmy_tuple = (C, G, M, Y).
    """
    import math
    import scipy.sparse.linalg as _spla
    from scipy.special import gamma as _sc_gamma

    from cgmy_bspline import matrices as _mats
    from cgmy_bspline.grid import make_grid_manual as _make_grid_manual
    from cgmy_bspline.projection import assemble_put_rhs as _put_rhs, eval_solution as _eval
    from cgmy_bspline.parameters import (
        CGMYParams as _SolverP,
        omega_stock as _omega_stock,
        lambda0 as _lambda0,
    )

    gkw     = grid_kwargs or {}
    N       = gkw.get("N",       64)
    p       = gkw.get("p",        3)
    N_tau   = gkw.get("N_tau",   10)
    bw      = gkw.get("bandwidth", 2 * N)
    if "domain_half_width" in gkw:
        half_w = gkw["domain_half_width"]
    elif theta_hint is not None:
        half_w = adaptive_half_width(theta_hint[1], theta_hint[2])
    else:
        half_w = _DEFAULT_HALF_WIDTH

    K_ref    = float(S0)
    log_Kref = math.log(K_ref)

    # ── Pre-build per-maturity fixed matrices (one-time cost) ─────────────────
    _per_mat: dict = {}

    for df_mat in surface_by_maturity.values():
        T     = float(df_mat["T"].iloc[0])
        T_key = round(T, 6)
        if T_key in _per_mat:
            continue

        grid  = _make_grid_manual(log_Kref - half_w, log_Kref + half_w, N)
        h, x_min = grid["h"], grid["x_min"]
        n_dof = N + p - 1

        # CGMY-independent matrices — expensive (scipy.quad), built once
        M_h = _mats.mass_matrix(N, p, h, x_min)
        D_h = _mats.derivative_matrix(N, p, h, x_min)

        # BC indices (same for every call on this maturity)
        bc_left  = list(range(p - 1))
        bc_right = list(range(n_dof - p + 1, n_dof))
        bc_dofs  = bc_left + bc_right

        # Initial condition: payoff projection for K_ref (CGMY-independent)
        _sp_ref = _SolverP(C=1.0, G=1.0, M=2.0, Y=1.5, r=0.0, K=K_ref, T=T)
        rhs_pay = _put_rhs(_sp_ref, grid, p)
        c_raw   = _spla.spsolve(M_h, rhs_pay)

        c_0     = c_raw.copy()
        c_L0    = K_ref * math.sqrt(h)
        for i in bc_left:  c_0[i] = c_L0
        for i in bc_right: c_0[i] = 0.0

        _per_mat[T_key] = dict(
            grid=grid, M_h=M_h, D_h=D_h,
            bc_left=bc_left, bc_right=bc_right, bc_dofs=bc_dofs,
            c_0=c_0, h=h, x_min=x_min,
        )

    # ── Returned closure — recomputes only K_G, K_M per call ─────────────────
    def _pricer_fn(cgmy_tuple, T_mat, K_array, option_type="auto"):
        C, G, M, Y = cgmy_tuple
        T_key   = round(T_mat, 6)
        r_e, q_e = rates_by_T[T_key]
        pm      = _per_mat[T_key]

        grid    = pm["grid"]
        M_h     = pm["M_h"]
        D_h     = pm["D_h"]
        bc_left = pm["bc_left"]
        bc_right= pm["bc_right"]
        bc_dofs = pm["bc_dofs"]
        h       = pm["h"]
        c       = pm["c_0"].copy()

        # Build solver-compatible params for fractional matrices
        sp = _SolverP(C=C, G=G, M=M, Y=Y, r=r_e, K=K_ref, T=T_mat)

        # Fast: IFFT-based fractional stiffness (~5 ms for N=64)
        K_G, K_M = _mats.fractional_stiffness_matrices(N, p, h, sp, bw)

        # Assemble A_h from cached M_h, D_h plus new K_G, K_M
        gY    = _sc_gamma(-Y)
        om_S  = _omega_stock(sp)
        lam0  = _lambda0(sp)
        A_h   = (-(r_e - om_S) * D_h - C * gY * K_M - C * gY * K_G + lam0 * M_h)

        # CN system matrices
        dt    = T_mat / N_tau
        F_mat = (M_h + (dt / 2.0) * A_h).tolil()
        for i in bc_dofs:
            F_mat[i, :] = 0.0
            F_mat[i, i] = 1.0
        B_mat = (M_h - (dt / 2.0) * A_h).tocsr()
        lu    = _spla.splu(F_mat.tocsc())

        # Crank-Nicolson time stepping (boundary_forcing_vector = 0 by design)
        tau = 0.0
        for step in range(N_tau):
            tau_n = (step + 1) * dt
            rhs   = B_mat @ c
            c_L   = K_ref * math.exp(-r_e * tau_n) * math.sqrt(h)
            for i in bc_left:  rhs[i] = c_L
            for i in bc_right: rhs[i] = 0.0
            c   = lu.solve(rhs)
            tau = tau_n

        # Homogeneity evaluation (dividend-adjusted)
        K_array    = np.asarray(K_array, dtype=float)
        log_S0_eff = math.log(S0) - q_e * T_mat
        x_eval     = log_S0_eff + log_Kref - np.log(K_array)
        x_clipped  = np.clip(x_eval, grid["x_min"] + 1e-10, grid["x_max"] - 1e-10)

        V_ref      = _eval(c, grid, p, x_clipped)
        put_prices = (K_array / K_ref) * V_ref

        D_disc = math.exp(-r_e * T_mat)
        F_fwd  = S0 * math.exp((r_e - q_e) * T_mat)  # = parity F exactly

        if option_type == "put":
            return put_prices
        elif option_type == "call":
            return put_prices + D_disc * (F_fwd - K_array)
        else:  # 'auto'
            call_prices = put_prices + D_disc * (F_fwd - K_array)
            return np.where(K_array < F_fwd, put_prices, call_prices)

    return _pricer_fn


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

    # COS reference pricer (independent benchmark)
    from cgmy_bspline.src.cgmy_params import CGMYParams as SrcP
    from cgmy_bspline.src.cos_pricer import cos_put_price

    print("=" * 65)
    print("Step 6 self-test: price_surface vs COS pricer")
    print("=" * 65)

    # ── Test parameters ───────────────────────────────────────────────────────
    C, G, M, Y = 1.0, 10.0, 10.0, 1.5
    r, q, S0   = 0.02, 0.0, 100.0
    T          = 0.5
    K_array    = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

    theta = (C, G, M, Y, r, q, S0)

    print(f"\nParameters: C={C}, G={G}, M={M}, Y={Y}, r={r}, q={q}, S0={S0}, T={T}")
    print(f"Strikes:    {K_array.tolist()}")

    # ── B-Spline prices (one PDE solve, N=256) ────────────────────────────────
    bspline_prices = price_surface(
        theta, T, K_array,
        option_type="put",
        grid_kwargs={"N": 256, "N_tau": 25, "bandwidth": 512},
    )

    # ── COS reference prices (one call per strike) ────────────────────────────
    cos_prices = np.array([
        cos_put_price(SrcP(C=C, G=G, M=M, Y=Y, r=r, T=T, K=K, S0=S0), N_cos=4096)
        for K in K_array
    ])

    # ── Table ─────────────────────────────────────────────────────────────────
    hdr = f"{'K':>6}  {'B-Spline':>12}  {'COS ref':>12}  {'rel err':>9}"
    print()
    print(hdr)
    print("-" * len(hdr))
    all_pass = True
    for K, bp, cp in zip(K_array, bspline_prices, cos_prices):
        rel_err = abs(bp - cp) / cp if cp > 1e-10 else float("nan")
        ok  = (rel_err < 0.01) if np.isfinite(rel_err) else False
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"{K:>6.0f}  {bp:>12.6f}  {cp:>12.6f}  {rel_err:>8.4%}  {tag}")

    print()
    print("=" * 65)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    print("=" * 65)
    sys.exit(0 if all_pass else 1)
