import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from cgmy_bspline.bspline_basis import dphi, ddphi, active_indices
from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.grid import make_grid


def compute_greeks(c, grid_info, p, x_eval):
    """
    Compute Delta and Gamma at each x in x_eval.

    partial_x  V_h(x) = sum_k c_k * dphi_{h,k}(x)
    partial_xx V_h(x) = sum_k c_k * ddphi_{h,k}(x)

    Delta = exp(-x)  * partial_x  V_h
    Gamma = exp(-2x) * (partial_xx V_h - partial_x V_h)

    Note: accurate Gamma requires p >= 3.
    """
    h     = grid_info['h']
    x_min = grid_info['x_min']
    N     = grid_info['N']
    ks    = active_indices(N, p)

    x_eval = np.asarray(x_eval, dtype=float)
    dV  = np.zeros_like(x_eval)
    ddV = np.zeros_like(x_eval)
    for i, k in enumerate(ks):
        dV  += c[i] * dphi (x_eval, k, p, h, x_min)
        ddV += c[i] * ddphi(x_eval, k, p, h, x_min)

    Delta = np.exp(-x_eval)        * dV
    Gamma = np.exp(-2.0 * x_eval)  * (ddV - dV)
    return Delta, Gamma


def _bs_put(sigma, S, K, T, r):
    """Black-Scholes European put price."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_implied_vol(price, S, K, T, r, option_type='put'):
    """
    Black-Scholes implied volatility for a European put.

    Uses scipy.optimize.brentq on [1e-4, 10.0].
    Returns NaN if the price is at or below intrinsic value, or if brentq fails.
    """
    intrinsic = max(K * np.exp(-r * T) - S, 0.0)
    if price <= intrinsic + 1e-10:
        return float('nan')

    try:
        return float(brentq(
            lambda s: _bs_put(s, S, K, T, r) - price,
            1e-4, 10.0, xtol=1e-8, maxiter=100,
        ))
    except (ValueError, RuntimeError):
        return float('nan')


def iv_surface(params, N, p, strikes, maturities, N_tau_per_unit=200,
               bandwidth=64, mode='banded'):
    """
    Generate the implied-volatility surface.

    For each maturity T_i:
      1. Solve the backward FPDE with params.T = T_i.
      2. Evaluate V_h at x = log(strikes).
      3. Convert each price to IV via bs_implied_vol(price, S=K_j, K=params.K, T_i, r).

    The FPDE solution V_h(log(K_j)) is the put price when the current stock price
    is S0 = K_j and the option strike is params.K (the reference strike). The 'strikes'
    array therefore sweeps moneyness around params.K.

    Parameters
    ----------
    params          : CGMYParams  (params.K is the fixed option strike)
    N               : spatial intervals
    p               : B-spline order
    strikes         : array of spot prices S0 at which to evaluate
    maturities      : list of maturities T_i
    N_tau_per_unit  : time steps per unit of maturity
    bandwidth       : banded-mode bandwidth (or preconditioner bandwidth)
    mode            : passed to solve_cn

    Returns
    -------
    iv_matrix : ndarray of shape (len(maturities), len(strikes)), entries are IVs
    """
    strikes    = np.asarray(strikes, dtype=float)
    maturities = list(maturities)
    x_eval     = np.log(strikes)

    iv_mat = np.full((len(maturities), len(strikes)), np.nan)

    for i, T_i in enumerate(maturities):
        N_tau = max(1, round(N_tau_per_unit * T_i))
        params_i = CGMYParams(
            C=params.C, G=params.G, M=params.M, Y=params.Y,
            r=params.r, K=params.K, T=T_i,
        )
        result = solve_cn(params_i, N=N, p=p, N_tau=N_tau,
                          bandwidth=bandwidth, mode=mode)
        g      = make_grid(params_i, N)
        prices = eval_solution(result['c_final'], g, p, x_eval)

        for j, (K_j, price_j) in enumerate(zip(strikes, prices)):
            iv_mat[i, j] = bs_implied_vol(
                price_j, S=K_j, K=params.K, T=T_i, r=params.r
            )

    return iv_mat
