"""
Calibration objective function for the CGMY B-Spline solver (Section 7, Step 7).

Provides three public functions:

  implied_vol_from_price  -- single-point Black-forward IV inversion via Brent
  vega_normalized_loss    -- optimizer-facing vega-weighted MSE objective
  iv_rmse_and_max_err     -- post-calibration diagnostic (IV-space errors)

pricer_fn convention
--------------------
Both vega_normalized_loss and iv_rmse_and_max_err expect a callable with signature:

    pricer_fn(cgmy_tuple, T, K_array, option_type='auto') -> np.ndarray

where cgmy_tuple = (C, G, M, Y).  Fixed market parameters (r, q, S0) and any
solver settings are captured in the closure when the caller constructs pricer_fn,
for example:

    def make_pricer(S0, r, q, grid_kwargs=None):
        def _pricer(cgmy_tuple, T, K_array, option_type='auto'):
            C, G, M, Y = cgmy_tuple
            return price_surface((C, G, M, Y, r, q, S0), T, K_array,
                                 option_type=option_type,
                                 grid_kwargs=grid_kwargs)
        return _pricer

surface_by_maturity convention
--------------------------------
Both objective functions accept a dict whose values are DataFrames with at least:
  T, K, mid, bs_vega                 (required by vega_normalized_loss)
  T, K, F, D, mid, bs_vega,          (also needed by iv_rmse_and_max_err)
      market_iv, log_moneyness, date
All rows in one DataFrame must share the same maturity (T is constant per slice).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from calibration.params import CGMYParams, inverse_transform


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _black_otm(sigma: float, F: float, K: float, T: float, D: float) -> float:
    """OTM Black-1976 forward price: put if K < F, call if K >= F."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if K < F:
        return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def implied_vol_from_price(
    price: float,
    F: float,
    K: float,
    T: float,
    D: float,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Invert the OTM Black-forward formula to recover implied volatility.

    Uses Brent's method on the bracket [lo, hi].  Automatically selects put
    (K < F) or call (K >= F) Black formula.

    Returns np.nan — never raises — when:
      - any input is non-finite or out of domain (price < 0, T <= 0, …)
      - the target price lies outside the achievable range [Black(lo), Black(hi)]
        (i.e. no sign change in the bracket)
      - the Brent solver fails for any other numerical reason

    Parameters
    ----------
    price : float  market or model mid-price of the OTM option
    F     : float  forward price (S0 * exp((r-q)*T))
    K     : float  strike
    T     : float  time to expiry in years
    D     : float  discount factor  exp(-r*T)
    lo, hi: float  vol search bracket (default [1e-4, 5.0])
    """
    # Guard against degenerate inputs
    if not (
        np.isfinite(price) and price >= 0.0
        and np.isfinite(F)  and F > 0.0
        and np.isfinite(K)  and K > 0.0
        and T > 0.0
        and D > 0.0
    ):
        return np.nan

    try:
        f_lo = _black_otm(lo, F, K, T, D) - price
        f_hi = _black_otm(hi, F, K, T, D) - price
    except Exception:
        return np.nan

    # No sign change → price is outside the bracket's achievable range
    if f_lo * f_hi >= 0.0:
        return np.nan

    try:
        return float(brentq(
            lambda s: _black_otm(s, F, K, T, D) - price,
            lo, hi,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=200,
        ))
    except Exception:
        return np.nan


def vega_normalized_loss(
    theta_unconstrained: np.ndarray,
    surface_by_maturity: dict,
    pricer_fn,
    vega_floor: float = 1e-3,
) -> float:
    """Vega-normalized MSE over the full IV surface.

    Implements the Section 6 recommended objective:

        L(theta) = (1/N) * sum_i ((V_model_i - V_mid_i) / max(vega_i, vega_floor))^2

    where the sum runs over every surviving OTM quote across all maturities.
    pricer_fn is called ONCE per maturity with the full strike array for that
    maturity — not once per strike.

    Parameters
    ----------
    theta_unconstrained : array-like of length 4
        (eta1, eta2, eta3, eta4) from transform_to_unconstrained(C, G, M, Y).
    surface_by_maturity : dict
        Values are DataFrames with columns T, K, mid, bs_vega.
    pricer_fn : callable
        pricer_fn(cgmy_tuple, T, K_array, option_type='auto') -> np.ndarray
        where cgmy_tuple = (C, G, M, Y).
    vega_floor : float
        Lower bound applied to bs_vega before division (prevents blow-up near
        zero-vega deep OTM quotes).

    Returns
    -------
    float  loss value; returns 1e10 on parameter or pricer failure.
    """
    try:
        C, G, M, Y = inverse_transform(*theta_unconstrained)
    except Exception:
        return 1e10

    theta_cgmy = (C, G, M, Y)
    total_sq = 0.0
    total_n  = 0

    for df_mat in surface_by_maturity.values():
        if df_mat is None or len(df_mat) == 0:
            continue

        T      = float(df_mat["T"].iloc[0])
        K_arr  = df_mat["K"].to_numpy(dtype=float)
        mid    = df_mat["mid"].to_numpy(dtype=float)
        vega   = df_mat["bs_vega"].to_numpy(dtype=float)

        try:
            model = np.asarray(
                pricer_fn(theta_cgmy, T, K_arr, option_type="auto"),
                dtype=float,
            )
        except Exception:
            return 1e10

        eff_v = np.maximum(
            np.where(np.isfinite(vega), vega, vega_floor),
            vega_floor,
        )
        resid = (model - mid) / eff_v

        finite = np.isfinite(resid)
        total_sq += float(np.sum(resid[finite] ** 2))
        total_n  += int(finite.sum())

    return total_sq / total_n if total_n > 0 else 1e10


def iv_rmse_and_max_err(
    theta: CGMYParams,
    surface_by_maturity: dict,
    pricer_fn,
) -> tuple[float, float, pd.DataFrame]:
    """Post-calibration diagnostic: per-row IV errors.

    Prices every OTM quote with pricer_fn, inverts to model IV via
    implied_vol_from_price, and compares against the market IV.  Rows where
    inversion fails (np.nan) are excluded from the aggregate statistics but
    retained in the returned DataFrame so callers can inspect failures.

    Parameters
    ----------
    theta : CGMYParams  (calibration.params)  full calibrated parameter set
    surface_by_maturity : dict
        Values are DataFrames with columns T, K, F, D, mid, bs_vega,
        market_iv, log_moneyness, date.
    pricer_fn : callable  (same signature as for vega_normalized_loss)

    Returns
    -------
    rmse    : float  RMSE in vol points (decimal), NaN if no valid rows
    max_err : float  max absolute IV error in vol points
    df_diag : pd.DataFrame  with columns date, T, K, log_moneyness,
                             market_iv, model_iv, abs_err
    """
    theta_cgmy = (theta.C, theta.G, theta.M, theta.Y)
    rows: list[dict] = []

    for df_mat in surface_by_maturity.values():
        if df_mat is None or len(df_mat) == 0:
            continue

        T     = float(df_mat["T"].iloc[0])
        K_arr = df_mat["K"].to_numpy(dtype=float)
        F_arr = df_mat["F"].to_numpy(dtype=float)
        D_arr = df_mat["D"].to_numpy(dtype=float)

        try:
            model = np.asarray(
                pricer_fn(theta_cgmy, T, K_arr, option_type="auto"),
                dtype=float,
            )
        except Exception:
            model = np.full(len(K_arr), np.nan)

        mkt_iv   = df_mat["market_iv"].to_numpy(dtype=float)
        lm_col   = df_mat["log_moneyness"].to_numpy(dtype=float)
        t_col    = df_mat["T"].to_numpy(dtype=float)
        date_col = df_mat["date"].to_numpy()

        for K, F, D, mp, miv, lm_i, t_i, d_i in zip(
            K_arr, F_arr, D_arr, model, mkt_iv, lm_col, t_col, date_col
        ):
            model_iv = (
                implied_vol_from_price(mp, F, K, T, D)
                if np.isfinite(mp)
                else np.nan
            )
            abs_e = (
                float(abs(model_iv - miv))
                if (np.isfinite(model_iv) and np.isfinite(miv))
                else np.nan
            )
            rows.append(
                {
                    "date":          d_i,
                    "T":             t_i,
                    "K":             K,
                    "log_moneyness": lm_i,
                    "market_iv":     miv,
                    "model_iv":      model_iv,
                    "abs_err":       abs_e,
                }
            )

    df_diag = pd.DataFrame(rows)

    if df_diag.empty or df_diag["abs_err"].notna().sum() == 0:
        return np.nan, np.nan, df_diag

    errs    = df_diag["abs_err"].dropna().to_numpy(dtype=float)
    rmse    = float(np.sqrt(np.mean(errs ** 2)))
    max_err = float(errs.max())
    return rmse, max_err, df_diag


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    from calibration.params import transform_to_unconstrained

    all_pass = True

    # ── Test 1a: exact recovery of sigma from a synthetic Black price ─────────
    print("=" * 65)
    print("Test 1a: implied_vol_from_price — exact recovery (put and call)")
    print("=" * 65)

    for sigma_true, F_, K_, T_, D_, label in [
        (0.20, 100.0,  95.0, 0.5, 0.990, "OTM put  (K=95,  F=100)"),
        (0.25, 100.0, 105.0, 0.5, 0.990, "OTM call (K=105, F=100)"),
        (0.30, 100.0, 100.0, 1.0, 0.980, "ATM call (K=100, F=100)"),
    ]:
        true_price = _black_otm(sigma_true, F_, K_, T_, D_)
        recovered  = implied_vol_from_price(true_price, F_, K_, T_, D_)
        err = abs(recovered - sigma_true)
        ok  = np.isfinite(recovered) and err < 1e-6
        if not ok:
            all_pass = False
        print(
            f"  {label}: sigma_true={sigma_true:.2f} "
            f"price={true_price:.6f}  recovered={recovered:.8f}  "
            f"err={err:.2e}  {'PASS' if ok else 'FAIL'}"
        )

    # ── Test 1b: nan path — price outside achievable range ───────────────────
    print()
    print("=" * 65)
    print("Test 1b: implied_vol_from_price — nan path")
    print("=" * 65)

    nan_cases = [
        # (description, price, F, K, T, D)
        ("negative price",
         implied_vol_from_price(-1.0, 100.0, 95.0, 0.5, 0.99)),
        ("price > max achievable call (D*F≈99 at sigma→∞, but even sigma=5 gives 91)",
         implied_vol_from_price(110.0, 100.0, 100.0, 0.5, 0.99)),
        ("non-finite (nan) price",
         implied_vol_from_price(np.nan, 100.0, 95.0, 0.5, 0.99)),
        ("T <= 0",
         implied_vol_from_price(5.0,   100.0, 95.0, 0.0, 0.99)),
    ]

    for desc, result in nan_cases:
        ok = isinstance(result, float) and np.isnan(result)
        if not ok:
            all_pass = False
        print(f"  {desc}:  result={result}  {'PASS' if ok else 'FAIL'}")

    # ── Test 2a: vega_normalized_loss — perfect fit gives exactly 0 ──────────
    print()
    print("=" * 65)
    print("Test 2a: vega_normalized_loss — perfect fit → loss = 0")
    print("=" * 65)

    K_syn    = np.array([90.0,  95.0, 105.0, 110.0])
    mid_syn  = np.array([ 3.5,   6.2,   7.8,   5.1])
    vega_syn = np.array([ 0.15,  0.18,  0.17,  0.14])

    df_syn = pd.DataFrame({
        "T":       0.5,
        "K":       K_syn,
        "F":       100.0,
        "D":       0.99,
        "mid":     mid_syn,
        "bs_vega": vega_syn,
    })
    surface_syn = {"mat1": df_syn}

    def _perfect_pricer(theta_cgmy, T, K_array, option_type="auto"):
        return mid_syn.copy()

    theta_unc  = transform_to_unconstrained(1.0, 5.0, 5.0, 1.5)
    loss_zero  = vega_normalized_loss(theta_unc, surface_syn, _perfect_pricer)
    ok_zero    = abs(loss_zero) < 1e-14
    if not ok_zero:
        all_pass = False
    print(f"  loss (perfect fit) = {loss_zero:.2e}  {'PASS' if ok_zero else 'FAIL'}")

    # ── Test 2b: vega_normalized_loss — perturbed fit matches analytic ────────
    print()
    print("=" * 65)
    print("Test 2b: vega_normalized_loss — perturbed fit matches hand value")
    print("=" * 65)

    delta = 0.01

    def _perturbed_pricer(theta_cgmy, T, K_array, option_type="auto"):
        return mid_syn + delta

    loss_pert = vega_normalized_loss(theta_unc, surface_syn, _perturbed_pricer)

    eff_v_exp = np.maximum(vega_syn, 1e-3)   # vega_floor default = 1e-3
    expected  = float(np.mean((delta / eff_v_exp) ** 2))

    ok_pert = (loss_pert > 0.0) and (abs(loss_pert - expected) < 1e-12)
    if not ok_pert:
        all_pass = False
    print(f"  loss (perturbed)   = {loss_pert:.10f}")
    print(f"  expected           = {expected:.10f}")
    print(f"  abs difference     = {abs(loss_pert - expected):.2e}  {'PASS' if ok_pert else 'FAIL'}")

    print()
    print("=" * 65)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    print("=" * 65)
    sys.exit(0 if all_pass else 1)
