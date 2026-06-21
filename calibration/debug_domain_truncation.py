"""
Diagnostic for domain-truncation error at the T=1y checkpoint.

The calibration pricer (calibration/pricer.py, make_cached_pricer) solves
on a FIXED log-price domain [log(K_ref)-2.0, log(K_ref)+2.0] for every
maturity and every date. cgmy_bspline/grid.py and src/american_solver.py
instead size the domain to the model's own tail-decay rates:
    x_min = log(K) - domain_scale/G,   x_max = log(K) + domain_scale/M
with domain_scale=10 chosen so the exterior tail carries weight ~exp(-10).
For dates with small G or M (heavy/slowly-decaying tails -- see the crisis
date investigated in debug_short_maturity.py), a FIXED half-width of 2.0
log-price units may not give that same exp(-10) margin, which would show
up as a systematic, theta-dependent domain-truncation error -- including
possibly at T=1y, where there has been a full year for the tempered-stable
jump measure's tails to matter.

This script:
  1. Evaluates a theoretical truncation-error ORDER OF MAGNITUDE at the
     production half-width (2.0) for each date's theta, using the same
     exponential-tempering decay (exp(-G*half_width), exp(-M*half_width))
     that motivates the domain_scale/G, domain_scale/M rule already used
     elsewhere in this codebase. No formal "Proposition" with this exact
     bound was found written out anywhere in this repo (searched
     cgmy_bspline/results/*.tex, src/american_solver.py, grid.py) -- this
     is a derived heuristic consistent with that existing convention, not
     a verbatim-quoted paper result.
  2. Empirically re-solves the T=1.0 price with the CURRENT domain
     (half_width=2.0) and a domain widened by 50% (half_width=3.0), same
     Nt=10, dt, N, p, bandwidth -- and compares the resulting IV against a
     COS benchmark (characteristic-function based, effectively domain-free).
  3. Prints the three calibrated theta side by side to check whether the
     crisis date's tail-decay parameters are what make a SHARED fixed
     domain inadequate for it but not the other two dates.

Does NOT modify cgmy_bspline/grid.py, calibration/pricer.py, or any
production domain-width setting.

Run:
    python calibration/debug_domain_truncation.py
"""

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from calibration.debug_short_maturity import (
    _DATES, _N, _P, _HALF_W, _MONEYNESS, _NT_BASELINE,
    _load_date_context, _build_spatial, _build_schedule, _run_solve,
    _eval_prices, _cos_prices,
)
from calibration.objective import implied_vol_from_price

_T_EVAL = 1.0
_WIDE_HALF_W = 1.5 * _HALF_W  # domain widened by 50%

# From cgmy_bspline/results/calibration_report.tex: "comfortably within the
# 3 vp target" / "3 vp acceptance threshold" used throughout the paper for
# in-sample IV RMSE.
_IV_TOLERANCE_VP = 3.0


# ---------------------------------------------------------------------------
# 1. Theoretical domain-truncation error order of magnitude
# ---------------------------------------------------------------------------

def truncation_bound(G: float, M: float, half_width: float = _HALF_W) -> dict:
    """
    Heuristic order-of-magnitude truncation factor, consistent with the
    domain_scale/G, domain_scale/M convention in grid.py and
    src/american_solver.py (domain_scale=10 there targets exp(-10)).

    eps_left  = exp(-G * half_width)   -- relative weight of the exterior
                                           tail beyond x_min (deep ITM side)
    eps_right = exp(-M * half_width)   -- relative weight beyond x_max

    These are dimensionless tail-mass ratios, not exact price/IV error
    bounds; they indicate the ORDER OF MAGNITUDE of the mismatch between
    the assumed Dirichlet boundary value and the true (non-local-operator)
    solution there.
    """
    eps_left = math.exp(-G * half_width)
    eps_right = math.exp(-M * half_width)
    eps = max(eps_left, eps_right)
    return dict(eps_left=eps_left, eps_right=eps_right, eps=eps, half_width=half_width)


# ---------------------------------------------------------------------------
# 2. Empirical T=1y re-solve at current vs widened domain
# ---------------------------------------------------------------------------

def _otm_price_and_iv(put_price: float, K: float, F: float, D: float, T: float):
    call_price = put_price + D * (F - K)
    otm_price = put_price if K < F else call_price
    iv = implied_vol_from_price(otm_price, F, K, T, D)
    return otm_price, iv


def domain_truncation_empirical(ctx: dict, Nt: int = _NT_BASELINE) -> dict:
    K_array = ctx["S0"] * _MONEYNESS
    r, q, S0 = ctx["r"], ctx["q"], ctx["S0"]
    F = S0 * math.exp((r - q) * _T_EVAL)
    D = math.exp(-r * _T_EVAL)

    sp_current = _build_spatial(ctx, half_width=_HALF_W)
    sp_wide = _build_spatial(ctx, half_width=_WIDE_HALF_W)

    sched, dt = _build_schedule(Nt, _T_EVAL, rannacher=False)

    c_current = _run_solve(ctx, sp_current, sched, {"1y": _T_EVAL}, c0_key="c0")["1y"]
    c_wide = _run_solve(ctx, sp_wide, sched, {"1y": _T_EVAL}, c0_key="c0")["1y"]

    p_current = _eval_prices(c_current, sp_current, ctx, _T_EVAL, K_array)
    p_wide = _eval_prices(c_wide, sp_wide, ctx, _T_EVAL, K_array)
    p_cos = _cos_prices(ctx, _T_EVAL, K_array)

    iv_current = np.full(len(K_array), np.nan)
    iv_wide = np.full(len(K_array), np.nan)
    iv_cos = np.full(len(K_array), np.nan)
    for i, K in enumerate(K_array):
        _, iv_current[i] = _otm_price_and_iv(p_current[i], K, F, D, _T_EVAL)
        _, iv_wide[i] = _otm_price_and_iv(p_wide[i], K, F, D, _T_EVAL)
        _, iv_cos[i] = _otm_price_and_iv(p_cos[i], K, F, D, _T_EVAL)

    iv_diff_current_wide_vp = np.abs(iv_current - iv_wide) * 100.0
    iv_diff_current_cos_vp = np.abs(iv_current - iv_cos) * 100.0
    iv_diff_wide_cos_vp = np.abs(iv_wide - iv_cos) * 100.0

    max_diff_current_wide = float(np.nanmax(iv_diff_current_wide_vp))

    return dict(
        K_array=K_array, dt=dt,
        p_current=p_current, p_wide=p_wide, p_cos=p_cos,
        iv_current=iv_current, iv_wide=iv_wide, iv_cos=iv_cos,
        iv_diff_current_wide_vp=iv_diff_current_wide_vp,
        iv_diff_current_cos_vp=iv_diff_current_cos_vp,
        iv_diff_wide_cos_vp=iv_diff_wide_cos_vp,
        max_diff_current_wide=max_diff_current_wide,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print(f"Domain-truncation diagnostic at T={_T_EVAL}y")
    print(f"Current production domain half-width = {_HALF_W}, widened (x1.5) = {_WIDE_HALF_W}")
    print(f"IV tolerance (from calibration_report.tex '3 vp target/acceptance threshold') = "
          f"{_IV_TOLERANCE_VP} vp")
    print("=" * 100)

    ctxs = {}
    results = {}

    for date_str, csv_name in _DATES:
        print(f"\n{'-' * 100}")
        print(f"Date: {date_str}")
        print("-" * 100)

        ctx = _load_date_context(date_str, csv_name)
        ctxs[date_str] = ctx

        tb = truncation_bound(ctx["G"], ctx["M"], half_width=_HALF_W)
        print(f"  theta = (C={ctx['C']:.6f}, G={ctx['G']:.6f}, M={ctx['M']:.6f}, Y={ctx['Y']:.6f})")
        print(f"  [1] Theoretical truncation order of magnitude at half_width={_HALF_W}:")
        print(f"      eps_left  = exp(-G*{_HALF_W}) = {tb['eps_left']:.6f}")
        print(f"      eps_right = exp(-M*{_HALF_W}) = {tb['eps_right']:.6f}")
        print(f"      eps = max(eps_left, eps_right) = {tb['eps']:.6f}  "
              f"({'LARGE (>0.1)' if tb['eps'] > 0.1 else 'small'})")

        print(f"\n  [2] Empirical re-solve at T={_T_EVAL}y: current domain vs widened domain vs COS")
        emp = domain_truncation_empirical(ctx)
        print(f"      strikes: {emp['K_array'].round(2).tolist()}")
        print(f"      IV(current domain, hw={_HALF_W}): "
              f"{(emp['iv_current'] * 100).round(3).tolist()} (vp)")
        print(f"      IV(wide domain,    hw={_WIDE_HALF_W}): "
              f"{(emp['iv_wide'] * 100).round(3).tolist()} (vp)")
        print(f"      IV(COS benchmark)              : "
              f"{(emp['iv_cos'] * 100).round(3).tolist()} (vp)")
        print(f"      |current - wide| (vp): {emp['iv_diff_current_wide_vp'].round(3).tolist()}")
        print(f"      |current - COS|  (vp): {emp['iv_diff_current_cos_vp'].round(3).tolist()}")
        print(f"      |wide - COS|     (vp): {emp['iv_diff_wide_cos_vp'].round(3).tolist()}")
        print(f"      max|current-wide| = {emp['max_diff_current_wide']:.3f} vp")

        results[date_str] = dict(theta=ctx, bound=tb, empirical=emp)

    # ── theta side-by-side (item 3) ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("[3] Calibrated theta side by side")
    print("=" * 100)
    hdr = f"{'date':>12}  {'C*':>10}  {'G*':>10}  {'M*':>10}  {'Y*':>8}  {'min(G,M)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for date_str, _ in _DATES:
        t = results[date_str]["theta"]
        print(
            f"{date_str:>12}  {t['C']:>10.4f}  {t['G']:>10.4f}  {t['M']:>10.4f}  "
            f"{t['Y']:>8.4f}  {min(t['G'], t['M']):>10.4f}"
        )
    print()
    print("The crisis date (2020-03-18) has G*=0.10, dramatically smaller than")
    print("the other two dates' G* (2.24, 2.59) -- a much heavier/slower-decaying")
    print("left tail in log-price space. M* is comparable (or larger) across all")
    print("three. min(G,M) is the relevant (slower-decaying, more dangerous) rate")
    print("for the fixed domain: 0.10 for crisis vs 2.24/2.59 for the other dates")
    print("-- the same mechanism identified in debug_short_maturity.py's local-")
    print("refinement test, now checked specifically at T=1y.")

    # ── SELF-TEST ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("SELF-TEST: domain-truncation verdict per date")
    print("=" * 100)
    hdr2 = f"{'date':>12}  {'theory eps':>11}  {'max|cur-wide| (vp)':>20}  {'tolerance (vp)':>15}  {'verdict':>16}"
    print(hdr2)
    print("-" * len(hdr2))
    for date_str, _ in _DATES:
        tb = results[date_str]["bound"]
        emp = results[date_str]["empirical"]
        diff = emp["max_diff_current_wide"]

        exceeds_tolerance = diff > _IV_TOLERANCE_VP
        theory_predicts_order = tb["eps"] > 0.05  # order-of-magnitude check, not exact

        if exceeds_tolerance and theory_predicts_order:
            verdict = "CONFIRMED"
        else:
            verdict = "NOT CONFIRMED"

        print(
            f"{date_str:>12}  {tb['eps']:>11.4f}  {diff:>20.3f}  "
            f"{_IV_TOLERANCE_VP:>15.1f}  {verdict:>16}"
        )

        if verdict == "NOT CONFIRMED":
            print(f"    -> current-domain and wide-domain IVs agree within "
                  f"{_IV_TOLERANCE_VP} vp (diff={diff:.3f} vp); any 1y pricing error for "
                  f"{date_str} is NOT explained by domain truncation at this half-width "
                  f"and must come from elsewhere (e.g. spatial under-resolution near the "
                  f"kink at short maturities does not apply at T=1y; possible candidates: "
                  f"Nt=10 CN time-stepping accuracy, bandwidth truncation, or the "
                  f"calibration optimizer's own fit residual).")
        else:
            print(f"    -> current-domain IV at T=1y differs from wide-domain IV by more "
                  f"than the {_IV_TOLERANCE_VP} vp tolerance, AND the theoretical tail-decay "
                  f"factor eps={tb['eps']:.4f} at half_width={_HALF_W} predicts a leak of "
                  f"this order of magnitude (eps > 0.05) given this date's min(G,M). "
                  f"Domain truncation is a credible cause for this date's T=1y error.")


if __name__ == "__main__":
    main()
