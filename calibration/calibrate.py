"""
CGMY calibration to an OTM IV surface (Section 7, Step 8).

calibrate_one_date
------------------
Two-phase calibration pipeline:

  Phase 1 – Differential Evolution (global search, N=32 pricer, fast)
  Phase 2 – L-BFGS-B (local refinement from DE result, N=64 pricer, accurate)

The optimisation variable is the *unconstrained* transform
    eta = (log C, log G, log(M-1), log((Y-1)/(2-Y)))
with bounds derived programmatically from the physical bounds:
    C ∈ [1e-3, 20], G ∈ [0.1, 80], M ∈ [1.01, 80], Y ∈ [1.05, 1.95].

Per-maturity (r_eff, q_eff) are computed from the parity-fitted (F, D) in the
surface CSV so that the forward exactly matches market parity.

S0 proxy: S0 = F_shortest * D_shortest (exact for q=0; error O(q * T_short)).

Performance notes
-----------------
Mass matrix M_h and derivative matrix D_h are CGMY-independent and expensive
(~500 ms each via scipy.quad).  make_cached_pricer pre-builds them once per
maturity so each objective evaluation only rebuilds the fractional stiffness
matrices (~5 ms via IFFT), giving a ~100× speedup.

Typical timing on a single core:
  N=32 cached pricer: pre-build 4.4 s, eval 25 ms  → DE ≈ 455 s (7.6 min)
  N=64 cached pricer: pre-build 8.6 s, eval 51 ms  → L-BFGS-B ≈ 25 s
"""

import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from calibration.params import (
    CGMYParams,
    transform_to_unconstrained,
    inverse_transform,
)
from calibration.pricer import make_cached_pricer
from calibration.objective import vega_normalized_loss, iv_rmse_and_max_err


# ---------------------------------------------------------------------------
# Physical parameter bounds for CGMY calibration
# ---------------------------------------------------------------------------

_C_LO, _C_HI = 1e-3, 20.0
_G_LO, _G_HI = 0.1,  80.0
_M_LO, _M_HI = 1.01, 80.0
_Y_LO, _Y_HI = 1.05, 1.95


def _eta_bounds() -> list[tuple[float, float]]:
    """Derive eta-space bounds from physical bounds via the transform.

    Never guess eta bounds — always derive them from the physical ones so
    any change to the physical bounds propagates automatically.
    """
    eta_lo = transform_to_unconstrained(_C_LO, _G_LO, _M_LO, _Y_LO)
    eta_hi = transform_to_unconstrained(_C_HI, _G_HI, _M_HI, _Y_HI)
    return [(lo, hi) for lo, hi in zip(eta_lo, eta_hi)]


# ---------------------------------------------------------------------------
# Surface loader helpers
# ---------------------------------------------------------------------------


def _load_surface(csv_path: str) -> tuple[dict, float, dict]:
    """Load a processed surface CSV and return (surface_by_maturity, S0, rates).

    surface_by_maturity : dict  exdate-string → DataFrame
    S0                  : float proxy spot = F_short * D_short
    rates               : dict  round(T,6) → (r_eff, q_eff)
    """
    df = pd.read_csv(csv_path, parse_dates=["date", "exdate"])

    surface: dict = {}
    for exdate, grp in df.groupby("exdate", sort=True):
        surface[str(exdate.date())] = grp.reset_index(drop=True)

    # S0 proxy from shortest-dated maturity
    sk = min(surface, key=lambda k: float(surface[k]["T"].iloc[0]))
    S0 = float(surface[sk]["F"].iloc[0]) * float(surface[sk]["D"].iloc[0])

    # Per-maturity rates from parity-fitted (F, D)
    rates: dict = {}
    for grp in surface.values():
        T = float(grp["T"].iloc[0])
        F = float(grp["F"].iloc[0])
        D = float(grp["D"].iloc[0])
        r_eff = -math.log(D) / T
        q_eff = r_eff - math.log(F / S0) / T
        rates[round(T, 6)] = (r_eff, q_eff)

    return surface, S0, rates


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------


def calibrate_one_date(
    surface_csv_path: str,
    r: float,
    q: float,
    seed: int = 0,
    de_maxiter: int = 300,
    de_popsize: int = 15,
) -> dict:
    """Calibrate CGMY parameters to one IV surface snapshot.

    Parameters
    ----------
    surface_csv_path : str
        Path to a processed surface CSV (output of build_surface.py).
    r : float
        Nominal risk-free rate (accepted for API compatibility; overridden per
        maturity by rates derived from parity-fitted F and D in the CSV).
    q : float
        Nominal dividend yield (same caveat — overridden per maturity).
    seed : int
        Random seed for Differential Evolution.
    de_maxiter : int
        Maximum number of DE generations (default 300).
    de_popsize : int
        DE population size multiplier; total population = de_popsize * 4
        (default 15 → 60 individuals for 4 free parameters).

    Returns
    -------
    dict with keys:
        theta_cgmy    CGMYParams  calibrated parameters
        de_loss       float       vega-normalised loss after Phase 1
        lbfgs_loss    float       vega-normalised loss after Phase 2
        iv_rmse       float       IV RMSE in decimal vol (e.g. 0.015 = 1.5 vp)
        max_iv_err    float       max absolute IV error across all quotes
        df_diag       pd.DataFrame  per-quote diagnostics
        total_time_sec float      wall-clock seconds for entire calibration
        de_nfev       int         number of Phase-1 objective evaluations
        lbfgs_nfev    int         number of Phase-2 objective evaluations
        n_quotes      int         number of OTM quotes in the surface
        S0            float       spot proxy used
    """
    t_start = time.time()

    # ── 1. Load surface ────────────────────────────────────────────────────────
    surface, S0, rates = _load_surface(surface_csv_path)
    n_quotes = sum(len(v) for v in surface.values())
    n_mats = len(surface)
    print(f"Loaded surface: {n_quotes} quotes across {n_mats} maturities")
    print(f"S0 proxy = {S0:.2f}")
    for T_key, (r_e, q_e) in sorted(rates.items()):
        print(f"  T={T_key:.4f}y  r_eff={r_e:.4f}  q_eff={q_e:.4f}")

    # ── 2. Pre-build cached pricers (N=32 for DE, N=64 for L-BFGS-B) ─────────
    print("\nBuilding cached pricers...")
    t_build = time.time()

    gkw32 = dict(N=32, p=3, N_tau=5,  bandwidth=64,  domain_half_width=2.0)
    gkw64 = dict(N=64, p=3, N_tau=10, bandwidth=128, domain_half_width=2.0)

    pf32 = make_cached_pricer(surface, S0, rates, grid_kwargs=gkw32)
    t_b32 = time.time() - t_build
    print(f"  N=32 pricer built in {t_b32:.1f}s")

    pf64 = make_cached_pricer(surface, S0, rates, grid_kwargs=gkw64)
    t_b64 = time.time() - t_build - t_b32
    print(f"  N=64 pricer built in {t_b64:.1f}s")

    # ── Timing estimate ────────────────────────────────────────────────────────
    _t0 = time.time()
    _theta_test = transform_to_unconstrained(1.0, 5.0, 10.0, 1.5)
    vega_normalized_loss(_theta_test, surface, pf32)
    t_eval32 = time.time() - _t0
    de_total_evals = (de_maxiter + 1) * de_popsize * 4
    de_est_sec = de_total_evals * t_eval32
    print(f"\nDE timing estimate:")
    print(f"  eval (N=32) = {t_eval32*1000:.1f} ms")
    print(f"  max evals   = {de_total_evals:,}")
    print(f"  estimate    = {de_est_sec:.0f}s ({de_est_sec/60:.1f} min)")
    if de_est_sec > 600:
        print(
            f"  WARNING: estimated DE time ({de_est_sec/60:.1f} min) exceeds "
            f"10-minute budget. Per-eval cost is {t_eval32*1000:.1f} ms. "
            f"Consider reducing de_maxiter or de_popsize."
        )

    # ── 3. Eta bounds (derived programmatically from physical bounds) ──────────
    bounds = _eta_bounds()
    print(f"\nEta bounds (unconstrained space):")
    for i, (lo, hi) in enumerate(bounds):
        print(f"  eta{i+1}: [{lo:.3f}, {hi:.3f}]")

    # ── 4. Phase 1: Differential Evolution (global search) ────────────────────
    print(f"\nPhase 1: Differential Evolution (seed={seed}, "
          f"maxiter={de_maxiter}, popsize={de_popsize})...")
    t_de_start = time.time()

    _loss32 = lambda eta: vega_normalized_loss(eta, surface, pf32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        de_result = differential_evolution(
            _loss32,
            bounds,
            seed=seed,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-7,
            polish=False,
            mutation=(0.5, 1.5),
            recombination=0.7,
            workers=1,
        )

    t_de = time.time() - t_de_start
    de_loss = float(de_result.fun)
    de_nfev = int(de_result.nfev)
    print(f"  converged={de_result.success}  nfev={de_nfev}  "
          f"loss={de_loss:.6f}  time={t_de:.1f}s ({t_de/60:.1f}min)")
    _C_de, _G_de, _M_de, _Y_de = inverse_transform(*de_result.x)
    print(f"  DE result: C={_C_de:.4f} G={_G_de:.4f} M={_M_de:.4f} Y={_Y_de:.4f}")

    # ── 5. Phase 2: L-BFGS-B local refinement (higher-accuracy N=64 pricer) ──
    print(f"\nPhase 2: L-BFGS-B (from DE result, N=64 pricer)...")
    t_lbfgs_start = time.time()

    _loss64 = lambda eta: vega_normalized_loss(eta, surface, pf64)

    lbfgs_result = minimize(
        _loss64,
        de_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    t_lbfgs = time.time() - t_lbfgs_start
    lbfgs_loss = float(lbfgs_result.fun)
    lbfgs_nfev = int(lbfgs_result.nfev)
    print(f"  converged={lbfgs_result.success}  nfev={lbfgs_nfev}  "
          f"loss={lbfgs_loss:.6f}  time={t_lbfgs:.1f}s")

    # ── 6. Inverse transform to physical parameters ────────────────────────────
    eta_star = lbfgs_result.x
    C_cal, G_cal, M_cal, Y_cal = inverse_transform(*eta_star)
    print(f"\nCalibrated CGMY parameters:")
    print(f"  C = {C_cal:.6f}")
    print(f"  G = {G_cal:.6f}")
    print(f"  M = {M_cal:.6f}")
    print(f"  Y = {Y_cal:.6f}")

    # Use the average of per-maturity rates as the nominal r, q for CGMYParams
    r_avg = float(np.mean([rv[0] for rv in rates.values()]))
    q_avg = float(np.mean([rv[1] for rv in rates.values()]))
    theta_cal = CGMYParams(C=C_cal, G=G_cal, M=M_cal, Y=Y_cal,
                           r=r_avg, q=q_avg, S0=S0)

    # ── 7. Diagnostics: IV RMSE and per-quote errors ──────────────────────────
    print("\nComputing IV RMSE...")
    iv_rmse, max_iv_err, df_diag = iv_rmse_and_max_err(theta_cal, surface, pf64)
    print(f"  IV RMSE   = {iv_rmse:.4f} ({iv_rmse*100:.2f} vol pts)")
    print(f"  max |err| = {max_iv_err:.4f} ({max_iv_err*100:.2f} vol pts)")

    total_time = time.time() - t_start
    print(f"\nTotal calibration time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Per-maturity RMSE breakdown (always print for visibility) ─────────────
    print("\nPer-maturity IV RMSE:")
    valid = df_diag.dropna(subset=["abs_err"])
    for T_val, grp in valid.groupby("T", sort=True):
        rmse_t = float(np.sqrt(np.mean(grp["abs_err"].to_numpy() ** 2)))
        print(f"  T={T_val:.4f}y  n={len(grp):3d}  rmse={rmse_t:.4f} "
              f"({rmse_t*100:.2f} vp)")

    if iv_rmse >= 0.03:
        print("\nWARNING: iv_rmse >= 0.03 — worst 10 quotes:")
        worst = df_diag.dropna(subset=["abs_err"]).nlargest(10, "abs_err")
        pd.set_option("display.float_format", "{:.4f}".format)
        print(worst[["T", "K", "log_moneyness",
                      "market_iv", "model_iv", "abs_err"]].to_string(index=False))

    return {
        "theta_cgmy":     theta_cal,
        "de_loss":        de_loss,
        "lbfgs_loss":     lbfgs_loss,
        "iv_rmse":        iv_rmse,
        "max_iv_err":     max_iv_err,
        "df_diag":        df_diag,
        "total_time_sec": total_time,
        "de_nfev":        de_nfev,
        "lbfgs_nfev":     lbfgs_nfev,
        "n_quotes":       n_quotes,
        "S0":             S0,
    }


# ---------------------------------------------------------------------------
# __main__: run on 2021-09-15 surface and print PASS/FAIL per criterion
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

    csv_path = str(
        pathlib.Path(__file__).parents[1]
        / "data" / "processed" / "surface_2021-09-15_spread05.csv"
    )

    print("=" * 70)
    print("CGMY Calibration — 2021-09-15 (Section 7, Step 8)")
    print("=" * 70)
    print(f"Surface: {csv_path}")
    print()

    result = calibrate_one_date(
        surface_csv_path=csv_path,
        r=0.0,   # nominal only — overridden per maturity from parity F,D
        q=0.0,
        seed=0,
        de_maxiter=300,
        de_popsize=15,
    )

    theta = result["theta_cgmy"]

    print()
    print("=" * 70)
    print("PIVOT CRITERION (Step 8)")
    print("=" * 70)
    all_pass = True

    # ── (a) Parameter validity ────────────────────────────────────────────────
    valid_c = theta.C > 0
    valid_g = theta.G > 0
    valid_m = theta.M > 1
    valid_y = 1.0 < theta.Y < 2.0
    ok_params = valid_c and valid_g and valid_m and valid_y
    if not ok_params:
        all_pass = False
    tag = "PASS" if ok_params else "FAIL"
    print(f"(a) Parameter validity: C={theta.C:.4f}>0, G={theta.G:.4f}>0, "
          f"M={theta.M:.4f}>1, Y={theta.Y:.4f}∈(1,2)  [{tag}]")

    # ── (b) IV RMSE < 0.03 ───────────────────────────────────────────────────
    iv_rmse = result["iv_rmse"]
    ok_rmse = np.isfinite(iv_rmse) and iv_rmse < 0.03
    if not ok_rmse:
        all_pass = False
    tag = "PASS" if ok_rmse else "FAIL"
    print(f"(b) IV RMSE < 0.03: {iv_rmse:.4f}  [{tag}]")

    # ── (c) L-BFGS-B loss ≤ DE loss ──────────────────────────────────────────
    ok_order = result["lbfgs_loss"] <= result["de_loss"]
    if not ok_order:
        all_pass = False
    tag = "PASS" if ok_order else "FAIL"
    print(f"(c) lbfgs_loss ≤ de_loss: {result['lbfgs_loss']:.6f} ≤ "
          f"{result['de_loss']:.6f}  [{tag}]")

    # ── (d) Timing ───────────────────────────────────────────────────────────
    t_sec = result["total_time_sec"]
    ok_time = t_sec < 600
    tag = "PASS" if ok_time else "WARN"
    if not ok_time:
        print(f"(d) Timing: {t_sec:.1f}s ({t_sec/60:.1f} min) — "
              f"EXCEEDS 10-minute budget  [{tag}]")
        print(f"    Suggestion: reduce de_maxiter (current {300}) or "
              f"de_popsize (current {15})")
    else:
        print(f"(d) Timing: {t_sec:.1f}s ({t_sec/60:.1f} min)  [{tag}]")

    print()
    print("=" * 70)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)
